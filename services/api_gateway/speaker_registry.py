# services/api_gateway/speaker_registry.py
"""
NEXUS Speaker Registry — cross-session speaker identity matching.

Public functions called by _run_pipeline in main.py:

  match_or_create_speakers     — fused face + voice matching; falls back to
      voice-only when face embeddings are unavailable (backward-compatible).

  match_or_create_by_face_only — used for video-only sessions where no voice
      diarization was performed.

  update_appearance_stats      — back-fills per-session stats after all signals
      are persisted.
"""
import base64
import logging
import re
import uuid as _uuid
from typing import Optional

logger = logging.getLogger("nexus.gateway.speaker_registry")

# ── Similarity thresholds ─────────────────────────────────────────────────────
DEFAULT_SIMILARITY_THRESHOLD = 0.75   # voice-only cosine similarity floor
FACE_SIMILARITY_THRESHOLD    = 0.62   # was 0.55; below ArcFace's reliable operating range
FUSED_SIMILARITY_THRESHOLD   = 0.50   # lower bar when both modalities agree
DEDUP_THRESHOLD              = 0.50   # near-duplicate check at INSERT time; lower than match
                                      # threshold by design — false negatives here are permanent
                                      # (creates duplicate row), false positives are recoverable
                                      # via the existing /speakers/{id}/merge endpoint.
# MFCC fallback embeddings capture recording environment, not just speaker identity.
# Cosine similarities > 0.97 between DIFFERENT speakers are MFCC artifacts — multiple
# speakers in the same call often all score 99%+ against each other.
# Ceiling prevents all speakers collapsing to one registry entry.
VOICE_SIMILARITY_CEILING     = 0.97   # above this → likely MFCC artifact, create new

# ── Fusion weights (face is more discriminative than MFCC voice) ──────────────
FACE_WEIGHT  = 0.55
VOICE_WEIGHT = 0.45

# ── Label helpers ─────────────────────────────────────────────────────────────
GENERIC_LABEL = re.compile(r"^(Face|Speaker|Person)_\d+$")


def _meaningful_name(label) -> str:
    """Return label only if it's a real human name. Empty string otherwise."""
    if not label:
        return ""
    if GENERIC_LABEL.match(label):
        return ""
    return label


async def match_or_create_speakers(
    pool,
    session_id: str,
    speaker_embeddings: dict[str, list[float]],
    voice_speakers: list[dict],
    speaker_map: dict[str, str],
    org_id: str,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    face_embeddings: Optional[dict[str, dict]] = None,
) -> dict[str, dict]:
    """
    Match speakers to persistent identities using voice embeddings,
    face embeddings, or both.

    Matching priority:
      1. Fused (face + voice, same person): weighted score >= 0.50 → match
      2. Face-only: cosine similarity >= 0.62 → match
      3. Voice-only: cosine similarity >= 0.75 → match (existing behaviour)
      4. No match → dedup check at 0.50, then INSERT if no near-duplicate found

    Fused score = face_similarity * 0.55 + voice_similarity * 0.45.
    Face is weighted higher because ArcFace is more discriminative than MFCC.

    Returns:
        {speaker_label: {registry_id, display_name, match_method, confidence,
                         voice_similarity, face_similarity}}
    """
    face_embeddings = face_embeddings or {}
    result: dict[str, dict] = {}

    for speaker_label, voice_embedding in speaker_embeddings.items():
        voice_emb_str = "[" + ",".join(str(v) for v in voice_embedding) + "]"

        face_data    = face_embeddings.get(speaker_label, {})
        face_emb     = face_data.get("embedding")
        face_emb_str = "[" + ",".join(str(v) for v in face_emb) + "]" if face_emb else None

        # ── Query voice match ─────────────────────────────────────────────
        voice_match = await pool.fetchrow("""
            SELECT id, display_name,
                   1 - (voice_embedding <=> $1::vector) AS voice_sim
            FROM   speakers_registry
            WHERE  org_id = $2
              AND  voice_embedding IS NOT NULL
            ORDER  BY voice_embedding <=> $1::vector
            LIMIT  1
        """, voice_emb_str, _coerce_uuid(org_id))

        voice_sim = float(voice_match["voice_sim"]) if voice_match else 0.0

        # ── Query face match ──────────────────────────────────────────────
        face_sim   = 0.0
        face_match = None
        if face_emb_str:
            face_match = await pool.fetchrow("""
                SELECT id, display_name,
                       1 - (face_embedding <=> $1::vector) AS face_sim
                FROM   speakers_registry
                WHERE  org_id = $2
                  AND  face_embedding IS NOT NULL
                ORDER  BY face_embedding <=> $1::vector
                LIMIT  1
            """, face_emb_str, _coerce_uuid(org_id))
            face_sim = float(face_match["face_sim"]) if face_match else 0.0

        # ── Decide: fused / face-only / voice-only / new ─────────────────
        best_match   = None
        best_score   = 0.0
        match_method = "new_registration"

        if face_emb_str and voice_match and face_match:
            voice_id = str(voice_match["id"])
            face_id  = str(face_match["id"])
            if voice_id == face_id:
                # Both modalities point to the same person — fused score
                combined = face_sim * FACE_WEIGHT + voice_sim * VOICE_WEIGHT
                if combined >= FUSED_SIMILARITY_THRESHOLD:
                    best_match   = voice_match
                    best_score   = combined
                    match_method = "face_voice_fused"
            else:
                # For a speaking Speaker_N, strong voice evidence defines the
                # identity. Face evidence may support it, but should not override
                # a conflicting strong voice match.
                if similarity_threshold <= voice_sim <= VOICE_SIMILARITY_CEILING:
                    best_match   = voice_match
                    best_score   = voice_sim
                    match_method = "voice_embedding"
                    logger.warning(
                        "Registry conflict for %s: voice=%s(%.3f) face=%s(%.3f) "
                        "-> keeping voice identity",
                        speaker_label,
                        voice_id, voice_sim,
                        face_id, face_sim,
                    )
                elif face_sim >= FACE_SIMILARITY_THRESHOLD:
                    best_match   = face_match
                    best_score   = face_sim
                    match_method = "face_embedding"
        elif face_emb_str and face_match and face_sim >= FACE_SIMILARITY_THRESHOLD:
            best_match   = face_match
            best_score   = face_sim
            match_method = "face_embedding"
        elif voice_match and similarity_threshold <= voice_sim <= VOICE_SIMILARITY_CEILING:
            best_match   = voice_match
            best_score   = voice_sim
            match_method = "voice_embedding"

        # ── Apply match or create ─────────────────────────────────────────
        # Lookup speaker_info once, before deciding match vs INSERT.
        speaker_info = next(
            (s for s in (voice_speakers or []) if s.get("speaker_id") == speaker_label), {}
        )

        if best_match:
            registry_id  = str(best_match["id"])
            display_name = best_match["display_name"]
            confidence   = best_score

            # Update registry — always increment voice; update face only if NULL
            update_parts  = [
                "session_count      = session_count + 1",
                "voice_sample_count = voice_sample_count + 1",
                "last_seen_at       = NOW()",
                "updated_at         = NOW()",
            ]
            update_params = [best_match["id"]]

            # Upgrade display_name only when the registry has no real name yet.
            # Never overwrite a human-meaningful name — admin corrections take priority
            # over automated diarization output (which can be wrong or hallucinated).
            new_name = _meaningful_name(speaker_info.get("name")) if voice_speakers else ""
            if new_name:
                existing_name = best_match["display_name"] or ""
                if not existing_name or GENERIC_LABEL.match(existing_name):
                    p = len(update_params) + 1
                    update_parts.append(f"display_name = ${p}")
                    update_params.append(new_name)
                    display_name = new_name

            if face_emb_str:
                p = len(update_params) + 1
                update_parts.append(
                    f"face_embedding    = CASE WHEN face_embedding IS NULL "
                    f"THEN ${p}::vector ELSE face_embedding END"
                )
                update_parts.append("face_sample_count = face_sample_count + 1")
                update_params.append(face_emb_str)

            # voice_embedding: write when NULL — fixes face-only → speaking upgrade path
            # (person was listener in session 1, speaks in session 2)
            p = len(update_params) + 1
            update_parts.append(
                f"voice_embedding = CASE WHEN voice_embedding IS NULL "
                f"THEN ${p}::vector ELSE voice_embedding END"
            )
            update_params.append(voice_emb_str)

            await pool.execute(
                f"UPDATE speakers_registry SET {', '.join(update_parts)} WHERE id = $1",
                *update_params,
            )
            logger.info(
                "Registry match: %s → %s (method=%s score=%.3f voice=%.3f face=%.3f)",
                speaker_label, display_name, match_method, confidence, voice_sim, face_sim,
            )
        else:
            # Before creating a new row, look for a near-duplicate by face embedding.
            # Dedup applies only when we have a face embedding to compare; voice-only
            # registrations skip dedup because MFCC false-positives (see VOICE_SIMILARITY_CEILING)
            # would cause cross-speaker merges.
            near_dup = None
            if face_emb_str:
                near_dup = await pool.fetchrow("""
                    SELECT id, display_name,
                           1 - (face_embedding <=> $1::vector) AS sim
                    FROM   speakers_registry
                    WHERE  org_id = $2 AND face_embedding IS NOT NULL
                    ORDER  BY face_embedding <=> $1::vector
                    LIMIT  1
                """, face_emb_str, _coerce_uuid(org_id))

            if near_dup and float(near_dup["sim"]) >= DEDUP_THRESHOLD:
                # Same person, just below the match threshold. Merge into the existing row.
                registry_id   = str(near_dup["id"])
                existing_name = near_dup["display_name"] or ""
                new_name      = _meaningful_name(speaker_info.get("name"))
                display_name  = new_name if (new_name and (not existing_name or GENERIC_LABEL.match(existing_name))) else existing_name
                confidence    = float(near_dup["sim"])
                match_method  = "face_embedding"

                # Update the existing row — add voice if missing, increment counters.
                dup_parts = [
                    "session_count      = session_count + 1",
                    "voice_sample_count = voice_sample_count + 1",
                    "face_sample_count  = face_sample_count + 1",
                    "last_seen_at       = NOW()",
                    "updated_at         = NOW()",
                    "voice_embedding    = CASE WHEN voice_embedding IS NULL THEN $2::vector ELSE voice_embedding END",
                ]
                dup_params = [near_dup["id"], voice_emb_str]
                if display_name != existing_name:
                    dup_parts.append("display_name = $3")
                    dup_params.append(display_name)
                await pool.execute(
                    f"UPDATE speakers_registry SET {', '.join(dup_parts)} WHERE id = $1",
                    *dup_params,
                )
                logger.info(
                    "Registry dedup-merge: %s → %s (sim=%.3f, avoided INSERT)",
                    speaker_label, display_name or "(unnamed)", near_dup["sim"],
                )
            else:
                # No near-duplicate. Create a new row.
                display_name = _meaningful_name(speaker_info.get("name")) or _meaningful_name(speaker_label)
                row = await pool.fetchrow("""
                    INSERT INTO speakers_registry
                        (org_id, display_name, role,
                         voice_embedding, voice_sample_count,
                         face_embedding,  face_sample_count,
                         session_count)
                    VALUES ($1, $2, $3, $4::vector, 1, $5::vector, $6, 1)
                    RETURNING id
                """,
                    _coerce_uuid(org_id),
                    display_name,
                    speaker_info.get("role", ""),
                    voice_emb_str,
                    face_emb_str,
                    1 if face_emb_str else 0,
                )
                registry_id  = str(row["id"])
                confidence   = 1.0
                match_method = "new_registration"
                logger.info(
                    "Registry new: %s registered as %r (has_face=%s)",
                    speaker_label, display_name or "(unnamed)", bool(face_emb_str),
                )

        # ── Store face thumbnail ──────────────────────────────────────────
        # Only when the match used face evidence — a voice-only match means the
        # thumbnail came from lip-sync and could belong to a DIFFERENT person than
        # the registry entry (MFCC false-positives store wrong faces under wrong IDs).
        if face_data.get("thumbnail_b64") and match_method != "voice_embedding":
            await _store_thumbnail(pool, registry_id, session_id, face_data, best_score)

        # ── Appearance record ─────────────────────────────────────────────
        await _upsert_appearance(
            pool, registry_id, session_id,
            speaker_map.get(speaker_label), speaker_label,
            match_method, confidence,
        )

        result[speaker_label] = {
            "registry_id":      registry_id,
            "display_name":     display_name,
            "match_method":     match_method,
            "confidence":       confidence,
            "voice_similarity": round(voice_sim, 4),
            "face_similarity":  round(face_sim, 4),
        }

    return result


async def match_or_create_by_face_only(
    pool,
    session_id: str,
    face_embeddings: dict[str, dict],
    speaker_map: dict[str, str],
    org_id: str,
) -> dict[str, dict]:
    """
    Match speakers using face embeddings only — used when no voice diarization
    is available (video-only sessions, screen recordings with webcam).
    """
    result: dict[str, dict] = {}

    for speaker_label, face_data in face_embeddings.items():
        embedding = face_data.get("embedding")
        if not embedding:
            continue

        emb_str = "[" + ",".join(str(v) for v in embedding) + "]"

        match = await pool.fetchrow("""
            SELECT id, display_name,
                   1 - (face_embedding <=> $1::vector) AS similarity
            FROM   speakers_registry
            WHERE  org_id = $2
              AND  face_embedding IS NOT NULL
            ORDER  BY face_embedding <=> $1::vector
            LIMIT  1
        """, emb_str, _coerce_uuid(org_id))

        if match and float(match["similarity"]) >= FACE_SIMILARITY_THRESHOLD:
            registry_id  = str(match["id"])
            display_name = match["display_name"]
            confidence   = float(match["similarity"])
            match_method = "face_embedding"
            await pool.execute("""
                UPDATE speakers_registry
                SET session_count     = session_count + 1,
                    face_sample_count = face_sample_count + 1,
                    last_seen_at      = NOW(),
                    updated_at        = NOW()
                WHERE id = $1
            """, match["id"])
        else:
            # Before INSERT, check for near-duplicate by face embedding.
            near_dup = await pool.fetchrow("""
                SELECT id, display_name,
                       1 - (face_embedding <=> $1::vector) AS sim
                FROM   speakers_registry
                WHERE  org_id = $2 AND face_embedding IS NOT NULL
                ORDER  BY face_embedding <=> $1::vector
                LIMIT  1
            """, emb_str, _coerce_uuid(org_id))

            if near_dup and float(near_dup["sim"]) >= DEDUP_THRESHOLD:
                registry_id  = str(near_dup["id"])
                display_name = near_dup["display_name"] or ""
                confidence   = float(near_dup["sim"])
                match_method = "face_embedding"
                await pool.execute("""
                    UPDATE speakers_registry
                    SET session_count     = session_count + 1,
                        face_sample_count = face_sample_count + 1,
                        last_seen_at      = NOW(),
                        updated_at        = NOW()
                    WHERE id = $1
                """, near_dup["id"])
                logger.info(
                    "Face-only dedup-merge: %s → %s (sim=%.3f)",
                    speaker_label, display_name or "(unnamed)", near_dup["sim"],
                )
            else:
                row = await pool.fetchrow("""
                    INSERT INTO speakers_registry
                        (org_id, display_name, face_embedding, face_sample_count, session_count)
                    VALUES ($1, $2, $3::vector, 1, 1)
                    RETURNING id
                """, _coerce_uuid(org_id), _meaningful_name(speaker_label), emb_str)
                registry_id  = str(row["id"])
                display_name = _meaningful_name(speaker_label)
                confidence   = 1.0
                match_method = "new_registration"

        if face_data.get("thumbnail_b64"):
            await _store_thumbnail(pool, registry_id, session_id, face_data, confidence)

        await _upsert_appearance(
            pool, registry_id, session_id,
            speaker_map.get(speaker_label), speaker_label,
            match_method, confidence,
        )

        result[speaker_label] = {
            "registry_id":  registry_id,
            "display_name": display_name,
            "match_method": match_method,
            "confidence":   confidence,
        }

    return result


async def _store_thumbnail(
    pool, registry_id: str, session_id: str, face_data: dict, quality_score: float
) -> None:
    """Insert a face thumbnail; mark as primary when none exists yet."""
    try:
        thumb_bytes = base64.b64decode(face_data["thumbnail_b64"])
        await pool.execute("""
            INSERT INTO face_thumbnails
                (registry_id, session_id, thumbnail, quality_score, is_primary)
            VALUES ($1, $2, $3, $4,
                    NOT EXISTS(SELECT 1 FROM face_thumbnails WHERE registry_id = $1))
        """,
            _coerce_uuid(registry_id),
            _coerce_uuid(session_id),
            thumb_bytes,
            quality_score,
        )
    except Exception as exc:
        logger.debug(f"Thumbnail store failed (non-fatal): {exc}")


async def _upsert_appearance(
    pool,
    registry_id: str,
    session_id: str,
    speaker_db_id: Optional[str],
    speaker_label: str,
    match_method: str,
    confidence: float,
) -> None:
    """Insert or update a speaker_appearances row for this session."""
    await pool.execute("""
        INSERT INTO speaker_appearances
            (registry_id, session_id, speaker_id,
             speaker_label, match_method, match_confidence)
        VALUES ($1, $2, $3, $4, $5, $6)
        ON CONFLICT (registry_id, session_id, speaker_label) DO UPDATE
            SET match_method     = EXCLUDED.match_method,
                match_confidence = EXCLUDED.match_confidence
    """,
        _coerce_uuid(registry_id),
        _coerce_uuid(session_id),
        _coerce_uuid(speaker_db_id) if speaker_db_id else None,
        speaker_label,
        match_method,
        confidence,
    )


async def update_appearance_stats(pool, session_id: str) -> None:
    """
    Back-fill speaker_appearances with aggregate stats computed from the
    signals table for this session.

    Call this after ALL agent signals (voice, language, conversation,
    video, fusion) have been persisted so the aggregates are complete.

    Stats updated:
      avg_stress       — mean vocal_stress_score value
      avg_engagement   — mean conversation_engagement value
      filler_rate      — mean filler_detection value
    """
    await pool.execute(
        """
        UPDATE speaker_appearances sa
        SET    avg_stress     = sub.avg_stress,
               avg_engagement = sub.avg_engagement,
               filler_rate    = sub.filler_rate
        FROM (
            SELECT
                sp.speaker_label,
                AVG(CASE WHEN s.signal_type = 'vocal_stress_score'      THEN s.value END) AS avg_stress,
                AVG(CASE WHEN s.signal_type = 'conversation_engagement'  THEN s.value END) AS avg_engagement,
                AVG(CASE WHEN s.signal_type = 'filler_detection'         THEN s.value END) AS filler_rate
            FROM   signals  s
            JOIN   speakers sp ON sp.id = s.speaker_id
            WHERE  s.session_id = $1
            GROUP  BY sp.speaker_label
        ) sub
        WHERE  sa.session_id   = $1
          AND  sa.speaker_label = sub.speaker_label
        """,
        _coerce_uuid(session_id),
    )
    logger.debug("Appearance stats updated for session %s", session_id)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _coerce_uuid(value: Optional[str]):
    """Convert a string UUID to asyncpg-compatible uuid.UUID, or return None."""
    if value is None:
        return None
    if isinstance(value, _uuid.UUID):
        return value
    try:
        return _uuid.UUID(str(value))
    except (ValueError, AttributeError):
        return None
