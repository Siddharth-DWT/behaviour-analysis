# backend/core/speaker_registry.py
"""
NEXUS Speaker Registry — cross-session speaker identity matching.
Copied verbatim from services/api_gateway/speaker_registry.py.
No import path changes required.
"""
import base64
import logging
import re
import uuid as _uuid
from typing import Optional

logger = logging.getLogger("nexus.gateway.speaker_registry")

# ── Similarity thresholds ─────────────────────────────────────────────────────
DEFAULT_SIMILARITY_THRESHOLD = 0.75
FACE_SIMILARITY_THRESHOLD    = 0.62
FUSED_SIMILARITY_THRESHOLD   = 0.50
DEDUP_THRESHOLD              = 0.50
VOICE_SIMILARITY_CEILING     = 0.97

# ── Fusion weights ────────────────────────────────────────────────────────────
FACE_WEIGHT  = 0.55
VOICE_WEIGHT = 0.45

# ── Label helpers ─────────────────────────────────────────────────────────────
GENERIC_LABEL = re.compile(r"^(Face|Speaker|Person)_\d+$")


def _meaningful_name(label) -> str:
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
    face_embeddings = face_embeddings or {}
    result: dict[str, dict] = {}

    for speaker_label, voice_embedding in speaker_embeddings.items():
        voice_emb_str = "[" + ",".join(str(v) for v in voice_embedding) + "]"

        face_data    = face_embeddings.get(speaker_label, {})
        face_emb     = face_data.get("embedding")
        face_emb_str = "[" + ",".join(str(v) for v in face_emb) + "]" if face_emb else None

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

        best_match   = None
        best_score   = 0.0
        match_method = "new_registration"

        if face_emb_str and voice_match and face_match:
            voice_id = str(voice_match["id"])
            face_id  = str(face_match["id"])
            if voice_id == face_id:
                combined = face_sim * FACE_WEIGHT + voice_sim * VOICE_WEIGHT
                if combined >= FUSED_SIMILARITY_THRESHOLD:
                    best_match   = voice_match
                    best_score   = combined
                    match_method = "face_voice_fused"
            else:
                if similarity_threshold <= voice_sim <= VOICE_SIMILARITY_CEILING:
                    best_match   = voice_match
                    best_score   = voice_sim
                    match_method = "voice_embedding"
                    logger.warning(
                        "Registry conflict for %s: voice=%s(%.3f) face=%s(%.3f) -> keeping voice identity",
                        speaker_label, voice_id, voice_sim, face_id, face_sim,
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

        speaker_info = next(
            (s for s in (voice_speakers or []) if s.get("speaker_id") == speaker_label), {}
        )

        if best_match:
            registry_id  = str(best_match["id"])
            display_name = best_match["display_name"]
            confidence   = best_score

            update_parts  = [
                "session_count      = session_count + 1",
                "voice_sample_count = voice_sample_count + 1",
                "last_seen_at       = NOW()",
                "updated_at         = NOW()",
            ]
            update_params = [best_match["id"]]

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
                registry_id   = str(near_dup["id"])
                existing_name = near_dup["display_name"] or ""
                new_name      = _meaningful_name(speaker_info.get("name"))
                display_name  = new_name if (new_name and (not existing_name or GENERIC_LABEL.match(existing_name))) else existing_name
                confidence    = float(near_dup["sim"])
                match_method  = "face_embedding"

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
                    "Registry dedup-merge: %s → %s (sim=%.3f)",
                    speaker_label, display_name or "(unnamed)", near_dup["sim"],
                )
            else:
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

        if face_data.get("thumbnail_b64") and match_method != "voice_embedding":
            await _store_thumbnail(pool, registry_id, session_id, face_data, best_score)

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


def _coerce_uuid(value: Optional[str]):
    if value is None:
        return None
    if isinstance(value, _uuid.UUID):
        return value
    try:
        return _uuid.UUID(str(value))
    except (ValueError, AttributeError):
        return None
