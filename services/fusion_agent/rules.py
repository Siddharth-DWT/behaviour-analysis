# services/fusion_agent/rules.py
"""
NEXUS Fusion Agent - Pairwise Cross-Modal Rules
Phase 1 (audio-only) + Phase 2E (audio × video) pairwise rules.

Each rule takes signals from two or more agents within a temporal window
and detects patterns that neither agent can detect alone.

Rules implemented:
  Audio-only (Phase 1):
    FUSION-02: Speech Content × Voice Stress → Credibility assessment
    FUSION-07: Hedge Language × Positive Sentiment → Incongruence detection
    FUSION-13: Persuasion Language × Speech Pace → Urgency authenticity

  Audio × Video (Phase 2E):
    FUSION-01: Tone × Facial Emotion → Masking detection
    FUSION-03: Voice Stress × Facial Stress → Suppression detection
    FUSION-04: Filler Words × Gaze Break → Cognitive load
    FUSION-05: Head Nod/Shake × Objection → Disagreement signal
    FUSION-06: Body Lean × Attention Level → Physical engagement
    FUSION-08: Gaze Break × Hedge Language → False confidence
    FUSION-09: Smile Quality × Negative Sentiment → Sarcasm / masking
    FUSION-10: Response Latency × Facial Stress → Cognitive processing
    FUSION-11: Dominance Score × Gaze Avoidance → Anxiety under pressure
    FUSION-12: Interruption × Body Lean → Intent (cooperative vs competitive)
    FUSION-14: Empathy Language × Head Nod → Rapport confirmation

Research references:
  - Bond & DePaulo 2006 (deception detection accuracy ceiling)
  - Levine 2014 (truth-default theory)
  - Rackham 1988 (persuasion + pace in sales)
  - Lakoff 1975 (powerless language as incongruence marker)
  - Ekman 1969 (leakage hypothesis — face leaks suppressed emotion)
  - Mehrabian 1972 (body language channels)
  - Tickle-Degnen & Rosenthal 1990 (rapport = nodding + gaze + lean)
  - Navarro 2008 (body language clusters)

Confidence caps:
  FUSION-01: 0.65   FUSION-02: 0.65   FUSION-03: 0.65
  FUSION-04: 0.70   FUSION-05: 0.55   FUSION-06: 0.60
  FUSION-07: 0.70   FUSION-08: 0.55   FUSION-09: 0.60
  FUSION-10: 0.60   FUSION-11: 0.65   FUSION-12: 0.55
  FUSION-13: 0.60   FUSION-14: 0.70
"""
import logging
from typing import Optional

try:
    from shared.config.content_type_profile import ContentTypeProfile
except ImportError:
    ContentTypeProfile = None

try:
    from shared.utils.conversions import to_float as _to_float, to_int as _to_int
except ImportError:
    def _to_float(v) -> float:
        if v is None or v == "":
            return 0.0
        try:
            return float(v)
        except (ValueError, TypeError):
            return 0.0

    def _to_int(v) -> int:
        if v is None or v == "":
            return 0
        try:
            return int(float(v))
        except (ValueError, TypeError):
            return 0

logger = logging.getLogger("nexus.fusion.rules")


class FusionRuleEngine:
    """
    Evaluates cross-modal pairwise rules by correlating signals
    from different agents within temporal windows.
    """

    def evaluate(
        self,
        speaker_id: str,
        voice_signals: list[dict],
        language_signals: list[dict],
        video_signals: Optional[list[dict]] = None,
        window_start_ms: int = 0,
        window_end_ms: int = 0,
        content_type: str = "sales_call",
        profile: "ContentTypeProfile | None" = None,
    ) -> list[dict]:
        """
        Run all fusion rules for a speaker given their recent signals.

        Args:
            speaker_id: Speaker identifier
            voice_signals: Recent voice agent signals for this speaker
            language_signals: Recent language agent signals for this speaker
            video_signals: Recent video agent signals (facial/gaze/body) — Phase 2E
            window_start_ms: Fusion window start
            window_end_ms: Fusion window end
            content_type: Meeting type for content-aware gating
            profile: ContentTypeProfile instance (created from content_type if not passed)

        Returns:
            List of fusion signal dicts
        """
        if profile is None and ContentTypeProfile is not None:
            profile = ContentTypeProfile(content_type)

        video_signals = video_signals or []
        signals = []

        def _maybe_append(rule_id: str, result: dict | None, signal_type: str):
            if result is None:
                return
            if profile and profile.is_gated(rule_id):
                return
            conf = round(result["confidence"], 4)
            if profile:
                conf = round(profile.apply_confidence(rule_id, conf), 4)
            renamed = profile.rename_signal(signal_type) if profile else signal_type
            signals.append({
                "agent": "fusion",
                "speaker_id": speaker_id,
                "signal_type": renamed,
                "value": round(result["score"], 4),
                "value_text": result["level"],
                "confidence": conf,
                "window_start_ms": window_start_ms,
                "window_end_ms": window_end_ms,
                "metadata": result["evidence"],
            })

        # ── FUSION-02: Content × Stress → Stress-Sentiment Incongruence ──
        if not (profile and profile.is_gated("FUSION-02")):
            max_conf = profile.get_threshold("FUSION-02", "max_confidence", 0.65) if profile else 0.65
            cred = self._rule_fusion_02(voice_signals, language_signals, max_confidence=max_conf)
            _maybe_append("FUSION-02", cred, "stress_sentiment_incongruence")

        # ── FUSION-07: Hedge × Positive Sentiment → Incongruence ──
        if not (profile and profile.is_gated("FUSION-07")):
            conf_floor = profile.get_threshold("FUSION-07", "confidence_floor", 0.10) if profile else 0.10
            max_conf = profile.get_threshold("FUSION-07", "max_confidence", 0.70) if profile else 0.70
            incong = self._rule_fusion_07(language_signals, voice_signals, content_type, conf_floor=conf_floor, max_conf=max_conf)
            _maybe_append("FUSION-07", incong, "verbal_incongruence")

        # ── FUSION-13: Persuasion × Pace → Urgency Authenticity ──
        if not (profile and profile.is_gated("FUSION-13")):
            urg = self._rule_fusion_13(voice_signals, language_signals, content_type)
            _maybe_append("FUSION-13", urg, "urgency_authenticity")

        # ── Phase 2E: Audio × Video pairwise rules (skipped if no video signals) ──
        if video_signals:
            # FUSION-01: Tone × Facial Emotion → Masking
            if not (profile and profile.is_gated("FUSION-01")):
                max_conf = profile.get_threshold("FUSION-01", "max_confidence", 0.65) if profile else 0.65
                _maybe_append("FUSION-01", self._rule_fusion_01(voice_signals, video_signals, max_conf), "tone_face_masking")

            # FUSION-03: Voice Stress × Facial Stress → Suppression
            if not (profile and profile.is_gated("FUSION-03")):
                max_conf = profile.get_threshold("FUSION-03", "max_confidence", 0.65) if profile else 0.65
                _maybe_append("FUSION-03", self._rule_fusion_03(voice_signals, video_signals, max_conf), "stress_suppression")

            # FUSION-04: Filler × Gaze Break → Cognitive Load
            if not (profile and profile.is_gated("FUSION-04")):
                max_conf = profile.get_threshold("FUSION-04", "max_confidence", 0.70) if profile else 0.70
                _maybe_append("FUSION-04", self._rule_fusion_04(voice_signals, video_signals, max_conf), "cognitive_load")

            # FUSION-05: Head Nod/Shake × Objection → Disagreement
            if not (profile and profile.is_gated("FUSION-05")):
                max_conf = profile.get_threshold("FUSION-05", "max_confidence", 0.55) if profile else 0.55
                _maybe_append("FUSION-05", self._rule_fusion_05(language_signals, video_signals, max_conf), "nonverbal_disagreement")

            # FUSION-06: Body Lean × Attention Level → Physical Engagement
            if not (profile and profile.is_gated("FUSION-06")):
                max_conf = profile.get_threshold("FUSION-06", "max_confidence", 0.60) if profile else 0.60
                _maybe_append("FUSION-06", self._rule_fusion_06(video_signals, max_conf), "physical_engagement")

            # FUSION-08: Gaze Break × Hedge Language → False Confidence
            if not (profile and profile.is_gated("FUSION-08")):
                max_conf = profile.get_threshold("FUSION-08", "max_confidence", 0.55) if profile else 0.55
                _maybe_append("FUSION-08", self._rule_fusion_08(language_signals, video_signals, max_conf), "false_confidence")

            # FUSION-09: Smile Quality × Negative Sentiment → Masking / Sarcasm
            if not (profile and profile.is_gated("FUSION-09")):
                max_conf = profile.get_threshold("FUSION-09", "max_confidence", 0.60) if profile else 0.60
                _maybe_append("FUSION-09", self._rule_fusion_09(language_signals, video_signals, max_conf), "smile_sentiment_incongruence")

            # FUSION-10: Response Latency × Facial Stress → Processing Load
            if not (profile and profile.is_gated("FUSION-10")):
                max_conf = profile.get_threshold("FUSION-10", "max_confidence", 0.60) if profile else 0.60
                _maybe_append("FUSION-10", self._rule_fusion_10(language_signals, video_signals, max_conf), "processing_load")

            # FUSION-11: Dominance Score × Gaze Avoidance → Anxiety Under Pressure
            if not (profile and profile.is_gated("FUSION-11")):
                max_conf = profile.get_threshold("FUSION-11", "max_confidence", 0.65) if profile else 0.65
                _maybe_append("FUSION-11", self._rule_fusion_11(language_signals, video_signals, max_conf), "dominance_anxiety")

            # FUSION-12: Interruption × Body Lean → Interrupt Intent
            if not (profile and profile.is_gated("FUSION-12")):
                max_conf = profile.get_threshold("FUSION-12", "max_confidence", 0.55) if profile else 0.55
                _maybe_append("FUSION-12", self._rule_fusion_12(language_signals, voice_signals, video_signals, max_conf), "interrupt_intent")

            # FUSION-14: Empathy Language × Head Nod → Rapport
            if not (profile and profile.is_gated("FUSION-14")):
                max_conf = profile.get_threshold("FUSION-14", "max_confidence", 0.70) if profile else 0.70
                _maybe_append("FUSION-14", self._rule_fusion_14(language_signals, video_signals, max_conf), "rapport_confirmation")

        return signals

    # ════════════════════════════════════════════════════════
    # FUSION-02: Speech Content × Voice Stress → Stress-Sentiment Incongruence
    # Research: Bond & DePaulo 2006, Levine 2014
    # Max confidence: 0.65 (raised — stress-sentiment gap ≠ deception claim)
    # ════════════════════════════════════════════════════════

    def _rule_fusion_02(
        self,
        voice_signals: list[dict],
        language_signals: list[dict],
        max_confidence: float = 0.65,
    ) -> Optional[dict]:
        """
        Cross-modal stress-sentiment incongruence check:
        When the CONTENT says something positive/certain but the VOICE
        shows elevated stress, the gap flags incongruence for human review.

        Scans all time-aligned stress+sentiment pairs (within 10s window)
        and returns the strongest credibility concern found.

        Does NOT claim deception — only flags incongruence for human review.
        """
        stress_signals = [
            s for s in voice_signals
            if s.get("signal_type") == "vocal_stress_score"
        ]
        sentiment_signals = [
            s for s in language_signals
            if s.get("signal_type") == "sentiment_score"
        ]

        if not stress_signals or not sentiment_signals:
            return None

        # Check for reinforcing signals (fillers, pitch elevation)
        filler_signals = [
            s for s in voice_signals
            if s.get("signal_type") == "filler_detection"
            and s.get("value_text") in ("filler_spike", "filler_elevated")
        ]
        pitch_signals = [
            s for s in voice_signals
            if s.get("signal_type") == "pitch_elevation_flag"
        ]
        reinforcing_count = (1 if filler_signals else 0) + (1 if pitch_signals else 0)

        # Scan all time-aligned pairs within 10s window
        ALIGN_MS = 10_000
        best_result = None
        best_gap = 0

        for stress_sig in stress_signals:
            stress_value = _to_float(stress_sig.get("value", 0))
            if stress_value <= 0.40:
                continue
            stress_time = _to_int(stress_sig.get("window_start_ms", 0))

            for sent_sig in sentiment_signals:
                sentiment_value = _to_float(sent_sig.get("value", 0))
                if sentiment_value <= 0.30:
                    continue
                sent_time = _to_int(sent_sig.get("window_start_ms", 0))

                # Time alignment check
                if abs(stress_time - sent_time) > ALIGN_MS:
                    continue

                # Credibility gap
                gap = (sentiment_value - 0.30) * (stress_value - 0.30)
                if gap <= best_gap:
                    continue
                best_gap = gap

                credibility_score = 1.0 - min(gap / 0.49, 1.0)

                if credibility_score >= 0.70:
                    continue

                if credibility_score < 0.40:
                    level = "credibility_concern"
                elif credibility_score < 0.60:
                    level = "mild_incongruence"
                else:
                    level = "mostly_congruent"

                raw_confidence = min(0.45 + (1.0 - credibility_score) * 0.20, max_confidence)
                if reinforcing_count > 0:
                    raw_confidence = min(raw_confidence + 0.05 * reinforcing_count, max_confidence)

                best_result = {
                    "score": credibility_score,
                    "level": level,
                    "confidence": raw_confidence,
                    "evidence": {
                        "sentiment_value": round(sentiment_value, 3),
                        "stress_value": round(stress_value, 3),
                        "gap_magnitude": round(1.0 - credibility_score, 3),
                        "reinforcing_signals": reinforcing_count,
                        "filler_elevated": len(filler_signals) > 0,
                        "pitch_elevated": len(pitch_signals) > 0,
                        "aligned_window_ms": abs(stress_time - sent_time),
                    },
                }

        return best_result

    # ════════════════════════════════════════════════════════
    # FUSION-07: Hedge Language × Positive Sentiment → Incongruence
    # Audio-only simplification of Head Shake × Affirmative Language
    # Research: Lakoff 1975, Navarro 2008
    # Max confidence: 0.70
    # ════════════════════════════════════════════════════════

    def _rule_fusion_07(
        self,
        language_signals: list[dict],
        voice_signals: list[dict] = None,
        content_type: str = "sales_call",
        conf_floor: float = 0.10,
        max_conf: float = 0.70,
    ) -> Optional[dict]:
        """
        Audio-only adaptation of FUSION-07 (Head Shake × Affirmative Language).

        True incongruence requires ALL of:
          a) Sentiment > +0.4 (clearly positive)
          b) At least ONE of: objection (conf > 0.5), power < 0.25, filler_rate > 3%
          c) Voice stress > 0.35 for same speaker/window
          d) Combined confidence > 0.45

        If only (a)+(b) met but NOT (c): downgrade to "hedged_agreement"
        with confidence capped at 0.35 — logged but not alerted.
        """
        voice_signals = voice_signals or []

        sentiment_signals = [
            s for s in language_signals
            if s.get("signal_type") == "sentiment_score"
        ]
        power_signals = [
            s for s in language_signals
            if s.get("signal_type") == "power_language_score"
        ]

        if not sentiment_signals:
            return None

        latest_sentiment = max(
            sentiment_signals, key=lambda s: _to_int(s.get("window_start_ms", 0))
        )
        sentiment_value = _to_float(latest_sentiment.get("value", 0))

        # (a) Sentiment must be clearly positive (> 0.4, not just > 0.3)
        if sentiment_value <= 0.40:
            return None

        # Get power value
        power_value = 0.5
        if power_signals:
            latest_power = max(
                power_signals, key=lambda s: _to_int(s.get("window_start_ms", 0))
            )
            power_value = _to_float(latest_power.get("value", 0.5))

        # (b) At least ONE linguistic marker:
        #   - objection with confidence > 0.5
        #   - power < 0.25 (very weak)
        #   - filler elevated
        objection_signals = [
            s for s in language_signals
            if s.get("signal_type") == "objection_signal"
            and _to_float(s.get("confidence", 0)) > 0.50
        ]
        filler_signals = [
            s for s in voice_signals
            if s.get("signal_type") == "filler_detection"
            and s.get("value_text") in ("filler_spike", "filler_elevated", "elevated", "high")
        ]

        has_objection = len(objection_signals) > 0
        has_weak_power = power_value < 0.25
        has_filler_spike = len(filler_signals) > 0

        if not (has_objection or has_weak_power or has_filler_spike):
            return None

        # (c) Check voice stress > 0.35
        stress_signals = [
            s for s in voice_signals
            if s.get("signal_type") == "vocal_stress_score"
        ]
        stress_value = 0.0
        if stress_signals:
            latest_stress = max(
                stress_signals, key=lambda s: _to_int(s.get("window_start_ms", 0))
            )
            stress_value = _to_float(latest_stress.get("value", 0))

        has_voice_stress = stress_value > 0.35

        # Build evidence
        evidence = {
            "sentiment_value": round(sentiment_value, 3),
            "power_value": round(power_value, 3),
            "stress_value": round(stress_value, 3),
            "objection_present": has_objection,
            "weak_power": has_weak_power,
            "filler_spike": has_filler_spike,
            "voice_stress_aligned": has_voice_stress,
        }

        # Score based on how many markers fire
        marker_count = sum([has_objection, has_weak_power, has_filler_spike])
        incongruence = (sentiment_value - 0.40) * (marker_count * 0.20)
        score = min(incongruence / 0.36, 1.0)

        if not has_voice_stress:
            # Downgrade: polite hedging, not true incongruence
            hedged_conf = min(0.35, score * 0.50)
            if hedged_conf < conf_floor:
                return None  # Below noise floor
            return {
                "score": min(score, 0.30),
                "level": "hedged_agreement",
                "confidence": hedged_conf,
                "evidence": evidence,
            }

        # True incongruence: sentiment + markers + voice stress
        if score > 0.60:
            level = "strong_verbal_incongruence"
        elif score > 0.35:
            level = "moderate_verbal_incongruence"
        else:
            level = "mild_verbal_incongruence"

        raw_confidence = min(0.40 + score * 0.30, max_conf)

        # (d) Minimum confidence threshold
        if raw_confidence < conf_floor:
            return None

        return {
            "score": score,
            "level": level,
            "confidence": raw_confidence,
            "evidence": evidence,
        }

    # ════════════════════════════════════════════════════════
    # FUSION-13: Persuasion Language × Speech Pace → Urgency
    # Research: Rackham 1988, Apple et al. 1979
    # Max confidence: 0.60
    # ════════════════════════════════════════════════════════

    def _rule_fusion_13(
        self,
        voice_signals: list[dict],
        language_signals: list[dict],
        content_type: str = "sales_call",
    ) -> Optional[dict]:
        """
        Urgency authenticity check:
        When someone uses buying/persuasion language while also speaking
        faster than baseline, are they genuinely excited or artificially
        creating urgency?

        Logic:
          IF buying_signal or positive_intent detected
             AND speech_rate_anomaly = rate_elevated
          THEN assess urgency authenticity

        Authentic urgency:
          - Rate elevated + tone confident + sentiment positive
          → genuine excitement about the topic

        Manufactured urgency:
          - Rate elevated + stress elevated + filler elevated
          → artificial pressure or anxiety-driven rushing

        This is commercially valuable: it helps distinguish a salesperson
        genuinely enthusiastic about their product from one artificially
        creating time pressure.
        """
        # Need rate anomaly signal (elevated = speaking faster than baseline)
        rate_signals = [
            s for s in voice_signals
            if s.get("signal_type") == "speech_rate_anomaly"
            and s.get("value_text") == "rate_elevated"
        ]
        if not rate_signals:
            return None

        # Need persuasion/engagement language:
        # buying signals, specific intents, OR strong positive sentiment
        buying_signals = [
            s for s in language_signals
            if s.get("signal_type") == "buying_signal"
        ]
        intent_signals = [
            s for s in language_signals
            if s.get("signal_type") == "intent_classification"
            and s.get("value_text") in ("PROPOSE", "CLOSE", "NEGOTIATE", "COMMIT")
        ]
        positive_sentiment = [
            s for s in language_signals
            if s.get("signal_type") == "sentiment_score"
            and _to_float(s.get("value", 0)) > 0.40
        ]

        persuasion_present = (
            len(buying_signals) > 0
            or len(intent_signals) > 0
            or len(positive_sentiment) > 0
        )

        if not persuasion_present:
            return None

        # Find time-aligned rate + persuasion pair (within 10s)
        ALIGN_MS = 10_000
        all_persuasion = buying_signals + intent_signals + positive_sentiment
        latest_rate = None
        for rs in sorted(rate_signals, key=lambda s: _to_int(s.get("window_start_ms", 0)), reverse=True):
            rs_time = _to_int(rs.get("window_start_ms", 0))
            for ps in all_persuasion:
                ps_time = _to_int(ps.get("window_start_ms", 0))
                if abs(rs_time - ps_time) <= ALIGN_MS:
                    latest_rate = rs
                    break
            if latest_rate:
                break

        if not latest_rate:
            return None

        # Check what's driving the acceleration
        sub_class = ""
        if isinstance(latest_rate.get("metadata"), dict):
            sub_class = latest_rate["metadata"].get("sub_classification", "")
        elif isinstance(latest_rate.get("metadata"), str):
            try:
                import json
                meta = json.loads(latest_rate["metadata"])
                sub_class = meta.get("sub_classification", "")
            except (json.JSONDecodeError, TypeError):
                pass

        # Check tone
        tone_signals = [
            s for s in voice_signals
            if s.get("signal_type") == "tone_classification"
        ]
        latest_tone = None
        if tone_signals:
            latest_tone = max(
                tone_signals, key=lambda s: _to_int(s.get("window_start_ms", 0))
            )

        tone_text = latest_tone.get("value_text", "neutral") if latest_tone else "neutral"

        # Check stress
        stress_signals = [
            s for s in voice_signals
            if s.get("signal_type") == "vocal_stress_score"
        ]
        latest_stress_val = 0.0
        if stress_signals:
            latest_stress = max(
                stress_signals, key=lambda s: _to_int(s.get("window_start_ms", 0))
            )
            latest_stress_val = _to_float(latest_stress.get("value", 0))

        # Check fillers
        filler_signals = [
            s for s in voice_signals
            if s.get("signal_type") == "filler_detection"
            and s.get("value_text") in ("filler_spike", "filler_elevated")
        ]

        # ── Classify authenticity ──
        authenticity_score = 0.50  # Start neutral

        # Authentic indicators
        if tone_text == "confident":
            authenticity_score += 0.20
        if sub_class == "enthusiasm_driven_acceleration":
            authenticity_score += 0.20
        if latest_stress_val < 0.30:
            authenticity_score += 0.10

        # Manufactured indicators
        if tone_text == "nervous":
            authenticity_score -= 0.20
        if sub_class == "anxiety_driven_acceleration":
            authenticity_score -= 0.25
        if latest_stress_val > 0.50:
            authenticity_score -= 0.15
        if filler_signals:
            authenticity_score -= 0.10

        authenticity_score = max(0.0, min(1.0, authenticity_score))

        # Classify
        if authenticity_score >= 0.65:
            level = "authentic_urgency"
        elif authenticity_score >= 0.40:
            level = "ambiguous_urgency"
        else:
            level = "manufactured_urgency"

        # Confidence (cap at 0.60)
        raw_confidence = min(0.40 + abs(authenticity_score - 0.50) * 0.40, 0.60)

        # Rate magnitude boosts confidence
        rate_delta = abs(_to_float(latest_rate.get("value", 0)))
        if rate_delta > 40.0:  # >40% rate change is very significant
            raw_confidence = min(raw_confidence + 0.05, 0.60)

        evidence = {
            "rate_delta_pct": round(rate_delta, 1),
            "rate_sub_classification": sub_class,
            "tone": tone_text,
            "stress_level": round(latest_stress_val, 3),
            "filler_elevated": len(filler_signals) > 0,
            "buying_signals_present": len(buying_signals) > 0,
            "persuasion_intents": [s.get("value_text", "") for s in intent_signals[:3]],
        }

        return {
            "score": authenticity_score,
            "level": level,
            "confidence": raw_confidence,
            "evidence": evidence,
        }

    # ════════════════════════════════════════════════════════════════════════════
    # PHASE 2E — AUDIO × VIDEO PAIRWISE RULES
    # ════════════════════════════════════════════════════════════════════════════

    # ════════════════════════════════════════════════════════
    # FUSION-01: Tone × Facial Emotion → Masking Detection
    # Research: Ekman 1969 (leakage), Ekman & Friesen 1974
    # Max confidence: 0.65
    # ════════════════════════════════════════════════════════

    def _rule_fusion_01(
        self,
        voice_signals: list[dict],
        video_signals: list[dict],
        max_confidence: float = 0.65,
    ) -> Optional[dict]:
        """
        Detects fake positivity: voice tone says positive/confident but face
        shows a contradicting primary emotion (fear, disgust, contempt, sadness).

        Ekman's leakage hypothesis: the face is harder to fully control under
        stress — microexpressions and subtle incongruence reveal suppressed state.
        """
        ALIGN_MS = 8_000
        tone_sigs = [s for s in voice_signals if s.get("signal_type") == "tone_classification"]
        face_sigs = [s for s in video_signals if s.get("signal_type") == "facial_emotion"]

        if not tone_sigs or not face_sigs:
            return None

        POSITIVE_TONES = {"confident", "enthusiastic", "warm", "positive"}
        NEGATIVE_EMOTIONS = {"fear", "disgust", "contempt", "sadness", "anger"}

        best: dict | None = None
        best_score = 0.0

        for tone_sig in tone_sigs:
            tone_val = tone_sig.get("value_text", "").lower()
            if tone_val not in POSITIVE_TONES:
                continue
            tone_conf = _to_float(tone_sig.get("confidence", 0))
            t_time = _to_int(tone_sig.get("window_start_ms", 0))

            for face_sig in face_sigs:
                emotion = (face_sig.get("value_text") or "").lower()
                if emotion not in NEGATIVE_EMOTIONS:
                    continue
                f_conf = _to_float(face_sig.get("confidence", 0))
                f_time = _to_int(face_sig.get("window_start_ms", 0))

                if abs(t_time - f_time) > ALIGN_MS:
                    continue

                score = min(tone_conf * f_conf * 2.0, 1.0)
                if score <= best_score:
                    continue
                best_score = score

                if score >= 0.70:
                    level = "strong_masking"
                elif score >= 0.40:
                    level = "moderate_masking"
                else:
                    level = "mild_masking"

                raw_conf = min(0.40 + score * 0.25, max_confidence)
                best = {
                    "score": score,
                    "level": level,
                    "confidence": raw_conf,
                    "evidence": {
                        "voice_tone": tone_val,
                        "facial_emotion": emotion,
                        "tone_confidence": round(tone_conf, 3),
                        "face_confidence": round(f_conf, 3),
                        "aligned_window_ms": abs(t_time - f_time),
                    },
                }

        return best

    # ════════════════════════════════════════════════════════
    # FUSION-03: Voice Stress × Facial Stress → Suppression
    # Research: Ekman 1969, Harrigan 2005 (suppression channels)
    # Max confidence: 0.65
    # ════════════════════════════════════════════════════════

    def _rule_fusion_03(
        self,
        voice_signals: list[dict],
        video_signals: list[dict],
        max_confidence: float = 0.65,
    ) -> Optional[dict]:
        """
        Corroborating stress across channels → higher confidence stress detection.
        Opposing stress → suppression: voice calm but face stressed = hiding it.

        Both high: corroborated stress (both channels agree).
        Voice high + face low: stress leaking through voice only (verbal suppression).
        Voice low + face high: stress leaking through face only (vocal suppression).
        """
        ALIGN_MS = 10_000
        v_stress = [s for s in voice_signals if s.get("signal_type") == "vocal_stress_score"]
        f_stress = [s for s in video_signals if s.get("signal_type") == "facial_stress"]

        if not v_stress or not f_stress:
            return None

        latest_v = max(v_stress, key=lambda s: _to_int(s.get("window_start_ms", 0)))
        v_val = _to_float(latest_v.get("value", 0))
        v_time = _to_int(latest_v.get("window_start_ms", 0))

        best: dict | None = None
        best_score = 0.0

        for fs in f_stress:
            f_val = _to_float(fs.get("value", 0))
            f_time = _to_int(fs.get("window_start_ms", 0))

            if abs(v_time - f_time) > ALIGN_MS:
                continue

            gap = abs(v_val - f_val)
            combined = (v_val + f_val) / 2.0

            if v_val > 0.50 and f_val > 0.50:
                level = "corroborated_stress"
                score = combined
                raw_conf = min(0.45 + combined * 0.20, max_confidence)
            elif gap > 0.30 and max(v_val, f_val) > 0.45:
                level = "stress_suppression"
                score = gap
                raw_conf = min(0.40 + gap * 0.25, max_confidence)
            else:
                continue

            if score <= best_score:
                continue
            best_score = score

            best = {
                "score": round(score, 4),
                "level": level,
                "confidence": raw_conf,
                "evidence": {
                    "voice_stress": round(v_val, 3),
                    "facial_stress": round(f_val, 3),
                    "channel_gap": round(gap, 3),
                    "suppression_channel": "vocal" if f_val > v_val else "facial",
                    "aligned_window_ms": abs(v_time - f_time),
                },
            }

        return best

    # ════════════════════════════════════════════════════════
    # FUSION-04: Filler Words × Gaze Break → Cognitive Load
    # Research: Goldman-Eisler 1968, Rayner 1998
    # Max confidence: 0.70
    # ════════════════════════════════════════════════════════

    def _rule_fusion_04(
        self,
        voice_signals: list[dict],
        video_signals: list[dict],
        max_confidence: float = 0.70,
    ) -> Optional[dict]:
        """
        Co-occurring filler spikes + gaze breaks indicate cognitive overload:
        the person is searching for information while simultaneously losing
        visual contact — a reliable multi-channel overwhelm signal.
        """
        ALIGN_MS = 8_000
        filler_sigs = [
            s for s in voice_signals
            if s.get("signal_type") == "filler_detection"
            and s.get("value_text") in ("filler_spike", "filler_elevated", "elevated", "high")
        ]
        gaze_sigs = [
            s for s in video_signals
            if s.get("signal_type") == "gaze_direction_shift"
        ]

        if not filler_sigs or not gaze_sigs:
            return None

        # Also check for sustained distraction which amplifies the signal
        distraction_sigs = [s for s in video_signals if s.get("signal_type") == "sustained_distraction"]

        aligned_pairs = 0
        for fs in filler_sigs:
            f_time = _to_int(fs.get("window_start_ms", 0))
            for gs in gaze_sigs:
                g_time = _to_int(gs.get("window_start_ms", 0))
                if abs(f_time - g_time) <= ALIGN_MS:
                    aligned_pairs += 1
                    break

        if aligned_pairs == 0:
            return None

        # Score based on filler severity + gaze break count + distraction
        filler_rate = max(_to_float(s.get("value", 0)) for s in filler_sigs)
        gaze_shift_count = len(gaze_sigs)
        has_distraction = len(distraction_sigs) > 0

        score = min(0.30 + filler_rate * 0.40 + min(gaze_shift_count, 5) * 0.06, 1.0)
        if has_distraction:
            score = min(score + 0.10, 1.0)

        if score >= 0.70:
            level = "high_cognitive_load"
        elif score >= 0.45:
            level = "moderate_cognitive_load"
        else:
            level = "mild_cognitive_load"

        raw_conf = min(0.40 + score * 0.30, max_confidence)

        return {
            "score": round(score, 4),
            "level": level,
            "confidence": raw_conf,
            "evidence": {
                "filler_rate": round(filler_rate, 3),
                "gaze_shift_count": gaze_shift_count,
                "aligned_pairs": aligned_pairs,
                "sustained_distraction": has_distraction,
            },
        }

    # ════════════════════════════════════════════════════════
    # FUSION-05: Head Nod/Shake × Objection → Disagreement Signal
    # Research: Burgoon et al. 1995, Navarro 2008
    # Max confidence: 0.55 (subtle — nod + objection could be "I hear you but…")
    # ════════════════════════════════════════════════════════

    def _rule_fusion_05(
        self,
        language_signals: list[dict],
        video_signals: list[dict],
        max_confidence: float = 0.55,
    ) -> Optional[dict]:
        """
        Head shake co-occurring with objection language = clear disagreement.
        Head nod co-occurring with objection language = polite disagreement
        ("I understand but no") — a softer but commercially important signal.
        """
        ALIGN_MS = 10_000
        objection_sigs = [
            s for s in language_signals
            if s.get("signal_type") == "objection_signal"
            and _to_float(s.get("confidence", 0)) > 0.40
        ]
        nod_sigs = [s for s in video_signals if s.get("signal_type") == "head_nod"]
        shake_sigs = [s for s in video_signals if s.get("signal_type") == "head_shake"]

        if not objection_sigs or not (nod_sigs or shake_sigs):
            return None

        has_aligned_shake = False
        has_aligned_nod = False

        for obj in objection_sigs:
            o_time = _to_int(obj.get("window_start_ms", 0))
            for s in shake_sigs:
                if abs(o_time - _to_int(s.get("window_start_ms", 0))) <= ALIGN_MS:
                    has_aligned_shake = True
                    break
            for n in nod_sigs:
                if abs(o_time - _to_int(n.get("window_start_ms", 0))) <= ALIGN_MS:
                    has_aligned_nod = True
                    break

        if not (has_aligned_shake or has_aligned_nod):
            return None

        obj_conf = max(_to_float(s.get("confidence", 0)) for s in objection_sigs)

        if has_aligned_shake:
            level = "explicit_disagreement"
            score = min(0.60 + obj_conf * 0.30, 1.0)
            raw_conf = min(0.45 + obj_conf * 0.10, max_confidence)
        else:
            level = "polite_disagreement"
            score = min(0.40 + obj_conf * 0.20, 1.0)
            raw_conf = min(0.35 + obj_conf * 0.10, max_confidence)

        return {
            "score": round(score, 4),
            "level": level,
            "confidence": raw_conf,
            "evidence": {
                "head_shake_aligned": has_aligned_shake,
                "head_nod_aligned": has_aligned_nod,
                "objection_confidence": round(obj_conf, 3),
                "objection_count": len(objection_sigs),
            },
        }

    # ════════════════════════════════════════════════════════
    # FUSION-06: Body Lean × Attention Level → Physical Engagement
    # Research: Mehrabian 1972, Tickle-Degnen & Rosenthal 1990
    # Max confidence: 0.60
    # ════════════════════════════════════════════════════════

    def _rule_fusion_06(
        self,
        video_signals: list[dict],
        max_confidence: float = 0.60,
    ) -> Optional[dict]:
        """
        Body lean direction combined with gaze attention level measures
        physical engagement: forward lean + high screen contact = engaged;
        backward lean + low attention = disengaged.
        """
        ALIGN_MS = 10_000
        lean_sigs = [s for s in video_signals if s.get("signal_type") == "body_lean"]
        attention_sigs = [s for s in video_signals if s.get("signal_type") == "attention_level"]

        if not lean_sigs or not attention_sigs:
            return None

        latest_lean = max(lean_sigs, key=lambda s: _to_int(s.get("window_start_ms", 0)))
        lean_dir = (latest_lean.get("value_text") or "").lower()
        lean_time = _to_int(latest_lean.get("window_start_ms", 0))

        best_attn: dict | None = None
        for a in attention_sigs:
            if abs(lean_time - _to_int(a.get("window_start_ms", 0))) <= ALIGN_MS:
                if best_attn is None or _to_float(a.get("value", 0)) > _to_float(best_attn.get("value", 0)):
                    best_attn = a

        if best_attn is None:
            return None

        attn_val = _to_float(best_attn.get("value", 0))
        lean_val = _to_float(latest_lean.get("value", 0))

        forward = "forward" in lean_dir
        backward = "backward" in lean_dir or "back" in lean_dir

        if forward and attn_val > 0.60:
            level = "high_engagement"
            score = min((lean_val + attn_val) / 2.0, 1.0)
        elif backward and attn_val < 0.40:
            level = "disengagement"
            score = 1.0 - min((lean_val + attn_val) / 2.0, 1.0)
        elif forward and attn_val < 0.40:
            level = "body_engaged_mind_elsewhere"
            score = 0.50
        else:
            return None

        raw_conf = min(0.35 + score * 0.25, max_confidence)

        return {
            "score": round(score, 4),
            "level": level,
            "confidence": raw_conf,
            "evidence": {
                "lean_direction": lean_dir,
                "lean_magnitude": round(lean_val, 3),
                "attention_level": round(attn_val, 3),
            },
        }

    # ════════════════════════════════════════════════════════
    # FUSION-08: Gaze Break × Hedge Language → False Confidence
    # Research: Lakoff 1975, Vrij 2008
    # Max confidence: 0.55
    # ════════════════════════════════════════════════════════

    def _rule_fusion_08(
        self,
        language_signals: list[dict],
        video_signals: list[dict],
        max_confidence: float = 0.55,
    ) -> Optional[dict]:
        """
        Hedged language (power language low) co-occurring with gaze breaks
        indicates the speaker lacks genuine confidence in what they are saying.
        """
        ALIGN_MS = 10_000
        power_sigs = [
            s for s in language_signals
            if s.get("signal_type") == "power_language_score"
            and _to_float(s.get("value", 1.0)) < 0.35
        ]
        gaze_sigs = [
            s for s in video_signals
            if s.get("signal_type") in ("gaze_direction_shift", "sustained_distraction")
        ]

        if not power_sigs or not gaze_sigs:
            return None

        aligned = False
        for p in power_sigs:
            p_time = _to_int(p.get("window_start_ms", 0))
            for g in gaze_sigs:
                if abs(p_time - _to_int(g.get("window_start_ms", 0))) <= ALIGN_MS:
                    aligned = True
                    break
            if aligned:
                break

        if not aligned:
            return None

        avg_power = sum(_to_float(s.get("value", 0)) for s in power_sigs) / len(power_sigs)
        gaze_break_count = len(gaze_sigs)

        score = min((0.35 - avg_power) * 2.0 + min(gaze_break_count, 4) * 0.08, 1.0)
        score = max(score, 0.0)

        if score >= 0.60:
            level = "low_confidence_detected"
        elif score >= 0.35:
            level = "mild_uncertainty"
        else:
            level = "hedged_statement"

        raw_conf = min(0.35 + score * 0.20, max_confidence)

        return {
            "score": round(score, 4),
            "level": level,
            "confidence": raw_conf,
            "evidence": {
                "avg_power_score": round(avg_power, 3),
                "gaze_break_count": gaze_break_count,
                "hedge_signals": len(power_sigs),
            },
        }

    # ════════════════════════════════════════════════════════
    # FUSION-09: Smile Quality × Negative Sentiment → Masking / Sarcasm
    # Research: Ekman & Friesen 1982 (Duchenne smile), Niedenthal 2007
    # Max confidence: 0.60
    # ════════════════════════════════════════════════════════

    def _rule_fusion_09(
        self,
        language_signals: list[dict],
        video_signals: list[dict],
        max_confidence: float = 0.60,
    ) -> Optional[dict]:
        """
        Non-Duchenne (social/fake) smile co-occurring with negative sentiment
        or objection language = masking displeasure.
        Duchenne smile + negative sentiment = possible sarcasm or irony.
        """
        ALIGN_MS = 10_000
        smile_sigs = [s for s in video_signals if s.get("signal_type") == "smile_type"]
        sentiment_sigs = [
            s for s in language_signals
            if s.get("signal_type") == "sentiment_score"
            and _to_float(s.get("value", 0)) < -0.20
        ]
        objection_sigs = [s for s in language_signals if s.get("signal_type") == "objection_signal"]

        if not smile_sigs or not (sentiment_sigs or objection_sigs):
            return None

        negative_signals = sentiment_sigs + objection_sigs

        best: dict | None = None
        best_score = 0.0

        for smile in smile_sigs:
            smile_type = (smile.get("value_text") or "").lower()
            smile_conf = _to_float(smile.get("confidence", 0))
            s_time = _to_int(smile.get("window_start_ms", 0))

            for neg in negative_signals:
                if abs(s_time - _to_int(neg.get("window_start_ms", 0))) > ALIGN_MS:
                    continue

                neg_val = abs(_to_float(neg.get("value", 0.3)))
                is_social = "social" in smile_type or "non_duchenne" in smile_type or "fake" in smile_type
                is_duchenne = "duchenne" in smile_type or "genuine" in smile_type

                if is_social:
                    level = "emotion_masking"
                    score = min(smile_conf * neg_val * 2.0, 1.0)
                elif is_duchenne:
                    level = "possible_sarcasm"
                    score = min(smile_conf * neg_val * 1.5, 1.0)
                else:
                    continue

                if score <= best_score:
                    continue
                best_score = score

                raw_conf = min(0.35 + score * 0.25, max_confidence)
                best = {
                    "score": round(score, 4),
                    "level": level,
                    "confidence": raw_conf,
                    "evidence": {
                        "smile_type": smile_type,
                        "smile_confidence": round(smile_conf, 3),
                        "sentiment_value": round(_to_float(neg.get("value", 0)), 3),
                        "negative_signal_type": neg.get("signal_type", ""),
                    },
                }

        return best

    # ════════════════════════════════════════════════════════
    # FUSION-10: Response Latency × Facial Stress → Processing Load
    # Research: Greene 1984, Vrij 2008 (latency as cognitive load indicator)
    # Max confidence: 0.60
    # ════════════════════════════════════════════════════════

    def _rule_fusion_10(
        self,
        language_signals: list[dict],
        video_signals: list[dict],
        max_confidence: float = 0.60,
    ) -> Optional[dict]:
        """
        Long response latency + elevated facial stress = processing overload,
        not just thoughtfulness. The face confirms it's stress-driven delay.
        """
        ALIGN_MS = 15_000
        latency_sigs = [
            s for s in language_signals
            if s.get("signal_type") == "response_latency"
            and _to_float(s.get("value", 0)) > 0.50
        ]
        f_stress_sigs = [
            s for s in video_signals
            if s.get("signal_type") == "facial_stress"
            and _to_float(s.get("value", 0)) > 0.35
        ]

        if not latency_sigs or not f_stress_sigs:
            return None

        aligned = False
        for lat in latency_sigs:
            l_time = _to_int(lat.get("window_start_ms", 0))
            for fs in f_stress_sigs:
                if abs(l_time - _to_int(fs.get("window_start_ms", 0))) <= ALIGN_MS:
                    aligned = True
                    break
            if aligned:
                break

        if not aligned:
            return None

        max_latency = max(_to_float(s.get("value", 0)) for s in latency_sigs)
        max_face_stress = max(_to_float(s.get("value", 0)) for s in f_stress_sigs)

        score = min((max_latency + max_face_stress) / 2.0, 1.0)

        if score >= 0.70:
            level = "high_processing_load"
        elif score >= 0.50:
            level = "elevated_processing_load"
        else:
            level = "mild_processing_load"

        raw_conf = min(0.35 + score * 0.25, max_confidence)

        return {
            "score": round(score, 4),
            "level": level,
            "confidence": raw_conf,
            "evidence": {
                "max_latency_score": round(max_latency, 3),
                "max_facial_stress": round(max_face_stress, 3),
                "latency_event_count": len(latency_sigs),
            },
        }

    # ════════════════════════════════════════════════════════
    # FUSION-11: Dominance Score × Gaze Avoidance → Anxiety Under Pressure
    # Research: Burgoon & Dunbar 2006, Knapp & Hall 2009
    # Max confidence: 0.65
    # ════════════════════════════════════════════════════════

    def _rule_fusion_11(
        self,
        language_signals: list[dict],
        video_signals: list[dict],
        max_confidence: float = 0.65,
    ) -> Optional[dict]:
        """
        A speaker using dominant language while simultaneously showing gaze
        avoidance is exhibiting anxiety under their dominant persona — the
        body betrays what the words are trying to project.
        """
        ALIGN_MS = 12_000
        dominance_sigs = [
            s for s in language_signals
            if s.get("signal_type") == "dominance_score"
            and _to_float(s.get("value", 0)) > 0.55
        ]
        gaze_avoid_sigs = [
            s for s in video_signals
            if s.get("signal_type") in ("gaze_direction_shift", "sustained_distraction")
        ]

        if not dominance_sigs or not gaze_avoid_sigs:
            return None

        aligned_pairs = 0
        for d in dominance_sigs:
            d_time = _to_int(d.get("window_start_ms", 0))
            for g in gaze_avoid_sigs:
                if abs(d_time - _to_int(g.get("window_start_ms", 0))) <= ALIGN_MS:
                    aligned_pairs += 1
                    break

        if aligned_pairs == 0:
            return None

        avg_dominance = sum(_to_float(s.get("value", 0)) for s in dominance_sigs) / len(dominance_sigs)
        gaze_break_count = len(gaze_avoid_sigs)

        # High dominance + many gaze breaks = stronger anxiety signal
        score = min((avg_dominance - 0.55) * 2.0 + min(gaze_break_count, 6) * 0.07, 1.0)
        score = max(score, 0.0)

        if score >= 0.60:
            level = "dominance_anxiety"
        elif score >= 0.35:
            level = "mild_dominance_anxiety"
        else:
            level = "dominance_with_uncertainty"

        raw_conf = min(0.40 + score * 0.25, max_confidence)

        return {
            "score": round(score, 4),
            "level": level,
            "confidence": raw_conf,
            "evidence": {
                "avg_dominance": round(avg_dominance, 3),
                "gaze_avoidance_count": gaze_break_count,
                "aligned_pairs": aligned_pairs,
            },
        }

    # ════════════════════════════════════════════════════════
    # FUSION-12: Interruption × Body Lean → Interrupt Intent
    # Research: Tannen 1994, Beattie 1982 (turn-taking signals)
    # Max confidence: 0.55
    # ════════════════════════════════════════════════════════

    def _rule_fusion_12(
        self,
        language_signals: list[dict],
        voice_signals: list[dict],
        video_signals: list[dict],
        max_confidence: float = 0.55,
    ) -> Optional[dict]:
        """
        Interruption + forward lean = competitive/assertive interrupt (dominance).
        Interruption + backward lean = reactive interrupt (defensive or confused).
        Distinguishes between cooperative and competitive turn-taking.
        """
        ALIGN_MS = 8_000
        interrupt_sigs = [
            s for s in language_signals
            if s.get("signal_type") == "interruption_event"
        ]
        if not interrupt_sigs:
            interrupt_sigs = [
                s for s in voice_signals
                if s.get("signal_type") == "interruption_event"
            ]

        lean_sigs = [s for s in video_signals if s.get("signal_type") == "body_lean"]

        if not interrupt_sigs or not lean_sigs:
            return None

        forward_aligned = 0
        backward_aligned = 0

        for intr in interrupt_sigs:
            i_time = _to_int(intr.get("window_start_ms", 0))
            for lean in lean_sigs:
                if abs(i_time - _to_int(lean.get("window_start_ms", 0))) > ALIGN_MS:
                    continue
                lean_dir = (lean.get("value_text") or "").lower()
                if "forward" in lean_dir:
                    forward_aligned += 1
                elif "back" in lean_dir:
                    backward_aligned += 1

        if forward_aligned == 0 and backward_aligned == 0:
            return None

        total = forward_aligned + backward_aligned
        if forward_aligned >= backward_aligned:
            level = "competitive_interrupt"
            score = min(0.50 + (forward_aligned / total) * 0.50, 1.0)
        else:
            level = "reactive_interrupt"
            score = min(0.50 + (backward_aligned / total) * 0.50, 1.0)

        raw_conf = min(0.35 + (total / max(len(interrupt_sigs), 1)) * 0.20, max_confidence)

        return {
            "score": round(score, 4),
            "level": level,
            "confidence": raw_conf,
            "evidence": {
                "interrupt_count": len(interrupt_sigs),
                "forward_lean_aligned": forward_aligned,
                "backward_lean_aligned": backward_aligned,
            },
        }

    # ════════════════════════════════════════════════════════
    # FUSION-14: Empathy Language × Head Nod → Rapport Confirmation
    # Research: Tickle-Degnen & Rosenthal 1990, Bavelas et al. 1987
    # Max confidence: 0.70 (strongest fusion rule — research well-supported)
    # ════════════════════════════════════════════════════════

    def _rule_fusion_14(
        self,
        language_signals: list[dict],
        video_signals: list[dict],
        max_confidence: float = 0.70,
    ) -> Optional[dict]:
        """
        Empathy language co-occurring with head nods is the strongest
        multi-channel rapport signal. Tickle-Degnen & Rosenthal 1990: rapport
        is characterised by mutual attention, positivity, and co-ordination.
        Nodding while using empathy language confirms genuine alignment.
        """
        ALIGN_MS = 12_000
        empathy_sigs = [
            s for s in language_signals
            if s.get("signal_type") in ("empathy_language", "rapport_signal", "sentiment_score")
            and _to_float(s.get("value", 0)) > 0.30
        ]
        nod_sigs = [s for s in video_signals if s.get("signal_type") == "head_nod"]

        if not empathy_sigs or not nod_sigs:
            return None

        aligned_pairs = 0
        for emp in empathy_sigs:
            e_time = _to_int(emp.get("window_start_ms", 0))
            for nod in nod_sigs:
                if abs(e_time - _to_int(nod.get("window_start_ms", 0))) <= ALIGN_MS:
                    aligned_pairs += 1
                    break

        if aligned_pairs == 0:
            return None

        avg_empathy = sum(_to_float(s.get("value", 0)) for s in empathy_sigs) / len(empathy_sigs)
        nod_intensity = sum(_to_float(s.get("value", 0)) for s in nod_sigs) / len(nod_sigs)

        # Both channels reinforce each other
        score = min((avg_empathy + nod_intensity) / 2.0 + (aligned_pairs / max(len(empathy_sigs), 1)) * 0.15, 1.0)

        if score >= 0.70:
            level = "strong_rapport"
        elif score >= 0.45:
            level = "building_rapport"
        else:
            level = "rapport_indicator"

        # Rapport is positive — cap applies only to prevent over-certainty
        raw_conf = min(0.45 + score * 0.25, max_confidence)

        return {
            "score": round(score, 4),
            "level": level,
            "confidence": raw_conf,
            "evidence": {
                "avg_empathy_score": round(avg_empathy, 3),
                "avg_nod_intensity": round(nod_intensity, 3),
                "aligned_pairs": aligned_pairs,
                "empathy_signal_count": len(empathy_sigs),
                "nod_count": len(nod_sigs),
            },
        }

    # ════════════════════════════════════════════════════════
    # GRAPH-BASED FUSION RULES
    # ════════════════════════════════════════════════════════

    def evaluate_graph_insights(
        self,
        graph_insights: dict,
        speakers: list[str],
        existing_fusion_signals: list[dict],
        content_type: str = "sales_call",
        profile: "ContentTypeProfile | None" = None,
    ) -> list[dict]:
        """Generate fusion signals from graph analytics."""
        if profile is None and ContentTypeProfile is not None:
            profile = ContentTypeProfile(content_type)

        signals = []

        # FUSION-GRAPH-01: Tension Cluster Detection
        if not (profile and profile.is_gated("FUSION-GRAPH-01")):
            min_signals = int(profile.get_threshold("FUSION-GRAPH-01", "min_signals", 3)) if profile else 3
            for cluster in graph_insights.get("tension_clusters", []):
                if cluster["signal_count"] >= min_signals:
                    conf = min(0.50 + (cluster["signal_count"] - min_signals) * 0.05, 0.75)
                    signals.append({
                        "agent": "fusion",
                        "speaker_id": cluster["speaker_id"],
                        "signal_type": "tension_cluster",
                        "value": round(cluster["signal_count"] / 10.0, 3),
                        "value_text": "high_tension" if cluster["signal_count"] >= min_signals + 2 else "moderate_tension",
                        "confidence": round(conf, 3),
                        "window_start_ms": cluster["timestamp_ms"],
                        "window_end_ms": cluster["timestamp_ms"] + cluster["duration_ms"],
                        "metadata": cluster,
                    })

        # FUSION-GRAPH-02: Momentum Shift Detection
        momentum = graph_insights.get("momentum", {})
        if (
            momentum.get("turning_point_ms")
            and momentum.get("overall_trajectory") in ("positive", "negative")
        ):
            signals.append({
                "agent": "fusion",
                "speaker_id": "all",
                "signal_type": "momentum_shift",
                "value": momentum.get("momentum_score", 0),
                "value_text": f"{momentum['overall_trajectory']}_trajectory",
                "confidence": 0.55,
                "window_start_ms": momentum["turning_point_ms"],
                "window_end_ms": momentum["turning_point_ms"],
                "metadata": momentum,
            })

        # FUSION-GRAPH-03: Persistent Incongruence
        if not (profile and profile.is_gated("FUSION-GRAPH-03")):
            for speaker_id, pattern in graph_insights.get("incongruence_patterns", {}).items():
                if pattern.get("consistency") == "persistent":
                    signals.append({
                        "agent": "fusion",
                        "speaker_id": speaker_id,
                        "signal_type": "persistent_incongruence",
                        "value": round(pattern["total_contradicts_edges"] / 10.0, 3),
                        "value_text": "persistent_incongruence",
                        "confidence": 0.60,
                        "window_start_ms": pattern.get("worst_incongruence_ms", 0),
                        "window_end_ms": pattern.get("worst_incongruence_ms", 0),
                        "metadata": pattern,
                    })

        return signals
