# services/conversation_agent/rules.py
"""
NEXUS Conversation Agent - Rule Engine
Implements 7 rules from the NEXUS Rule Engine specification for dialogue dynamics.

Each rule takes conversation features and produces Signal dicts.
Thresholds are hardcoded defaults matching the Rule Engine document;
in production these load from the rule_config database table.

Rules implemented:
  CONVO-TURN-01: Turn-taking pattern (rapid exchange / monologue dominated / normal)
  CONVO-LAT-01:  Response latency pattern per speaker pair
  CONVO-DOM-01:  Dominance score per speaker
  CONVO-INT-01:  Interruption pattern per speaker
  CONVO-RAP-01:  Rapport indicator per speaker pair
  CONVO-ENG-01:  Conversation engagement per speaker
  CONVO-BAL-01:  Conversation balance (session-level)

Research references:
  - Sacks, Schegloff & Jefferson 1974 (turn-taking organisation)
  - Tannen 1994 (conversational style)
  - Gravano & Hirschberg 2011 (turn-taking cues)
  - Heldner & Edlund 2010 (pauses, gaps, overlaps)
  - Tickle-Degnen & Rosenthal 1990 (rapport components)
  - Dunbar & Burgoon 2005 (dominance in interaction)
"""
import logging
import math
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger("nexus.conversation.rules")

try:
    from shared.models.signals import Signal
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from shared.models.signals import Signal

try:
    from shared.config.content_type_profile import ContentTypeProfile
except ImportError:
    ContentTypeProfile = None


def _make_signal(
    speaker_id: str, signal_type: str, value: float, value_text: str,
    confidence: float, window_start_ms: int, window_end_ms: int,
    metadata: dict = None,
) -> dict:
    """Create a validated signal dict via the Signal model."""
    return Signal(
        agent="conversation",
        speaker_id=speaker_id,
        signal_type=signal_type,
        value=round(value, 4),
        value_text=value_text,
        confidence=round(min(confidence, 0.85), 4),  # Enforce 0.85 cap
        window_start_ms=window_start_ms,
        window_end_ms=window_end_ms,
        metadata=metadata,
    ).to_dict()


class ConversationRuleEngine:
    """
    Evaluates conversation dynamics features and produces signals
    using research-derived detection rules.

    All thresholds match the NEXUS Rule Engine specification.
    """

    def __init__(self):
        logger.info("ConversationRuleEngine initialised")

    def evaluate(
        self,
        features: dict,
        content_type: str = "sales_call",
        language_signals: Optional[list] = None,
        profile=None,
    ) -> list[dict]:
        """
        Run all conversation rules against extracted features.

        Args:
            features: Output from ConversationFeatureExtractor.extract_all()
            content_type: Type of meeting (sales_call, interview, presentation, etc.)
            language_signals: Optional list of Language Agent signals
            profile: ContentTypeProfile for content-aware gating/renaming

        Returns:
            List of Signal dicts.
        """
        if profile is None and ContentTypeProfile is not None:
            profile = ContentTypeProfile(content_type)

        signals = []

        def _apply_profile(rule_id: str, raw_signals: list[dict]) -> list[dict]:
            """Filter, rename, and adjust confidence for a batch of signals."""
            if not profile:
                return raw_signals
            if profile.is_gated(rule_id):
                return []
            result = []
            for s in raw_signals:
                s["confidence"] = profile.apply_confidence(rule_id, s.get("confidence", 0.5))
                s["value_text"] = profile.rename_signal(s.get("value_text", ""))
                result.append(s)
            return result

        per_speaker = features.get("per_speaker", {})
        per_pair = features.get("per_pair", {})
        session = features.get("session", {})

        window_start_ms = 0
        window_end_ms = session.get("total_duration_ms", 0)
        speakers = list(per_speaker.keys())
        duration_minutes = window_end_ms / 60000.0 if window_end_ms > 0 else 0

        # Read content-type thresholds once; pass to rule methods below
        mono_per_min   = profile.get_threshold("CONVO-TURN-01", "monologue_per_min", 2.0) if profile else 2.0
        delayed_ms     = profile.get_threshold("CONVO-LAT-01",  "delayed_ms",        1500.0) if profile else 1500.0
        dom_pct        = profile.get_threshold("CONVO-DOM-01",  "expected_dominant_pct", 65.0) if profile else 65.0
        gini_low       = profile.get_threshold("CONVO-BAL-01",  "expected_gini_low",  0.0) if profile else 0.0
        gini_high      = profile.get_threshold("CONVO-BAL-01",  "expected_gini_high", 0.0) if profile else 0.0
        min_indicators = int(profile.get_threshold("CONVO-CONF-01", "min_indicators", 2.0)) if profile else 2

        # CONVO-TURN-01: Turn-taking pattern (session-level)
        signals.extend(_apply_profile("CONVO-TURN-01",
            self._rule_turn_taking(session, window_start_ms, window_end_ms, mono_per_min)))

        # CONVO-LAT-01: Response latency pattern (per pair)
        signals.extend(_apply_profile("CONVO-LAT-01",
            self._rule_response_latency(per_pair, window_start_ms, window_end_ms, delayed_ms)))

        # CONVO-DOM-01: Dominance score (per speaker)
        signals.extend(_apply_profile("CONVO-DOM-01", self._rule_dominance(
            per_speaker, session, speakers, duration_minutes,
            window_start_ms, window_end_ms, dom_pct / 100.0,
        )))

        # CONVO-INT-01: Interruption pattern (per speaker)
        signals.extend(_apply_profile("CONVO-INT-01", self._rule_interruption(
            per_speaker, speakers, duration_minutes,
            window_start_ms, window_end_ms,
        )))

        # CONVO-RAP-01: Rapport indicator (per pair)
        signals.extend(_apply_profile("CONVO-RAP-01", self._rule_rapport(
            per_speaker, per_pair, session, speakers, duration_minutes,
            window_start_ms, window_end_ms,
        )))

        # CONVO-ENG-01: Conversation engagement (per speaker)
        signals.extend(_apply_profile("CONVO-ENG-01", self._rule_engagement(
            per_speaker, per_pair, session, speakers, duration_minutes,
            window_start_ms, window_end_ms,
        )))

        # CONVO-BAL-01: Conversation balance (session-level)
        signals.extend(_apply_profile("CONVO-BAL-01", self._rule_balance(
            session, content_type, window_start_ms, window_end_ms, gini_low, gini_high,
        )))

        # CONVO-CONF-01: Conflict detection (cross-modal with Language Agent)
        signals.extend(_apply_profile("CONVO-CONF-01", self._rule_convo_conf_01(
            per_speaker, session, speakers, duration_minutes,
            language_signals, window_start_ms, window_end_ms, min_indicators,
        )))

        logger.info(f"ConversationRuleEngine produced {len(signals)} signals")
        return signals

    # ──────────────────────────────────────────────────────
    # CONVO-TURN-01: Turn-Taking Pattern
    # ──────────────────────────────────────────────────────

    def _rule_turn_taking(
        self, session: dict, window_start_ms: int, window_end_ms: int,
        mono_per_min: float = 2.0,
    ) -> list[dict]:
        """
        Classify turn-taking pattern based on turn rate per minute.
        > 10/min → rapid_exchange (Stivers 2009), < 2/min → monologue_dominated, else → normal_conversation
        """
        turn_rate = session.get("turn_rate_per_minute", 0)

        if turn_rate > 10:
            label = "rapid_exchange"
            confidence = min(0.55 + (turn_rate - 10) * 0.03, 0.80)
        elif turn_rate < mono_per_min:
            label = "monologue_dominated"
            confidence = min(0.55 + (mono_per_min - turn_rate) * 0.10, 0.80)
        else:
            label = "normal_conversation"
            confidence = 0.65

        return [_make_signal(
            speaker_id="session",
            signal_type="turn_taking_pattern",
            value=turn_rate,
            value_text=label,
            confidence=confidence,
            window_start_ms=window_start_ms,
            window_end_ms=window_end_ms,
            metadata={
                "rule": "CONVO-TURN-01",
                "total_turns": session.get("total_turns", 0),
                "avg_turn_duration_ms": session.get("avg_turn_duration_ms", 0),
            },
        )]

    # ──────────────────────────────────────────────────────
    # CONVO-LAT-01: Response Latency Pattern
    # ──────────────────────────────────────────────────────

    def _rule_response_latency(
        self, per_pair: dict, window_start_ms: int, window_end_ms: int,
        delayed_ms: float = 1500.0,
    ) -> list[dict]:
        """
        Classify response latency pattern per speaker pair.
        < 200ms → overlapping, 200-600ms → highly_engaged,
        600-1500ms → normal, > 1500ms → delayed_responses.
        Minimum 5 exchanges needed.
        """
        signals = []

        for pair_key, pair_data in per_pair.items():
            exchanges = pair_data.get("turn_exchanges", 0)
            if exchanges < 5:
                continue  # Insufficient data

            avg_latency = pair_data.get("response_latency_ms_avg", 0)

            if avg_latency < 200:
                label = "overlapping"
                confidence = 0.60
            elif avg_latency < 600:
                label = "highly_engaged"
                confidence = 0.65
            elif avg_latency <= delayed_ms:
                label = "normal"
                confidence = 0.60
            else:
                label = "delayed_responses"
                confidence = min(0.55 + (avg_latency - delayed_ms) / 5000, 0.75)

            signals.append(_make_signal(
                speaker_id=pair_key,
                signal_type="response_latency_pattern",
                value=avg_latency,
                value_text=label,
                confidence=confidence,
                window_start_ms=window_start_ms,
                window_end_ms=window_end_ms,
                metadata={
                    "rule": "CONVO-LAT-01",
                    "turn_exchanges": exchanges,
                    "median_latency_ms": pair_data.get("response_latency_ms_median", 0),
                    "overlap_count": pair_data.get("overlap_count", 0),
                },
            ))

        return signals

    # ──────────────────────────────────────────────────────
    # CONVO-DOM-01: Dominance Score
    # ──────────────────────────────────────────────────────

    def _rule_dominance(
        self,
        per_speaker: dict,
        session: dict,
        speakers: list[str],
        duration_minutes: float,
        window_start_ms: int,
        window_end_ms: int,
        dom_threshold: float = 0.65,
    ) -> list[dict]:
        """
        Compute dominance score per speaker.
        dominance = 0.5*talk_time_pct + 0.15*interruption_ratio + 0.25*monologue_ratio + 0.1*(1-question_ratio)
        Weights: monologue (0.25) > interruption (0.15) per Dunbar 1996 (speaking time > interruptions as dominance).
        > 0.65 → dominant, 0.35-0.65 → balanced, < 0.35 → passive
        """
        signals = []
        total_turns = session.get("total_turns", 0)
        total_monologues = sum(
            per_speaker[s].get("monologue_count", 0) for s in speakers
        )
        total_interruptions = sum(
            per_speaker[s].get("interruption_count", 0) for s in speakers
        )
        total_questions = sum(
            per_speaker[s].get("questions_asked", 0) for s in speakers
        )

        for spk in speakers:
            data = per_speaker[spk]

            # Normalised talk time (0-1 scale, where 1/n = equal share)
            talk_pct_norm = data.get("talk_time_pct", 0) / 100.0

            # Interruption ratio: fraction of all interruptions made by this speaker
            int_count = data.get("interruption_count", 0)
            interruption_ratio = (
                int_count / total_interruptions if total_interruptions > 0 else 0
            )

            # Monologue ratio: fraction of all monologues by this speaker
            mono_count = data.get("monologue_count", 0)
            monologue_ratio = (
                mono_count / total_monologues if total_monologues > 0 else 0
            )

            # Question ratio (inverse contributes to dominance)
            q_count = data.get("questions_asked", 0)
            question_ratio = (
                q_count / total_questions if total_questions > 0 else 0
            )

            dominance = (
                0.5  * talk_pct_norm +
                0.15 * interruption_ratio +
                0.25 * monologue_ratio +
                0.1  * (1.0 - question_ratio)
            )
            dominance = max(0.0, min(1.0, dominance))

            passive_threshold = 1.0 - dom_threshold
            if dominance > dom_threshold:
                label = "dominant"
                confidence = min(0.55 + (dominance - dom_threshold) * 0.5, 0.80)
            elif dominance < passive_threshold:
                label = "passive"
                confidence = min(0.55 + (passive_threshold - dominance) * 0.5, 0.75)
            else:
                label = "balanced"
                confidence = 0.60

            signals.append(_make_signal(
                speaker_id=spk,
                signal_type="dominance_score",
                value=dominance,
                value_text=label,
                confidence=confidence,
                window_start_ms=window_start_ms,
                window_end_ms=window_end_ms,
                metadata={
                    "rule": "CONVO-DOM-01",
                    "talk_time_pct": data.get("talk_time_pct", 0),
                    "interruption_count": int_count,
                    "monologue_count": mono_count,
                    "questions_asked": q_count,
                },
            ))

        return signals

    # ──────────────────────────────────────────────────────
    # CONVO-INT-01: Interruption Pattern
    # ──────────────────────────────────────────────────────

    def _rule_interruption(
        self,
        per_speaker: dict,
        speakers: list[str],
        duration_minutes: float,
        window_start_ms: int,
        window_end_ms: int,
    ) -> list[dict]:
        """
        Flag speakers with notable interruption patterns.
        interruption_rate > 3/min → frequent_interrupter
        interrupted_rate > 3/min → frequently_interrupted
        Only emit if rate > 1/min.
        """
        signals = []

        if duration_minutes <= 0:
            return signals

        for spk in speakers:
            data = per_speaker[spk]
            int_count = data.get("interruption_count", 0)
            was_int_count = data.get("was_interrupted_count", 0)

            int_rate = int_count / duration_minutes
            was_int_rate = was_int_count / duration_minutes

            # Interrupter signal
            if int_rate > 1.0:
                if int_rate > 3.0:
                    label = "frequent_interrupter"
                    confidence = min(0.55 + (int_rate - 3) * 0.05, 0.80)
                else:
                    label = "moderate_interrupter"
                    confidence = 0.55

                signals.append(_make_signal(
                    speaker_id=spk,
                    signal_type="interruption_pattern",
                    value=round(int_rate, 2),
                    value_text=label,
                    confidence=confidence,
                    window_start_ms=window_start_ms,
                    window_end_ms=window_end_ms,
                    metadata={
                        "rule": "CONVO-INT-01",
                        "interruption_count": int_count,
                        "rate_per_minute": round(int_rate, 2),
                        "direction": "made",
                    },
                ))

            # Interrupted signal
            if was_int_rate > 1.0:
                if was_int_rate > 3.0:
                    label = "frequently_interrupted"
                    confidence = min(0.55 + (was_int_rate - 3) * 0.05, 0.80)
                else:
                    label = "moderately_interrupted"
                    confidence = 0.55

                signals.append(_make_signal(
                    speaker_id=spk,
                    signal_type="interruption_pattern",
                    value=round(was_int_rate, 2),
                    value_text=label,
                    confidence=confidence,
                    window_start_ms=window_start_ms,
                    window_end_ms=window_end_ms,
                    metadata={
                        "rule": "CONVO-INT-01",
                        "was_interrupted_count": was_int_count,
                        "rate_per_minute": round(was_int_rate, 2),
                        "direction": "received",
                    },
                ))

        return signals

    # ──────────────────────────────────────────────────────
    # CONVO-RAP-01: Rapport Indicator
    # ──────────────────────────────────────────────────────

    def _rule_rapport(
        self,
        per_speaker: dict,
        per_pair: dict,
        session: dict,
        speakers: list[str],
        duration_minutes: float,
        window_start_ms: int,
        window_end_ms: int,
    ) -> list[dict]:
        """
        Compute rapport indicator per speaker pair.
        rapport = 0.3*backchannel_freq + 0.25*latency_consistency +
                  0.25*turn_balance + 0.2*qa_reciprocity
        > 0.65 → high_rapport, 0.4-0.65 → moderate_rapport, < 0.4 → low_rapport
        Confidence range: 0.45-0.70
        """
        signals = []

        for pair_key, pair_data in per_pair.items():
            parts = pair_key.split("__")
            if len(parts) != 2:
                continue
            spk_a, spk_b = parts

            exchanges = pair_data.get("turn_exchanges", 0)
            if exchanges < 3:
                continue  # Too few exchanges for rapport assessment

            # 1. Backchannel frequency (normalised 0-1)
            bc_a = per_speaker.get(spk_a, {}).get("back_channel_count", 0)
            bc_b = per_speaker.get(spk_b, {}).get("back_channel_count", 0)
            total_bc = bc_a + bc_b
            # High backchannel = good rapport; cap at 1.0
            bc_freq = min(total_bc / max(exchanges, 1) * 2.0, 1.0)

            # 2. Latency consistency (low variance = good rapport)
            avg_latency = pair_data.get("response_latency_ms_avg", 0)
            median_latency = pair_data.get("response_latency_ms_median", 0)
            # If avg ~ median, variance is low → good consistency
            if avg_latency > 0:
                latency_ratio = min(median_latency, avg_latency) / max(median_latency, avg_latency, 1)
            else:
                latency_ratio = 0.5
            latency_consistency = latency_ratio

            # 3. Turn balance between the pair
            turns_a = per_speaker.get(spk_a, {}).get("segment_count", 0)
            turns_b = per_speaker.get(spk_b, {}).get("segment_count", 0)
            total_pair_turns = turns_a + turns_b
            if total_pair_turns > 0:
                minority = min(turns_a, turns_b)
                turn_balance = minority / (total_pair_turns / 2.0)
                turn_balance = min(turn_balance, 1.0)
            else:
                turn_balance = 0.5

            # 4. QA reciprocity
            qa_pairs = pair_data.get("question_answer_pairs", 0)
            qa_reciprocity = min(qa_pairs / max(exchanges, 1) * 3.0, 1.0)

            rapport = (
                0.30 * bc_freq +
                0.25 * latency_consistency +
                0.25 * turn_balance +
                0.20 * qa_reciprocity
            )
            rapport = max(0.0, min(1.0, rapport))

            if rapport > 0.65:
                label = "high_rapport"
                confidence = min(0.55 + (rapport - 0.65) * 0.3, 0.70)
            elif rapport >= 0.4:
                label = "moderate_rapport"
                confidence = 0.55
            else:
                label = "low_rapport"
                confidence = max(0.45, 0.55 - (0.4 - rapport) * 0.3)

            signals.append(_make_signal(
                speaker_id=pair_key,
                signal_type="rapport_indicator",
                value=rapport,
                value_text=label,
                confidence=confidence,
                window_start_ms=window_start_ms,
                window_end_ms=window_end_ms,
                metadata={
                    "rule": "CONVO-RAP-01",
                    "backchannel_freq": round(bc_freq, 3),
                    "latency_consistency": round(latency_consistency, 3),
                    "turn_balance": round(turn_balance, 3),
                    "qa_reciprocity": round(qa_reciprocity, 3),
                    "turn_exchanges": exchanges,
                },
            ))

        return signals

    # ──────────────────────────────────────────────────────
    # CONVO-ENG-01: Conversation Engagement
    # ──────────────────────────────────────────────────────

    def _rule_engagement(
        self,
        per_speaker: dict,
        per_pair: dict,
        session: dict,
        speakers: list[str],
        duration_minutes: float,
        window_start_ms: int,
        window_end_ms: int,
    ) -> list[dict]:
        """
        Compute engagement score per speaker.
        engagement = 0.25*response_speed + 0.20*backchannel_freq +
                     0.20*question_rate + 0.20*turn_participation + 0.15*segment_length_trend
        Levels: highly_engaged / engaged / passive / disengaged
        """
        signals = []

        total_turns = session.get("total_turns", 0)
        total_segments = sum(
            per_speaker[s].get("segment_count", 0) for s in speakers
        )

        for spk in speakers:
            data = per_speaker[spk]

            # 1. Response speed (based on average silence after speaker's turns)
            # Lower silence = faster response = higher engagement
            avg_silence = data.get("silence_after_ms_avg", 1000)
            if avg_silence <= 300:
                response_speed = 1.0
            elif avg_silence <= 800:
                response_speed = 0.7
            elif avg_silence <= 1500:
                response_speed = 0.4
            else:
                response_speed = max(0.1, 1.0 - avg_silence / 5000)

            # 2. Backchannel frequency (normalised by duration)
            bc_count = data.get("back_channel_count", 0)
            bc_freq = min(bc_count / max(duration_minutes, 0.1) / 5.0, 1.0)

            # 3. Question rate (normalised)
            questions = data.get("questions_asked", 0)
            question_rate = min(questions / max(duration_minutes, 0.1) / 3.0, 1.0)

            # 4. Turn participation: fraction of total turns
            seg_count = data.get("segment_count", 0)
            expected_share = 1.0 / max(len(speakers), 1)
            actual_share = seg_count / max(total_segments, 1)
            turn_participation = min(actual_share / max(expected_share, 0.01), 1.0)

            # 5. Segment length trend (longer = more substantive engagement)
            avg_words = data.get("avg_words_per_turn", 0)
            if avg_words >= 20:
                seg_length_trend = 1.0
            elif avg_words >= 10:
                seg_length_trend = 0.7
            elif avg_words >= 5:
                seg_length_trend = 0.4
            else:
                seg_length_trend = 0.2

            engagement = (
                0.25 * response_speed +
                0.20 * bc_freq +
                0.20 * question_rate +
                0.20 * turn_participation +
                0.15 * seg_length_trend
            )
            engagement = max(0.0, min(1.0, engagement))

            if engagement > 0.70:
                label = "highly_engaged"
                confidence = min(0.55 + (engagement - 0.70) * 0.5, 0.80)
            elif engagement > 0.45:
                label = "engaged"
                confidence = 0.60
            elif engagement > 0.25:
                label = "passive"
                confidence = 0.55
            else:
                label = "disengaged"
                confidence = min(0.55 + (0.25 - engagement) * 0.5, 0.75)

            signals.append(_make_signal(
                speaker_id=spk,
                signal_type="conversation_engagement",
                value=engagement,
                value_text=label,
                confidence=confidence,
                window_start_ms=window_start_ms,
                window_end_ms=window_end_ms,
                metadata={
                    "rule": "CONVO-ENG-01",
                    "response_speed": round(response_speed, 3),
                    "backchannel_freq": round(bc_freq, 3),
                    "question_rate": round(question_rate, 3),
                    "turn_participation": round(turn_participation, 3),
                    "segment_length_trend": round(seg_length_trend, 3),
                },
            ))

        return signals

    # ──────────────────────────────────────────────────────
    # CONVO-BAL-01: Conversation Balance
    # ──────────────────────────────────────────────────────

    def _rule_balance(
        self,
        session: dict,
        content_type: str,
        window_start_ms: int,
        window_end_ms: int,
        expected_gini_low: float = 0.0,
        expected_gini_high: float = 0.0,
    ) -> list[dict]:
        """
        Session-level conversation balance based on dominance index (Gini).
        < 0.15 → well_balanced, 0.15-0.35 → moderately_balanced, > 0.35 → imbalanced
        Skip flagging for lecture/presentation content types.
        """
        dominance_index = session.get("dominance_index", 0)

        # For lectures/presentations, imbalance is expected — don't flag
        skip_types = {"lecture", "presentation", "keynote", "webinar", "training"}
        if content_type and content_type.lower() in skip_types:
            label = "expected_imbalance"
            confidence = 0.50
        elif expected_gini_low > 0 and expected_gini_high > 0:
            # Interview / podcast: compare to expected distribution, not symmetry.
            # e.g. interview expects Gini 0.20-0.40 (candidate talks more).
            if dominance_index < expected_gini_low:
                label = "more_balanced_than_expected"
                confidence = min(0.55 + (expected_gini_low - dominance_index) * 2.0, 0.75)
            elif dominance_index > expected_gini_high:
                label = "more_imbalanced_than_expected"
                confidence = min(0.55 + (dominance_index - expected_gini_high) * 2.0, 0.75)
            else:
                label = "expected_distribution"
                confidence = 0.60
        elif dominance_index < 0.15:
            label = "well_balanced"
            confidence = min(0.60 + (0.15 - dominance_index) * 1.0, 0.80)
        elif dominance_index <= 0.35:
            label = "moderately_balanced"
            confidence = 0.60
        else:
            label = "imbalanced"
            confidence = min(0.55 + (dominance_index - 0.35) * 0.5, 0.80)

        return [_make_signal(
            speaker_id="session",
            signal_type="conversation_balance",
            value=dominance_index,
            value_text=label,
            confidence=confidence,
            window_start_ms=window_start_ms,
            window_end_ms=window_end_ms,
            metadata={
                "rule": "CONVO-BAL-01",
                "speaker_count": session.get("speaker_count", 0),
                "content_type": content_type,
            },
        )]

    # ──────────────────────────────────────────────────────
    # CONVO-CONF-01: Conflict Detection (cross-modal)
    # Combines interruption rate, dominance, and Gottman
    # horsemen from Language Agent signals.
    # ──────────────────────────────────────────────────────

    def _rule_convo_conf_01(
        self,
        per_speaker: dict,
        session: dict,
        speakers: list[str],
        duration_minutes: float,
        language_signals: Optional[list],
        window_start_ms: int,
        window_end_ms: int,
        min_indicators: int = 2,
    ) -> list[dict]:
        """
        Detect conflict by combining conversation dynamics with language signals.

        Indicators:
          - interruption_rate > 3/min → +2, > 1.5/min → +1
          - dominance_index > 0.40 → +1
          - gottman_horsemen signals: 2+ → +2, 1 → +1

        Requires at least 2 indicators to fire.
        conflict_score = min(1.0, indicators * 0.20)
        """
        if duration_minutes <= 0:
            return []

        indicators = 0

        # ── Interruption rate (sum across all speakers) ──
        total_interruptions = sum(
            per_speaker.get(s, {}).get("interruption_count", 0) for s in speakers
        )
        interruption_rate = total_interruptions / duration_minutes if duration_minutes > 0 else 0

        if interruption_rate > 3.0:
            indicators += 2
        elif interruption_rate > 1.5:
            indicators += 1

        # ── Dominance index ──
        dominance_index = session.get("dominance_index", 0)
        if dominance_index > 0.40:
            indicators += 1

        # ── Gottman horsemen from Language Agent signals ──
        gottman_count = 0
        if language_signals:
            for sig in language_signals:
                if sig.get("signal_type") == "gottman_horsemen":
                    gottman_count += 1

        if gottman_count >= 2:
            indicators += 2
        elif gottman_count >= 1:
            indicators += 1

        # Need at least min_indicators to fire (default 2; interview 3 to reduce false positives)
        if indicators < min_indicators:
            return []

        conflict_score = min(1.0, indicators * 0.20)

        if conflict_score >= 0.80:
            label = "high_conflict"
        elif conflict_score >= 0.40:
            label = "moderate_conflict"
        else:
            label = "low_conflict"

        confidence = min(0.50 + indicators * 0.05, 0.80)

        return [_make_signal(
            speaker_id="session",
            signal_type="conflict_score",
            value=conflict_score,
            value_text=label,
            confidence=confidence,
            window_start_ms=window_start_ms,
            window_end_ms=window_end_ms,
            metadata={
                "rule": "CONVO-CONF-01",
                "indicators": indicators,
                "interruption_rate": round(interruption_rate, 2),
                "dominance_index": round(dominance_index, 3),
                "gottman_horsemen_count": gottman_count,
            },
        )]
