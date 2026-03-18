"""
NEXUS Language Agent - Rule Engine
Implements 5 core rules from the NEXUS Rule Engine specification.

Each rule takes linguistic features and produces Signal dicts.
Thresholds are hardcoded defaults matching the Rule Engine document;
in production these load from the rule_config database table.

Rules implemented:
  LANG-SENT-01: Per-sentence sentiment (DistilBERT)
  LANG-BUY-01:  Buying signal detection (SPIN keyword patterns)
  LANG-OBJ-01:  Objection signal detection (hedges + resistance)
  LANG-PWR-01:  Power language score (Lakoff/O'Barr)
  LANG-INTENT-01: Intent classification (Claude API batch)

Research references:
  - Liu 2012 (sentiment)
  - Pennebaker 2015 (word-level affect)
  - Rackham 1988 (SPIN Selling — 35,000 calls)
  - Lakoff 1975 (language & gender / powerless speech)
  - O'Barr & Atkins 1982 (powerless speech in courtrooms)
"""
import os
import json
import logging
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger("nexus.language.rules")

try:
    from shared.models.signals import Signal
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from shared.models.signals import Signal

# ── LLM client (supports Anthropic + OpenAI via LLM_PROVIDER env var) ──
_llm_ready = None  # None = not checked yet, True/False = checked


def _check_llm_ready() -> bool:
    """Check if the LLM client (shared/utils/llm_client) is configured."""
    global _llm_ready
    if _llm_ready is not None:
        return _llm_ready
    try:
        from shared.utils.llm_client import is_configured, get_provider_info
        _llm_ready = is_configured()
        if _llm_ready:
            info = get_provider_info()
            logger.info(
                f"LLM client ready: provider={info['provider']}, model={info['model']}"
            )
        else:
            logger.warning("LLM not configured — LANG-INTENT-01 will be skipped")
    except ImportError:
        logger.warning("shared.utils.llm_client not available — LANG-INTENT-01 will be skipped")
        _llm_ready = False
    return _llm_ready


INTENT_BATCH_SIZE = 15  # 10-20 utterances per LLM call


class LanguageRuleEngine:
    """
    Evaluates linguistic features against detection rules.
    Produces Signal dicts for each fired rule.

    Content-type aware: buying signal and objection rules only run
    for sales_call content type. Sentiment, power, and intent rules
    run for all content types.
    """

    # Content types where sales-specific rules (buying/objection) are active
    SALES_TYPES = {"sales_call"}

    def __init__(self):
        # TODO: Load thresholds from rule_config DB table
        self._content_type = "sales_call"  # Default for backward compat

    def set_content_type(self, content_type: str):
        """Set the content type to adjust which rules are active."""
        self._content_type = content_type
        logger.info(f"Rule engine content type set to: {content_type}")

    def evaluate(
        self,
        features: dict,
        speaker_id: str,
        content_type: Optional[str] = None,
    ) -> list[dict]:
        """
        Run all rules against a single segment's features.

        Args:
            features: Feature dict from LanguageFeatureExtractor
            speaker_id: Speaker identifier
            content_type: Optional override for content type

        Returns:
            List of Signal dicts (one per fired rule)
        """
        signals = []
        start_ms = features.get("start_ms", 0)
        end_ms = features.get("end_ms", 0)
        active_type = content_type or self._content_type

        # ── LANG-SENT-01: Per-Sentence Sentiment (all content types) ──
        sent = self._rule_sentiment_01(features)
        if sent is not None:
            signals.append({
                "agent": "language",
                "speaker_id": speaker_id,
                "signal_type": "sentiment_score",
                "value": round(sent["value"], 4),
                "value_text": sent["label"],
                "confidence": round(sent["confidence"], 4),
                "window_start_ms": start_ms,
                "window_end_ms": end_ms,
                "metadata": {
                    "raw_label": sent["raw_label"],
                    "raw_score": sent["raw_score"],
                    "text_preview": features.get("text", "")[:80],
                },
            })

        # ── LANG-BUY-01: Buying Signal Detection (sales_call only) ──
        if active_type in self.SALES_TYPES:
            buy = self._rule_buying_01(features)
            if buy is not None:
                signals.append({
                    "agent": "language",
                    "speaker_id": speaker_id,
                    "signal_type": "buying_signal",
                    "value": round(buy["strength"], 4),
                    "value_text": buy["level"],
                    "confidence": round(buy["confidence"], 4),
                    "window_start_ms": start_ms,
                    "window_end_ms": end_ms,
                    "metadata": {
                        "categories": buy["categories"],
                        "match_count": buy["match_count"],
                        "matches": buy["matches"][:5],
                        "text_preview": features.get("text", "")[:80],
                    },
                })

        # ── LANG-OBJ-01: Objection Signal Detection (sales_call only) ──
        if active_type in self.SALES_TYPES:
            obj = self._rule_objection_01(features)
            if obj is not None:
                signals.append({
                    "agent": "language",
                    "speaker_id": speaker_id,
                    "signal_type": "objection_signal",
                    "value": round(obj["strength"], 4),
                    "value_text": obj["level"],
                    "confidence": round(obj["confidence"], 4),
                    "window_start_ms": start_ms,
                    "window_end_ms": end_ms,
                    "metadata": {
                        "categories": obj["categories"],
                        "hedge_count": obj.get("hedge_count", 0),
                        "matches": obj["matches"][:5],
                        "text_preview": features.get("text", "")[:80],
                    },
                })

        # ── LANG-PWR-01: Power Language Score (all content types) ──
        pwr = self._rule_power_01(features)
        if pwr is not None:
            signals.append({
                "agent": "language",
                "speaker_id": speaker_id,
                "signal_type": "power_language_score",
                "value": round(pwr["score"], 4),
                "value_text": pwr["level"],
                "confidence": round(pwr["confidence"], 4),
                "window_start_ms": start_ms,
                "window_end_ms": end_ms,
                "metadata": {
                    "powerless_feature_count": pwr["powerless_count"],
                    "features_found": pwr["features_found"],
                    "word_count": pwr["word_count"],
                    "text_preview": features.get("text", "")[:80],
                },
            })

        return signals

    def evaluate_batch_intent(
        self,
        features_list: list[dict],
    ) -> list[dict]:
        """
        LANG-INTENT-01: Batch intent classification via LLM API.
        Sends 10-20 utterances per API call to reduce cost.
        Supports both Anthropic Claude and OpenAI via LLM_PROVIDER env var.

        Returns list of Signal dicts for utterances with clear intent.
        """
        if not _check_llm_ready():
            return []

        signals = []

        # Process in batches of INTENT_BATCH_SIZE
        for batch_start in range(0, len(features_list), INTENT_BATCH_SIZE):
            batch = features_list[batch_start:batch_start + INTENT_BATCH_SIZE]
            batch_signals = self._classify_intent_batch(batch)
            signals.extend(batch_signals)

        return signals

    # ════════════════════════════════════════════════════════
    # LANG-SENT-01: Per-Sentence Sentiment
    # Research: Liu 2012, Pennebaker 2015
    # ════════════════════════════════════════════════════════

    def _rule_sentiment_01(self, f: dict) -> Optional[dict]:
        """
        Convert DistilBERT sentiment into a -1.0 to +1.0 signal.
        Only emits signals for non-neutral sentiment (strong positive/negative).
        """
        raw_label = f.get("sentiment_label", "NEUTRAL")
        raw_score = f.get("sentiment_score", 0.5)
        value = f.get("sentiment_value", 0.0)

        # Classify intensity
        abs_value = abs(value)
        if abs_value > 0.90:
            label = "strong_positive" if value > 0 else "strong_negative"
            confidence = 0.80
        elif abs_value > 0.75:
            label = "positive" if value > 0 else "negative"
            confidence = 0.70
        elif abs_value > 0.60:
            label = "mild_positive" if value > 0 else "mild_negative"
            confidence = 0.55
        else:
            label = "neutral"
            confidence = 0.40

        return {
            "value": value,
            "label": label,
            "confidence": min(confidence, 0.85),  # Hard cap per NEXUS rules
            "raw_label": raw_label,
            "raw_score": raw_score,
        }

    # ════════════════════════════════════════════════════════
    # LANG-BUY-01: Buying Signal Detection
    # Research: Rackham 1988 (SPIN Selling, 35,000 calls)
    # ════════════════════════════════════════════════════════

    def _rule_buying_01(self, f: dict) -> Optional[dict]:
        """
        Detect buying signals from keyword pattern matches.
        Only fires when at least one buying pattern is matched.
        """
        match_count = f.get("buying_signal_count", 0)
        categories = f.get("buying_categories", [])
        matches = f.get("buying_signals", [])

        if match_count == 0:
            return None

        # Strength based on number of distinct categories matched
        # (multiple categories in one utterance = stronger signal)
        num_categories = len(categories)
        if num_categories >= 3:
            strength = 0.80
            level = "strong_buying_signal"
            confidence = 0.75
        elif num_categories >= 2:
            strength = 0.60
            level = "moderate_buying_signal"
            confidence = 0.65
        else:
            strength = 0.40
            level = "weak_buying_signal"
            confidence = 0.55

        # Boost for high-intent categories
        high_intent = {
            "price_terms", "implementation_question", "specification_question",
            "specification_question_conversational", "next_step_acceptance",
            "information_sharing",
        }
        if high_intent.intersection(categories):
            strength = min(strength + 0.10, 0.85)
            confidence = min(confidence + 0.05, 0.80)

        return {
            "strength": strength,
            "level": level,
            "confidence": confidence,
            "match_count": match_count,
            "categories": categories,
            "matches": matches,
        }

    # ════════════════════════════════════════════════════════
    # LANG-OBJ-01: Objection Signal Detection
    # Research: Rackham 1988
    # ════════════════════════════════════════════════════════

    def _rule_objection_01(self, f: dict) -> Optional[dict]:
        """
        Detect objection/resistance signals from pattern matches
        combined with hedge counting and negative sentiment.
        """
        match_count = f.get("objection_signal_count", 0)
        categories = f.get("objection_categories", [])
        matches = f.get("objection_signals", [])
        hedge_count = f.get("powerless_feature_count", 0)
        sentiment_value = f.get("sentiment_value", 0.0)

        if match_count == 0:
            return None

        # Strength based on directness and reinforcing signals
        num_categories = len(categories)
        if "direct_objection" in categories:
            strength = 0.70
            level = "direct_objection"
            confidence = 0.75
        elif num_categories >= 2:
            strength = 0.55
            level = "moderate_objection"
            confidence = 0.60
        elif "hedge_cluster" in categories:
            strength = 0.30
            level = "hedged_resistance"
            confidence = 0.45
        else:
            strength = 0.40
            level = "mild_objection"
            confidence = 0.50

        # Boost if negative sentiment reinforces objection
        if sentiment_value < -0.60:
            strength = min(strength + 0.10, 0.85)
            confidence = min(confidence + 0.05, 0.80)

        # Boost if many hedges accompany the objection
        if hedge_count >= 3:
            strength = min(strength + 0.05, 0.85)

        return {
            "strength": strength,
            "level": level,
            "confidence": confidence,
            "match_count": match_count,
            "categories": categories,
            "matches": matches,
            "hedge_count": hedge_count,
        }

    # ════════════════════════════════════════════════════════
    # LANG-PWR-01: Power Language Score
    # Research: Lakoff 1975, O'Barr & Atkins 1982
    # ════════════════════════════════════════════════════════

    def _rule_power_01(self, f: dict) -> Optional[dict]:
        """
        Score powerfulness of speech based on Lakoff/O'Barr features.
        Only fires on utterances with enough words to be meaningful.
        """
        word_count = f.get("power_word_count", 0)
        if word_count < 5:
            return None  # Too short to assess

        score = f.get("power_score", 0.5)
        powerless_count = f.get("powerless_feature_count", 0)
        features_found = f.get("powerless_features_found", [])

        # Classify
        if score >= 0.80:
            level = "powerful"
            confidence = 0.70
        elif score >= 0.60:
            level = "moderate_power"
            confidence = 0.60
        elif score >= 0.40:
            level = "neutral_power"
            confidence = 0.50
        elif score >= 0.20:
            level = "weak_power"
            confidence = 0.60
        else:
            level = "powerless"
            confidence = 0.70

        return {
            "score": score,
            "level": level,
            "confidence": min(confidence, 0.85),
            "powerless_count": powerless_count,
            "features_found": features_found,
            "word_count": word_count,
        }

    # ════════════════════════════════════════════════════════
    # LANG-INTENT-01: Intent Classification (Claude API)
    # Batched: 10-20 utterances per API call
    # ════════════════════════════════════════════════════════

    def _classify_intent_batch(
        self, batch: list[dict]
    ) -> list[dict]:
        """
        Send a batch of utterances to the LLM for intent classification.
        Uses the shared llm_client (supports Anthropic Claude or OpenAI).
        Returns Signal dicts for each classified utterance.
        """
        from shared.utils.llm_client import complete

        # Build numbered utterance list for the prompt
        utterance_lines = []
        for i, f in enumerate(batch):
            speaker = f.get("speaker_id", "unknown")
            text = f.get("text", "").strip()
            if text:
                utterance_lines.append(f"{i+1}. [{speaker}]: {text}")

        if not utterance_lines:
            return []

        utterance_block = "\n".join(utterance_lines)

        system_prompt = (
            "You are a conversation intent classifier for a behavioural analysis system. "
            "You classify utterances into exactly one intent category and return structured JSON."
        )

        user_prompt = f"""Classify each utterance's primary intent. Use EXACTLY one of these categories:

- INFORM: Sharing facts, data, or status updates
- QUESTION: Asking for information
- REQUEST: Asking someone to do something
- PROPOSE: Suggesting an idea, plan, or solution
- AGREE: Expressing agreement or acceptance
- DISAGREE: Expressing disagreement or pushback
- NEGOTIATE: Bargaining, counter-offering, or discussing terms
- COMMIT: Making a promise, commitment, or decision
- DEFLECT: Avoiding, redirecting, or evading a topic
- RAPPORT: Small talk, relationship building, empathy
- CLOSE: Attempting to close a deal or reach a decision
- OBJECTION: Raising a concern or barrier

Respond with a JSON array of objects: [{{"id": 1, "intent": "CATEGORY", "confidence": 0.0-1.0}}]
Only include utterances where you have confidence > 0.4. Return ONLY the JSON array, no other text.

Utterances:
{utterance_block}"""

        try:
            response_text = complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=1024,
            )

            # Handle potential markdown wrapping
            if response_text.startswith("```"):
                response_text = response_text.split("\n", 1)[1]
                response_text = response_text.rsplit("```", 1)[0].strip()

            classifications = json.loads(response_text)

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM intent response: {e}")
            return []
        except Exception as e:
            logger.warning(f"LLM API call failed for intent classification: {e}")
            return []

        # Convert to signals
        signals = []
        for cls in classifications:
            idx = cls.get("id", 0) - 1  # Convert 1-based to 0-based
            if idx < 0 or idx >= len(batch):
                continue

            f = batch[idx]
            raw_confidence = min(float(cls.get("confidence", 0.5)), 0.85)

            if raw_confidence < 0.40:
                continue

            signals.append({
                "agent": "language",
                "speaker_id": f.get("speaker_id", "unknown"),
                "signal_type": "intent_classification",
                "value": raw_confidence,
                "value_text": cls.get("intent", "UNKNOWN"),
                "confidence": round(raw_confidence, 4),
                "window_start_ms": f.get("start_ms", 0),
                "window_end_ms": f.get("end_ms", 0),
                "metadata": {
                    "intent": cls.get("intent", "UNKNOWN"),
                    "text_preview": f.get("text", "")[:80],
                },
            })

        return signals
