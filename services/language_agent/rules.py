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

def _make_signal(
    speaker_id: str, signal_type: str, value: float, value_text: str,
    confidence: float, window_start_ms: int, window_end_ms: int,
    metadata: dict = None,
) -> dict:
    """Create a validated signal dict via the Signal model."""
    return Signal(
        agent="language",
        speaker_id=speaker_id,
        signal_type=signal_type,
        value=round(value, 4),
        value_text=value_text,
        confidence=round(min(confidence, 0.85), 4),  # Enforce 0.85 cap
        window_start_ms=window_start_ms,
        window_end_ms=window_end_ms,
        metadata=metadata,
    ).to_dict()


# ── LLM client (supports Anthropic + OpenAI via LLM_PROVIDER env var) ──
try:
    from shared.utils.llm_client import is_configured, get_provider_info, complete as llm_complete
    _LLM_IMPORT_OK = True
except ImportError:
    _LLM_IMPORT_OK = False
    llm_complete = None  # type: ignore[assignment]

_llm_ready = None  # None = not checked yet, True/False = checked


def _check_llm_ready() -> bool:
    """Check if the LLM client (shared/utils/llm_client) is configured."""
    global _llm_ready
    if _llm_ready is not None:
        return _llm_ready
    if not _LLM_IMPORT_OK:
        logger.warning("shared.utils.llm_client not available — LANG-INTENT-01 will be skipped")
        _llm_ready = False
        return _llm_ready
    _llm_ready = is_configured()
    if _llm_ready:
        info = get_provider_info()
        logger.info(f"LLM client ready: provider={info['provider']}, model={info['model']}")
    else:
        logger.warning("LLM not configured — LANG-INTENT-01 will be skipped")
    return _llm_ready


INTENT_BATCH_SIZE = 15  # 10-20 utterances per LLM call


class LanguageRuleEngine:
    # Per RULES.md: no single-domain signal may exceed 0.85
    _MAX_CONFIDENCE = 0.85
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
        speaker_role: Optional[str] = None,
    ) -> list[dict]:
        """
        Run all rules against a single segment's features.

        Args:
            features: Feature dict from LanguageFeatureExtractor
            speaker_id: Speaker identifier
            content_type: Optional override for content type
            speaker_role: Optional speaker role (e.g. "Seller", "Prospect")
                          Buying signals are only valid from Prospect/Buyer.

        Returns:
            List of Signal dicts (one per fired rule)
        """
        signals = []
        start_ms = features.get("start_ms", 0)
        end_ms = features.get("end_ms", 0)
        active_type = content_type or self._content_type

        # ── LANG-SENT-01: Per-Sentence Sentiment (all content types) ──
        sent = self._rule_sentiment_01(features)
        signals.append(_make_signal(
            speaker_id, "sentiment_score",
            sent["value"], sent["label"],
            sent["confidence"],
            start_ms, end_ms,
            {
                "raw_label": sent["raw_label"],
                "raw_score": sent["raw_score"],
                "text_preview": features.get("text", "")[:80],
            },
        ))

        # ── LANG-BUY-01: Buying Signal Detection (sales_call only) ──
        # Buying signals are only meaningful from the Prospect/Buyer, never the Seller.
        role_lower = (speaker_role or "").lower()
        is_seller = role_lower in ("seller", "agent", "rep", "salesperson")

        if active_type in self.SALES_TYPES and not is_seller:
            buy = self._rule_buying_01(features)
            if buy is not None:
                meta = {
                    "categories": buy["categories"],
                    "match_count": buy["match_count"],
                    "matches": buy["matches"][:5],
                    "text_preview": features.get("text", "")[:80],
                }
                if not speaker_role:
                    meta["needs_role_validation"] = True
                signals.append(_make_signal(
                    speaker_id, "buying_signal",
                    buy["strength"], buy["level"],
                    buy["confidence"],
                    start_ms, end_ms, meta,
                ))

        # ── LANG-OBJ-01: Objection Signal Detection (sales_call only) ──
        if active_type in self.SALES_TYPES:
            obj = self._rule_objection_01(features)
            if obj is not None:
                signals.append(_make_signal(
                    speaker_id, "objection_signal",
                    obj["strength"], obj["level"],
                    obj["confidence"],
                    start_ms, end_ms,
                    {
                        "categories": obj["categories"],
                        "hedge_count": obj.get("hedge_count", 0),
                        "matches": obj["matches"][:5],
                        "text_preview": features.get("text", "")[:80],
                    },
                ))

        # ── LANG-PWR-01: Power Language Score (all content types) ──
        pwr = self._rule_power_01(features)
        if pwr is not None:
            signals.append(_make_signal(
                speaker_id, "power_language_score",
                pwr["score"], pwr["level"],
                pwr["confidence"],
                start_ms, end_ms,
                {
                    "powerless_feature_count": pwr["powerless_count"],
                    "features_found": pwr["features_found"],
                    "word_count": pwr["word_count"],
                    "text_preview": features.get("text", "")[:80],
                },
            ))

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

    def _rule_sentiment_01(self, f: dict) -> dict:
        """
        Convert DistilBERT sentiment into a -1.0 to +1.0 signal.
        Always fires: neutral at confidence 0.40 (dashboard subtle indicator per spec).
        """
        raw_label = f.get("sentiment_label", "NEUTRAL")
        raw_score = f.get("sentiment_score", 0.5)
        value = f.get("sentiment_value", 0.0)

        # Classify intensity (thresholds aligned with LLM ±0.35 neutral zone)
        abs_value = abs(value)
        if abs_value > 0.80:
            label = "strong_positive" if value > 0 else "strong_negative"
            confidence = 0.80
        elif abs_value > 0.55:
            label = "positive" if value > 0 else "negative"
            confidence = 0.70
        elif abs_value > 0.35:
            label = "mild_positive" if value > 0 else "mild_negative"
            confidence = 0.55
        else:
            label = "neutral"
            confidence = 0.40

        return {
            "value": value,
            "label": label,
            "confidence": min(confidence, self._MAX_CONFIDENCE),
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
        Skips greetings, questions, introductions, and single-word responses.
        """
        word_count = f.get("power_word_count", 0)
        if word_count < 8:
            return None  # Too short to assess meaningfully

        text = f.get("text", "").strip()
        text_lower = text.lower()

        # Skip short questions — power scoring doesn't apply to "Have you worked in education?"
        if text.rstrip().endswith("?") and word_count < 12:
            return None

        # Skip greetings and single-word acknowledgements
        first_word = text_lower.split()[0] if text_lower else ""
        if first_word in {"hello", "hi", "hey", "yes", "sure", "thank", "thanks", "okay", "bye"}:
            if word_count < 12:
                return None

        # Skip introductions ("this is X from Y", "calling you from")
        if ("calling" in text_lower and "from" in text_lower) or \
           ("this is" in text_lower and "from" in text_lower) or \
           "my name is" in text_lower:
            return None

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
            "confidence": min(confidence, self._MAX_CONFIDENCE),
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
        # Build numbered utterance list for the prompt.
        # Track which batch indices have valid text so LLM IDs map back correctly.
        utterance_lines = []
        valid_batch_indices = []  # Maps prompt ID (1-based) → batch index
        for i, f in enumerate(batch):
            speaker = f.get("speaker_id", "unknown")
            text = f.get("text", "").strip()
            if text:
                valid_batch_indices.append(i)
                utterance_lines.append(f"{len(utterance_lines)+1}. [{speaker}]: {text}")

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
            response_text = llm_complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=1024,
            )

            # Robust JSON extraction — handles markdown, extra text, etc.
            import re
            # Try direct parse first
            try:
                classifications = json.loads(response_text)
            except json.JSONDecodeError:
                # Try extracting JSON array from markdown or surrounding text
                json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                if json_match:
                    classifications = json.loads(json_match.group())
                else:
                    logger.warning(f"No JSON array found in LLM response: {response_text[:200]}")
                    return []
        except Exception as e:
            logger.warning(f"LLM API call failed for intent classification: {e}")
            return []

        # Convert to signals — map LLM IDs back via valid_batch_indices
        signals = []
        for cls in classifications:
            prompt_id = cls.get("id", 0) - 1  # Convert 1-based to 0-based
            if prompt_id < 0 or prompt_id >= len(valid_batch_indices):
                continue

            batch_idx = valid_batch_indices[prompt_id]
            f = batch[batch_idx]
            raw_confidence = min(float(cls.get("confidence", 0.5)), 0.85)

            if raw_confidence < 0.40:
                continue

            signals.append(_make_signal(
                f.get("speaker_id", "unknown"),
                "intent_classification",
                1.0,  # Intent is categorical: 1.0 = detected
                cls.get("intent", "UNKNOWN"),
                raw_confidence,
                f.get("start_ms", 0),
                f.get("end_ms", 0),
                {
                    "intent": cls.get("intent", "UNKNOWN"),
                    "text_preview": f.get("text", "")[:80],
                },
            ))

        return signals
