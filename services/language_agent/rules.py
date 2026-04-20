# services/language_agent/rules.py
"""
NEXUS Language Agent - Rule Engine
Implements 12 core rules from the NEXUS Rule Engine specification.

Each rule takes linguistic features and produces Signal dicts.
Thresholds are hardcoded defaults matching the Rule Engine document;
in production these load from the rule_config database table.

Rules implemented:
  LANG-SENT-01: Per-sentence sentiment (DistilBERT)
  LANG-SENT-02: Emotional intensity (positive/negative word density)
  LANG-BUY-01:  Buying signal detection (SPIN keyword patterns)
  LANG-OBJ-01:  Objection signal detection (hedges + resistance)
  LANG-PWR-01:  Power language score (Lakoff/O'Barr)
  LANG-PERS-01: Persuasion technique detection (Cialdini 2006)
  LANG-QUES-01: Question type classification (SPIN)
  LANG-NEG-01:  Gottman Four Horsemen detection
  LANG-EMP-01:  Empathy language detection
  LANG-CLAR-01: Clarity score
  LANG-TOPIC-01: Topic shift detection
  LANG-INTENT-01: Intent classification (Claude API batch)

Research references:
  - Liu 2012 (sentiment)
  - Pennebaker 2015 (word-level affect)
  - Rackham 1988 (SPIN Selling — 35,000 calls)
  - Lakoff 1975 (language & gender / powerless speech)
  - O'Barr & Atkins 1982 (powerless speech in courtrooms)
  - Cialdini 2006 (influence: science and practice)
  - Gottman 1994 (four horsemen of the apocalypse)
  - Rogers 1957 (empathic understanding)
"""
import json
import logging
import re
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
    from shared.utils.llm_client import is_configured, get_provider_info, complete as llm_complete, acomplete as llm_acomplete
    _LLM_IMPORT_OK = True
except ImportError:
    _LLM_IMPORT_OK = False
    llm_complete = None  # type: ignore[assignment]
    llm_acomplete = None  # type: ignore[assignment]

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
SHORT_UTTERANCE_WORDS = 5  # ≤5 words need surrounding context for classification


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

    # ── Word sets for emotional intensity (LANG-SENT-02) ──
    POSITIVE_EMOTION_WORDS = {
        "happy", "great", "love", "wonderful", "fantastic", "amazing",
        "excellent", "perfect", "beautiful", "awesome", "glad", "pleased",
        "delighted", "thrilled", "grateful", "proud", "joy", "excited",
        "brilliant", "superb",
    }
    NEGATIVE_EMOTION_WORDS = {
        "angry", "hate", "terrible", "awful", "horrible", "disgusting",
        "frustrated", "annoyed", "upset", "worried", "scared", "anxious",
        "sad", "depressed", "miserable", "furious", "disappointed", "hurt",
        "painful", "dreadful",
    }

    # ── Persuasion regex patterns (LANG-PERS-01, Cialdini 2006) ──
    PERSUASION_PATTERNS = {
        "scarcity": re.compile(
            r"limited\s+time|only\s+\d+\s+left|offer\s+expires|last\s+chance|act\s+now|don'?t\s+miss",
            re.IGNORECASE,
        ),
        "social_proof": re.compile(
            r"other\s+compan(ies|y)|other\s+clients?|most\s+of\s+our|industry\s+standard|trusted\s+by",
            re.IGNORECASE,
        ),
        "authority": re.compile(
            r"research\s+shows|studies\s+indicate|experts?\s+recommend|certified|award[- ]winning",
            re.IGNORECASE,
        ),
        "reciprocity": re.compile(
            r"free.{0,20}send|let\s+me\s+share|as\s+a\s+thank\s+you|complimentary|no\s+obligation",
            re.IGNORECASE,
        ),
        "commitment": re.compile(
            r"you\s+mentioned\s+earlier|as\s+you\s+agreed|you\s+said\s+that|building\s+on\s+what\s+you",
            re.IGNORECASE,
        ),
        "scarcity_manufactured": re.compile(
            r"need\s+to\s+know\s+by|other\s+people\s+are\s+looking|can'?t\s+hold\s+this|decision\s+today",
            re.IGNORECASE,
        ),
    }

    # ── Gottman Four Horsemen patterns (LANG-NEG-01) ──
    GOTTMAN_PATTERNS = {
        "criticism": re.compile(
            r"you\s+always|you\s+never|what'?s\s+wrong\s+with\s+you|why\s+can'?t\s+you",
            re.IGNORECASE,
        ),
        "contempt": re.compile(
            r"\bwhatever\b|that'?s\s+ridiculous|i\s+don'?t\s+care|yeah\s+right|good\s+luck\s+with\s+that",
            re.IGNORECASE,
        ),
        "defensiveness": re.compile(
            r"that'?s\s+not\s+my\s+fault|yes\s+but\s+you|if\s+you\s+had\s+just|that'?s\s+not\s+what\s+i\s+said",
            re.IGNORECASE,
        ),
    }
    # Stonewalling: very short responses (handled separately in the rule)
    STONEWALLING_RESPONSES = {"fine", "whatever", "i don't know", "okay", "k"}

    # ── Empathy patterns (LANG-EMP-01, Rogers 1957) ──
    VALIDATION_PHRASES = [
        "i understand", "that makes sense", "i can see why", "i hear you",
        "you're right", "that must be", "i appreciate", "i can imagine",
    ]
    REFLECTION_PHRASES = [
        "so what you're saying", "it sounds like",
        "if i understand correctly", "in other words",
    ]

    # ── Stop words for topic shift (LANG-TOPIC-01) ──
    STOP_WORDS = {
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
        "your", "yours", "yourself", "yourselves", "he", "him", "his",
        "himself", "she", "her", "hers", "herself", "it", "its", "itself",
        "they", "them", "their", "theirs", "themselves", "what", "which",
        "who", "whom", "this", "that", "these", "those", "am", "is", "are",
        "was", "were", "be", "been", "being", "have", "has", "had", "having",
        "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
        "or", "because", "as", "until", "while", "of", "at", "by", "for",
        "with", "about", "against", "between", "through", "during", "before",
        "after", "above", "below", "to", "from", "up", "down", "in", "out",
        "on", "off", "over", "under", "again", "further", "then", "once",
        "here", "there", "when", "where", "why", "how", "all", "both",
        "each", "few", "more", "most", "other", "some", "such", "no", "nor",
        "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
        "can", "will", "just", "don", "should", "now", "d", "ll", "m", "o",
        "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn",
        "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", "shan",
        "shouldn", "wasn", "weren", "won", "wouldn", "yeah", "yes", "no",
        "okay", "ok", "um", "uh", "like", "know", "think", "right", "well",
        "going", "got", "get", "would", "could", "really", "actually",
        "thing", "things", "gonna", "want", "let", "say", "said",
    }

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
        all_features_list: Optional[list] = None,
        current_index: int = 0,
        profile=None,
    ) -> list[dict]:
        """
        Run all rules against a single segment's features.

        Args:
            features: Feature dict from LanguageFeatureExtractor
            speaker_id: Speaker identifier
            content_type: Optional override for content type
            speaker_role: Optional speaker role (e.g. "Seller", "Prospect")
            all_features_list: Full list of features for all segments (for topic shift)
            current_index: Index of current segment in all_features_list
            profile: ContentTypeProfile for content-aware gating/renaming

        Returns:
            List of Signal dicts (one per fired rule)
        """
        signals = []
        start_ms = features.get("start_ms", 0)
        end_ms = features.get("end_ms", 0)
        active_type = content_type or self._content_type

        def _add(rule_id: str, signal_type: str, value, value_text: str,
                 confidence: float, metadata: dict):
            if profile and profile.is_gated(rule_id):
                return
            conf = confidence
            if profile:
                conf = profile.apply_confidence(rule_id, conf)
            renamed_type = profile.rename_signal(signal_type) if profile else signal_type
            renamed_text = profile.rename_signal(value_text) if profile else value_text
            signals.append(_make_signal(
                speaker_id, renamed_type, value, renamed_text, conf,
                start_ms, end_ms, metadata,
            ))

        # ── LANG-SENT-01: Per-Sentence Sentiment (all content types) ──
        sent = self._rule_sentiment_01(features)
        _add("LANG-SENT-01", "sentiment_score",
             sent["value"], sent["label"], sent["confidence"],
             {"raw_label": sent["raw_label"], "raw_score": sent["raw_score"],
              "text_preview": features.get("text", "")[:80]})

        # ── LANG-SENT-02: Emotional Intensity (all content types) ──
        # high_pct / suppressed_pct are content-type-aware.
        # Internal meetings: >6% high, <1.5% suppressed (Tausczik 2010: professional speech).
        emo_high_pct       = profile.get_threshold("LANG-SENT-02", "high_pct",       0.08) if profile else 0.08
        emo_suppressed_pct = profile.get_threshold("LANG-SENT-02", "suppressed_pct", 0.02) if profile else 0.02
        emo = self._rule_sent_02(features, high_pct=emo_high_pct, suppressed_pct=emo_suppressed_pct)
        if emo is not None:
            _add("LANG-SENT-02", "emotional_intensity",
                 emo["value"], emo["label"], emo["confidence"],
                 {"rule": "LANG-SENT-02", "density": emo["density"],
                  "positive_count": emo["positive_count"],
                  "negative_count": emo["negative_count"],
                  "word_count": emo["word_count"],
                  "text_preview": features.get("text", "")[:80]})

        # ── LANG-BUY-01: Buying Signal Detection ──
        # Profile handles gating (internal/podcast suppressed) and renaming
        role_lower = (speaker_role or "").lower()
        is_seller = role_lower in ("seller", "agent", "rep", "salesperson")

        if not is_seller:
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
                _add("LANG-BUY-01", "buying_signal",
                     buy["strength"], buy["level"],
                     buy["confidence"], meta)

        # ── LANG-OBJ-01: Objection Signal Detection ──
        # Profile handles gating (podcast suppressed) and renaming
        obj = self._rule_objection_01(features)
        if obj is not None:
            _add("LANG-OBJ-01", "objection_signal",
                 obj["strength"], obj["level"], obj["confidence"],
                 {"categories": obj["categories"],
                  "hedge_count": obj.get("hedge_count", 0),
                  "matches": obj["matches"][:5],
                  "text_preview": features.get("text", "")[:80]})

        # ── LANG-PWR-01: Power Language Score (all content types) ──
        pwr = self._rule_power_01(features)
        if pwr is not None:
            _add("LANG-PWR-01", "power_language_score",
                 pwr["score"], pwr["level"], pwr["confidence"],
                 {"powerless_feature_count": pwr["powerless_count"],
                  "features_found": pwr["features_found"],
                  "word_count": pwr["word_count"],
                  "text_preview": features.get("text", "")[:80]})

        # ── LANG-PERS-01: Persuasion Detection ──
        pers = self._rule_pers_01(features, active_type)
        if pers is not None:
            _add("LANG-PERS-01", "persuasion_technique",
                 pers["value"], pers["value_text"], pers["confidence"],
                 {"rule": "LANG-PERS-01", "techniques_found": pers["techniques_found"],
                  "detected_count": pers["detected_count"],
                  "text_preview": features.get("text", "")[:80]})

        # ── LANG-QUES-01: Question Type ──
        ques = self._rule_ques_01(features, active_type)
        if ques is not None:
            _add("LANG-QUES-01", "question_type",
                 ques["value"], ques["label"], ques["confidence"],
                 {"rule": "LANG-QUES-01", "question_category": ques["category"],
                  "spin_type": ques.get("spin_type"),
                  "text_preview": features.get("text", "")[:80]})

        # ── LANG-NEG-01: Gottman Four Horsemen ──
        neg = self._rule_neg_01(features, active_type, all_features_list, current_index)
        if neg is not None:
            _add("LANG-NEG-01", "gottman_horsemen",
                 neg["value"], neg["label"], neg["confidence"],
                 {"rule": "LANG-NEG-01", "horseman": neg["horseman"],
                  "matched_pattern": neg.get("matched_pattern", ""),
                  "text_preview": features.get("text", "")[:80]})

        # ── LANG-EMP-01: Empathy Language ──
        emp = self._rule_emp_01(features, active_type)
        if emp is not None:
            _add("LANG-EMP-01", "empathy_language",
                 emp["value"], emp["label"], emp["confidence"],
                 {"rule": "LANG-EMP-01", "validation_count": emp["validation_count"],
                  "reflection_count": emp["reflection_count"],
                  "text_preview": features.get("text", "")[:80]})

        # ── LANG-CLAR-01: Clarity Score ──
        clar = self._rule_clar_01(features, active_type)
        if clar is not None:
            _add("LANG-CLAR-01", "clarity_score",
                 clar["value"], clar["label"], clar["confidence"],
                 {"rule": "LANG-CLAR-01", "penalties": clar.get("penalties", []),
                  "bonuses": clar.get("bonuses", []),
                  "text_preview": features.get("text", "")[:80]})

        # ── LANG-TOPIC-01: Topic Shift (requires full features list) ──
        if all_features_list is not None:
            topic = self._rule_topic_01(features, all_features_list, current_index)
            if topic is not None:
                _add("LANG-TOPIC-01", "topic_shift",
                     topic["value"], topic["label"], topic["confidence"],
                     {"rule": "LANG-TOPIC-01", "overlap_pct": topic["overlap_pct"],
                      "threshold_pct": topic.get("threshold_pct", 0.15),
                      "method": topic.get("method", "fixed"),
                      "current_keywords": topic.get("current_keywords", [])[:10],
                      "text_preview": features.get("text", "")[:80]})

        return signals

    async def evaluate_batch_intent(
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
            batch_signals = await self._classify_intent_batch(
                batch, all_features=features_list, batch_offset=batch_start,
            )
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
    # LANG-SENT-02: Emotional Intensity
    # Research: Pennebaker 2015 (word-level affect)
    # ════════════════════════════════════════════════════════

    def _rule_sent_02(self, f: dict, high_pct: float = 0.08, suppressed_pct: float = 0.02) -> Optional[dict]:
        """
        Measure emotional word density as a percentage of total words.
        Fires when density is notably high, moderate, or suppressed.
        Also detects negative emotional shifts.
        """
        text = f.get("text", "").strip().lower()
        words = text.split()
        word_count = len(words)
        if word_count == 0:
            return None

        pos_count = sum(1 for w in words if w in self.POSITIVE_EMOTION_WORDS)
        neg_count = sum(1 for w in words if w in self.NEGATIVE_EMOTION_WORDS)
        total_emotion = pos_count + neg_count
        density = total_emotion / word_count

        label = None

        # Check for negative emotional shift first
        if neg_count > pos_count * 1.5 and neg_count >= 2:
            label = "negative_emotional_shift"
        elif density > high_pct:
            label = "high"
        elif density > 0.04:
            label = "moderate"
        elif density < suppressed_pct and word_count >= 15:
            label = "suppressed"

        if label is None:
            return None  # Normal range — don't emit

        return {
            "value": round(density, 4),
            "label": label,
            "confidence": 0.60,
            "density": round(density, 4),
            "positive_count": pos_count,
            "negative_count": neg_count,
            "word_count": word_count,
        }

    # ════════════════════════════════════════════════════════
    # LANG-PERS-01: Persuasion Detection (Cialdini 2006)
    # ════════════════════════════════════════════════════════

    def _rule_pers_01(self, f: dict, content_type: str) -> Optional[dict]:
        """
        Detect persuasion techniques based on Cialdini's 6 principles.
        Uses regex patterns per category; emits on first match.
        """
        text = f.get("text", "").strip()
        if not text:
            return None

        techniques_found = []
        for technique, pattern in self.PERSUASION_PATTERNS.items():
            if pattern.search(text):
                techniques_found.append(technique)

        if not techniques_found:
            return None

        detected_count = len(techniques_found)
        value = min(detected_count / 7.0, 1.0)
        primary_technique = techniques_found[0]

        return {
            "value": round(value, 4),
            "value_text": primary_technique,
            "confidence": 0.55,
            "techniques_found": techniques_found,
            "detected_count": detected_count,
        }

    # ════════════════════════════════════════════════════════
    # LANG-QUES-01: Question Type Classification (SPIN)
    # Research: Rackham 1988
    # ════════════════════════════════════════════════════════

    _QUESTION_STARTERS = re.compile(
        r"^(how|what|why|when|where|who|do you|have you|can you|could you|would you|is it|are you|did you|will you|shall we)",
        re.IGNORECASE,
    )
    _TAG_QUESTION = re.compile(
        r"(right\s*\??|don'?t\s+you\s+think\s*\??|isn'?t\s+it\s*\??|aren'?t\s+you\s*\??|won'?t\s+you\s*\??)$",
        re.IGNORECASE,
    )

    # SPIN sub-classification keywords (for sales_call)
    _SPIN_SITUATION = re.compile(
        r"(how\s+many|how\s+long|how\s+often|currently|right\s+now|at\s+the\s+moment|what\s+do\s+you\s+use|tell\s+me\s+about\s+your)",
        re.IGNORECASE,
    )
    _SPIN_PROBLEM = re.compile(
        r"(difficult|challenge|problem|issue|frustrat|struggle|dissatisfied|concern|worry|trouble|pain\s+point)",
        re.IGNORECASE,
    )
    _SPIN_IMPLICATION = re.compile(
        r"(what\s+happens\s+if|what\s+would|how\s+does\s+that\s+affect|impact|consequence|result\s+in|lead\s+to|cost\s+you)",
        re.IGNORECASE,
    )
    _SPIN_NEED_PAYOFF = re.compile(
        r"(would\s+it\s+help|how\s+would|what\s+if\s+you\s+could|imagine|benefit|value|useful|important\s+to\s+you|solve)",
        re.IGNORECASE,
    )

    def _rule_ques_01(self, f: dict, content_type: str) -> Optional[dict]:
        """
        Detect and classify question types.
        For sales_call: sub-classify into SPIN categories.
        """
        text = f.get("text", "").strip()
        if not text:
            return None

        is_question = text.rstrip().endswith("?") or bool(self._QUESTION_STARTERS.match(text))
        if not is_question:
            return None

        # Classify question type
        if self._TAG_QUESTION.search(text):
            category = "tag"
        elif re.match(r"^(how|why|tell\s+me)", text, re.IGNORECASE):
            category = "open"
        else:
            category = "closed"

        result = {
            "value": 1.0,
            "label": category,
            "confidence": 0.70,
            "category": category,
        }

        # SPIN sub-classification for sales calls
        if content_type in self.SALES_TYPES:
            spin_type = None
            if self._SPIN_NEED_PAYOFF.search(text):
                spin_type = "need_payoff"
            elif self._SPIN_IMPLICATION.search(text):
                spin_type = "implication"
            elif self._SPIN_PROBLEM.search(text):
                spin_type = "problem"
            elif self._SPIN_SITUATION.search(text):
                spin_type = "situation"

            if spin_type:
                result["label"] = f"{category}_{spin_type}"
                result["spin_type"] = spin_type

        return result

    # ════════════════════════════════════════════════════════
    # LANG-NEG-01: Gottman Four Horsemen
    # Research: Gottman 1994
    # ════════════════════════════════════════════════════════

    _QUESTION_RE = re.compile(r"\?|^(what|where|when|why|who|how|do|does|did|can|could|will|would|is|are|was|were|have|has)\b", re.IGNORECASE)

    def _rule_neg_01(
        self, f: dict, content_type: str,
        all_features_list: Optional[list] = None, current_index: int = 0,
    ) -> Optional[dict]:
        """
        Detect the Four Horsemen of relationship breakdown:
        criticism, contempt, defensiveness, stonewalling.
        Stonewalling only fires when preceded by a substantive question from another speaker.
        """
        text = f.get("text", "").strip()
        if not text:
            return None

        text_lower = text.lower().strip()

        # Check stonewalling first: very short responses
        if text_lower in self.STONEWALLING_RESPONSES:
            # Only fire if the previous segment from a different speaker was a question
            prev_was_question = False
            if all_features_list is not None and current_index > 0:
                current_speaker = f.get("speaker_id", "")
                for look_back in range(1, min(4, current_index + 1)):
                    prev_f = all_features_list[current_index - look_back]
                    if prev_f.get("speaker_id", "") != current_speaker:
                        prev_text = prev_f.get("text", "").strip()
                        if prev_text and self._QUESTION_RE.search(prev_text):
                            prev_was_question = True
                        break
            else:
                prev_was_question = True  # No context available — preserve prior behaviour

            if not prev_was_question:
                return None  # "fine"/"okay" without a preceding question is not stonewalling

            return {
                "value": 0.4,
                "label": "stonewalling",
                "confidence": 0.35,
                "horseman": "stonewalling",
                "matched_pattern": text_lower,
            }

        # Check the three regex-based horsemen
        for horseman, pattern in self.GOTTMAN_PATTERNS.items():
            match = pattern.search(text)
            if match:
                return {
                    "value": 0.6,
                    "label": horseman,
                    "confidence": 0.50,
                    "horseman": horseman,
                    "matched_pattern": match.group(0),
                }

        return None

    # ════════════════════════════════════════════════════════
    # LANG-EMP-01: Empathy Language
    # Research: Rogers 1957 (empathic understanding)
    # ════════════════════════════════════════════════════════

    def _rule_emp_01(self, f: dict, content_type: str) -> Optional[dict]:
        """
        Detect empathy language: validation and reflection phrases.
        Score = validation*0.3 + reflection*0.4, capped at 1.0.
        """
        text = f.get("text", "").strip().lower()
        if not text:
            return None

        validation_count = sum(1 for phrase in self.VALIDATION_PHRASES if phrase in text)
        reflection_count = sum(1 for phrase in self.REFLECTION_PHRASES if phrase in text)

        if validation_count == 0 and reflection_count == 0:
            return None

        score = min(validation_count * 0.3 + reflection_count * 0.4, 1.0)

        if score >= 0.7:
            label = "high_empathy"
        elif score >= 0.3:
            label = "moderate_empathy"
        else:
            label = "low_empathy"

        return {
            "value": round(score, 4),
            "label": label,
            "confidence": 0.55,
            "validation_count": validation_count,
            "reflection_count": reflection_count,
        }

    # ════════════════════════════════════════════════════════
    # LANG-CLAR-01: Clarity Score
    # ════════════════════════════════════════════════════════

    _PASSIVE_INDICATORS = re.compile(
        r"\b(was|were|is|are|been|being)\s+\w+ed\b", re.IGNORECASE,
    )
    _STRUCTURE_MARKERS = re.compile(
        r"\b(first|second|third|finally|in\s+conclusion|to\s+summarize|next|lastly)\b",
        re.IGNORECASE,
    )
    _NUMBERS = re.compile(r"\b\d+\.?\d*\b")

    def _rule_clar_01(self, f: dict, content_type: str) -> Optional[dict]:
        """
        Score clarity of speech. Penalises long sentences and passive voice.
        Bonuses for structure markers and specificity (numbers).
        Only emits for notably high (>0.80) or low (<0.40) clarity.
        """
        text = f.get("text", "").strip()
        words = text.split()
        word_count = len(words)
        if word_count < 10:
            return None

        score = 1.0
        penalties = []
        bonuses = []

        # Penalty: average sentence length
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            if avg_sentence_length > 30:
                score -= 0.30
                penalties.append(f"very_long_sentences ({avg_sentence_length:.1f} avg)")
            elif avg_sentence_length > 18:
                score -= 0.15
                penalties.append(f"long_sentences ({avg_sentence_length:.1f} avg)")

        # Penalty: passive voice indicators
        passive_matches = self._PASSIVE_INDICATORS.findall(text)
        passive_pct = len(passive_matches) / word_count if word_count > 0 else 0
        if passive_pct > 0.05:
            score -= 0.10
            penalties.append(f"passive_voice ({len(passive_matches)} instances)")

        # Bonus: structure markers
        if self._STRUCTURE_MARKERS.search(text):
            score += 0.10
            bonuses.append("structure_markers")

        # Bonus: numbers/specificity (max 3 bonuses at 0.05 each)
        number_count = min(len(self._NUMBERS.findall(text)), 3)
        if number_count > 0:
            score += number_count * 0.05
            bonuses.append(f"specificity ({number_count} numbers)")

        score = max(0.0, min(1.0, score))

        # Only emit if notably high or low
        if score > 0.80:
            label = "high_clarity"
        elif score < 0.40:
            label = "low_clarity"
        else:
            return None  # Normal range — don't emit

        return {
            "value": round(score, 4),
            "label": label,
            "confidence": 0.50,
            "penalties": penalties,
            "bonuses": bonuses,
        }

    # ════════════════════════════════════════════════════════
    # LANG-TOPIC-01: Topic Shift Detection
    # ════════════════════════════════════════════════════════

    def _rule_topic_01(
        self, f: dict, all_features_list: list, current_index: int,
    ) -> Optional[dict]:
        """
        Detect topic shifts using an adaptive threshold (Hearst 1997 TextTiling).
        Computes rolling 3-segment overlaps for all prior windows, then sets
        threshold = mean(overlaps) - 1*std(overlaps), floored at 0.05.
        Falls back to fixed 0.15 when fewer than 4 prior segments exist.
        """
        if current_index == 0:
            return None

        def _content_words(text: str) -> set:
            words = re.findall(r"[a-z]+", text.lower())
            return {w for w in words if w not in self.STOP_WORDS and len(w) > 2}

        current_words = _content_words(f.get("text", ""))
        if len(current_words) < 3:
            return None

        # Gather content words from last 3 segments (immediate context window)
        lookback = max(0, current_index - 3)
        previous_words: set = set()
        for idx in range(lookback, current_index):
            prev_text = all_features_list[idx].get("text", "")
            previous_words.update(_content_words(prev_text))

        if not previous_words:
            return None

        overlap = current_words & previous_words
        overlap_pct = len(overlap) / len(current_words)

        # Adaptive threshold: mean - 1SD across all prior windows (Hearst 1997)
        threshold_pct = 0.15
        method = "fixed"
        if current_index >= 4:
            prior_overlaps = []
            for i in range(1, current_index):
                win_words = _content_words(all_features_list[i].get("text", ""))
                if len(win_words) < 3:
                    continue
                ctx_start = max(0, i - 3)
                ctx_words: set = set()
                for j in range(ctx_start, i):
                    ctx_words.update(_content_words(all_features_list[j].get("text", "")))
                if ctx_words:
                    ov = len(win_words & ctx_words) / len(win_words)
                    prior_overlaps.append(ov)
            if len(prior_overlaps) >= 3:
                mean_ov = sum(prior_overlaps) / len(prior_overlaps)
                variance = sum((x - mean_ov) ** 2 for x in prior_overlaps) / len(prior_overlaps)
                std_ov = variance ** 0.5
                threshold_pct = max(0.05, mean_ov - std_ov)
                method = "adaptive"

        if overlap_pct >= threshold_pct:
            return None  # Enough continuity — no topic shift

        return {
            "value": round(1.0 - overlap_pct, 4),
            "label": "topic_change_detected",
            "confidence": 0.55,
            "overlap_pct": round(overlap_pct, 4),
            "threshold_pct": round(threshold_pct, 4),
            "method": method,
            "current_keywords": list(current_words)[:10],
        }

    # ════════════════════════════════════════════════════════
    # LANG-INTENT-01: Intent Classification (Claude API)
    # Batched: 10-20 utterances per API call
    # ════════════════════════════════════════════════════════

    async def _classify_intent_batch(
        self, batch: list[dict],
        all_features: list[dict] | None = None,
        batch_offset: int = 0,
    ) -> list[dict]:
        """
        Send a batch of utterances to the LLM for intent classification.
        Short utterances (≤5 words) are enriched with surrounding context
        so the LLM can classify them accurately.
        """
        # Build numbered utterance list for the prompt.
        # Track which batch indices have valid text so LLM IDs map back correctly.
        utterance_lines = []
        valid_batch_indices = []  # Maps prompt ID (1-based) → batch index
        for i, f in enumerate(batch):
            speaker = f.get("speaker_id", "unknown")
            text = f.get("text", "").strip()
            if not text:
                continue

            valid_batch_indices.append(i)
            words = text.split()

            # Short utterance: enrich with surrounding context
            if len(words) <= SHORT_UTTERANCE_WORDS and all_features is not None:
                global_idx = batch_offset + i
                parts = []
                # Previous 2 segments
                for prev_off in [2, 1]:
                    prev_idx = global_idx - prev_off
                    if 0 <= prev_idx < len(all_features):
                        prev_f = all_features[prev_idx]
                        prev_text = prev_f.get("text", "").strip()
                        if prev_text:
                            parts.append(f"{prev_f.get('speaker_id', '?')}: {prev_text}")
                # Current (marked)
                parts.append(f">>> {speaker}: {text} <<<")
                # Next 1 segment
                next_idx = global_idx + 1
                if next_idx < len(all_features):
                    nxt_f = all_features[next_idx]
                    nxt_text = nxt_f.get("text", "").strip()
                    if nxt_text:
                        parts.append(f"{nxt_f.get('speaker_id', '?')}: {nxt_text}")
                utterance_lines.append(f"{len(utterance_lines)+1}. " + " | ".join(parts))
            else:
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
- ACKNOWLEDGE: Confirming receipt, accepting a point ("Fine.", "Okay.", "Right.", "Sure.")
- NEGOTIATE: Bargaining, counter-offering, or discussing terms
- COMMIT: Making a promise, commitment, or decision
- DEFLECT: Avoiding, redirecting, or evading a topic
- RAPPORT: Small talk, relationship building, empathy
- CLOSE: Attempting to close a deal or reach a decision
- OBJECTION: Raising a concern or barrier

IMPORTANT for context-enriched utterances (marked with >>> and <<<):
- Classify ONLY the marked segment. Surrounding text is context.
- Short responses after questions: "Fine.", "Okay.", "Sure." = ACKNOWLEDGE or AGREE
- "No.", "Nope." after a yes/no question = DISAGREE or INFORM
- "Yes.", "Yeah." = AGREE
- When in doubt for short utterances, prefer ACKNOWLEDGE over REQUEST.

Respond with a JSON array of objects: [{{"id": 1, "intent": "CATEGORY", "confidence": 0.0-1.0}}]
Only include utterances where you have confidence > 0.4. Return ONLY the JSON array, no other text.

Utterances:
{utterance_block}"""

        try:
            response_text = await llm_acomplete(
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
