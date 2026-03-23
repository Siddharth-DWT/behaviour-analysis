"""
tests/test_language_agent.py
Unit tests for Language Agent rule engine and feature extractor.
Feeds transcript text directly — no audio or API calls needed.
"""
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from services.language_agent.feature_extractor import LanguageFeatureExtractor
from services.language_agent.rules import LanguageRuleEngine


def make_features(text: str, sentiment_value: float = 0.0,
                  sentiment_label: str = "NEUTRAL",
                  buying_count: int = 0, buying_cats: list = None,
                  obj_count: int = 0, obj_cats: list = None,
                  power_score: float = 0.6, word_count: int = 10,
                  powerless_count: int = 0, start_ms: int = 0, end_ms: int = 3000):
    """Helper to build feature dicts directly (bypasses DistilBERT)."""
    return {
        "text": text,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "sentiment_label": sentiment_label,
        "sentiment_score": abs(sentiment_value),
        "sentiment_value": sentiment_value,
        "buying_signals": [],
        "buying_signal_count": buying_count,
        "buying_categories": buying_cats or [],
        "objection_signals": [],
        "objection_signal_count": obj_count,
        "objection_categories": obj_cats or [],
        "power_score": power_score,
        "powerless_feature_count": powerless_count,
        "powerless_features_found": [],
        "power_word_count": word_count,
    }


# ═══════════════════════════════════════════════════════════════
# LANG-SENT-01: Sentiment
# ═══════════════════════════════════════════════════════════════

class TestSentiment01:

    def setup_method(self):
        self.engine = LanguageRuleEngine()

    def test_sentiment_strong_positive(self):
        f = make_features("Amazing product", sentiment_value=0.95, sentiment_label="POSITIVE")
        result = self.engine._rule_sentiment_01(f)
        assert result is not None
        assert result["value"] == pytest.approx(0.95)
        assert result["label"] == "strong_positive"
        assert result["confidence"] >= 0.70

    def test_sentiment_negative(self):
        f = make_features("Terrible results", sentiment_value=-0.80, sentiment_label="NEGATIVE")
        result = self.engine._rule_sentiment_01(f)
        assert result is not None
        assert result["value"] < -0.30
        assert "negative" in result["label"]

    def test_sentiment_neutral(self):
        f = make_features("Meeting on Tuesday", sentiment_value=0.50, sentiment_label="NEUTRAL")
        result = self.engine._rule_sentiment_01(f)
        assert result is not None
        # abs(0.50) is not > 0.60, so it should be neutral
        assert result["label"] == "neutral"

    def test_sentiment_confidence_cap(self):
        """No signal confidence exceeds 0.85."""
        f = make_features("text", sentiment_value=1.0, sentiment_label="POSITIVE")
        result = self.engine._rule_sentiment_01(f)
        assert result["confidence"] <= 0.85


# ═══════════════════════════════════════════════════════════════
# LANG-BUY-01: Buying Signals (via feature extractor patterns)
# ═══════════════════════════════════════════════════════════════

class TestBuying01:

    def setup_method(self):
        self.extractor = LanguageFeatureExtractor()
        self.engine = LanguageRuleEngine()

    def _detect(self, text: str):
        """Extract buying features from text and run the rule."""
        buying = self.extractor._detect_buying_signals(text)
        f = make_features(text,
                          buying_count=buying["count"],
                          buying_cats=buying["categories"])
        return self.engine._rule_buying_01(f)

    def test_buying_specification(self):
        result = self._detect("Have you worked in education before?")
        assert result is not None, "Expected buying signal for specification question"

    def test_buying_next_step(self):
        result = self._detect("Send me your email and let's schedule a call.")
        assert result is not None, "Expected buying signal for next step acceptance"

    def test_buying_pricing(self):
        result = self._detect("What's the cost for 50 users?")
        assert result is not None, "Expected buying signal for pricing question"

    def test_buying_risk_eval(self):
        result = self._detect("Worst case scenario, what happens if it doesn't work?")
        assert result is not None, "Expected buying signal for risk evaluation"

    def test_no_buying_signal(self):
        result = self._detect("Thanks for calling, we're not interested.")
        assert result is None, "Expected no buying signal for rejection"

    def test_buying_signal_structure(self):
        result = self._detect("How much does the enterprise plan cost?")
        assert result is not None
        assert "strength" in result
        assert "level" in result
        assert "confidence" in result
        assert result["confidence"] <= 0.85


# ═══════════════════════════════════════════════════════════════
# LANG-OBJ-01: Objection Detection (via feature extractor patterns)
# ═══════════════════════════════════════════════════════════════

class TestObjection01:

    def setup_method(self):
        self.extractor = LanguageFeatureExtractor()
        self.engine = LanguageRuleEngine()

    def _detect(self, text: str, sentiment_value: float = -0.3):
        obj = self.extractor._detect_objection_signals(text)
        f = make_features(text,
                          obj_count=obj["count"],
                          obj_cats=obj["categories"],
                          sentiment_value=sentiment_value)
        return self.engine._rule_objection_01(f)

    def test_direct_objection(self):
        result = self._detect("We're not looking for anyone right now.")
        assert result is not None
        assert result["level"] == "direct_objection"

    def test_budget_objection(self):
        result = self._detect("We can't afford that, it's way too expensive.")
        assert result is not None

    def test_hedge_objection(self):
        result = self._detect("Well, I'm not sure, maybe, perhaps sort of, I guess it might not work for us.")
        assert result is not None

    def test_no_objection(self):
        result = self._detect("Tell me more about the enterprise plan.", sentiment_value=0.5)
        assert result is None

    def test_objection_boosted_by_sentiment(self):
        """Objection + strong negative sentiment → higher confidence."""
        result_neutral = self._detect("We're not interested.", sentiment_value=-0.3)
        result_negative = self._detect("We're not interested.", sentiment_value=-0.85)
        if result_neutral and result_negative:
            assert result_negative["confidence"] >= result_neutral["confidence"]


# ═══════════════════════════════════════════════════════════════
# LANG-PWR-01: Power Language Score
# ═══════════════════════════════════════════════════════════════

class TestPower01:

    def setup_method(self):
        self.extractor = LanguageFeatureExtractor()
        self.engine = LanguageRuleEngine()

    def _score(self, text: str):
        power = self.extractor._score_power_language(text)
        f = make_features(text,
                          power_score=power["score"],
                          word_count=power["word_count"],
                          powerless_count=power["powerless_count"])
        return self.engine._rule_power_01(f)

    def test_powerful(self):
        result = self._score("We will implement this by Friday. The results speak for themselves.")
        assert result is not None
        assert result["score"] >= 0.60, f"Expected score >= 0.60, got {result['score']}"

    def test_powerless(self):
        result = self._score("I kind of think maybe we could sort of try this, you know, I suppose, perhaps?")
        assert result is not None
        assert result["score"] < 0.50, f"Expected score < 0.50, got {result['score']}"

    def test_mixed(self):
        result = self._score("I think the data clearly shows significant improvement in the results.")
        assert result is not None
        # Mixed: has hedge "i think" but also strong words
        assert result is not None

    def test_too_short_skipped(self):
        """Utterances under 5 words return None."""
        f = make_features("Sure.", word_count=1, power_score=0.9, powerless_count=0)
        result = self.engine._rule_power_01(f)
        assert result is None


# ═══════════════════════════════════════════════════════════════
# LANG-INTENT-01: Intent Classification (mocked LLM)
# ═══════════════════════════════════════════════════════════════

class TestIntent01:

    def setup_method(self):
        self.engine = LanguageRuleEngine()

    def _run_intent(self, mock_llm, response_json: str, text: str, speaker="spk"):
        mock_llm.set_response(response_json)
        features = [{"text": text, "speaker_id": speaker, "start_ms": 0, "end_ms": 3000}]
        return self.engine.evaluate_batch_intent(features)

    def test_intent_question(self, mock_llm_client):
        mock_llm_client.set_response('[{"id": 1, "intent": "QUESTION", "confidence": 0.85}]')
        features = [{"text": "What features do you offer?", "speaker_id": "spk", "start_ms": 0, "end_ms": 3000}]
        signals = self.engine.evaluate_batch_intent(features)
        assert len(signals) == 1
        assert signals[0]["value_text"] == "QUESTION"

    def test_intent_objection(self, mock_llm_client):
        mock_llm_client.set_response('[{"id": 1, "intent": "OBJECTION", "confidence": 0.80}]')
        features = [{"text": "We can't afford that right now.", "speaker_id": "spk", "start_ms": 0, "end_ms": 3000}]
        signals = self.engine.evaluate_batch_intent(features)
        assert len(signals) == 1
        assert signals[0]["value_text"] == "OBJECTION"

    def test_intent_commit(self, mock_llm_client):
        mock_llm_client.set_response('[{"id": 1, "intent": "COMMIT", "confidence": 0.90}]')
        features = [{"text": "Let's go ahead with the enterprise plan.", "speaker_id": "spk", "start_ms": 0, "end_ms": 3000}]
        signals = self.engine.evaluate_batch_intent(features)
        assert len(signals) == 1
        assert signals[0]["value_text"] == "COMMIT"

    def test_intent_low_confidence_filtered(self, mock_llm_client):
        """Responses with confidence < 0.40 are filtered out."""
        mock_llm_client.set_response('[{"id": 1, "intent": "DEFLECT", "confidence": 0.30}]')
        features = [{"text": "Hmm, interesting.", "speaker_id": "spk", "start_ms": 0, "end_ms": 3000}]
        signals = self.engine.evaluate_batch_intent(features)
        assert signals == []

    def test_intent_llm_not_configured_returns_empty(self, monkeypatch):
        """If LLM not configured, evaluate_batch_intent returns []."""
        import services.language_agent.rules as rules_mod
        monkeypatch.setattr(rules_mod, "_llm_ready", False)
        features = [{"text": "Some text", "speaker_id": "spk", "start_ms": 0, "end_ms": 3000}]
        signals = self.engine.evaluate_batch_intent(features)
        assert signals == []


# ═══════════════════════════════════════════════════════════════
# Feature Extractor — Pattern matching smoke tests
# ═══════════════════════════════════════════════════════════════

class TestFeatureExtractorPatterns:

    def setup_method(self):
        self.extractor = LanguageFeatureExtractor()

    def test_buying_patterns_compile(self):
        """Compiled patterns should not raise on typical text."""
        result = self.extractor._detect_buying_signals("What is the pricing for this?")
        assert isinstance(result, dict)
        assert result["count"] >= 0

    def test_objection_patterns_compile(self):
        result = self.extractor._detect_objection_signals("We already have a solution.")
        assert isinstance(result, dict)
        assert result["count"] >= 1

    def test_power_score_range(self):
        """Power score must always be 0.0 to 1.0."""
        texts = [
            "Yes.",
            "I think maybe sort of perhaps we could kind of try, you know?",
            "We will execute this plan immediately and deliver results.",
        ]
        for text in texts:
            result = self.extractor._score_power_language(text)
            assert 0.0 <= result["score"] <= 1.0, f"Score out of range for: {text}"
