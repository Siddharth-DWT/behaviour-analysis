"""
tests/test_fusion_agent.py
Unit tests for Fusion Agent cross-modal rules.
Feeds pre-built signal dicts directly — no audio or API calls needed.
"""
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from services.fusion_agent.rules import FusionRuleEngine


def voice_signal(signal_type, value, value_text="", window_start_ms=0, window_end_ms=5000, metadata=None):
    return {
        "agent": "voice",
        "speaker_id": "Speaker_0",
        "signal_type": signal_type,
        "value": value,
        "value_text": value_text,
        "confidence": 0.7,
        "window_start_ms": window_start_ms,
        "window_end_ms": window_end_ms,
        "metadata": metadata or {},
    }


def lang_signal(signal_type, value, value_text="", window_start_ms=0, window_end_ms=5000, metadata=None):
    return {
        "agent": "language",
        "speaker_id": "Speaker_0",
        "signal_type": signal_type,
        "value": value,
        "value_text": value_text,
        "confidence": 0.7,
        "window_start_ms": window_start_ms,
        "window_end_ms": window_end_ms,
        "metadata": metadata or {},
    }


@pytest.fixture
def engine():
    return FusionRuleEngine()


# ═══════════════════════════════════════════════════════════════
# FUSION-02: Content × Stress → Credibility
# ═══════════════════════════════════════════════════════════════

class TestFusion02:

    def test_credibility_match(self, engine):
        """Positive sentiment + LOW stress → no credibility concern (rule returns None)."""
        voice = [voice_signal("vocal_stress_score", 0.20, "low_stress")]
        lang = [lang_signal("sentiment_score", 0.80, "strong_positive")]
        result = engine._rule_fusion_02(voice, lang)
        # stress=0.20 ≤ 0.40 threshold → rule does not fire
        assert result is None

    def test_credibility_mismatch(self, engine):
        """Positive sentiment + HIGH stress → credibility concern signal."""
        voice = [voice_signal("vocal_stress_score", 0.75, "high_stress")]
        lang = [lang_signal("sentiment_score", 0.90, "strong_positive")]
        result = engine._rule_fusion_02(voice, lang)
        assert result is not None
        assert result["level"] in ("credibility_concern", "mild_incongruence")
        assert result["confidence"] <= 0.55  # Hard cap

    def test_credibility_insufficient_voice(self, engine):
        """No voice signals → rule returns None gracefully."""
        lang = [lang_signal("sentiment_score", 0.90, "strong_positive")]
        result = engine._rule_fusion_02([], lang)
        assert result is None

    def test_credibility_insufficient_language(self, engine):
        """No language signals → rule returns None gracefully."""
        voice = [voice_signal("vocal_stress_score", 0.75, "high_stress")]
        result = engine._rule_fusion_02(voice, [])
        assert result is None

    def test_credibility_confidence_hard_cap(self, engine):
        """Confidence must never exceed 0.55 for this deception-adjacent rule."""
        voice = [voice_signal("vocal_stress_score", 1.0, "high_stress")]
        lang = [lang_signal("sentiment_score", 1.0, "strong_positive")]
        result = engine._rule_fusion_02(voice, lang)
        if result is not None:
            assert result["confidence"] <= 0.55

    def test_credibility_negative_sentiment_no_fire(self, engine):
        """Negative sentiment + high stress → rule doesn't fire (logic requires positive)."""
        voice = [voice_signal("vocal_stress_score", 0.80, "high_stress")]
        lang = [lang_signal("sentiment_score", -0.80, "strong_negative")]
        result = engine._rule_fusion_02(voice, lang)
        assert result is None


# ═══════════════════════════════════════════════════════════════
# FUSION-07: Hedge Language × Positive Sentiment → Incongruence
# ═══════════════════════════════════════════════════════════════

class TestFusion07:

    def test_incongruence_detected(self, engine):
        """Positive sentiment + weak power → verbal incongruence."""
        lang = [
            lang_signal("sentiment_score", 0.80, "strong_positive"),
            lang_signal("power_language_score", 0.25, "weak_power",
                        metadata={"powerless_feature_count": 5}),
        ]
        result = engine._rule_fusion_07(lang)
        assert result is not None
        assert "incongruence" in result["level"]
        assert result["confidence"] <= 0.70

    def test_no_incongruence_high_power(self, engine):
        """Positive sentiment + HIGH power → no incongruence signal."""
        lang = [
            lang_signal("sentiment_score", 0.80, "strong_positive"),
            lang_signal("power_language_score", 0.75, "powerful"),
        ]
        result = engine._rule_fusion_07(lang)
        # power_value=0.75 >= 0.40 threshold → no signal
        assert result is None

    def test_no_incongruence_negative_sentiment(self, engine):
        """Negative sentiment + weak power → no signal (requires positive sentiment)."""
        lang = [
            lang_signal("sentiment_score", -0.70, "negative"),
            lang_signal("power_language_score", 0.20, "powerless"),
        ]
        result = engine._rule_fusion_07(lang)
        assert result is None

    def test_incongruence_single_source_no_voice(self, engine):
        """FUSION-07 is language-only — works without voice signals."""
        lang = [
            lang_signal("sentiment_score", 0.90, "strong_positive"),
            lang_signal("power_language_score", 0.15, "powerless"),
        ]
        result = engine._rule_fusion_07(lang)
        assert result is not None

    def test_incongruence_missing_signals(self, engine):
        """Missing either sentiment or power → returns None."""
        lang_no_power = [lang_signal("sentiment_score", 0.80, "strong_positive")]
        assert engine._rule_fusion_07(lang_no_power) is None

        lang_no_sent = [lang_signal("power_language_score", 0.20, "powerless")]
        assert engine._rule_fusion_07(lang_no_sent) is None

    def test_incongruence_boosted_by_objection(self, engine):
        """Objection present alongside incongruence → boosted confidence and renamed level."""
        lang = [
            lang_signal("sentiment_score", 0.80, "strong_positive"),
            lang_signal("power_language_score", 0.20, "powerless"),
            lang_signal("objection_signal", 0.70, "direct_objection"),
        ]
        result = engine._rule_fusion_07(lang)
        assert result is not None
        assert result["level"] == "incongruence_with_objection"


# ═══════════════════════════════════════════════════════════════
# FUSION-13: Persuasion Language × Speech Pace → Urgency
# ═══════════════════════════════════════════════════════════════

class TestFusion13:

    def test_authentic_urgency(self, engine):
        """Rate elevated + confident tone + enthusiasm sub-class → authentic_urgency."""
        voice = [
            voice_signal("speech_rate_anomaly", 45.0, "rate_elevated",
                         metadata={"sub_classification": "enthusiasm_driven_acceleration"}),
            voice_signal("tone_classification", 0.65, "confident"),
            voice_signal("vocal_stress_score", 0.20, "low_stress"),
        ]
        lang = [lang_signal("buying_signal", 0.60, "moderate_buying_signal")]
        result = engine._rule_fusion_13(voice, lang)
        assert result is not None
        assert result["level"] == "authentic_urgency"
        assert result["confidence"] <= 0.60

    def test_manufactured_urgency(self, engine):
        """Rate elevated + nervous tone + anxiety sub-class + high stress → manufactured."""
        voice = [
            voice_signal("speech_rate_anomaly", 50.0, "rate_elevated",
                         metadata={"sub_classification": "anxiety_driven_acceleration"}),
            voice_signal("tone_classification", 0.65, "nervous"),
            voice_signal("vocal_stress_score", 0.70, "high_stress"),
            voice_signal("filler_detection", 2.0, "filler_spike"),
        ]
        lang = [lang_signal("buying_signal", 0.60, "moderate_buying_signal")]
        result = engine._rule_fusion_13(voice, lang)
        assert result is not None
        assert result["level"] == "manufactured_urgency"

    def test_no_rate_signal_no_fire(self, engine):
        """No speech_rate_anomaly signal → rule does not fire."""
        voice = [voice_signal("tone_classification", 0.65, "confident")]
        lang = [lang_signal("buying_signal", 0.60, "moderate_buying_signal")]
        result = engine._rule_fusion_13(voice, lang)
        assert result is None

    def test_no_persuasion_no_fire(self, engine):
        """Rate elevated but no buying/persuasion signal → rule does not fire."""
        voice = [voice_signal("speech_rate_anomaly", 40.0, "rate_elevated")]
        lang = [lang_signal("sentiment_score", 0.50, "positive")]
        result = engine._rule_fusion_13(voice, lang)
        assert result is None

    def test_urgency_confidence_cap(self, engine):
        """Confidence must never exceed 0.60."""
        voice = [
            voice_signal("speech_rate_anomaly", 100.0, "rate_elevated",
                         metadata={"sub_classification": "enthusiasm_driven_acceleration"}),
            voice_signal("tone_classification", 1.0, "confident"),
            voice_signal("vocal_stress_score", 0.05, "low_stress"),
        ]
        lang = [lang_signal("buying_signal", 1.0, "strong_buying_signal")]
        result = engine._rule_fusion_13(voice, lang)
        assert result is not None
        assert result["confidence"] <= 0.60


# ═══════════════════════════════════════════════════════════════
# FusionRuleEngine.evaluate() — full output tests
# ═══════════════════════════════════════════════════════════════

class TestEvaluateOutput:

    def test_evaluate_returns_list(self, engine):
        signals = engine.evaluate("Speaker_0", [], [])
        assert isinstance(signals, list)

    def test_evaluate_with_empty_signals(self, engine):
        """No voice or language signals → empty fusion output."""
        result = engine.evaluate("Speaker_0", [], [])
        assert result == []

    def test_evaluate_signal_structure(self, engine):
        """Each fusion signal has required fields."""
        voice = [
            voice_signal("vocal_stress_score", 0.75, "high_stress"),
            voice_signal("speech_rate_anomaly", 40.0, "rate_elevated",
                         metadata={"sub_classification": "anxiety_driven_acceleration"}),
        ]
        lang = [
            lang_signal("sentiment_score", 0.85, "strong_positive"),
            lang_signal("buying_signal", 0.60, "moderate_buying_signal"),
            lang_signal("power_language_score", 0.25, "weak_power"),
        ]
        signals = engine.evaluate("Speaker_0", voice, lang)
        for sig in signals:
            assert sig["agent"] == "fusion"
            assert "signal_type" in sig
            assert "value" in sig
            assert "confidence" in sig
            assert 0.0 <= sig["confidence"] <= 1.0
