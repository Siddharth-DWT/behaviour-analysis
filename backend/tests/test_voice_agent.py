"""
tests/test_voice_agent.py
Unit tests for Voice Agent rules engine and calibration module.
Feeds synthetic feature dicts directly — no audio files needed.
"""
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from services.voiceAgent.rules import VoiceRuleEngine
from services.voiceAgent.calibration import CalibrationModule
from shared.models.signals import SpeakerBaseline


# ═══════════════════════════════════════════════════════════════
# VOICE-STRESS-01: Composite Vocal Stress Score
# ═══════════════════════════════════════════════════════════════

class TestStress01:

    def setup_method(self):
        self.engine = VoiceRuleEngine()

    def _make_baseline(self, **kwargs):
        b = SpeakerBaseline(speaker_id="spk", session_id="sess")
        b.f0_mean = kwargs.get("f0_mean", 150.0)
        b.f0_std = 20.0
        b.f0_variance = 400.0
        b.speech_rate_wpm = kwargs.get("speech_rate_wpm", 160.0)
        b.energy_rms_db = -22.0
        b.jitter_pct = kwargs.get("jitter_pct", 1.5)
        b.shimmer_pct = 8.0
        b.hnr_db = 18.0
        b.filler_rate_pct = kwargs.get("filler_rate_pct", 0.5)
        b.pause_ratio_pct = 0.05
        b.calibration_confidence = 0.9
        return b

    def test_stress_low(self, fake_baseline, fake_features_calm):
        result = self.engine._rule_stress_01(fake_features_calm, fake_baseline)
        assert result is not None
        assert result["score"] < 0.30
        assert result["level"] == "low_stress"

    def test_stress_moderate(self, fake_baseline):
        """F0 15% up + jitter 30% up → moderate stress band."""
        f = {
            "f0_mean": fake_baseline.f0_mean * 1.15,
            "f0_variance": fake_baseline.f0_variance,
            "speech_rate_wpm": fake_baseline.speech_rate_wpm,
            "energy_rms_db": fake_baseline.energy_rms_db,
            "jitter_local_pct": fake_baseline.jitter_pct * 1.30,
            "shimmer_local_pct": fake_baseline.shimmer_pct,
            "hnr_db": fake_baseline.hnr_db,
            "filler_rate_pct": fake_baseline.filler_rate_pct,
            "pause_ratio": fake_baseline.pause_ratio_pct,
        }
        result = self.engine._rule_stress_01(f, fake_baseline)
        assert result is not None
        assert 0.20 <= result["score"] <= 0.55, f"Expected 0.20-0.55, got {result['score']}"

    def test_stress_high(self, fake_baseline, fake_features_stressed):
        """All stress indicators elevated → high stress."""
        result = self.engine._rule_stress_01(fake_features_stressed, fake_baseline)
        assert result is not None
        assert result["score"] > 0.50, f"Expected > 0.50, got {result['score']}"

    def test_stress_baseline_zero(self):
        """Baseline F0=0 → rule returns None (graceful skip)."""
        b = self._make_baseline(f0_mean=0)
        f = {"f0_mean": 150.0, "speech_rate_wpm": 160.0}
        result = self.engine._rule_stress_01(f, b)
        assert result is None


# ═══════════════════════════════════════════════════════════════
# VOICE-FILLER-01/02: Filler Detection
# ═══════════════════════════════════════════════════════════════

class TestFiller01:

    def setup_method(self):
        self.engine = VoiceRuleEngine()

    def _baseline_with_fillers(self, rate=0.5):
        b = SpeakerBaseline(speaker_id="spk", session_id="sess")
        b.f0_mean = 150.0
        b.filler_rate_pct = rate
        b.calibration_confidence = 0.9
        return b

    def test_filler_normal(self):
        """Normal filler rate with no credibility impact → returns None (no signal)."""
        b = self._baseline_with_fillers(rate=0.5)
        f = {"filler_rate_pct": 0.5, "filler_count": 1, "um_count": 1, "uh_count": 0}
        result = self.engine._rule_filler_01(f, b)
        assert result is None

    def test_filler_spike(self):
        """Filler rate 60% above baseline → filler_spike status."""
        b = self._baseline_with_fillers(rate=1.0)
        f = {"filler_rate_pct": 1.65, "filler_count": 5, "um_count": 4, "uh_count": 1}
        result = self.engine._rule_filler_01(f, b)
        assert result["status"] == "filler_spike"

    def test_filler_credibility_severe(self):
        """Filler rate > 4.0% → severe credibility impact."""
        b = self._baseline_with_fillers(rate=1.0)
        f = {"filler_rate_pct": 4.5, "filler_count": 10, "um_count": 8, "uh_count": 2}
        result = self.engine._rule_filler_01(f, b)
        assert result["credibility_impact"] == "severe"

    def test_filler_credibility_noticeable(self):
        """Filler rate 1.3-2.5% → noticeable credibility impact."""
        b = self._baseline_with_fillers(rate=0.5)
        f = {"filler_rate_pct": 1.5, "filler_count": 4, "um_count": 3, "uh_count": 1}
        result = self.engine._rule_filler_01(f, b)
        assert result["credibility_impact"] == "noticeable"


# ═══════════════════════════════════════════════════════════════
# VOICE-PITCH-01: Pitch Elevation Flag
# ═══════════════════════════════════════════════════════════════

class TestPitch01:

    def setup_method(self):
        self.engine = VoiceRuleEngine()

    def _baseline(self, f0=150.0):
        b = SpeakerBaseline(speaker_id="spk", session_id="sess")
        b.f0_mean = f0
        b.calibration_confidence = 0.9
        return b

    def test_pitch_no_flag(self):
        """F0 only 5% above baseline → returns None (within normal range)."""
        b = self._baseline(150.0)
        f = {"f0_mean": 157.5}  # +5%
        result = self.engine._rule_pitch_01(f, b)
        assert result is None

    def test_pitch_mild(self):
        """F0 10% above → mild elevation."""
        b = self._baseline(150.0)
        f = {"f0_mean": 165.0}  # +10%
        result = self.engine._rule_pitch_01(f, b)
        assert result is not None
        assert result["level"] == "pitch_elevated_mild"

    def test_pitch_significant(self):
        """F0 18% above → significant elevation."""
        b = self._baseline(150.0)
        f = {"f0_mean": 177.0}  # +18%
        result = self.engine._rule_pitch_01(f, b)
        assert result is not None
        assert result["level"] == "pitch_elevated_significant"

    def test_pitch_extreme(self):
        """F0 30% above → extreme elevation."""
        b = self._baseline(150.0)
        f = {"f0_mean": 195.0}  # +30%
        result = self.engine._rule_pitch_01(f, b)
        assert result is not None
        assert result["level"] == "pitch_elevated_extreme"


# ═══════════════════════════════════════════════════════════════
# VOICE-RATE-01: Speech Rate Anomaly Detection
# ═══════════════════════════════════════════════════════════════

class TestRate01:

    def setup_method(self):
        self.engine = VoiceRuleEngine()

    def _baseline(self, wpm=160.0, f0=150.0, energy=-22.0):
        b = SpeakerBaseline(speaker_id="spk", session_id="sess")
        b.f0_mean = f0
        b.speech_rate_wpm = wpm
        b.energy_rms_db = energy
        b.pause_ratio_pct = 0.05
        b.calibration_confidence = 0.9
        return b

    def test_rate_normal(self):
        """WPM 15% above baseline → within normal range, returns None."""
        b = self._baseline(160.0)
        f = {"speech_rate_wpm": 184.0, "f0_mean": 150.0, "energy_rms_db": -22.0, "pause_ratio": 0.05}
        result = self.engine._rule_rate_01(f, b)
        assert result is None

    def test_rate_elevated(self):
        """WPM 35% above + F0 elevated → rate_elevated / anxiety_driven_acceleration."""
        b = self._baseline(160.0, f0=150.0)
        f = {
            "speech_rate_wpm": 216.0,   # +35%
            "f0_mean": 168.0,           # +12% (triggers anxiety sub-class)
            "energy_rms_db": -22.0,
            "pause_ratio": 0.05,
        }
        result = self.engine._rule_rate_01(f, b)
        assert result is not None
        assert result["classification"] == "rate_elevated"
        assert result["sub_classification"] == "anxiety_driven_acceleration"

    def test_rate_depressed(self):
        """WPM 30% below + energy down → disengagement_deceleration."""
        b = self._baseline(160.0, energy=-22.0)
        f = {
            "speech_rate_wpm": 112.0,   # -30%
            "f0_mean": 150.0,
            "energy_rms_db": -26.0,     # -18% energy drop
            "pause_ratio": 0.05,
        }
        result = self.engine._rule_rate_01(f, b)
        assert result is not None
        assert result["classification"] == "rate_depressed"
        assert result["sub_classification"] == "disengagement_deceleration"


# ═══════════════════════════════════════════════════════════════
# VOICE-TONE-03/04: Tone Classification
# ═══════════════════════════════════════════════════════════════

class TestTone:

    def setup_method(self):
        self.engine = VoiceRuleEngine()

    def _baseline(self):
        b = SpeakerBaseline(speaker_id="spk", session_id="sess")
        b.f0_mean = 150.0
        b.f0_variance = 400.0
        b.speech_rate_wpm = 160.0
        b.energy_rms_db = -22.0
        b.filler_rate_pct = 0.5
        b.calibration_confidence = 0.9
        return b

    def test_tone_nervous(self):
        """F0 up + variance narrower + rate up + high jitter → nervous."""
        b = self._baseline()
        f = {
            "f0_mean": 168.0,          # +12% — pitch elevated
            "f0_variance": 310.0,      # -22% narrower
            "speech_rate_wpm": 178.0,  # +11%
            "energy_rms_db": -22.0,
            "jitter_local_pct": 2.5,   # Elevated
            "filler_rate_pct": 1.2,    # 140% above baseline
        }
        result = self.engine._rule_tone(f, b)
        assert result is not None
        assert result["tone"] == "nervous"

    def test_tone_confident(self):
        """F0 at/below baseline + wider variance + energy up + clean voice → confident."""
        b = self._baseline()
        f = {
            "f0_mean": 148.0,          # Slightly below — pitch stable
            "f0_variance": 480.0,      # +20% wider
            "speech_rate_wpm": 160.0,  # Controlled (within 10% → rate_controlled)
            "energy_rms_db": -19.5,    # +12% energy
            "jitter_local_pct": 1.1,   # Clean phonation < 1.5
            "filler_rate_pct": 0.3,    # Minimal < 0.5
        }
        result = self.engine._rule_tone(f, b)
        assert result is not None
        assert result["tone"] == "confident"

    def test_tone_neutral(self):
        """Mixed signals that don't clearly fit nervous or confident."""
        b = self._baseline()
        f = {
            "f0_mean": 153.0,          # Only +2% pitch
            "f0_variance": 400.0,      # Same variance
            "speech_rate_wpm": 165.0,  # Only +3%
            "energy_rms_db": -22.0,
            "jitter_local_pct": 1.8,
            "filler_rate_pct": 0.6,
        }
        result = self.engine._rule_tone(f, b)
        assert result is not None
        assert result["tone"] == "neutral"


# ═══════════════════════════════════════════════════════════════
# Calibration Module
# ═══════════════════════════════════════════════════════════════

class TestCalibration:

    def setup_method(self):
        self.cal = CalibrationModule()

    def _make_feature_window(self, f0=150.0, wpm=160.0, energy=-22.0, jitter=1.5,
                              shimmer=8.0, hnr=18.0, fillers=0.5):
        return {
            "f0_mean": f0,
            "f0_variance": 400.0,
            "speech_rate_wpm": wpm,
            "energy_rms_db": energy,
            "jitter_local_pct": jitter,
            "shimmer_local_pct": shimmer,
            "hnr_db": hnr,
            "filler_rate_pct": fillers,
            "pause_ratio": 0.05,
            "speaking_time_sec": 4.5,
        }

    def test_baseline_build(self):
        """10 windows → reasonable baseline values and confidence > 0.5."""
        windows = [self._make_feature_window() for _ in range(10)]
        baseline = self.cal.build_baseline("spk", "sess", windows)
        assert baseline.f0_mean == pytest.approx(150.0, rel=0.05)
        assert baseline.speech_rate_wpm == pytest.approx(160.0, rel=0.05)
        assert baseline.calibration_confidence > 0.5

    def test_baseline_insufficient(self):
        """Only 2 windows → calibration confidence very low."""
        windows = [self._make_feature_window() for _ in range(2)]
        baseline = self.cal.build_baseline("spk", "sess", windows)
        assert baseline.calibration_confidence < 0.2

    def test_compute_delta(self):
        """current=120, baseline=100 → delta = 0.20."""
        delta = CalibrationModule.compute_delta(120.0, 100.0)
        assert delta == pytest.approx(0.20)

    def test_compute_sigma(self):
        """current=120, mean=100, std=10 → sigma = 2.0."""
        sigma = CalibrationModule.compute_sigma(120.0, 100.0, 10.0)
        assert sigma == pytest.approx(2.0)

    def test_compute_delta_zero_baseline(self):
        """Baseline=0 → delta = 0.0 (no divide-by-zero)."""
        delta = CalibrationModule.compute_delta(100.0, 0.0)
        assert delta == 0.0

    def test_normalise_delta_capped(self):
        """Delta exceeding max_delta is capped at 1.0."""
        val = CalibrationModule.normalise_delta_to_01(0.60, max_delta=0.30)
        assert val == 1.0


# ═══════════════════════════════════════════════════════════════
# Full evaluate() output structure
# ═══════════════════════════════════════════════════════════════

class TestEvaluateOutput:

    def setup_method(self):
        self.engine = VoiceRuleEngine()

    def test_evaluate_returns_list(self, fake_baseline, fake_features_calm):
        signals = self.engine.evaluate(fake_features_calm, fake_baseline, "Speaker_0")
        assert isinstance(signals, list)

    def test_evaluate_low_calibration_skipped(self):
        """If cal_confidence < 0.1, evaluate returns empty."""
        b = SpeakerBaseline(speaker_id="spk", session_id="sess")
        b.f0_mean = 150.0
        b.calibration_confidence = 0.05
        signals = self.engine.evaluate({"f0_mean": 180.0, "speech_rate_wpm": 200.0}, b, "spk")
        assert signals == []

    def test_evaluate_signal_structure(self, fake_baseline, fake_features_stressed):
        signals = self.engine.evaluate(fake_features_stressed, fake_baseline, "Speaker_0")
        for sig in signals:
            assert "agent" in sig
            assert sig["agent"] == "voice"
            assert "signal_type" in sig
            assert "value" in sig
            assert "confidence" in sig
            assert 0.0 <= sig["confidence"] <= 1.0
