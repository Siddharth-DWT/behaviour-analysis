"""
Shared pytest fixtures for NEXUS test suite.
"""
import sys
from pathlib import Path
import pytest

# Add project root to path so all shared/services imports resolve
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from shared.models.signals import SpeakerBaseline


# ── Baseline fixtures ──────────────────────────────────────────

@pytest.fixture
def fake_baseline():
    """Realistic speaker baseline: calm, normal-rate speaker."""
    b = SpeakerBaseline(speaker_id="Speaker_0", session_id="test-session")
    b.f0_mean = 150.0          # Hz — typical male
    b.f0_std = 20.0
    b.f0_variance = 400.0
    b.speech_rate_wpm = 160.0  # Normal pace
    b.energy_rms_db = -22.0
    b.jitter_pct = 1.5
    b.shimmer_pct = 8.0
    b.hnr_db = 18.0
    b.filler_rate_pct = 0.5
    b.pause_ratio_pct = 0.05
    b.speech_seconds = 120.0
    b.sample_count = 24
    b.calibration_confidence = 0.9
    return b


@pytest.fixture
def fake_features_calm(fake_baseline):
    """Feature dict representing calm, baseline-level speech."""
    return {
        "window_start_ms": 0,
        "window_end_ms": 5000,
        "f0_mean": fake_baseline.f0_mean,           # At baseline
        "f0_variance": fake_baseline.f0_variance,
        "speech_rate_wpm": fake_baseline.speech_rate_wpm,
        "energy_rms_db": fake_baseline.energy_rms_db,
        "jitter_local_pct": fake_baseline.jitter_pct,
        "shimmer_local_pct": fake_baseline.shimmer_pct,
        "hnr_db": fake_baseline.hnr_db,
        "filler_rate_pct": fake_baseline.filler_rate_pct,
        "filler_count": 1,
        "um_count": 1,
        "uh_count": 0,
        "pause_ratio": fake_baseline.pause_ratio_pct,
        "speaking_time_sec": 4.5,
    }


@pytest.fixture
def fake_features_stressed(fake_baseline):
    """Feature dict representing stressed speech (F0 up, jitter up, rate up, HNR down)."""
    return {
        "window_start_ms": 5000,
        "window_end_ms": 10000,
        "f0_mean": fake_baseline.f0_mean * 1.30,       # +30% pitch
        "f0_variance": fake_baseline.f0_variance * 0.7, # Narrower
        "speech_rate_wpm": fake_baseline.speech_rate_wpm * 1.40,  # +40% rate
        "energy_rms_db": fake_baseline.energy_rms_db,
        "jitter_local_pct": fake_baseline.jitter_pct * 1.60,      # +60% jitter
        "shimmer_local_pct": fake_baseline.shimmer_pct * 1.40,
        "hnr_db": fake_baseline.hnr_db * 0.70,                    # -30% HNR
        "filler_rate_pct": fake_baseline.filler_rate_pct * 2.50,  # 2.5x fillers
        "filler_count": 4,
        "um_count": 3,
        "uh_count": 1,
        "pause_ratio": fake_baseline.pause_ratio_pct * 1.40,
        "speaking_time_sec": 4.0,
    }


# ── Language fixtures ──────────────────────────────────────────

@pytest.fixture
def fake_transcript_sales():
    """10-segment sales call transcript for integration tests."""
    return [
        {"speaker": "Speaker_0", "start_ms": 0,     "end_ms": 4000,  "text": "Hi, thanks for taking the time to connect today."},
        {"speaker": "Speaker_1", "start_ms": 4000,  "end_ms": 8000,  "text": "Of course. Have you worked with companies in the education sector before?"},
        {"speaker": "Speaker_0", "start_ms": 8000,  "end_ms": 13000, "text": "Yes, we have several clients in education. The results have been excellent."},
        {"speaker": "Speaker_1", "start_ms": 13000, "end_ms": 18000, "text": "What's the pricing for 50 users? And what's included in the base plan?"},
        {"speaker": "Speaker_0", "start_ms": 18000, "end_ms": 23000, "text": "Our enterprise plan covers unlimited users and includes onboarding support."},
        {"speaker": "Speaker_1", "start_ms": 23000, "end_ms": 28000, "text": "That's not going to work for us. We already have a solution in place."},
        {"speaker": "Speaker_0", "start_ms": 28000, "end_ms": 33000, "text": "I understand. What if we offered a pilot with no commitment?"},
        {"speaker": "Speaker_1", "start_ms": 33000, "end_ms": 38000, "text": "Worst case scenario, what happens if it doesn't deliver results?"},
        {"speaker": "Speaker_0", "start_ms": 38000, "end_ms": 43000, "text": "We guarantee outcomes. Let's schedule a follow-up call to walk through specifics."},
        {"speaker": "Speaker_1", "start_ms": 43000, "end_ms": 48000, "text": "Sounds good. Send me your email and we can go from there."},
    ]


# ── LLM mock fixture ───────────────────────────────────────────

@pytest.fixture
def mock_llm_client(monkeypatch):
    """
    Patches shared.utils.llm_client so no real API call is made.
    Mocks both sync (complete) and async (acomplete) paths.
    Returns a factory: call mock_llm_client.set_response(text) to set the response.
    """
    import shared.utils.llm_client as llm_mod

    class _Mock:
        def __init__(self):
            self._response = '[{"id": 1, "intent": "QUESTION", "confidence": 0.85}]'

        def set_response(self, text: str):
            self._response = text

        def complete(self, system_prompt, user_prompt, **kwargs):
            return self._response

        async def acomplete(self, system_prompt, user_prompt, **kwargs):
            return self._response

    mock = _Mock()
    monkeypatch.setattr(llm_mod, "complete", mock.complete)
    monkeypatch.setattr(llm_mod, "acomplete", mock.acomplete)
    monkeypatch.setattr(llm_mod, "is_configured", lambda: True)

    # Also patch the module-level alias in rules.py so async intent classification works
    try:
        import services.language_agent.rules as rules_mod
        monkeypatch.setattr(rules_mod, "llm_complete", mock.complete)
        monkeypatch.setattr(rules_mod, "llm_acomplete", mock.acomplete)
    except (ImportError, AttributeError):
        pass

    return mock
