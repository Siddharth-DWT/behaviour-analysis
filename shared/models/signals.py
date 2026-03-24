"""
Shared Signal Models
Core data model for all agent outputs. Every agent (Voice, Language, Fusion)
produces Signal instances. The FusionSignalInput is the typed input to Fusion Agent.
UnifiedSpeakerState is the Fusion Agent's per-speaker output (dataclass for asdict()).
"""
from typing import Optional
from dataclasses import dataclass, field
from pydantic import BaseModel


class Signal(BaseModel):
    """
    Universal signal produced by any agent.
    Use this as the output type in all agent rule engines.
    """
    agent: str                          # "voice" | "language" | "fusion"
    speaker_id: str = "unknown"
    signal_type: str = ""               # e.g. "vocal_stress_score", "buying_signal"
    value: Optional[float] = None       # Numeric value (0-1 or domain-specific)
    value_text: str = ""                # Human-readable label or category
    confidence: float = 0.5            # 0.0–0.85 (never 1.0 per design principles)
    window_start_ms: int = 0
    window_end_ms: int = 0
    metadata: Optional[dict] = None    # Rule-specific extra data

    def to_dict(self) -> dict:
        return self.model_dump()


@dataclass
class SpeakerBaseline:
    """
    Per-speaker acoustic baseline computed from the first N feature windows.
    All rule engine detections operate as deviations from these values.
    Implements VOICE-CAL-01.
    """
    speaker_id: str
    session_id: str = ""

    # Pitch
    f0_mean: float = 0.0            # Mean fundamental frequency (Hz)
    f0_std: float = 0.0             # Std dev of F0
    f0_variance: float = 0.0        # Mean frame-level F0 variance

    # Prosody
    speech_rate_wpm: float = 0.0    # Words per minute
    energy_rms_db: float = 0.0      # RMS energy (dB)

    # Voice quality
    jitter_pct: float = 0.0         # Jitter local %
    shimmer_pct: float = 0.0        # Shimmer local %
    hnr_db: float = 0.0             # Harmonics-to-noise ratio (dB)

    # Behavioural
    filler_rate_pct: float = 0.0    # Fillers per 100 words
    pause_ratio_pct: float = 0.0    # Fraction of window in silence

    # Calibration metadata
    speech_seconds: float = 0.0     # Total speech used to build baseline
    sample_count: int = 0           # Number of feature windows used
    calibration_confidence: float = 0.0  # 0.0–0.90

    def update_confidence(self):
        """
        Recompute calibration_confidence from speech_seconds and sample_count.
        Scales 0.20 (at 5 windows / ~25s) → 0.90 (at 36 windows / ~90s).
        """
        if self.sample_count < 5:
            self.calibration_confidence = 0.10
        else:
            # Linear ramp: 25s → 0.20, 90s+ → 0.90
            self.calibration_confidence = round(
                min(0.20 + (self.speech_seconds / 90.0) * 0.70, 0.90), 3
            )

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


@dataclass
class UnifiedSpeakerState:
    """
    Aggregated per-speaker behavioural state computed by the Fusion Agent.
    Used with dataclasses.asdict() so it must stay a dataclass, not Pydantic.

    Fields map directly to dashboard gauge widgets (stress, confidence, etc.).
    All float fields are clamped 0.0–1.0 except sentiment_score (-1.0–1.0).
    """
    speaker_id: str
    speaker_name: str = ""
    timestamp_ms: int = 0

    # Core state dimensions
    stress_level: float = 0.0          # 0 = calm, 1 = high stress
    confidence_level: float = 0.50     # 0 = uncertain, 1 = highly confident
    sentiment_score: float = 0.0       # -1 = negative, 0 = neutral, +1 = positive
    engagement: float = 0.50           # 0 = disengaged, 1 = highly engaged
    authenticity_score: float = 0.85   # Reduced by credibility flags

    # Qualitative labels
    dominant_emotion: str = "neutral"  # positive | negative | neutral

    # Active alert summaries (capped at 5 most recent)
    active_alerts: list = field(default_factory=list)

    # Flags for incongruent signal clusters
    uncertainty_flags: list = field(default_factory=list)


class FusionSignalInput(BaseModel):
    """
    Typed signal input to the Fusion Agent.
    Accepted by POST /analyse on the Fusion Agent and also
    constructed by the API Gateway when calling the Fusion Agent.
    """
    agent: str                          # "voice" | "language"
    speaker_id: str = "unknown"
    signal_type: str = ""
    value: Optional[float] = None
    value_text: str = ""
    confidence: float = 0.5
    window_start_ms: int = 0
    window_end_ms: int = 0
    metadata: Optional[dict] = None
