"""
NEXUS Voice Agent - Calibration Module
Implements VOICE-CAL-01: Per-Speaker Baseline Calibration.

Research basis:
  - Goldman-Eisler (1968): articulation rate is stable within individuals
  - Laukka et al. (2011): speaker adaptation dramatically improves emotion classification
  - All stress/emotion research uses within-subject comparisons, not absolute thresholds

The calibration module uses the FIRST portion of features to establish
a per-speaker baseline. All subsequent rule engine detections operate
on DEVIATIONS from this baseline.
"""
import numpy as np
from typing import Optional
import logging
import sys
from pathlib import Path

try:
    from shared.models.signals import SpeakerBaseline
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from shared.models.signals import SpeakerBaseline

logger = logging.getLogger("nexus.voice.calibration")

# How many feature windows to use for calibration
# With 5s windows and 2.5s hop, 180s of speech ≈ 72 windows
CALIBRATION_MIN_WINDOWS = 5    # Minimum windows needed
CALIBRATION_TARGET_WINDOWS = 36  # ~90 seconds of speech at 2.5s hop


class CalibrationModule:
    """
    Builds per-speaker baselines from initial feature windows.
    
    Uses first N windows (up to ~3 minutes of speech) to compute
    mean and standard deviation for each acoustic feature.
    All subsequent analysis is expressed as deviation from baseline.
    """
    
    def build_baseline(
        self,
        speaker_id: str,
        session_id: str,
        features_list: list[dict],
        max_windows: int = CALIBRATION_TARGET_WINDOWS,
        transcript_speech_sec: float = 0.0,
    ) -> SpeakerBaseline:
        """
        Build a baseline from the first N feature windows.

        Args:
            speaker_id: Speaker identifier
            session_id: Session identifier
            features_list: List of feature dicts from VoiceFeatureExtractor
            max_windows: How many windows to use for calibration
            transcript_speech_sec: Total speaking time from transcript segments.
                Used as a floor for speech_seconds so that calibration confidence
                isn't penalised by short windows skipped during feature extraction.

        Returns:
            SpeakerBaseline with computed means, stds, and confidence
        """
        # Use first max_windows features for calibration
        cal_features = features_list[:max_windows]
        
        if len(cal_features) == 0:
            logger.warning(f"No calibration data for {speaker_id}")
            baseline = SpeakerBaseline(speaker_id=speaker_id, session_id=session_id)
            baseline.calibration_confidence = 0.1
            return baseline

        if len(cal_features) < CALIBRATION_MIN_WINDOWS:
            logger.warning(
                f"Limited data for {speaker_id}: "
                f"{len(cal_features)} windows (ideal {CALIBRATION_MIN_WINDOWS}) — "
                f"using available data with reduced confidence"
            )
            # Continue below — use whatever data is available, confidence will be low
        
        baseline = SpeakerBaseline(speaker_id=speaker_id, session_id=session_id)
        
        # ── Pitch baseline ──
        f0_values = [f["f0_mean"] for f in cal_features if f.get("f0_mean", 0) > 0]
        if f0_values:
            baseline.f0_mean = float(np.mean(f0_values))
            baseline.f0_std = float(np.std(f0_values))
            baseline.f0_variance = float(np.mean([
                f.get("f0_variance", 0) for f in cal_features if f.get("f0_variance", 0) > 0
            ]))
        
        # ── Speech rate baseline ──
        rate_values = [f["speech_rate_wpm"] for f in cal_features if f.get("speech_rate_wpm", 0) > 30]
        if rate_values:
            baseline.speech_rate_wpm = float(np.mean(rate_values))
        
        # ── Energy baseline ──
        energy_values = [f["energy_rms_db"] for f in cal_features if f.get("energy_rms_db", 0) != 0]
        if energy_values:
            baseline.energy_rms_db = float(np.mean(energy_values))
        
        # ── Voice quality baselines ──
        jitter_values = [f["jitter_local_pct"] for f in cal_features if f.get("jitter_local_pct", 0) > 0]
        if jitter_values:
            baseline.jitter_pct = float(np.mean(jitter_values))
        
        shimmer_values = [f["shimmer_local_pct"] for f in cal_features if f.get("shimmer_local_pct", 0) > 0]
        if shimmer_values:
            baseline.shimmer_pct = float(np.mean(shimmer_values))
        
        hnr_values = [f["hnr_db"] for f in cal_features if f.get("hnr_db", 0) != 0]
        if hnr_values:
            baseline.hnr_db = float(np.mean(hnr_values))
        
        # ── Filler rate baseline ──
        filler_values = [f["filler_rate_pct"] for f in cal_features if "filler_rate_pct" in f]
        if filler_values:
            baseline.filler_rate_pct = float(np.mean(filler_values))
        
        # ── Pause ratio baseline ──
        pause_values = [f["pause_ratio"] for f in cal_features if "pause_ratio" in f]
        if pause_values:
            baseline.pause_ratio_pct = float(np.mean(pause_values))
        
        # ── Compute speaking time and calibration confidence ──
        # Use transcript-derived speaking time as floor so that short windows
        # skipped by the feature extractor (<300ms) don't undercount speech.
        total_speaking = sum(f.get("speaking_time_sec", 0) for f in cal_features)
        baseline.speech_seconds = max(total_speaking, transcript_speech_sec)
        baseline.sample_count = len(cal_features)
        baseline.update_confidence()
        
        logger.info(
            f"Baseline for {speaker_id}: "
            f"F0={baseline.f0_mean:.1f}±{baseline.f0_std:.1f}Hz, "
            f"rate={baseline.speech_rate_wpm:.0f}wpm, "
            f"energy={baseline.energy_rms_db:.1f}dB, "
            f"jitter={baseline.jitter_pct:.2f}%, "
            f"fillers={baseline.filler_rate_pct:.2f}%, "
            f"from {baseline.speech_seconds:.0f}s speech → "
            f"confidence={baseline.calibration_confidence:.2f}"
        )
        
        return baseline
    
    @staticmethod
    def compute_delta(current_value: float, baseline_value: float) -> float:
        """
        Compute percentage deviation from baseline.
        Returns: (current - baseline) / baseline as a fraction.
        E.g., 0.15 means 15% above baseline.
        """
        if baseline_value == 0:
            return 0.0
        return (current_value - baseline_value) / abs(baseline_value)
    
    @staticmethod
    def compute_sigma(current_value: float, baseline_mean: float, baseline_std: float) -> float:
        """
        Compute how many standard deviations from baseline.
        Returns: number of standard deviations (signed).
        """
        if baseline_std == 0:
            return 0.0
        return (current_value - baseline_mean) / baseline_std
    
    @staticmethod
    def normalise_delta_to_01(delta: float, max_delta: float = 1.0) -> float:
        """
        Normalise a delta value to 0-1 range.
        0 = no change from baseline
        1 = max_delta or more deviation
        """
        return min(abs(delta) / max_delta, 1.0)
