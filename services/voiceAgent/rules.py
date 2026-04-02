"""
NEXUS Voice Agent - Rule Engine
Implements 5 core rules from the NEXUS Rule Engine specification.

Each rule takes raw features + speaker baseline and produces Signal objects.
Thresholds are loaded from the rule_config database table (configurable).
For now, defaults are hardcoded matching the Rule Engine document.

Rules implemented:
  VOICE-STRESS-01: Composite vocal stress score
  VOICE-FILLER-01: Filler word detection & classification
  VOICE-FILLER-02: Filler credibility threshold
  VOICE-PITCH-01: Pitch elevation flag
  VOICE-RATE-01: Speech rate anomaly detection
  VOICE-TONE-03: Nervous/Anxious tone classification
  VOICE-TONE-04: Confident/Authoritative tone classification
"""
import logging
import sys
from pathlib import Path
from typing import Optional

try:
    from shared.models.signals import Signal, SpeakerBaseline
    from calibration import CalibrationModule
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from shared.models.signals import Signal, SpeakerBaseline
    from services.voiceAgent.calibration import CalibrationModule

logger = logging.getLogger("nexus.voice.rules")


def _make_signal(
    speaker_id: str, signal_type: str, value: float, value_text: str,
    confidence: float, window_start_ms: int, window_end_ms: int,
    metadata: dict = None,
) -> dict:
    """Create a validated signal dict via the Signal model."""
    return Signal(
        agent="voice",
        speaker_id=speaker_id,
        signal_type=signal_type,
        value=round(value, 4),
        value_text=value_text,
        confidence=round(min(confidence, 0.85), 4),  # Enforce 0.85 cap
        window_start_ms=window_start_ms,
        window_end_ms=window_end_ms,
        metadata=metadata,
    ).to_dict()


class VoiceRuleEngine:
    """
    Evaluates acoustic features against per-speaker baselines
    using research-derived detection rules.

    All thresholds are expressed as deviations from baseline
    unless marked as absolute.
    """

    def __init__(self):
        self.cal = CalibrationModule()
        # TODO: Load thresholds from rule_config DB table
        # For now, using defaults from the Rule Engine document
    
    def evaluate(
        self,
        features: dict,
        baseline: SpeakerBaseline,
        speaker_id: str,
        transcript_segments: list[dict] = None
    ) -> list[dict]:
        """
        Run all rules against a single feature window.
        
        Args:
            features: Feature dict from VoiceFeatureExtractor
            baseline: SpeakerBaseline from CalibrationModule
            speaker_id: Speaker identifier
            transcript_segments: Transcript segments in this window
            
        Returns:
            List of Signal dicts (one per fired rule)
        """
        signals = []
        window_start = features.get("window_start_ms", 0)
        window_end = features.get("window_end_ms", 0)
        cal_conf = baseline.calibration_confidence
        
        # Skip if calibration is too low
        if cal_conf < 0.1:
            return signals
        
        # ── VOICE-STRESS-01: Composite Vocal Stress Score ──
        stress = self._rule_stress_01(features, baseline)
        if stress is not None:
            signals.append(_make_signal(
                speaker_id, "vocal_stress_score",
                stress["score"], stress["level"],
                stress["score"] * cal_conf,
                window_start, window_end,
                stress["components"],
            ))

        # ── VOICE-FILLER-01: Filler Detection ──
        filler = self._rule_filler_01(features, baseline)
        if filler is not None:
            signals.append(_make_signal(
                speaker_id, "filler_detection",
                filler["filler_rate_pct"], filler["status"],
                0.90 * cal_conf,
                window_start, window_end,
                {
                    "filler_count": filler["filler_count"],
                    "um_count": filler["um_count"],
                    "uh_count": filler["uh_count"],
                    "delta_from_baseline": filler.get("delta", 0),
                    "credibility_impact": filler.get("credibility_impact", "none"),
                },
            ))

        # ── VOICE-PITCH-01: Pitch Elevation Flag ──
        pitch = self._rule_pitch_01(features, baseline)
        if pitch is not None:
            signals.append(_make_signal(
                speaker_id, "pitch_elevation_flag",
                pitch["delta_pct"], pitch["level"],
                0.50 * cal_conf,
                window_start, window_end,
                {
                    "f0_current": pitch["f0_current"],
                    "f0_baseline": pitch["f0_baseline"],
                    "delta_pct": pitch["delta_pct"],
                },
            ))

        # ── VOICE-RATE-01: Speech Rate Anomaly ──
        rate = self._rule_rate_01(features, baseline)
        if rate is not None:
            signals.append(_make_signal(
                speaker_id, "speech_rate_anomaly",
                rate["delta_pct"], rate["classification"],
                0.40 * cal_conf,
                window_start, window_end,
                {
                    "wpm_current": rate["wpm_current"],
                    "wpm_baseline": rate["wpm_baseline"],
                    "sub_classification": rate.get("sub_classification", ""),
                },
            ))
        
        # ── VOICE-TONE-03/04: Tone Classification ──
        tone = self._rule_tone(features, baseline)
        if tone is not None:
            signals.append(_make_signal(
                speaker_id, "tone_classification",
                tone["confidence_raw"], tone["tone"],
                tone["confidence_raw"] * cal_conf,
                window_start, window_end,
                tone.get("evidence", {}),
            ))
        
        return signals
    
    # ════════════════════════════════════════════════════════
    # VOICE-STRESS-01: Composite Vocal Stress Score
    # Research: Streeter 1977, Laukka 2008, Kappen 2022
    # ════════════════════════════════════════════════════════
    
    def _rule_stress_01(self, f: dict, b: SpeakerBaseline) -> Optional[dict]:
        """
        Weighted composite stress score from 7 acoustic features.
        
        Weights (from research reliability ranking):
          pitch delta:   0.30 (most reliable - Streeter 1977)
          jitter delta:  0.20 (Kappen 2022)
          rate change:   0.15
          filler rate:   0.15
          pause freq:    0.10
          HNR inverse:   0.05 (spectral-flatness proxy, reduced weight)
          shimmer:       0.05
        """
        if b.f0_mean == 0:
            return None
        
        # Compute deltas
        f0_delta = self.cal.compute_delta(f.get("f0_mean", 0), b.f0_mean)
        jitter_delta = self.cal.compute_delta(f.get("jitter_local_pct", 0), b.jitter_pct) if b.jitter_pct > 0 else 0
        rate_delta = self.cal.compute_delta(f.get("speech_rate_wpm", 0), b.speech_rate_wpm) if b.speech_rate_wpm > 0 else 0
        filler_delta = self.cal.compute_delta(f.get("filler_rate_pct", 0), b.filler_rate_pct) if b.filler_rate_pct > 0 else 0
        pause_delta = self.cal.compute_delta(f.get("pause_ratio", 0), b.pause_ratio_pct) if b.pause_ratio_pct > 0 else 0
        hnr_delta = self.cal.compute_delta(f.get("hnr_db", 0), b.hnr_db) if b.hnr_db != 0 else 0
        shimmer_delta = self.cal.compute_delta(f.get("shimmer_local_pct", 0), b.shimmer_pct) if b.shimmer_pct > 0 else 0
        
        # Normalise each to 0-1 (0 = no change, 1 = 2σ+ deviation)
        # Stress indicators: pitch UP, jitter UP, rate UP or DOWN, filler UP, pause UP, HNR DOWN, shimmer UP
        components = {
            "f0_norm": self.cal.normalise_delta_to_01(max(f0_delta, 0), max_delta=0.30),         # Pitch rise
            "jitter_norm": self.cal.normalise_delta_to_01(max(jitter_delta, 0), max_delta=0.50),  # Jitter increase
            "rate_norm": self.cal.normalise_delta_to_01(abs(rate_delta), max_delta=0.40),          # Rate change (either direction)
            "filler_norm": self.cal.normalise_delta_to_01(max(filler_delta, 0), max_delta=1.00),  # Filler increase
            "pause_norm": self.cal.normalise_delta_to_01(max(pause_delta, 0), max_delta=0.50),    # Pause increase
            "hnr_norm": self.cal.normalise_delta_to_01(max(-hnr_delta, 0), max_delta=0.30),       # HNR decrease (inverted)
            "shimmer_norm": self.cal.normalise_delta_to_01(max(shimmer_delta, 0), max_delta=0.50), # Shimmer increase
        }
        
        # Weighted composite
        score = (
            0.30 * components["f0_norm"] +
            0.20 * components["jitter_norm"] +
            0.15 * components["rate_norm"] +
            0.15 * components["filler_norm"] +
            0.10 * components["pause_norm"] +
            0.05 * components["hnr_norm"] +
            0.05 * components["shimmer_norm"]
        )
        
        score = min(max(score, 0.0), 1.0)
        
        # Classify level
        if score > 0.70:
            level = "high_stress"
        elif score > 0.50:
            level = "elevated_stress"
        elif score > 0.30:
            level = "moderate_stress"
        else:
            level = "low_stress"
        
        return {
            "score": score,
            "level": level,
            "components": components,
        }
    
    # ════════════════════════════════════════════════════════
    # VOICE-FILLER-01 + VOICE-FILLER-02: Filler Detection
    # Research: Clark & Fox Tree 2002, Duvall 2014
    # ════════════════════════════════════════════════════════
    
    def _rule_filler_01(self, f: dict, b: SpeakerBaseline) -> Optional[dict]:
        """
        Filler word detection with baseline comparison and credibility assessment.
        """
        filler_count = f.get("filler_count", 0)
        filler_rate = f.get("filler_rate_pct", 0)
        um_count = f.get("um_count", 0)
        uh_count = f.get("uh_count", 0)
        
        # Delta from baseline
        delta = 0.0
        if b.filler_rate_pct > 0:
            delta = self.cal.compute_delta(filler_rate, b.filler_rate_pct)
        elif filler_rate > 0:
            # Zero-baseline speaker: any fillers are a spike.
            # Use absolute rate as the delta proxy (1% = 1.0 delta).
            delta = filler_rate / 1.0  # 1% filler rate = 100% delta

        # Status based on delta
        if delta > 0.50:  # 50% more fillers than baseline (or >0.5% absolute for zero-baseline)
            status = "filler_spike"
        elif delta > 0.25:
            status = "filler_elevated"
        else:
            status = "normal"
        
        # VOICE-FILLER-02: Absolute credibility thresholds
        if filler_rate > 4.0:
            credibility_impact = "severe"
        elif filler_rate > 2.5:
            credibility_impact = "significant"
        elif filler_rate > 1.3:
            credibility_impact = "noticeable"
        else:
            credibility_impact = "none"

        # Only emit a signal when something noteworthy is happening
        if status == "normal" and credibility_impact == "none":
            return None

        return {
            "filler_count": filler_count,
            "filler_rate_pct": round(filler_rate, 3),
            "um_count": um_count,
            "uh_count": uh_count,
            "delta": round(delta, 3),
            "status": status,
            "credibility_impact": credibility_impact,
        }
    
    # ════════════════════════════════════════════════════════
    # VOICE-PITCH-01: Pitch Elevation Flag
    # Research: Streeter et al. 1977, Laukka 2008, Weeks 2011
    # ════════════════════════════════════════════════════════
    
    def _rule_pitch_01(self, f: dict, b: SpeakerBaseline) -> Optional[dict]:
        """
        Detect pitch elevation from baseline.
        Pitch rise = arousal/stress indicator (NOT deception per se).
        
        Thresholds:
          > +8%  → mild (arousal increase)
          > +15% → significant (strong stress response)
          > +25% → extreme (acute stress or strong emotion)
        """
        f0_current = f.get("f0_mean", 0)
        if f0_current == 0 or b.f0_mean == 0:
            return None
        
        delta_pct = self.cal.compute_delta(f0_current, b.f0_mean) * 100
        
        # Only flag elevations (stress indicator). Drops are separate (confidence/disengagement).
        if delta_pct < 8.0:
            return None  # Within normal range, don't flag
        
        if delta_pct >= 25.0:
            level = "pitch_elevated_extreme"
        elif delta_pct >= 15.0:
            level = "pitch_elevated_significant"
        else:
            level = "pitch_elevated_mild"
        
        return {
            "f0_current": round(f0_current, 1),
            "f0_baseline": round(b.f0_mean, 1),
            "delta_pct": round(delta_pct, 2),
            "level": level,
        }
    
    # ════════════════════════════════════════════════════════
    # VOICE-RATE-01: Speech Rate Anomaly Detection
    # Research: Apple et al. 1979, Smith et al. 1975
    # ════════════════════════════════════════════════════════
    
    def _rule_rate_01(self, f: dict, b: SpeakerBaseline) -> Optional[dict]:
        """
        Detect speech rate deviations from baseline.
        
        > +25% → rate_elevated (anxiety, enthusiasm, or rushing)
        < -25% → rate_depressed (disengagement, cognitive load, or deliberation)
        """
        wpm_current = f.get("speech_rate_wpm", 0)
        if wpm_current < 30 or b.speech_rate_wpm < 30:  # Filter out near-silence
            return None
        
        delta_pct = self.cal.compute_delta(wpm_current, b.speech_rate_wpm) * 100
        
        if abs(delta_pct) < 25.0:
            return None  # Within normal range
        
        if delta_pct > 25.0:
            classification = "rate_elevated"
            # Sub-classify based on concurrent features
            f0_delta = self.cal.compute_delta(f.get("f0_mean", 0), b.f0_mean) if b.f0_mean > 0 else 0
            energy_delta = self.cal.compute_delta(f.get("energy_rms_db", 0), b.energy_rms_db) if b.energy_rms_db != 0 else 0
            
            if f0_delta > 0.10:
                sub = "anxiety_driven_acceleration"
            elif energy_delta > 0.10 and f0_delta < 0.05:
                sub = "enthusiasm_driven_acceleration"
            else:
                sub = "rushing"
        else:
            classification = "rate_depressed"
            energy_delta = self.cal.compute_delta(f.get("energy_rms_db", 0), b.energy_rms_db) if b.energy_rms_db != 0 else 0
            pause_delta = self.cal.compute_delta(f.get("pause_ratio", 0), b.pause_ratio_pct) if b.pause_ratio_pct > 0 else 0
            
            if energy_delta < -0.15:
                sub = "disengagement_deceleration"
            elif pause_delta > 0.30:
                sub = "cognitive_load_deceleration"
            else:
                sub = "deliberation_deceleration"
        
        return {
            "wpm_current": round(wpm_current, 1),
            "wpm_baseline": round(b.speech_rate_wpm, 1),
            "delta_pct": round(delta_pct, 2),
            "classification": classification,
            "sub_classification": sub,
        }
    
    # ════════════════════════════════════════════════════════
    # VOICE-TONE-03 + VOICE-TONE-04: Tone Classification
    # Research: Juslin & Laukka 2003, Tusing & Dillard 2000
    # ════════════════════════════════════════════════════════
    
    def _rule_tone(self, f: dict, b: SpeakerBaseline) -> Optional[dict]:
        """
        Classify tone as nervous, confident, or neutral based on
        the acoustic profile matching from the Rule Engine document.
        
        NERVOUS (VOICE-TONE-03):
          F0 above baseline + F0 variance narrower + rate faster + jitter elevated
          
        CONFIDENT (VOICE-TONE-04):
          F0 at/below baseline + F0 variance wider + energy above + rate controlled + low jitter
        """
        if b.f0_mean == 0:
            return None
        
        f0_delta = self.cal.compute_delta(f.get("f0_mean", 0), b.f0_mean)
        f0_var_delta = self.cal.compute_delta(f.get("f0_variance", 0), b.f0_variance) if b.f0_variance > 0 else 0
        rate_delta = self.cal.compute_delta(f.get("speech_rate_wpm", 0), b.speech_rate_wpm) if b.speech_rate_wpm > 0 else 0
        energy_delta = self.cal.compute_delta(f.get("energy_rms_db", 0), b.energy_rms_db) if b.energy_rms_db != 0 else 0
        jitter = f.get("jitter_local_pct", 0)
        filler_delta = self.cal.compute_delta(f.get("filler_rate_pct", 0), b.filler_rate_pct) if b.filler_rate_pct > 0 else 0
        
        # ── Score for NERVOUS ──
        nervous_score = 0.0
        nervous_evidence = {}
        
        if f0_delta > 0.10:  # Pitch elevated > 10%
            nervous_score += 0.30
            nervous_evidence["pitch_elevated"] = round(f0_delta * 100, 1)
        
        if f0_var_delta < -0.15:  # Pitch variance narrower (tight control)
            nervous_score += 0.20
            nervous_evidence["pitch_range_narrowed"] = round(f0_var_delta * 100, 1)
        
        if rate_delta > 0.10:  # Faster speech
            nervous_score += 0.20
            nervous_evidence["rate_faster"] = round(rate_delta * 100, 1)
        
        if jitter > 2.0:  # Elevated jitter
            nervous_score += 0.15
            nervous_evidence["jitter_elevated"] = round(jitter, 2)
        
        if filler_delta > 0.50:  # More fillers
            nervous_score += 0.15
            nervous_evidence["fillers_elevated"] = round(filler_delta * 100, 1)
        
        # ── Score for CONFIDENT ──
        confident_score = 0.0
        confident_evidence = {}
        
        if f0_delta <= 0.0:  # Pitch at or below baseline
            confident_score += 0.20
            confident_evidence["pitch_stable_or_low"] = round(f0_delta * 100, 1)
        
        if f0_var_delta > 0.15:  # Wider pitch variation (dynamic, expressive)
            confident_score += 0.20
            confident_evidence["pitch_range_wider"] = round(f0_var_delta * 100, 1)
        
        if energy_delta > 0.10:  # Louder than baseline
            confident_score += 0.20
            confident_evidence["energy_above_baseline"] = round(energy_delta * 100, 1)
        
        if abs(rate_delta) < 0.10:  # Controlled pace (not rushing)
            confident_score += 0.15
            confident_evidence["rate_controlled"] = round(rate_delta * 100, 1)
        
        if jitter < 1.5:  # Clean phonation
            confident_score += 0.15
            confident_evidence["voice_quality_clean"] = round(jitter, 2)
        
        if f.get("filler_rate_pct", 0) < 0.5:  # Very few fillers
            confident_score += 0.10
            confident_evidence["fillers_minimal"] = f.get("filler_rate_pct", 0)
        
        # ── Classify ──
        if nervous_score > 0.50 and nervous_score > confident_score + 0.15:
            return {
                "tone": "nervous",
                "confidence_raw": min(nervous_score, 0.75),
                "evidence": nervous_evidence,
            }
        elif confident_score > 0.50 and confident_score > nervous_score + 0.15:
            return {
                "tone": "confident",
                "confidence_raw": min(confident_score, 0.70),
                "evidence": confident_evidence,
            }
        else:
            # Neither pattern is clearly dominant
            return {
                "tone": "neutral",
                "confidence_raw": 0.30,
                "evidence": {
                    "nervous_score": round(nervous_score, 3),
                    "confident_score": round(confident_score, 3),
                    "note": "Neither pattern dominant (difference < 0.15)"
                },
            }
