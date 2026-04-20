# services/voiceAgent/rules.py
"""
NEXUS Voice Agent - Rule Engine
Implements 14 core rules from the NEXUS Rule Engine specification.

Each rule takes raw features + speaker baseline and produces Signal objects.
Thresholds are loaded from the rule_config database table (configurable).
For now, defaults are hardcoded matching the Rule Engine document.

Rules implemented:
  VOICE-STRESS-01: Composite vocal stress score
  VOICE-FILLER-01: Filler word detection & classification
  VOICE-FILLER-02: Filler credibility threshold
  VOICE-PITCH-01: Pitch elevation flag
  VOICE-PITCH-02: Monotone detection
  VOICE-RATE-01: Speech rate anomaly detection
  VOICE-TONE-03: Nervous/Anxious tone classification
  VOICE-TONE-04: Confident/Authoritative tone classification
  VOICE-TONE-05: Warm/Friendly tone classification
  VOICE-TONE-06: Cold/Distant tone classification
  VOICE-TONE-07: Aggressive tone classification
  VOICE-TONE-08: Excited tone classification
  VOICE-ENERGY-01: Energy level classification
  VOICE-VOL-01:   Volume shift from baseline
  VOICE-PAUSE-01: Pause classification (hesitation)
  VOICE-PAUSE-02: Strategic pause detection
  VOICE-INT-01:   Interruption detection
  VOICE-TALK-01:  Talk time ratio (session level)
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

try:
    from shared.config.content_type_profile import ContentTypeProfile
except ImportError:
    ContentTypeProfile = None

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
        transcript_segments: list[dict] = None,
        conversation_features: dict = None,
        profile: "ContentTypeProfile | None" = None,
    ) -> list[dict]:
        """
        Run all rules against a single feature window.

        Args:
            features: Feature dict from VoiceFeatureExtractor
            baseline: SpeakerBaseline from CalibrationModule
            speaker_id: Speaker identifier
            transcript_segments: Transcript segments in this window
            conversation_features: Reserved for future conversation-level features
            profile: ContentTypeProfile for content-aware thresholds/gating/renaming

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

        def _add(rule_id: str, signal_type: str, value, value_text: str,
                 confidence: float, metadata: dict = None):
            """Helper: apply profile gating, confidence multiplier, and renaming."""
            if profile and profile.is_gated(rule_id):
                return
            conf = confidence
            if profile:
                conf = profile.apply_confidence(rule_id, conf)
            renamed = profile.rename_signal(value_text) if profile else value_text
            signals.append(_make_signal(
                speaker_id, signal_type, value, renamed, conf,
                window_start, window_end, metadata or {},
            ))

        # ── VOICE-STRESS-01: Composite Vocal Stress Score ──
        stress = self._rule_stress_01(features, baseline)
        if stress is not None:
            stress_offset = profile.get_threshold("VOICE-STRESS-01", "stress_offset", 0.0) if profile else 0.0
            adjusted_score = max(0, stress["score"] - stress_offset)
            _add("VOICE-STRESS-01", "vocal_stress_score",
                 adjusted_score, stress["level"],
                 adjusted_score * cal_conf, stress["components"])

        # ── VOICE-FILLER-01 + VOICE-FILLER-02: Filler Detection ──
        # noticeable_pct: minimum filler rate to flag credibility impact.
        # Default 2.5% per matrix; internal raises to 3.0% (Bortfeld 2001).
        noticeable_pct = profile.get_threshold("VOICE-FILLER-02", "noticeable_pct", 2.5) if profile else 2.5
        filler = self._rule_filler_01(features, baseline, noticeable_pct=noticeable_pct)
        if filler is not None:
            _add("VOICE-FILLER-01", "filler_detection",
                 filler["filler_rate_pct"], filler["status"],
                 0.90 * cal_conf, {
                     "filler_count": filler["filler_count"],
                     "um_count": filler["um_count"],
                     "uh_count": filler["uh_count"],
                     "delta_from_baseline": filler.get("delta", 0),
                     "credibility_impact": filler.get("credibility_impact", "none"),
                 })

        # ── VOICE-PITCH-01: Pitch Elevation Flag ──
        # mild_pct: minimum delta% to flag as mild elevation. Default 7% (Pakosz 1983); interview 12%.
        mild_pct = profile.get_threshold("VOICE-PITCH-01", "mild_pct", 7.0) if profile else 7.0
        pitch = self._rule_pitch_01(features, baseline, mild_pct=mild_pct)
        if pitch is not None:
            _add("VOICE-PITCH-01", "pitch_elevation_flag",
                 pitch["delta_pct"], pitch["level"],
                 0.50 * cal_conf, {
                     "f0_current": pitch["f0_current"],
                     "f0_baseline": pitch["f0_baseline"],
                     "delta_pct": pitch["delta_pct"],
                 })

        # ── VOICE-PITCH-02: Monotone Detection ──
        monotone = self._rule_pitch_02(features, baseline)
        if monotone is not None:
            _add("VOICE-PITCH-02", "monotone_flag",
                 monotone["value"], monotone["value_text"],
                 monotone["confidence_raw"] * cal_conf,
                 monotone.get("evidence", {}))

        # ── VOICE-RATE-01: Speech Rate Anomaly ──
        # anomaly_pct: deviation % from baseline to flag. Default 20% (Apple 1979); interview 35%.
        anomaly_pct = profile.get_threshold("VOICE-RATE-01", "anomaly_pct", 20.0) if profile else 20.0
        rate = self._rule_rate_01(features, baseline, anomaly_pct=anomaly_pct)
        if rate is not None:
            _add("VOICE-RATE-01", "speech_rate_anomaly",
                 rate["delta_pct"], rate["classification"],
                 0.40 * cal_conf, {
                     "wpm_current": rate["wpm_current"],
                     "wpm_baseline": rate["wpm_baseline"],
                     "sub_classification": rate.get("sub_classification", ""),
                 })

        # ── VOICE-TONE-03/04: Tone Classification (nervous/confident) ──
        tone = self._rule_tone(features, baseline)
        tone_rule_id = "VOICE-TONE-03"  # nervous/confident
        if tone is not None and tone["tone"] not in ("neutral",):
            tone_rule_id = "VOICE-TONE-04" if tone["tone"] == "confident" else "VOICE-TONE-03"
            _add(tone_rule_id, "tone_classification",
                 tone["confidence_raw"], tone["tone"],
                 tone["confidence_raw"] * cal_conf,
                 tone.get("evidence", {}))
        else:
            new_tone = (
                self._rule_tone_aggressive(features, baseline)
                or self._rule_tone_excited(features, baseline)
                or self._rule_tone_warm(features, baseline)
                or self._rule_tone_cold(features, baseline)
            )
            if new_tone is not None:
                tone_label = new_tone["value_text"]
                tone_ids = {"warm": "VOICE-TONE-01", "excited": "VOICE-TONE-06",
                            "aggressive": "VOICE-TONE-05", "cold": "VOICE-TONE-02"}
                _add(tone_ids.get(tone_label, "VOICE-TONE-01"), "tone_classification",
                     new_tone["confidence_raw"], tone_label,
                     new_tone["confidence_raw"] * cal_conf,
                     new_tone.get("evidence", {}))
            elif tone is not None:
                _add("VOICE-TONE-03", "tone_classification",
                     tone["confidence_raw"], tone["tone"],
                     tone["confidence_raw"] * cal_conf,
                     tone.get("evidence", {}))

        # ── VOICE-ENERGY-01: Energy Level Classification ──
        energy = self._rule_energy_01(features, baseline)
        if energy is not None:
            _add("VOICE-ENERGY-01", "energy_level",
                 energy["value"], energy["value_text"],
                 energy["confidence_raw"] * cal_conf,
                 energy.get("evidence", {}))

        # ── VOICE-VOL-01: Volume Shift From Baseline ──
        vol = self._rule_vol_01(features, baseline)
        if vol is not None:
            _add("VOICE-VOL-01", "volume_shift",
                 vol["delta_db"], vol["level"],
                 vol["confidence_raw"] * cal_conf,
                 {
                     "delta_db": vol["delta_db"],
                     "energy_current_db": vol["energy_current_db"],
                     "energy_baseline_db": vol["energy_baseline_db"],
                 })

        # ── VOICE-PAUSE-01: Pause Classification ──
        # extended_pause_ms: threshold for extended_hesitation. Default 2000ms;
        # interview 3000ms (Stivers 2009: complex answers need formulation time).
        extended_pause_ms = int(profile.get_threshold("VOICE-PAUSE-01", "extended_pause_ms", 2000)) if profile else 2000
        pause_signals = self._rule_pause_01(features, baseline, extended_pause_ms=extended_pause_ms)
        for ps in pause_signals:
            _add("VOICE-PAUSE-01", "pause_classification",
                 ps["value"], ps["value_text"],
                 ps["confidence_raw"] * cal_conf,
                 ps.get("evidence", {}))

        # ── VOICE-PAUSE-02: Strategic Pause Detection ──
        strategic = self._rule_pause_02(features, baseline)
        if strategic is not None:
            _add("VOICE-PAUSE-02", "strategic_pause",
                 strategic["pause_ms"] / 1000.0, strategic["level"],
                 strategic["confidence_raw"] * cal_conf,
                 {
                     "pause_ms": strategic["pause_ms"],
                     "word_count_after": strategic["word_count_after"],
                     "baseline_pause_ratio": strategic["baseline_pause_ratio"],
                 })

        # ── VOICE-INT-01: Interruption Detection ──
        # min_overlap_ms: minimum overlap to count as interruption. Default 500ms (Levinson 1983);
        # podcast 400ms (crosstalk / simultaneous laughter is normal in podcasts).
        if transcript_segments:
            min_overlap_ms = int(profile.get_threshold("VOICE-INT-01", "overlap_ms", 500)) if profile else 500
            int_signals = self._rule_int_01(features, baseline, transcript_segments, min_overlap_ms=min_overlap_ms)
            if int_signals:
                for isig in int_signals:
                    _add("VOICE-INT-01", "interruption_event",
                         isig["value"], isig["value_text"],
                         isig["confidence_raw"] * cal_conf,
                         isig.get("evidence", {}))

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
          filler rate:   0.10 (Veiga 2025: lacks direct stress validation)
          pause freq:    0.10
          HNR inverse:   0.10 (Veiga 2025: spectral quality more diagnostic than filler)
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
        
        # Weighted composite (Veiga 2025: HNR raised to 0.10, filler lowered to 0.10)
        score = (
            0.30 * components["f0_norm"] +
            0.20 * components["jitter_norm"] +
            0.15 * components["rate_norm"] +
            0.10 * components["filler_norm"] +
            0.10 * components["pause_norm"] +
            0.10 * components["hnr_norm"] +
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
    
    def _rule_filler_01(self, f: dict, b: SpeakerBaseline, noticeable_pct: float = 2.5) -> Optional[dict]:
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
            # Zero-baseline speaker: use 3% normal floor (Bortfeld 2001 lower bound).
            delta = filler_rate / 3.0  # 3% filler rate = 100% delta

        # Status based on delta
        if delta > 0.50:  # 50% more fillers than baseline (or >0.5% absolute for zero-baseline)
            status = "filler_spike"
        elif delta > 0.25:
            status = "filler_elevated"
        else:
            status = "normal"
        
        # VOICE-FILLER-02: Absolute credibility thresholds (Bortfeld 2001: normal range 1.3–4.4%)
        # noticeable_pct is content-type-aware (default 2.5%, internal 3.0%)
        if filler_rate > 6.0:
            credibility_impact = "severe"
        elif filler_rate > 4.0:
            credibility_impact = "significant"
        elif filler_rate > noticeable_pct:
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
    
    def _rule_pitch_01(self, f: dict, b: SpeakerBaseline, mild_pct: float = 7.0) -> Optional[dict]:
        """
        Detect pitch elevation from baseline.
        Pitch rise = arousal/stress indicator (NOT deception per se).

        Thresholds:
          > +7%  → mild (arousal increase, Pakosz 1983)
          > +12% → significant (strong stress response, Veiga 2025 avg effect ~10.7%)
          > +20% → extreme (acute stress or strong emotion)
        """
        f0_current = f.get("f0_mean", 0)
        if f0_current == 0 or b.f0_mean == 0:
            return None
        
        delta_pct = self.cal.compute_delta(f0_current, b.f0_mean) * 100
        
        # Only flag elevations (stress indicator). Drops are separate (confidence/disengagement).
        if delta_pct < mild_pct:
            return None  # Within normal range, don't flag
        
        if delta_pct >= 20.0:
            level = "pitch_elevated_extreme"
        elif delta_pct >= 12.0:
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
    
    def _rule_rate_01(self, f: dict, b: SpeakerBaseline, anomaly_pct: float = 20.0) -> Optional[dict]:
        """
        Detect speech rate deviations from baseline.

        > +20% → rate_elevated (anxiety, enthusiasm, or rushing)
        < -20% → rate_depressed (disengagement, cognitive load, or deliberation)
        """
        wpm_current = f.get("speech_rate_wpm", 0)
        if wpm_current < 30 or b.speech_rate_wpm < 30:  # Filter out near-silence
            return None
        
        delta_pct = self.cal.compute_delta(wpm_current, b.speech_rate_wpm) * 100
        
        if abs(delta_pct) < anomaly_pct:
            return None  # Within normal range

        if delta_pct > anomaly_pct:
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

    # ════════════════════════════════════════════════════════
    # VOICE-TONE-05: Warm/Friendly Tone Classification
    # Research: Juslin & Laukka 2003 — warmth associated with
    # stable pitch near baseline, lower variance, controlled rate
    # ════════════════════════════════════════════════════════

    def _rule_tone_warm(self, f: dict, b: SpeakerBaseline) -> Optional[dict]:
        """
        Detect warm/friendly tone.

        Profile:
          - F0 within -10% to +5% of baseline (relaxed, not stressed)
          - F0 variance lower than baseline (smooth prosody)
          - Not faster than baseline (unhurried delivery)
          - Jitter < 1.5% (clean phonation, no tension)
        """
        if b.f0_mean == 0:
            return None

        f0_delta = self.cal.compute_delta(f.get("f0_mean", 0), b.f0_mean)
        f0_var_delta = self.cal.compute_delta(f.get("f0_variance", 0), b.f0_variance) if b.f0_variance > 0 else 0
        rate_delta = self.cal.compute_delta(f.get("speech_rate_wpm", 0), b.speech_rate_wpm) if b.speech_rate_wpm > 0 else 0
        jitter = f.get("jitter_local_pct", 0)

        score = 0.0
        evidence = {}

        # F0 within -10% to +5% of baseline
        if -0.10 <= f0_delta <= 0.05:
            score += 0.30
            evidence["pitch_near_baseline"] = round(f0_delta * 100, 1)

        # F0 variance lower (smooth prosody)
        if f0_var_delta < -0.10:
            score += 0.25
            evidence["pitch_variance_lower"] = round(f0_var_delta * 100, 1)

        # Not faster than baseline (unhurried)
        if rate_delta <= 0.05:
            score += 0.25
            evidence["rate_unhurried"] = round(rate_delta * 100, 1)

        # Clean phonation (low jitter)
        if jitter < 1.5:
            score += 0.20
            evidence["jitter_low"] = round(jitter, 2)

        if score < 0.50:
            return None

        return {
            "value_text": "warm",
            "confidence_raw": min(score, 0.60),
            "evidence": evidence,
        }

    # ════════════════════════════════════════════════════════
    # VOICE-TONE-06: Cold/Distant Tone Classification
    # Research: Juslin & Laukka 2003 — cold/distant voice marked
    # by low prosodic variation, reduced energy range
    # ════════════════════════════════════════════════════════

    def _rule_tone_cold(self, f: dict, b: SpeakerBaseline) -> Optional[dict]:
        """
        Detect cold/distant tone.

        Profile:
          - F0 variance 30%+ below baseline (flat, unexpressive)
          - Energy dynamic range 25%+ below baseline (narrow dynamics)
          - Slightly slower than baseline
        """
        if b.f0_mean == 0:
            return None

        f0_var_delta = self.cal.compute_delta(f.get("f0_variance", 0), b.f0_variance) if b.f0_variance > 0 else 0
        energy_range_current = f.get("energy_dynamic_range_db", 0)
        # Use energy_rms_db as proxy for energy range baseline if no explicit baseline
        energy_range_baseline = b.energy_rms_db  # Approximate: compare dynamic range to mean energy
        energy_range_delta = self.cal.compute_delta(energy_range_current, energy_range_baseline) if energy_range_baseline != 0 else 0
        rate_delta = self.cal.compute_delta(f.get("speech_rate_wpm", 0), b.speech_rate_wpm) if b.speech_rate_wpm > 0 else 0

        score = 0.0
        evidence = {}

        # F0 variance 30%+ below baseline
        if f0_var_delta <= -0.30:
            score += 0.35
            evidence["pitch_variance_flat"] = round(f0_var_delta * 100, 1)

        # Energy range 25%+ below baseline
        if energy_range_delta <= -0.25:
            score += 0.35
            evidence["energy_range_narrow"] = round(energy_range_delta * 100, 1)

        # Slightly slower
        if rate_delta < -0.05:
            score += 0.30
            evidence["rate_slower"] = round(rate_delta * 100, 1)

        if score < 0.50:
            return None

        return {
            "value_text": "cold",
            "confidence_raw": min(score, 0.55),
            "evidence": evidence,
        }

    # ════════════════════════════════════════════════════════
    # VOICE-TONE-07: Aggressive Tone Classification
    # Research: Banse & Scherer 1996 — anger/aggression marked by
    # high F0, high variance, high energy, fast rate
    # ════════════════════════════════════════════════════════

    def _rule_tone_aggressive(self, f: dict, b: SpeakerBaseline) -> Optional[dict]:
        """
        Detect aggressive tone.

        Profile:
          - F0 20%+ above baseline
          - F0 variance 25%+ above baseline
          - Energy 15%+ above baseline
          - Rate 15%+ above baseline
        """
        if b.f0_mean == 0:
            return None

        f0_delta = self.cal.compute_delta(f.get("f0_mean", 0), b.f0_mean)
        f0_var_delta = self.cal.compute_delta(f.get("f0_variance", 0), b.f0_variance) if b.f0_variance > 0 else 0
        energy_delta = self.cal.compute_delta(f.get("energy_rms_db", 0), b.energy_rms_db) if b.energy_rms_db != 0 else 0
        rate_delta = self.cal.compute_delta(f.get("speech_rate_wpm", 0), b.speech_rate_wpm) if b.speech_rate_wpm > 0 else 0

        score = 0.0
        evidence = {}

        # F0 20%+ above baseline
        if f0_delta >= 0.20:
            score += 0.25
            evidence["pitch_elevated"] = round(f0_delta * 100, 1)

        # F0 variance 25%+ above (wide, forceful)
        if f0_var_delta >= 0.25:
            score += 0.25
            evidence["pitch_variance_wide"] = round(f0_var_delta * 100, 1)

        # Energy 15%+ above baseline
        if energy_delta >= 0.15:
            score += 0.25
            evidence["energy_elevated"] = round(energy_delta * 100, 1)

        # Rate 15%+ above baseline
        if rate_delta >= 0.15:
            score += 0.25
            evidence["rate_fast"] = round(rate_delta * 100, 1)

        if score < 0.50:
            return None

        return {
            "value_text": "aggressive",
            "confidence_raw": min(score, 0.55),
            "evidence": evidence,
        }

    # ════════════════════════════════════════════════════════
    # VOICE-TONE-08: Excited Tone Classification
    # Research: Juslin & Laukka 2003 — excitement has elevated F0,
    # wide variance, high energy, fast rate, but GOOD voice quality
    # (key discriminator from nervous: HNR maintained, low jitter)
    # ════════════════════════════════════════════════════════

    def _rule_tone_excited(self, f: dict, b: SpeakerBaseline) -> Optional[dict]:
        """
        Detect excited tone.

        Profile:
          - F0 10%+ above baseline
          - F0 variance 20%+ above baseline (wide, not narrow like nervous)
          - Energy 10%+ above baseline
          - Rate 10%+ above baseline
          - HNR maintained (not degraded — key discriminator from nervous)
          - Jitter < 2.0% (voice quality intact)
        """
        if b.f0_mean == 0:
            return None

        f0_delta = self.cal.compute_delta(f.get("f0_mean", 0), b.f0_mean)
        f0_var_delta = self.cal.compute_delta(f.get("f0_variance", 0), b.f0_variance) if b.f0_variance > 0 else 0
        energy_delta = self.cal.compute_delta(f.get("energy_rms_db", 0), b.energy_rms_db) if b.energy_rms_db != 0 else 0
        rate_delta = self.cal.compute_delta(f.get("speech_rate_wpm", 0), b.speech_rate_wpm) if b.speech_rate_wpm > 0 else 0
        hnr_delta = self.cal.compute_delta(f.get("hnr_db", 0), b.hnr_db) if b.hnr_db != 0 else 0
        jitter = f.get("jitter_local_pct", 0)

        score = 0.0
        evidence = {}

        # F0 10%+ above baseline
        if f0_delta >= 0.10:
            score += 0.20
            evidence["pitch_elevated"] = round(f0_delta * 100, 1)

        # F0 variance 20%+ above (wide, expressive — not narrow like nervous)
        if f0_var_delta >= 0.20:
            score += 0.20
            evidence["pitch_variance_wide"] = round(f0_var_delta * 100, 1)

        # Energy 10%+ above
        if energy_delta >= 0.10:
            score += 0.15
            evidence["energy_elevated"] = round(energy_delta * 100, 1)

        # Rate 10%+ above
        if rate_delta >= 0.10:
            score += 0.15
            evidence["rate_fast"] = round(rate_delta * 100, 1)

        # HNR maintained (not degraded) — key discriminator from nervous
        if hnr_delta >= -0.05:
            score += 0.15
            evidence["hnr_maintained"] = round(hnr_delta * 100, 1)

        # Jitter < 2.0% (voice quality intact)
        if jitter < 2.0:
            score += 0.15
            evidence["jitter_controlled"] = round(jitter, 2)

        if score < 0.50:
            return None

        return {
            "value_text": "excited",
            "confidence_raw": min(score, 0.55),
            "evidence": evidence,
        }

    # ════════════════════════════════════════════════════════
    # VOICE-PITCH-02: Monotone Detection
    # Research: Apple et al. 1979 — monotone delivery reduces
    # persuasiveness and perceived engagement
    # ════════════════════════════════════════════════════════

    def _rule_pitch_02(self, f: dict, b: SpeakerBaseline) -> Optional[dict]:
        """
        Detect monotone speech: flat prosody indicating disengagement
        or deliberate emotional suppression.

        Thresholds:
          - F0 variance 40%+ below baseline
          - F0 range < 30 Hz (absolute)
          - If energy also drops 15%+, increase confidence
        """
        if b.f0_mean == 0 or b.f0_variance == 0:
            return None

        f0_var_delta = self.cal.compute_delta(f.get("f0_variance", 0), b.f0_variance)
        f0_range = f.get("f0_range", 0)

        # Must meet both: variance 40%+ below AND range < 30 Hz
        if f0_var_delta > -0.40 or f0_range >= 30.0:
            return None

        delta = abs(f0_var_delta)
        cal_conf = b.calibration_confidence
        confidence = delta * cal_conf

        evidence = {
            "f0_variance_delta_pct": round(f0_var_delta * 100, 1),
            "f0_range_hz": round(f0_range, 1),
        }

        # Energy drop boosts confidence
        energy_delta = self.cal.compute_delta(f.get("energy_rms_db", 0), b.energy_rms_db) if b.energy_rms_db != 0 else 0
        if energy_delta <= -0.15:
            confidence += 0.10
            evidence["energy_also_depressed"] = round(energy_delta * 100, 1)

        confidence = min(confidence, 0.65)

        return {
            "value": round(delta, 4),
            "value_text": "monotone_detected",
            "confidence_raw": confidence,
            "evidence": evidence,
        }

    # ════════════════════════════════════════════════════════
    # VOICE-ENERGY-01: Energy Level Classification
    # Research: Scherer 2003 — vocal energy correlates with arousal
    # ════════════════════════════════════════════════════════

    def _rule_energy_01(self, f: dict, b: SpeakerBaseline) -> Optional[dict]:
        """
        Classify energy level relative to baseline.

        Only emits a signal when energy is outside the ±6 dB normal range.
          > +6 dB → "elevated" (arousal or emphasis)
          < -6 dB → "depressed" (disengagement or fatigue)
        """
        if b.energy_rms_db == 0:
            return None

        energy_current = f.get("energy_rms_db", 0)
        if energy_current == 0:
            return None

        energy_delta_db = energy_current - b.energy_rms_db

        # Only emit if outside ±6 dB (normal range produces no signal)
        if abs(energy_delta_db) < 6.0:
            return None

        evidence = {
            "energy_current_db": round(energy_current, 1),
            "energy_baseline_db": round(b.energy_rms_db, 1),
            "delta_db": round(energy_delta_db, 1),
        }

        if energy_delta_db > 6.0:
            value_text = "elevated"
            # Sub-classify: arousal vs emphasis based on F0
            f0_delta = self.cal.compute_delta(f.get("f0_mean", 0), b.f0_mean) if b.f0_mean > 0 else 0
            if f0_delta > 0.10:
                evidence["sub_classification"] = "arousal"
            else:
                evidence["sub_classification"] = "emphasis"
        else:
            value_text = "depressed"

        # Confidence scales with how far outside the ±6 dB range
        confidence = min(abs(energy_delta_db) / 20.0, 0.70)

        return {
            "value": round(energy_delta_db, 4),
            "value_text": value_text,
            "confidence_raw": confidence,
            "evidence": evidence,
        }

    # ════════════════════════════════════════════════════════
    # VOICE-VOL-01: Volume Shift From Baseline
    # Research: Scherer 2003 — vocal effort correlates with
    # dominance, emphasis, and emotional arousal.
    # 6 dB ≈ doubling of perceived loudness (Stevens' power law).
    # ════════════════════════════════════════════════════════

    def _rule_vol_01(self, f: dict, b: SpeakerBaseline) -> Optional[dict]:
        """
        Detect significant volume shifts from speaker's baseline.

        Uses energy_rms_db already extracted by VoiceFeatureExtractor.
        Complements VOICE-ENERGY-01 with clinically named 4-level output
        and a separate signal_type ("volume_shift") for downstream fusion.

        Thresholds:
          > +10 dB → volume_increase_significant (strong emphasis / frustration)
          > +6 dB  → volume_increase (emphasis, dominance, arousal)
          < -6 dB  → volume_decrease (disengagement, submission, uncertainty)
          < -10 dB → volume_decrease_significant (withdrawal, very low confidence)
        """
        if b.energy_rms_db == 0:
            return None
        energy_db = f.get("energy_rms_db", 0)
        if energy_db == 0:
            return None

        delta_db = energy_db - b.energy_rms_db
        if abs(delta_db) < 6.0:
            return None

        if delta_db >= 10.0:
            level = "volume_increase_significant"
            confidence = min(0.65, 0.50 + (delta_db - 10.0) * 0.03)
        elif delta_db >= 6.0:
            level = "volume_increase"
            confidence = min(0.55, 0.40 + (delta_db - 6.0) * 0.04)
        elif delta_db <= -10.0:
            level = "volume_decrease_significant"
            confidence = min(0.60, 0.45 + (abs(delta_db) - 10.0) * 0.03)
        else:
            level = "volume_decrease"
            confidence = min(0.50, 0.35 + (abs(delta_db) - 6.0) * 0.04)

        return {
            "delta_db": round(delta_db, 2),
            "energy_current_db": round(energy_db, 2),
            "energy_baseline_db": round(b.energy_rms_db, 2),
            "level": level,
            "confidence_raw": round(confidence, 4),
        }

    # ════════════════════════════════════════════════════════
    # VOICE-PAUSE-01: Pause Classification
    # Research: Goldman-Eisler 1968, Campione & Véronis 2002
    # Returns a LIST (can emit multiple signals per window)
    # ════════════════════════════════════════════════════════

    def _rule_pause_01(self, f: dict, b: SpeakerBaseline, extended_pause_ms: int = 2000) -> list[dict]:
        """
        Classify pause patterns in the current window.

        Can emit multiple signals:
          - pause_ratio > 0.55 → "excessive_pausing"
          - pause_ratio < 0.20 when baseline > 0.30 → "reduced_pausing"
          - max_pause > 2000 ms → "extended_hesitation"
        """
        results = []
        pause_ratio = f.get("pause_ratio", 0)
        max_pause_ms = f.get("max_pause_ms", 0)

        # Excessive pausing
        if pause_ratio > 0.55:
            confidence = min(pause_ratio, 0.70)
            results.append({
                "value": round(pause_ratio, 4),
                "value_text": "excessive_pausing",
                "confidence_raw": confidence,
                "evidence": {
                    "pause_ratio": round(pause_ratio, 3),
                    "pause_count": f.get("pause_count", 0),
                    "total_pause_ms": f.get("total_pause_ms", 0),
                },
            })

        # Reduced pausing (only meaningful if baseline has significant pauses)
        if pause_ratio < 0.20 and b.pause_ratio_pct > 0.30:
            delta = b.pause_ratio_pct - pause_ratio
            confidence = min(delta, 0.60)
            results.append({
                "value": round(pause_ratio, 4),
                "value_text": "reduced_pausing",
                "confidence_raw": confidence,
                "evidence": {
                    "pause_ratio": round(pause_ratio, 3),
                    "baseline_pause_ratio": round(b.pause_ratio_pct, 3),
                    "note": "Speaker normally pauses more — reduced pausing may indicate rushing or rehearsal",
                },
            })

        # Extended hesitation (threshold is content-type-aware; interview=3000ms)
        if max_pause_ms > extended_pause_ms:
            confidence = min(max_pause_ms / 5000.0, 0.65)
            results.append({
                "value": round(max_pause_ms / 1000.0, 4),
                "value_text": "extended_hesitation",
                "confidence_raw": confidence,
                "evidence": {
                    "max_pause_ms": max_pause_ms,
                    "avg_pause_ms": f.get("avg_pause_ms", 0),
                },
            })

        return results

    # ════════════════════════════════════════════════════════
    # VOICE-PAUSE-02: Strategic Pause Detection
    # Research: Duez 2001 — political speakers use pre-content
    # pauses for emphasis. Goldman-Eisler 1968 — pauses before
    # high-information content are longer.
    # Distinct from PAUSE-01 (hesitation): intentional emphasis.
    # ════════════════════════════════════════════════════════

    def _rule_pause_02(
        self, f: dict, b: SpeakerBaseline, min_pause_ms: int = 500,
    ) -> Optional[dict]:
        """
        Detect strategic pauses: silence 500–3000 ms before substantive content.

        Distinguishes from hesitation (PAUSE-01) by requiring:
          1. No filler words in the window (fillers = uncertainty, not strategy)
          2. Substantive content following (>= 5 words)
          3. Pause in 500–3000 ms range (> 3000 ms is hesitation, handled by PAUSE-01)
          4. Speaker's baseline pause ratio is not already high (< 0.50)

        Levels:
          500–1500 ms → strategic_pause (emphasis before key point)
          1500–3000 ms → dramatic_pause (strong rhetorical emphasis)
        """
        max_pause_ms = f.get("max_pause_ms", 0)
        if max_pause_ms < min_pause_ms:
            return None
        if max_pause_ms > 3000:
            return None  # Too long — hesitation territory (PAUSE-01 handles this)

        # Strategic pause must NOT be accompanied by fillers (that's hesitation)
        if f.get("filler_count", 0) > 0:
            return None

        # Must have substantive content in the window
        if f.get("word_count", 0) < 5:
            return None

        # Speaker who already pauses a lot — pause is not strategic
        if b.pause_ratio_pct > 0.50:
            return None

        if max_pause_ms >= 1500:
            level = "dramatic_pause"
            confidence = min(0.55, 0.40 + (max_pause_ms - 1500) / 5000.0)
        else:
            level = "strategic_pause"
            confidence = min(0.50, 0.35 + (max_pause_ms - 500) / 3000.0)

        return {
            "pause_ms": max_pause_ms,
            "level": level,
            "confidence_raw": round(confidence, 4),
            "word_count_after": f.get("word_count", 0),
            "baseline_pause_ratio": round(b.pause_ratio_pct, 3),
        }

    # ════════════════════════════════════════════════════════
    # VOICE-INT-01: Interruption Detection
    # Research: Schegloff 2000 — competitive vs cooperative overlaps
    # Takes transcript_segments for overlap analysis
    # ════════════════════════════════════════════════════════

    def _rule_int_01(
        self, f: dict, b: SpeakerBaseline, transcript_segments: list[dict],
        min_overlap_ms: int = 500,
    ) -> Optional[list[dict]]:
        """
        Detect interruptions from transcript segment overlaps.

        Checks consecutive segments: if seg_b starts before seg_a ends
        AND they are different speakers AND overlap > 500ms (Levinson 1983),
        that's an interruption. If the interrupting segment has <= 2 words,
        it's a backchannel (skip).

        Returns:
            List of signal dicts, or None if no interruptions found.
        """
        if not transcript_segments or len(transcript_segments) < 2:
            return None

        # Sort all segments by start time
        sorted_segs = sorted(transcript_segments, key=lambda s: s["start_ms"])
        results = []

        for i in range(len(sorted_segs) - 1):
            seg_a = sorted_segs[i]
            seg_b = sorted_segs[i + 1]

            # Must be different speakers
            if seg_a.get("speaker") == seg_b.get("speaker"):
                continue

            # Check overlap: seg_b starts before seg_a ends
            if seg_b["start_ms"] >= seg_a["end_ms"]:
                continue

            overlap_ms = seg_a["end_ms"] - seg_b["start_ms"]
            if overlap_ms <= min_overlap_ms:
                continue  # Below content-type threshold (podcast=400ms, default=500ms)

            # Backchannel filter: if interrupting segment has <= 2 words, skip
            interrupter_words = len(seg_b.get("text", "").split())
            if interrupter_words <= 2:
                continue

            overlap_seconds = round(overlap_ms / 1000.0, 3)
            confidence = min(overlap_seconds / 2.0, 0.65)

            results.append({
                "value": overlap_seconds,
                "value_text": "competitive_interruption",
                "confidence_raw": confidence,
                "window_start_ms": seg_b["start_ms"],
                "window_end_ms": seg_a["end_ms"],
                "evidence": {
                    "overlap_ms": overlap_ms,
                    "interrupter": seg_b.get("speaker", "unknown"),
                    "interrupted": seg_a.get("speaker", "unknown"),
                    "interrupter_words": interrupter_words,
                    "interrupter_text": seg_b.get("text", "")[:100],
                },
            })

        return results if results else None

    # ════════════════════════════════════════════════════════
    # VOICE-TALK-01: Talk Time Ratio (Session Level)
    # Research: Mast 2002 — dominance correlates with talk time
    # This is a session-level method, not per-window
    # ════════════════════════════════════════════════════════

    @staticmethod
    def _emit_talk_time_signals(
        features_by_speaker: dict[str, list[dict]],
        duration_sec: float,
    ) -> list[dict]:
        """
        Compute per-speaker talk time ratios and flag imbalances.

        Args:
            features_by_speaker: {speaker_id: [feature_dicts]}
            duration_sec: Total session duration in seconds

        Returns:
            List of Signal dicts for talk time imbalances.
        """
        if not features_by_speaker or duration_sec <= 0:
            return []

        # Compute total speaking time per speaker
        speaker_times = {}
        for speaker_id, features_list in features_by_speaker.items():
            total_sec = sum(f.get("speaking_time_sec", 0) for f in features_list)
            speaker_times[speaker_id] = total_sec

        total_talk_time = sum(speaker_times.values())
        if total_talk_time <= 0:
            return []

        n_speakers = len(speaker_times)
        expected_pct = 1.0 / n_speakers if n_speakers > 0 else 1.0
        signals = []

        for speaker_id, talk_sec in speaker_times.items():
            talk_pct = talk_sec / total_talk_time

            value_text = None
            confidence = 0.0

            if n_speakers == 2 and talk_pct > 0.70:
                # 2-speaker case: significant imbalance
                value_text = "talk_imbalance_significant"
                confidence = min((talk_pct - 0.70) / 0.20 * 0.50 + 0.40, 0.65)
            elif n_speakers > 2:
                if talk_pct > 2.0 * expected_pct:
                    # Dominant talker (more than 2x expected share)
                    value_text = "dominant_talker"
                    confidence = min((talk_pct / expected_pct - 2.0) / 2.0 * 0.50 + 0.40, 0.65)
                elif talk_pct < 0.3 * expected_pct:
                    # Silent participant (less than 30% of expected share)
                    value_text = "silent_participant"
                    confidence = min((0.3 * expected_pct - talk_pct) / (0.3 * expected_pct) * 0.50 + 0.40, 0.65)

            if value_text is not None:
                signals.append(_make_signal(
                    speaker_id=speaker_id,
                    signal_type="talk_time_ratio",
                    value=round(talk_pct, 4),
                    value_text=value_text,
                    confidence=confidence,
                    window_start_ms=0,
                    window_end_ms=int(duration_sec * 1000),
                    metadata={
                        "talk_seconds": round(talk_sec, 1),
                        "talk_pct": round(talk_pct * 100, 1),
                        "total_talk_time_sec": round(total_talk_time, 1),
                        "n_speakers": n_speakers,
                        "expected_pct": round(expected_pct * 100, 1),
                    },
                ))

        return signals
