# services/voiceAgent/interrogation_rules.py
"""
Interrogation-specific voice rules (NEXUS INTERROGATION_UPDATES1.MD §5).

Applies audio-quality-based confidence adjustments to existing voice signals
when analysing interrogation_video sessions.  Interrogation rooms use HVAC
systems that elevate ambient noise — when SNR < 20 dB, pitch, rate, and
agitation signal confidences drop slightly (0.50 → 0.45 per §5 table).

Rules implemented:
  INTERROG-VOICE-SNR  Audio SNR assessment + confidence adjustment
                      (Pitch Elevation 0.50→0.45, Speech Rate 0.50→0.45,
                       Vocal Agitation 0.50→0.45 when SNR < 20 dB)

Quality is quality-invariant for transcript-based analysis (diarization,
segments, speakers). Only acoustic feature signals are affected.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("nexus.voice.interrogation")

# SNR threshold below which HVAC noise becomes significant (dB)
_CCTV_AUDIO_SNR_THRESHOLD = 20.0

# Voice signal types that degrade with HVAC noise per §5 table
# Full confidence / CCTV audio confidence
_VOICE_SNR_ADJUSTMENTS: dict[str, tuple[float, float]] = {
    "pitch_deviation":        (0.50, 0.45),
    "pitch_elevation":        (0.50, 0.45),
    "speech_rate_deviation":  (0.50, 0.45),
    "speech_rate":            (0.50, 0.45),
    "vocal_stress_score":     (0.50, 0.45),
    "voice_energy_change":    (0.50, 0.45),
    "vocal_agitation":        (0.50, 0.45),
    "tone_shift":             (0.50, 0.45),
}


def estimate_snr(audio_path: str) -> float:
    """
    Estimate audio signal-to-noise ratio in dB using librosa RMS energy.

    Approach: percentile-based — top 10% of short-time RMS frames is the
    signal level; bottom 10% is the noise floor. Returns 40.0 for clean
    audio when noise floor is near zero.

    Returns: SNR in dB (higher = cleaner audio).
    """
    try:
        import librosa
        import numpy as np

        y, sr = librosa.load(audio_path, sr=None, mono=True, duration=300)
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]

        signal_rms = float(np.percentile(rms, 90))
        noise_rms  = float(np.percentile(rms, 10))

        if noise_rms <= 1e-8:
            return 40.0
        return float(20.0 * np.log10(signal_rms / noise_rms))
    except Exception as exc:
        logger.warning("SNR estimation failed (non-fatal): %s", exc)
        return 40.0  # assume clean if we can't measure


class InterrogationVoiceRules:
    """
    Stateless post-processing step for interrogation_video voice analysis.

    Two responsibilities:
      apply()                       — SNR-based confidence adjustment for acoustic signals
      derive_evidence_response_delays() — voice-based evidence response delay detection
                                          (complements conversation-based INTERROG-CONV-01)
    """

    def apply(
        self,
        signals: list[dict],
        audio_path: str,
        session_id: str = "",
    ) -> list[dict]:
        """
        Measure audio SNR and adjust affected signal confidences.

        Returns the same list with confidences modified in-place where
        the signal type is in _VOICE_SNR_ADJUSTMENTS and SNR < threshold.
        """
        snr_db = estimate_snr(audio_path)
        logger.info(
            "[%s] Interrogation audio quality: SNR=%.1f dB (%s)",
            session_id, snr_db,
            "CCTV_AUDIO — adjusting voice confidences" if snr_db < _CCTV_AUDIO_SNR_THRESHOLD
            else "GOOD_AUDIO — no adjustment",
        )

        if snr_db >= _CCTV_AUDIO_SNR_THRESHOLD:
            return signals

        adjusted_count = 0
        for sig in signals:
            sig_type = sig.get("signal_type", "")
            if sig_type not in _VOICE_SNR_ADJUSTMENTS:
                continue

            full_conf, cctv_conf = _VOICE_SNR_ADJUSTMENTS[sig_type]
            current_conf = sig.get("confidence", 0.0)

            # Only adjust signals whose confidence is near the full-confidence value
            # (avoids touching calibration-reduced signals that are already lower)
            if current_conf >= full_conf * 0.85:
                scale = cctv_conf / full_conf
                sig["confidence"] = round(current_conf * scale, 4)
                meta = sig.get("metadata")
                if isinstance(meta, dict):
                    meta["audio_snr_db"]      = round(snr_db, 1)
                    meta["snr_quality_tier"]  = "CCTV_AUDIO"
                    meta["snr_disclaimer"]    = (
                        f"Confidence reduced (SNR={snr_db:.1f}dB < {_CCTV_AUDIO_SNR_THRESHOLD}dB). "
                        "Interrogation room HVAC noise may affect acoustic precision."
                    )
                adjusted_count += 1

        if adjusted_count:
            logger.info(
                "[%s] SNR adjustment: %d voice signals reduced (%.1fdB < %.0fdB threshold)",
                session_id, adjusted_count, snr_db, _CCTV_AUDIO_SNR_THRESHOLD,
            )
        return signals

    def derive_evidence_response_delays(
        self,
        voice_signals: list[dict],
        diar_segments: list[dict],
        session_id: str = "",
    ) -> list[dict]:
        """
        INTERROG-VOICE-02: Voice-based evidence response processing delay.

        Detects long pauses (> 3 s) at speaker-turn boundaries using existing
        VOICE-PAUSE-01 extended_hesitation signals.  Complements the text-based
        conversation rule INTERROG-CONV-01 (which fires on evidence keywords +
        latency measurement) by providing a parallel acoustic confirmation path.

        DSA: O(P × D) where P = qualifying pause signals, D = diar segments.
        Both are small in practice (< 100 each for a typical interrogation).
        """
        # Filter to extended_hesitation pauses >= 3 000 ms only
        pause_sigs = [
            s for s in voice_signals
            if s.get("signal_type") == "pause_classification"
            and s.get("value_text") == "extended_hesitation"
            and s.get("metadata", {}).get("evidence", {}).get("max_pause_ms", 0) >= 3_000
        ]
        if not pause_sigs or not diar_segments:
            return []

        # Normalise segments to ms (handles both start_ms/end_ms and start/end float seconds)
        normed: list[dict] = []
        for seg in diar_segments:
            if "start_ms" in seg:
                start = int(seg["start_ms"])
                end   = int(seg["end_ms"])
            else:
                start = int(float(seg.get("start", 0)) * 1_000)
                end   = int(float(seg.get("end",   0)) * 1_000)
            spk = str(seg.get("speaker") or seg.get("speaker_id") or "unknown")
            if end > start:
                normed.append({"speaker": spk, "start_ms": start, "end_ms": end})
        normed.sort(key=lambda s: s["end_ms"])

        results: list[dict] = []
        for ps in pause_sigs:
            ps_start   = ps.get("window_start_ms", 0)
            ps_speaker = ps.get("speaker_id", "")
            pause_ms   = ps.get("metadata", {}).get("evidence", {}).get("max_pause_ms", 3_000)

            # Walk backward through sorted segments to find the most recent
            # different-speaker turn that ended within 5 s before the pause window
            preceding: dict | None = None
            for seg in reversed(normed):
                if seg["speaker"] == ps_speaker:
                    continue
                if seg["end_ms"] > ps_start:
                    continue
                lag = ps_start - seg["end_ms"]
                if lag > 5_000:
                    break
                preceding = seg
                break

            if preceding is None:
                continue

            results.append({
                "agent":           "voice",
                "speaker_id":      ps_speaker,
                "signal_type":     "evidence_response_processing_delay",
                "value":           round(min(pause_ms / 10_000, 1.0), 4),
                "value_text":      "delayed_response_pause",
                # Confidence scales 0.25 → 0.50 with pause duration; cap=0.50
                # (acoustic proxy is lower-confidence than text+latency CONV-01)
                "confidence":      round(min(0.50, 0.25 + (pause_ms - 3_000) / 20_000), 4),
                "window_start_ms": ps_start,
                "window_end_ms":   ps.get("window_end_ms", 0),
                "metadata": {
                    "rule_id":           "INTERROG-VOICE-02",
                    "pause_ms":          pause_ms,
                    "preceding_speaker": preceding["speaker"],
                    "turn_gap_ms":       ps_start - preceding["end_ms"],
                    "interpretation": (
                        "Extended pause at speaker turn boundary — may indicate "
                        "processing time for unexpected information or fabrication latency."
                    ),
                    "research": "Hartwig et al. (2014-2016) SUE framework",
                },
            })

        if results:
            logger.info(
                "[%s] INTERROG-VOICE-02: %d evidence response delay signals",
                session_id, len(results),
            )
        return results
