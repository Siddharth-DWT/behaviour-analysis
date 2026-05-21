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
import re
from collections import defaultdict, deque

logger = logging.getLogger("nexus.voice.interrogation")

# Filled-pause pattern for vocal_hesitation_cluster (conservative: standalone filler words only)
_FILLER_RE = re.compile(r"\b(um+|uh+|er+|ah+|hmm+)\b", re.IGNORECASE)

# SNR threshold below which HVAC noise becomes significant (dB)
_CCTV_AUDIO_SNR_THRESHOLD = 20.0

# Voice signal types that degrade with HVAC noise per §5 table
# Full confidence / CCTV audio confidence
_VOICE_SNR_ADJUSTMENTS: dict[str, tuple[float, float]] = {
    "pitch_elevation_flag":   (0.50, 0.45),
    "speech_rate_anomaly":    (0.50, 0.45),
    "vocal_stress_score":     (0.50, 0.45),
    "energy_level":           (0.50, 0.45),
    "agitated_high_arousal_tone": (0.50, 0.45),
    "tone_classification":    (0.50, 0.45),
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

    def vocal_hesitation_cluster(
        self,
        voice_signals: list[dict],
        diar_segments: list[dict],
        session_id: str = "",
    ) -> list[dict]:
        """
        INTERROG-VOICE-03: Temporal cluster of filled-pause disfluencies.

        Counts filler words (um/uh/er/ah/hmm) in absolute 2s time bins per speaker.
        Sliding deque of 5 consecutive bins (10s total).  Fires when >= 3/5 bins
        are elevated AND the window-aggregate rate >= 2x speaker baseline.

        DSA: O(S × B) — speakers × bins; deque O(1) append/pop; regex filler scan.
        Confidence cap 0.40 per prompt.md (Sporer & Schwandt 2006: small effect).
        """
        WINDOW_MS     = 2_000   # 2s bins
        SLIDE_N       = 5       # 10s sliding window
        MIN_ELEVATED  = 3       # ≥3/5 bins must be elevated
        BASELINE_RATIO = 2.0    # cluster rate must be ≥ 2x baseline
        CONF_CAP      = 0.40

        if not diar_segments:
            return []

        # Build per-speaker filler timeline: (start_ms, end_ms, filler_count)
        by_speaker: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
        for seg in diar_segments:
            spk = str(seg.get("speaker") or seg.get("speaker_id") or "")
            if not spk:
                continue
            if "start_ms" in seg:
                start, end = int(seg["start_ms"]), int(seg["end_ms"])
            else:
                start = int(float(seg.get("start", 0)) * 1_000)
                end   = int(float(seg.get("end",   0)) * 1_000)
            text    = seg.get("text", "") or ""
            fillers = len(_FILLER_RE.findall(text))
            by_speaker[spk].append((start, end, fillers))

        results: list[dict] = []

        for spk, segs in by_speaker.items():
            segs.sort()
            t_start = segs[0][0]
            t_end   = segs[-1][1]
            if t_end - t_start < WINDOW_MS * SLIDE_N * 2:
                continue

            # Distribute filler counts proportionally into absolute 2s bins
            n_bins = (t_end - t_start + WINDOW_MS - 1) // WINDOW_MS
            bins: list[float] = [0.0] * n_bins
            for s_start, s_end, s_fillers in segs:
                if s_fillers == 0 or s_end <= s_start:
                    continue
                s_dur = s_end - s_start
                b0 = (s_start - t_start) // WINDOW_MS
                b1 = (s_end   - t_start) // WINDOW_MS
                for b in range(max(0, b0), min(n_bins, b1 + 1)):
                    bin_s = t_start + b * WINDOW_MS
                    bin_e = bin_s + WINDOW_MS
                    overlap = min(s_end, bin_e) - max(s_start, bin_s)
                    if overlap > 0:
                        bins[b] += s_fillers * (overlap / s_dur)

            # Baseline: mean filler count in first-third bins
            baseline_n = max(1, n_bins // 3)
            baseline   = max(0.1, sum(bins[:baseline_n]) / baseline_n)

            # Sliding deque — fire once per non-overlapping cluster
            dq: deque = deque()
            fired_ends: set[int] = set()

            for b, count in enumerate(bins):
                dq.append((b, count))
                if len(dq) > SLIDE_N:
                    dq.popleft()
                if len(dq) < SLIDE_N:
                    continue

                elevated = sum(1 for _, c in dq if c > baseline)
                total    = sum(c for _, c in dq)
                ratio    = total / (baseline * SLIDE_N)

                if elevated >= MIN_ELEVATED and ratio >= BASELINE_RATIO:
                    end_bin = dq[-1][0]
                    if end_bin in fired_ends:
                        continue
                    fired_ends.add(end_bin)

                    cluster_start_ms = t_start + dq[0][0]  * WINDOW_MS
                    cluster_end_ms   = t_start + (end_bin + 1) * WINDOW_MS
                    conf = round(min(CONF_CAP, 0.20 + (ratio - BASELINE_RATIO) * 0.05), 4)

                    results.append({
                        "agent":           "voice",
                        "speaker_id":      spk,
                        "signal_type":     "vocal_hesitation_cluster",
                        "value":           round(min(ratio / 4.0, 1.0), 4),
                        "value_text":      "hesitation_burst",
                        "confidence":      conf,
                        "window_start_ms": cluster_start_ms,
                        "window_end_ms":   cluster_end_ms,
                        "metadata": {
                            "rule_id":          "INTERROG-VOICE-03",
                            "baseline_rate":    round(baseline, 3),
                            "cluster_rate":     round(total / SLIDE_N, 3),
                            "elevation_ratio":  round(ratio, 2),
                            "elevated_windows": elevated,
                            "research": (
                                "Sporer & Schwandt 2006 Applied Cognitive Psychology "
                                "20:421-446 — 'speech errors positively related to "
                                "deception'. Effect sizes described as 'small'."
                            ),
                            "effect_note": (
                                "Direction validated. Exact effect size for filled pause "
                                "clusters not available. Cluster threshold (3+ in 10s) "
                                "is an engineering heuristic, not from research."
                            ),
                            "interpretation": (
                                "Burst of speech disfluencies indicating cognitive load "
                                "spike. Equally occurs during genuine confusion, "
                                "word-finding difficulty, or high emotional arousal."
                            ),
                        },
                    })

        if results:
            logger.info(
                "[%s] INTERROG-VOICE-03: %d vocal_hesitation_cluster signals",
                session_id, len(results),
            )
        return results

    def speech_rate_change(
        self,
        voice_signals: list[dict],
        diar_segments: list[dict],
        session_id: str = "",
    ) -> list[dict]:
        """
        INTERROG-VOICE-04: Sustained speech rate deviation from speaker baseline.

        Computes per-speaker WPM in 10s absolute-time bins.  Fires when 2+
        consecutive bins deviate > 30% from the speaker's first-third baseline
        WPM in the same direction (faster or slower).

        Direction is context-dependent per Sporer & Schwandt 2006 — both
        acceleration and deceleration are detected.  value_text records direction.

        DSA: O(S × B) — speakers × bins; consecutive-run detection via deque.
        Confidence cap 0.40 per prompt.md (r≈0.08, approximately d≈0.16).
        """
        WINDOW_MS           = 10_000  # 10s bins
        CONSECUTIVE         = 2       # ≥2 consecutive deviating bins to fire
        DEVIATION_THRESHOLD = 0.30    # >30% from baseline
        MIN_SPEAK_FRAC      = 0.30    # bin must have ≥30% actual speech to count
        CONF_CAP            = 0.40

        if not diar_segments:
            return []

        # Build per-speaker (start_ms, end_ms, text) list
        by_speaker: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
        for seg in diar_segments:
            spk = str(seg.get("speaker") or seg.get("speaker_id") or "")
            if not spk:
                continue
            if "start_ms" in seg:
                start, end = int(seg["start_ms"]), int(seg["end_ms"])
            else:
                start = int(float(seg.get("start", 0)) * 1_000)
                end   = int(float(seg.get("end",   0)) * 1_000)
            text = seg.get("text", "") or ""
            by_speaker[spk].append((start, end, text))

        results: list[dict] = []

        for spk, segs in by_speaker.items():
            segs.sort()
            t_start = segs[0][0]
            t_end   = segs[-1][1]
            if t_end - t_start < WINDOW_MS * CONSECUTIVE * 3:
                continue

            n_bins    = (t_end - t_start + WINDOW_MS - 1) // WINDOW_MS
            word_bins:  list[float] = [0.0] * n_bins
            speak_bins: list[float] = [0.0] * n_bins

            for s_start, s_end, text in segs:
                s_dur = max(s_end - s_start, 1)
                words = len(text.split())
                if words == 0:
                    continue
                b0 = (s_start - t_start) // WINDOW_MS
                b1 = (s_end   - t_start) // WINDOW_MS
                for b in range(max(0, b0), min(n_bins, b1 + 1)):
                    bin_s   = t_start + b * WINDOW_MS
                    bin_e   = bin_s + WINDOW_MS
                    overlap = max(0, min(s_end, bin_e) - max(s_start, bin_s))
                    if overlap > 0:
                        frac = overlap / s_dur
                        word_bins[b]  += words  * frac
                        speak_bins[b] += overlap

            # Convert bins to WPM (nan when speech fraction < threshold)
            wpm_bins: list[float] = []
            for b in range(n_bins):
                speak_frac = speak_bins[b] / WINDOW_MS
                if speak_frac < MIN_SPEAK_FRAC:
                    wpm_bins.append(float("nan"))
                else:
                    speak_min = speak_bins[b] / 60_000
                    wpm_bins.append(word_bins[b] / speak_min)

            # Baseline: mean of first-third valid bins
            baseline_n    = max(1, n_bins // 3)
            valid_baseline = [w for w in wpm_bins[:baseline_n] if w == w]  # not nan
            if not valid_baseline:
                continue
            baseline_wpm = sum(valid_baseline) / len(valid_baseline)
            if baseline_wpm < 10:
                continue

            # Consecutive-run detector via deque
            run_dq: deque = deque()
            fired_ends: set[int] = set()

            for b, wpm in enumerate(wpm_bins):
                if wpm != wpm:   # nan — reset run
                    run_dq.clear()
                    continue

                dev = (wpm - baseline_wpm) / baseline_wpm
                if abs(dev) > DEVIATION_THRESHOLD:
                    run_dq.append((b, wpm, dev))
                else:
                    run_dq.clear()

                if len(run_dq) < CONSECUTIVE:
                    continue

                # All deviations must share the same direction
                signs = {1 if d > 0 else -1 for _, _, d in run_dq}
                if len(signs) > 1:
                    run_dq.popleft()
                    continue

                end_bin = run_dq[-1][0]
                if end_bin in fired_ends:
                    continue
                fired_ends.add(end_bin)

                direction = "faster" if next(iter(signs)) > 0 else "slower"
                mean_dev  = sum(abs(d) for _, _, d in run_dq) / len(run_dq)
                mean_wpm  = sum(w for _, w, _ in run_dq) / len(run_dq)
                conf = round(min(CONF_CAP, 0.25 + mean_dev * 0.15), 4)

                results.append({
                    "agent":           "voice",
                    "speaker_id":      spk,
                    "signal_type":     "speech_rate_change",
                    "value":           round(min(mean_dev, 1.0), 4),
                    "value_text":      direction,
                    "confidence":      conf,
                    "window_start_ms": t_start + run_dq[0][0]  * WINDOW_MS,
                    "window_end_ms":   t_start + (end_bin + 1) * WINDOW_MS,
                    "metadata": {
                        "rule_id":            "INTERROG-VOICE-04",
                        "baseline_wpm":       round(baseline_wpm, 1),
                        "mean_wpm":           round(mean_wpm, 1),
                        "deviation_pct":      round(mean_dev * 100, 1),
                        "direction":          direction,
                        "consecutive_windows": len(run_dq),
                        "research": (
                            "Sporer & Schwandt 2006 Applied Cognitive Psychology "
                            "20:421-446 — 'speech rate slightly positively related "
                            "to deception after short preparation (r=.082), unrelated "
                            "after medium preparation'. DePaulo et al. 2003 includes "
                            "speech rate."
                        ),
                        "effect_note": (
                            "Direction is MIXED — liars may speak faster OR slower "
                            "depending on preparation time and context. r≈0.08 "
                            "(approximately d≈0.16). Signal detects significant "
                            "change in EITHER direction."
                        ),
                        "interpretation": (
                            "Significant speech rate shift from baseline. "
                            "Acceleration may indicate rehearsed delivery. "
                            "Deceleration may indicate careful word selection. "
                            "Both also occur from fatigue, topic change, or "
                            "emotional arousal."
                        ),
                    },
                })

        if results:
            logger.info(
                "[%s] INTERROG-VOICE-04: %d speech_rate_change signals",
                session_id, len(results),
            )
        return results
