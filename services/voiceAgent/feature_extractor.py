"""
NEXUS Voice Agent - Feature Extractor
Extracts acoustic/prosodic features from audio using librosa.

Produces per-speaker, per-window feature vectors containing:
  - F0 (pitch): mean, std, variance, max, min, range
  - Energy: RMS mean, dynamic range
  - Speech rate: syllables/sec proxy, words/min from transcript
  - Voice quality: jitter (local), shimmer, HNR
  - Temporal: pause count, pause duration, filler count
"""
import numpy as np
import librosa
import soundfile as sf
from typing import Optional
from pathlib import Path
import logging

logger = logging.getLogger("nexus.voice.features")

# Analysis window size in seconds
WINDOW_SIZE_SEC = 5.0
HOP_SIZE_SEC = 2.5  # 50% overlap


class VoiceFeatureExtractor:
    """Extract acoustic features from audio per speaker per time window."""
    
    def __init__(self, sr: int = 16000):
        self.sr = sr  # Target sample rate
    
    def extract_all(
        self, 
        audio_path: str, 
        segments: list[dict]
    ) -> dict[str, list[dict]]:
        """
        Extract features for all speakers across the full audio.
        
        Args:
            audio_path: Path to audio file
            segments: List of diarised transcript segments 
                      [{speaker, start_ms, end_ms, text, words}, ...]
        
        Returns:
            Dict of {speaker_id: [feature_vector, feature_vector, ...]}
            Each feature_vector covers a WINDOW_SIZE_SEC window.
        """
        # Load full audio
        y, sr = librosa.load(audio_path, sr=self.sr, mono=True)
        duration_sec = len(y) / sr
        
        # Group segments by speaker
        speakers = set(seg["speaker"] for seg in segments)
        
        features_by_speaker = {}
        
        for speaker_id in speakers:
            speaker_segments = [s for s in segments if s["speaker"] == speaker_id]
            speaker_features = []
            
            # Create windows across the full audio
            window_samples = int(WINDOW_SIZE_SEC * sr)
            hop_samples = int(HOP_SIZE_SEC * sr)
            
            for win_start_sample in range(0, len(y) - window_samples + 1, hop_samples):
                win_end_sample = win_start_sample + window_samples
                win_start_ms = int(win_start_sample / sr * 1000)
                win_end_ms = int(win_end_sample / sr * 1000)
                
                # Find segments from THIS speaker in THIS window
                win_segments = [
                    s for s in speaker_segments
                    if s["end_ms"] > win_start_ms and s["start_ms"] < win_end_ms
                ]
                
                if not win_segments:
                    continue  # Speaker not talking in this window
                
                # Extract audio for this speaker's segments in this window
                speaker_audio = self._extract_speaker_audio(
                    y, sr, win_segments, win_start_ms, win_end_ms
                )
                
                if len(speaker_audio) < sr * 0.3:  # Less than 300ms of speech
                    continue
                
                # Extract features
                features = self._extract_features(
                    speaker_audio, sr, win_segments, win_start_ms, win_end_ms
                )
                
                if features:
                    features["speaker_id"] = speaker_id
                    features["window_start_ms"] = win_start_ms
                    features["window_end_ms"] = win_end_ms
                    speaker_features.append(features)
            
            if speaker_features:
                features_by_speaker[speaker_id] = speaker_features
        
        return features_by_speaker
    
    def _extract_speaker_audio(
        self, y: np.ndarray, sr: int, 
        segments: list[dict], 
        win_start_ms: int, win_end_ms: int
    ) -> np.ndarray:
        """Concatenate audio samples from speaker's segments within window."""
        chunks = []
        for seg in segments:
            # Clamp segment to window boundaries
            seg_start_ms = max(seg["start_ms"], win_start_ms)
            seg_end_ms = min(seg["end_ms"], win_end_ms)
            
            start_sample = int(seg_start_ms / 1000 * sr)
            end_sample = int(seg_end_ms / 1000 * sr)
            
            start_sample = max(0, min(start_sample, len(y)))
            end_sample = max(0, min(end_sample, len(y)))
            
            if end_sample > start_sample:
                chunks.append(y[start_sample:end_sample])
        
        if chunks:
            return np.concatenate(chunks)
        return np.array([], dtype=np.float32)
    
    def _extract_features(
        self, audio: np.ndarray, sr: int,
        segments: list[dict],
        win_start_ms: int, win_end_ms: int
    ) -> Optional[dict]:
        """Extract all acoustic features from an audio chunk."""
        if len(audio) < 512:
            return None
        
        features = {}
        
        # ── Pitch (F0) ──
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio, 
                fmin=librosa.note_to_hz('C2'),   # ~65 Hz
                fmax=librosa.note_to_hz('C7'),   # ~2093 Hz
                sr=sr,
                frame_length=2048,
                hop_length=512
            )
            
            # Filter to voiced frames only
            f0_voiced = f0[~np.isnan(f0)]
            
            if len(f0_voiced) > 0:
                features["f0_mean"] = float(np.mean(f0_voiced))
                features["f0_std"] = float(np.std(f0_voiced))
                features["f0_variance"] = float(np.var(f0_voiced))
                features["f0_max"] = float(np.max(f0_voiced))
                features["f0_min"] = float(np.min(f0_voiced))
                features["f0_range"] = float(np.max(f0_voiced) - np.min(f0_voiced))
                features["voiced_fraction"] = float(len(f0_voiced) / len(f0))
            else:
                features["f0_mean"] = 0.0
                features["f0_std"] = 0.0
                features["f0_variance"] = 0.0
                features["f0_max"] = 0.0
                features["f0_min"] = 0.0
                features["f0_range"] = 0.0
                features["voiced_fraction"] = 0.0
        except Exception as e:
            logger.warning(f"Pitch extraction failed: {e}")
            features.update({k: 0.0 for k in [
                "f0_mean", "f0_std", "f0_variance", "f0_max", "f0_min", "f0_range", "voiced_fraction"
            ]})
        
        # ── Energy (RMS) ──
        try:
            rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
            rms_db = librosa.amplitude_to_db(rms + 1e-10)
            
            features["energy_rms_db"] = float(np.mean(rms_db))
            features["energy_std_db"] = float(np.std(rms_db))
            features["energy_max_db"] = float(np.max(rms_db))
            features["energy_min_db"] = float(np.min(rms_db))
            features["energy_dynamic_range_db"] = float(np.max(rms_db) - np.min(rms_db))
        except Exception as e:
            logger.warning(f"Energy extraction failed: {e}")
            features.update({k: 0.0 for k in [
                "energy_rms_db", "energy_std_db", "energy_max_db", "energy_min_db", "energy_dynamic_range_db"
            ]})
        
        # ── Speech Rate (proxy) ──
        try:
            # Use transcript word count as primary speech rate measure
            window_words = sum(
                len(s.get("text", "").split()) for s in segments
            )
            window_duration_sec = (win_end_ms - win_start_ms) / 1000.0
            
            # Only count time where this speaker was actually talking
            speaker_time_sec = sum(
                (min(s["end_ms"], win_end_ms) - max(s["start_ms"], win_start_ms)) / 1000.0
                for s in segments
            )
            
            if speaker_time_sec > 0.5:
                features["speech_rate_wpm"] = float(window_words / (speaker_time_sec / 60.0))
            else:
                features["speech_rate_wpm"] = 0.0
            
            features["word_count"] = window_words
            features["speaking_time_sec"] = float(speaker_time_sec)
            
            # Spectral flux as syllable rate proxy
            onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=512)
            features["onset_rate"] = float(
                len(librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=512))
                / (len(audio) / sr)
            )
        except Exception as e:
            logger.warning(f"Rate extraction failed: {e}")
            features.update({
                "speech_rate_wpm": 0.0, "word_count": 0, 
                "speaking_time_sec": 0.0, "onset_rate": 0.0
            })
        
        # ── Voice Quality: Jitter & Shimmer ──
        try:
            jitter, shimmer = self._compute_jitter_shimmer(audio, sr)
            features["jitter_local_pct"] = jitter
            features["shimmer_local_pct"] = shimmer
        except Exception as e:
            logger.warning(f"Jitter/shimmer extraction failed: {e}")
            features["jitter_local_pct"] = 0.0
            features["shimmer_local_pct"] = 0.0
        
        # ── HNR (Harmonics-to-Noise Ratio) ──
        try:
            features["hnr_db"] = self._compute_hnr(audio, sr)
        except Exception as e:
            logger.warning(f"HNR extraction failed: {e}")
            features["hnr_db"] = 0.0
        
        # ── Spectral Centroid (brightness/tension indicator) ──
        try:
            centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=512)[0]
            features["spectral_centroid_hz"] = float(np.mean(centroid))
        except Exception:
            features["spectral_centroid_hz"] = 0.0
        
        # ── Pause Detection ──
        try:
            pause_info = self._detect_pauses(audio, sr, segments, win_start_ms, win_end_ms)
            features.update(pause_info)
        except Exception:
            features.update({
                "pause_count": 0, "total_pause_ms": 0, 
                "avg_pause_ms": 0, "max_pause_ms": 0, "pause_ratio": 0.0
            })
        
        # ── Filler Words (from transcript) ──
        try:
            filler_info = self._count_fillers(segments)
            features.update(filler_info)
        except Exception:
            features.update({
                "filler_count": 0, "filler_rate_pct": 0.0,
                "um_count": 0, "uh_count": 0, "like_count": 0,
                "fillers_detected": []
            })
        
        return features
    
    def _compute_jitter_shimmer(self, audio: np.ndarray, sr: int) -> tuple[float, float]:
        """
        Compute local jitter (F0 perturbation) and shimmer (amplitude perturbation).
        Simplified implementation using librosa pitch tracking.
        """
        f0, voiced_flag, _ = librosa.pyin(
            audio, fmin=65, fmax=600, sr=sr,
            frame_length=2048, hop_length=512
        )
        f0_voiced = f0[~np.isnan(f0)]
        
        if len(f0_voiced) < 3:
            return 0.0, 0.0
        
        # Jitter: mean absolute difference between consecutive periods
        period_diffs = np.abs(np.diff(1.0 / (f0_voiced + 1e-10)))
        mean_period = np.mean(1.0 / (f0_voiced + 1e-10))
        jitter_pct = float(np.mean(period_diffs) / (mean_period + 1e-10) * 100)
        
        # Shimmer: mean absolute difference between consecutive amplitudes
        rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
        if len(rms) < 3:
            return jitter_pct, 0.0
        
        amp_diffs = np.abs(np.diff(rms))
        mean_amp = np.mean(rms)
        shimmer_pct = float(np.mean(amp_diffs) / (mean_amp + 1e-10) * 100)
        
        return min(jitter_pct, 20.0), min(shimmer_pct, 30.0)  # Clamp outliers
    
    def _compute_hnr(self, audio: np.ndarray, sr: int) -> float:
        """
        Compute Harmonics-to-Noise Ratio (voice quality indicator).
        Higher HNR = cleaner voice. Lower = breathier/tenser.
        Normal speech: 15-25 dB. Stressed: < 12 dB.
        """
        # Autocorrelation-based HNR estimation
        frame_length = int(0.04 * sr)  # 40ms frames
        hop_length = int(0.01 * sr)    # 10ms hop
        
        hnr_values = []
        for start in range(0, len(audio) - frame_length, hop_length):
            frame = audio[start:start + frame_length]
            
            # Autocorrelation
            autocorr = np.correlate(frame, frame, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            if autocorr[0] == 0:
                continue
            
            # Normalize
            autocorr = autocorr / autocorr[0]
            
            # Find peak in pitch range (50-500 Hz)
            min_lag = int(sr / 500)
            max_lag = int(sr / 50)
            
            if max_lag >= len(autocorr):
                continue
            
            peak_region = autocorr[min_lag:max_lag]
            if len(peak_region) == 0:
                continue
            
            peak_val = np.max(peak_region)
            
            if peak_val > 0 and peak_val < 1:
                hnr = 10 * np.log10(peak_val / (1 - peak_val + 1e-10))
                hnr_values.append(float(hnr))
        
        if hnr_values:
            return float(np.median(hnr_values))
        return 0.0
    
    def _detect_pauses(
        self, audio: np.ndarray, sr: int,
        segments: list[dict],
        win_start_ms: int, win_end_ms: int
    ) -> dict:
        """
        Detect pauses within speaker's speech.
        Goldman-Eisler (1968): pauses > 250ms are hesitation pauses.
        """
        pauses = []
        sorted_segs = sorted(segments, key=lambda s: s["start_ms"])
        
        for i in range(len(sorted_segs) - 1):
            gap_start = sorted_segs[i]["end_ms"]
            gap_end = sorted_segs[i + 1]["start_ms"]
            gap_ms = gap_end - gap_start
            
            if 250 < gap_ms < 10000:  # 250ms - 10s = meaningful pause
                pauses.append(gap_ms)
        
        total_window_ms = win_end_ms - win_start_ms
        total_pause_ms = sum(pauses) if pauses else 0
        
        return {
            "pause_count": len(pauses),
            "total_pause_ms": total_pause_ms,
            "avg_pause_ms": int(np.mean(pauses)) if pauses else 0,
            "max_pause_ms": int(max(pauses)) if pauses else 0,
            "pause_ratio": float(total_pause_ms / total_window_ms) if total_window_ms > 0 else 0.0,
        }
    
    def _count_fillers(self, segments: list[dict]) -> dict:
        """
        Count filler words from transcript segments.
        Clark & Fox Tree (2002): 'uh' = minor delay, 'um' = major delay.
        """
        PRIMARY_FILLERS = {"um", "uh", "er", "ah", "uhm", "erm"}
        SECONDARY_FILLERS = {"like", "you know", "i mean", "sort of", "kind of", 
                            "basically", "actually", "right", "okay so"}
        
        fillers_detected = []
        total_words = 0
        um_count = 0
        uh_count = 0
        like_count = 0
        
        for seg in segments:
            words = seg.get("text", "").lower().split()
            total_words += len(words)
            
            for i, word in enumerate(words):
                clean = word.strip(".,!?;:'\"")
                
                if clean in PRIMARY_FILLERS:
                    filler_type = "major_delay" if clean in {"um", "uhm"} else "minor_delay"
                    fillers_detected.append({
                        "word": clean,
                        "type": filler_type,
                        "segment_start_ms": seg.get("start_ms", 0),
                    })
                    if clean in {"um", "uhm"}:
                        um_count += 1
                    elif clean in {"uh", "er", "ah", "erm"}:
                        uh_count += 1
                
                # Check bigrams for secondary fillers
                if i < len(words) - 1:
                    bigram = f"{clean} {words[i+1].strip('.,!?;:')}"
                    if bigram in SECONDARY_FILLERS:
                        fillers_detected.append({
                            "word": bigram,
                            "type": "discourse_marker",
                            "segment_start_ms": seg.get("start_ms", 0),
                        })
                        if "like" in bigram:
                            like_count += 1
                
                elif clean == "like" and i > 0 and i < len(words) - 1:
                    # Standalone "like" as filler (not comparative)
                    prev_word = words[i-1].strip(".,!?;:'\"")
                    if prev_word not in {"would", "looks", "sounds", "feels", "seems", "is", "was"}:
                        like_count += 1
                        fillers_detected.append({
                            "word": "like",
                            "type": "discourse_marker",
                            "segment_start_ms": seg.get("start_ms", 0),
                        })
        
        filler_count = len(fillers_detected)
        filler_rate_pct = (filler_count / total_words * 100) if total_words > 0 else 0.0
        
        return {
            "filler_count": filler_count,
            "filler_rate_pct": round(filler_rate_pct, 3),
            "um_count": um_count,
            "uh_count": uh_count,
            "like_count": like_count,
            "fillers_detected": fillers_detected,
            "total_words_in_window": total_words,
        }
