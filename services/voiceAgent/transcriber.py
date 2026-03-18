"""
NEXUS Voice Agent - Transcriber
Handles audio transcription (Whisper) and speaker diarization.

Supports two backends:
  1. LOCAL  — faster-whisper on CPU (default, no external dependency)
  2. EXTERNAL — GPU-accelerated Whisper API on remote server (much faster)

Set EXTERNAL_WHISPER_URL + EXTERNAL_API_KEY to enable the GPU backend.
It will auto-detect and fall back to local if the external API is unreachable.

Diarization:
  - When num_speakers is specified, uses pitch + energy KMeans clustering
    (scikit-learn) to separate speakers from acoustic features.
  - Falls back to gap-based heuristic if clustering is unavailable.
  - For production: use pyannote (USE_PYANNOTE=true).

Models auto-download on first use (~1.5GB for medium model).
"""
import os
import sys
import logging
import numpy as np
from typing import Optional
from pathlib import Path

logger = logging.getLogger("nexus.voice.transcriber")

# ── Configuration ──
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium")
USE_PYANNOTE = os.getenv("USE_PYANNOTE", "false").lower() == "true"

# External Whisper API (GPU-accelerated)
EXTERNAL_WHISPER_URL = os.getenv("EXTERNAL_WHISPER_URL", "")
EXTERNAL_API_KEY = os.getenv("EXTERNAL_API_KEY", "")
EXTERNAL_WHISPER_MODEL = os.getenv("EXTERNAL_WHISPER_MODEL", "base")


class Transcriber:
    """
    Audio transcription with word-level timestamps and speaker identification.

    Automatically uses the GPU-accelerated external Whisper API when configured
    (EXTERNAL_WHISPER_URL + EXTERNAL_API_KEY). Falls back to local faster-whisper
    if the external API is unreachable.
    """

    def __init__(self):
        self._model = None
        self._diarization_pipeline = None
        self._external_client = None
        self._use_external = False
        self._num_speakers = None  # Speaker count hint (2-10), None = auto

        # Try to initialise external Whisper client
        if EXTERNAL_WHISPER_URL and EXTERNAL_API_KEY:
            self._init_external()

    def _init_external(self):
        """Try to connect to the external Whisper API."""
        try:
            # Add shared module to path if needed
            project_root = str(Path(__file__).parent.parent.parent)
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            from shared.utils.external_apis import WhisperClient

            client = WhisperClient(
                base_url=EXTERNAL_WHISPER_URL,
                api_key=EXTERNAL_API_KEY,
                model=EXTERNAL_WHISPER_MODEL,
            )

            if client.is_healthy():
                self._external_client = client
                self._use_external = True
                logger.info(
                    f"Using EXTERNAL Whisper API: {EXTERNAL_WHISPER_URL} "
                    f"(model={EXTERNAL_WHISPER_MODEL})"
                )
            else:
                logger.warning(
                    f"External Whisper API at {EXTERNAL_WHISPER_URL} is not healthy. "
                    f"Falling back to local faster-whisper."
                )
        except Exception as e:
            logger.warning(
                f"Could not initialise external Whisper client: {e}. "
                f"Falling back to local faster-whisper."
            )

    @property
    def backend(self) -> str:
        """Return which backend is active: 'external' or 'local'."""
        return "external" if self._use_external else "local"

    def _load_model(self):
        """Lazy-load local whisper model on first use."""
        if self._model is not None:
            return

        try:
            from faster_whisper import WhisperModel

            logger.info(f"Loading local Whisper model: {WHISPER_MODEL} ...")
            self._model = WhisperModel(
                WHISPER_MODEL,
                device="cpu",       # Use "cuda" if GPU available
                compute_type="int8" # Fastest on CPU; use "float16" on GPU
            )
            logger.info(f"Local Whisper model loaded.")
        except ImportError:
            logger.error("faster-whisper not installed. Run: pip install faster-whisper")
            raise

    def transcribe(self, audio_path: str, num_speakers: Optional[int] = None) -> dict:
        """
        Transcribe an audio file with word-level timestamps.

        Uses external GPU API if configured, otherwise local faster-whisper.

        Args:
            audio_path: Path to audio file
            num_speakers: Optional hint for number of speakers (2-10).
                         None = auto-detect (defaults to 2 for simple diarization).

        Returns:
            {
                "duration_seconds": float,
                "backend": "external" | "local",
                "model": str,
                "segments": [
                    {
                        "speaker": "Speaker_0",
                        "start_ms": int,
                        "end_ms": int,
                        "text": str,
                        "words": [{"word": str, "start": float, "end": float, "probability": float}]
                    }
                ]
            }
        """
        self._num_speakers = num_speakers
        if self._use_external:
            return self._transcribe_external(audio_path)
        else:
            return self._transcribe_local(audio_path)

    # ═══════════════════════════════════════════════════════════
    # EXTERNAL BACKEND (GPU Whisper API)
    # ═══════════════════════════════════════════════════════════

    def _transcribe_external(self, audio_path: str) -> dict:
        """Transcribe via the external GPU-accelerated Whisper API."""
        logger.info(f"Transcribing via EXTERNAL API: {audio_path}")

        try:
            result = self._external_client.transcribe(
                audio_path,
                model=EXTERNAL_WHISPER_MODEL,
                language="en",
                word_timestamps=True,
                vad_filter=True,
            )
        except Exception as e:
            logger.error(
                f"External Whisper API failed: {e}. "
                f"Falling back to local transcription."
            )
            self._use_external = False
            return self._transcribe_local(audio_path)

        # Convert external API response to our internal format
        segments = []
        for seg in result.get("segments", []):
            words = []
            # External API returns words nested in segments when word_timestamps=True
            for w in seg.get("words", []):
                words.append({
                    "word": w.get("word", "").strip(),
                    "start": w.get("start", 0),
                    "end": w.get("end", 0),
                    "probability": round(w.get("probability", 0), 3),
                })

            segments.append({
                "start_ms": int(seg.get("start", 0) * 1000),
                "end_ms": int(seg.get("end", 0) * 1000),
                "text": seg.get("text", "").strip(),
                "words": words,
            })

        duration = result.get("duration", 0)
        model_used = result.get("model", EXTERNAL_WHISPER_MODEL)
        proc_time = result.get("processing_time", 0)

        logger.info(
            f"External transcription complete: {duration:.1f}s audio, "
            f"{len(segments)} segments, model={model_used}, "
            f"server_time={proc_time:.2f}s"
        )

        # Apply speaker diarization (same as local)
        if USE_PYANNOTE:
            segments = self._diarize_pyannote(audio_path, segments)
        else:
            segments = self._diarize_simple(segments, self._num_speakers, audio_path)

        return {
            "duration_seconds": duration,
            "backend": "external",
            "model": model_used,
            "language": result.get("language", "en"),
            "language_probability": result.get("language_probability", 0),
            "processing_time": proc_time,
            "segments": segments,
        }

    # ═══════════════════════════════════════════════════════════
    # LOCAL BACKEND (faster-whisper on CPU)
    # ═══════════════════════════════════════════════════════════

    def _transcribe_local(self, audio_path: str) -> dict:
        """Transcribe using local faster-whisper."""
        self._load_model()

        logger.info(f"Transcribing via LOCAL faster-whisper: {audio_path}")

        # Run Whisper
        segments_raw, info = self._model.transcribe(
            audio_path,
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,           # Filter out non-speech
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=200,
            ),
            language="en",             # Force English (remove for auto-detect)
        )

        # Convert generator to list and extract segments
        segments = []
        for seg in segments_raw:
            words = []
            if seg.words:
                words = [
                    {
                        "word": w.word.strip(),
                        "start": w.start,
                        "end": w.end,
                        "probability": round(w.probability, 3),
                    }
                    for w in seg.words
                ]

            segments.append({
                "start_ms": int(seg.start * 1000),
                "end_ms": int(seg.end * 1000),
                "text": seg.text.strip(),
                "words": words,
            })

        duration = info.duration
        logger.info(f"Local transcription complete: {duration:.1f}s, {len(segments)} segments")

        # Apply speaker diarization
        if USE_PYANNOTE:
            segments = self._diarize_pyannote(audio_path, segments)
        else:
            segments = self._diarize_simple(segments, self._num_speakers, audio_path)

        return {
            "duration_seconds": duration,
            "backend": "local",
            "model": WHISPER_MODEL,
            "segments": segments,
        }

    # ═══════════════════════════════════════════════════════════
    # SPEAKER DIARIZATION
    # ═══════════════════════════════════════════════════════════

    def _diarize_simple(
        self,
        segments: list[dict],
        num_speakers: Optional[int] = None,
        audio_path: Optional[str] = None,
    ) -> list[dict]:
        """
        Speaker diarization using acoustic clustering.

        When num_speakers is specified AND audio_path is available:
          - Extracts mean pitch (F0) and mean energy (RMS) per segment
          - Runs KMeans(n_clusters=num_speakers) on [pitch, energy] features
          - Assigns Speaker_0, Speaker_1, ... based on cluster labels

        Falls back to gap-based heuristic when clustering is unavailable.

        For production: replace with pyannote diarization.
        """
        if not segments:
            return segments

        max_speakers = min(num_speakers or 2, 10)

        # Try acoustic KMeans clustering first (when we have the audio)
        if audio_path and max_speakers >= 2:
            try:
                clustered = self._diarize_acoustic_kmeans(
                    segments, max_speakers, audio_path
                )
                if clustered is not None:
                    return clustered
            except Exception as e:
                logger.warning(f"Acoustic KMeans diarization failed: {e}")

        # Fallback: gap-based heuristic
        if max_speakers == 2:
            return self._diarize_gap_two_speaker(segments)
        return self._diarize_simple_multi_speaker(segments, max_speakers)

    def _diarize_acoustic_kmeans(
        self,
        segments: list[dict],
        num_speakers: int,
        audio_path: str,
    ) -> Optional[list[dict]]:
        """
        Cluster transcript segments into speakers using pitch + energy features.

        For each segment:
          1. Extract the audio slice [start_ms..end_ms]
          2. Compute mean F0 (pitch) via librosa.pyin
          3. Compute mean RMS energy
        Then KMeans(n_clusters=num_speakers) on the 2D feature matrix.
        """
        import librosa
        from sklearn.cluster import KMeans

        logger.info(
            f"Running acoustic KMeans diarization: {num_speakers} speakers, "
            f"{len(segments)} segments"
        )

        # Load full audio once
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        total_samples = len(y)

        features = []  # (mean_pitch, mean_energy) per segment
        valid_mask = []  # Track which segments have valid features

        for seg in segments:
            start_sample = int(seg["start_ms"] / 1000 * sr)
            end_sample = int(seg["end_ms"] / 1000 * sr)
            # Clamp to audio bounds
            start_sample = max(0, min(start_sample, total_samples - 1))
            end_sample = max(start_sample + 1, min(end_sample, total_samples))

            chunk = y[start_sample:end_sample]

            if len(chunk) < sr * 0.1:  # Skip very short chunks (< 100ms)
                features.append([0.0, 0.0])
                valid_mask.append(False)
                continue

            # Mean pitch via pyin
            try:
                f0, voiced_flag, _ = librosa.pyin(
                    chunk, fmin=60, fmax=500, sr=sr,
                    frame_length=2048, hop_length=512,
                )
                # Take mean of voiced frames only
                voiced_f0 = f0[voiced_flag] if voiced_flag is not None else f0[~np.isnan(f0)]
                mean_pitch = float(np.nanmean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
            except Exception:
                mean_pitch = 0.0

            # Mean RMS energy
            rms = librosa.feature.rms(y=chunk, frame_length=2048, hop_length=512)[0]
            mean_energy = float(np.mean(rms)) if len(rms) > 0 else 0.0

            features.append([mean_pitch, mean_energy])
            valid_mask.append(mean_pitch > 0)

        features_arr = np.array(features, dtype=np.float64)

        # Check we have enough valid features
        n_valid = sum(valid_mask)
        if n_valid < num_speakers:
            logger.warning(
                f"Only {n_valid} segments with valid pitch — "
                f"need at least {num_speakers}. Falling back."
            )
            return None

        # Normalise features to [0, 1] for balanced clustering
        for col in range(features_arr.shape[1]):
            col_data = features_arr[:, col]
            valid_data = col_data[np.array(valid_mask)]
            if len(valid_data) > 0:
                col_min = valid_data.min()
                col_max = valid_data.max()
                col_range = col_max - col_min
                if col_range > 0:
                    features_arr[:, col] = (col_data - col_min) / col_range
                else:
                    features_arr[:, col] = 0.5

        # Fill invalid segments with column means
        for i, valid in enumerate(valid_mask):
            if not valid:
                for col in range(features_arr.shape[1]):
                    valid_vals = features_arr[np.array(valid_mask), col]
                    features_arr[i, col] = np.mean(valid_vals) if len(valid_vals) > 0 else 0.5

        # KMeans clustering
        kmeans = KMeans(
            n_clusters=num_speakers,
            n_init=10,
            random_state=42,
        )
        labels = kmeans.fit_predict(features_arr)

        # Assign speaker labels; order clusters by mean pitch (lower pitch = Speaker_0)
        cluster_pitches = {}
        for cluster_id in range(num_speakers):
            mask = labels == cluster_id
            cluster_pitches[cluster_id] = float(np.mean(features_arr[mask, 0]))

        # Sort clusters by pitch: lowest pitch → Speaker_0
        sorted_clusters = sorted(cluster_pitches, key=lambda c: cluster_pitches[c])
        cluster_to_speaker = {
            cluster_id: f"Speaker_{rank}"
            for rank, cluster_id in enumerate(sorted_clusters)
        }

        for i, seg in enumerate(segments):
            seg["speaker"] = cluster_to_speaker[labels[i]]

        # Log results
        speaker_counts = {}
        for seg in segments:
            spk = seg["speaker"]
            speaker_counts[spk] = speaker_counts.get(spk, 0) + 1

        logger.info(
            f"Acoustic KMeans diarization complete: "
            f"{len(speaker_counts)} speakers — "
            + ", ".join(f"{k}: {v} segs" for k, v in sorted(speaker_counts.items()))
        )

        return segments

    def _diarize_gap_two_speaker(self, segments: list[dict]) -> list[dict]:
        """Fallback: 2-speaker alternation based on gaps."""
        current_speaker = "Speaker_0"
        TURN_GAP_MS = 1000

        for i, seg in enumerate(segments):
            if i > 0:
                gap_ms = seg["start_ms"] - segments[i-1]["end_ms"]
                if gap_ms > TURN_GAP_MS:
                    current_speaker = (
                        "Speaker_1" if current_speaker == "Speaker_0"
                        else "Speaker_0"
                    )
            seg["speaker"] = current_speaker

        return segments

    def _diarize_simple_multi_speaker(
        self,
        segments: list[dict],
        max_speakers: int,
    ) -> list[dict]:
        """
        Multi-speaker heuristic diarization (3-10 speakers).

        Uses segment duration, gap patterns, and word characteristics
        to cluster segments into speaker groups. This is a rough
        heuristic — for accurate multi-speaker, use pyannote.

        Strategy:
        1. Detect turn changes via gaps (> 1s gap = new turn)
        2. Group consecutive same-speaker segments into turns
        3. Cluster turns by duration pattern and assign speakers
           round-robin up to max_speakers
        4. Merge adjacent turns from the same speaker
        """
        TURN_GAP_MS = 1000
        SHORT_GAP_MS = 300  # Short gaps within same speaker

        # Step 1: Detect turn boundaries
        turn_boundaries = [0]  # First segment always starts a turn
        for i in range(1, len(segments)):
            gap_ms = segments[i]["start_ms"] - segments[i-1]["end_ms"]
            if gap_ms > TURN_GAP_MS:
                turn_boundaries.append(i)

        # Step 2: Group segments into turns
        turns = []
        for t in range(len(turn_boundaries)):
            start_idx = turn_boundaries[t]
            end_idx = (
                turn_boundaries[t + 1] if t + 1 < len(turn_boundaries)
                else len(segments)
            )
            turn_segments = segments[start_idx:end_idx]
            total_duration = sum(
                s["end_ms"] - s["start_ms"] for s in turn_segments
            )
            total_words = sum(
                len(s.get("words", []) or s.get("text", "").split())
                for s in turn_segments
            )
            turns.append({
                "segment_indices": list(range(start_idx, end_idx)),
                "total_duration_ms": total_duration,
                "total_words": total_words,
                "start_ms": turn_segments[0]["start_ms"],
            })

        # Step 3: Assign speakers using round-robin with clustering
        # Group turns by similar duration patterns to try to keep
        # the same speaker for similar-length utterances
        if len(turns) <= max_speakers:
            # Fewer turns than speakers — each turn is a different speaker
            for i, turn in enumerate(turns):
                speaker = f"Speaker_{i}"
                for idx in turn["segment_indices"]:
                    segments[idx]["speaker"] = speaker
        else:
            # More turns than speakers — use round-robin assignment
            # with a heuristic: long gaps between turns with similar
            # characteristics suggest the same speaker returning
            speaker_idx = 0
            prev_duration = 0

            for i, turn in enumerate(turns):
                if i == 0:
                    speaker_idx = 0
                else:
                    # Check if this turn is very different from previous
                    # (different duration pattern suggests different speaker)
                    duration_ratio = (
                        turn["total_duration_ms"] / max(prev_duration, 1)
                    )
                    inter_turn_gap = (
                        turn["start_ms"] - turns[i-1]["start_ms"]
                        - turns[i-1]["total_duration_ms"]
                    )

                    if inter_turn_gap > 2000 or duration_ratio > 2.0 or duration_ratio < 0.5:
                        # Likely a different speaker
                        speaker_idx = (speaker_idx + 1) % max_speakers
                    # else: same speaker continues

                speaker = f"Speaker_{speaker_idx}"
                for idx in turn["segment_indices"]:
                    segments[idx]["speaker"] = speaker

                prev_duration = turn["total_duration_ms"]

        detected = len(set(seg.get("speaker", "") for seg in segments))
        logger.info(
            f"Multi-speaker simple diarization: {detected} speakers detected "
            f"(max={max_speakers}, {len(turns)} turns)"
        )

        return segments

    def _diarize_pyannote(self, audio_path: str, segments: list[dict]) -> list[dict]:
        """
        Full speaker diarization using pyannote.audio.
        Requires HuggingFace token with pyannote model access.

        Enable with: USE_PYANNOTE=true
        """
        try:
            if self._diarization_pipeline is None:
                from pyannote.audio import Pipeline

                hf_token = os.getenv("HF_TOKEN", "")
                if not hf_token:
                    logger.warning("HF_TOKEN not set, falling back to simple diarization")
                    return self._diarize_simple(segments)

                logger.info("Loading pyannote diarization pipeline...")
                self._diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=hf_token,
                )
                logger.info("Pyannote pipeline loaded.")

            # Run diarization
            diarization = self._diarization_pipeline(audio_path)

            # Create speaker timeline
            speaker_timeline = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_timeline.append({
                    "speaker": speaker,
                    "start_ms": int(turn.start * 1000),
                    "end_ms": int(turn.end * 1000),
                })

            # Assign speakers to transcript segments by overlap
            for seg in segments:
                seg_mid = (seg["start_ms"] + seg["end_ms"]) / 2
                best_speaker = "Speaker_0"
                best_overlap = 0

                for turn in speaker_timeline:
                    overlap_start = max(seg["start_ms"], turn["start_ms"])
                    overlap_end = min(seg["end_ms"], turn["end_ms"])
                    overlap = max(0, overlap_end - overlap_start)

                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_speaker = turn["speaker"]

                seg["speaker"] = best_speaker

            speakers = set(seg["speaker"] for seg in segments)
            logger.info(f"Pyannote diarization: {len(speakers)} speakers detected")

            return segments

        except Exception as e:
            logger.error(f"Pyannote diarization failed: {e}. Falling back to simple.")
            return self._diarize_simple(segments)
