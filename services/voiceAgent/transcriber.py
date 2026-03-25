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
import torchaudio as _ta
import torch
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger("nexus.voice.transcriber")
load_dotenv(Path(__file__).parent.parent.parent / ".env")

# ── Configuration ──
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium")
USE_PYANNOTE = os.getenv("USE_PYANNOTE", "false").lower() == "true"

# External Whisper API (GPU-accelerated)
EXTERNAL_WHISPER_URL = os.getenv("EXTERNAL_WHISPER_URL", "")
EXTERNAL_API_KEY = os.getenv("EXTERNAL_API_KEY", "")
EXTERNAL_WHISPER_MODEL = os.getenv("EXTERNAL_WHISPER_MODEL", "base")

# ── Pyannote Community-1 Pipeline (global singleton — loaded once) ──
_pyannote_community_pipeline = None
_pyannote_community_load_attempted = False


def get_pyannote_community_pipeline():
    """
    Load pyannote community-1 pipeline. Returns None if unavailable.
    Called once — result is cached globally. Requires HF_TOKEN.
    """
    global _pyannote_community_pipeline, _pyannote_community_load_attempted

    if _pyannote_community_load_attempted:
        return _pyannote_community_pipeline

    _pyannote_community_load_attempted = True

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        logger.warning(
            "No HuggingFace token found (HF_TOKEN or HUGGINGFACE_TOKEN). "
            "Pyannote community-1 disabled. Falling back to ECAPA-TDNN."
        )
        return None

    try:
        from pyannote.audio import Pipeline

        logger.info("Loading pyannote community-1 pipeline...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=hf_token,
        )

        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
            logger.info("Pyannote community-1 loaded on GPU")
        else:
            logger.info("Pyannote community-1 loaded on CPU")

        _pyannote_community_pipeline = pipeline
        return pipeline

    except Exception as e:
        logger.error(f"Failed to load pyannote community-1: {e}")
        logger.info("Falling back to ECAPA-TDNN diarization")
        return None


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
        self._embedding_model = None  # ECAPA-TDNN speaker embedding model
        self._external_client = None
        self._use_external = False
        self._num_speakers = None  # Speaker count hint (2-10), None = auto
        self._audio_data = None    # Pre-loaded (y, sr) tuple, avoids redundant disk reads
        self._last_diarization_backend = "uninitialized"

        # Try to initialise external Whisper client (API key is optional)
        if EXTERNAL_WHISPER_URL:
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

    def transcribe(self, audio_path: str, num_speakers: Optional[int] = None, audio_data: tuple = None, meeting_type: str = "sales_call") -> dict:
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
        self._audio_data = audio_data
        self._meeting_type = meeting_type
        try:
            if self._use_external:
                return self._transcribe_external(audio_path)
            else:
                return self._transcribe_local(audio_path)
        finally:
            # Clear per-request state to prevent leaking into next request
            self._num_speakers = None
            self._audio_data = None
            self._meeting_type = None

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
                stream=True,
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

        # Strip Whisper hallucination loops
        segments = self._strip_hallucinations(segments)

        logger.info(
            f"External transcription complete: {duration:.1f}s audio, "
            f"{len(segments)} segments, model={model_used}, "
            f"server_time={proc_time:.2f}s"
        )

        # Apply speaker diarization (cascade handles pyannote/embedding/kmeans)
        segments = self._diarize_simple(segments, self._num_speakers, audio_path)

        return {
            "duration_seconds": duration,
            "backend": "external",
            "model": model_used,
            "language": result.get("language", "en"),
            "language_probability": result.get("language_probability", 0),
            "processing_time": proc_time,
            "segments": segments,
            "diarization_backend": self._last_diarization_backend,
            "diarization_confidence": self._compute_diarization_confidence(segments),
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

        # Strip Whisper hallucination loops
        segments = self._strip_hallucinations(segments)

        logger.info(f"Local transcription complete: {duration:.1f}s, {len(segments)} segments")

        # Apply speaker diarization (cascade handles pyannote/embedding/kmeans)
        segments = self._diarize_simple(segments, self._num_speakers, audio_path)

        return {
            "duration_seconds": duration,
            "backend": "local",
            "model": WHISPER_MODEL,
            "segments": segments,
            "diarization_backend": self._last_diarization_backend,
            "diarization_confidence": self._compute_diarization_confidence(segments),
        }

    # ═══════════════════════════════════════════════════════════
    # HALLUCINATION FILTER
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _strip_hallucinations(segments: list[dict], max_repeat: int = 5) -> list[dict]:
        """
        Remove Whisper hallucination loops while preserving real speech.

        Whisper hallucinates repetitive short tokens ("yeah", "you",
        "thank you", punctuation) when audio is silent, noisy, or has
        music.  Detected by: N+ consecutive segments with identical
        short text (after lowering + stripping punctuation).

        Unlike the old tail-truncation approach, this scans the ENTIRE
        transcript and marks only the hallucinated runs for removal,
        so real speech on both sides of a hallucination gap is kept.

        Args:
            segments: transcript segments
            max_repeat: how many identical consecutive segments before
                        we consider it a hallucination loop (default 5)
        Returns:
            Cleaned segment list with hallucination runs excised.
        """
        if len(segments) < max_repeat + 1:
            return segments

        def _norm(text: str) -> str:
            return text.lower().strip(" .,!?;:'\"-")

        # Pass 1: find all runs of identical short text
        hallucinated = set()  # indices to remove
        i = 0
        while i < len(segments):
            # Look ahead for a run of identical short segments
            norm_i = _norm(segments[i].get("text", ""))
            if not norm_i or len(norm_i.split()) > 3:
                i += 1
                continue

            # Count how many consecutive segments share this text
            run_end = i + 1
            while run_end < len(segments):
                if _norm(segments[run_end].get("text", "")) == norm_i:
                    run_end += 1
                else:
                    break

            run_length = run_end - i
            if run_length >= max_repeat:
                # Mark the entire run as hallucinated
                for idx in range(i, run_end):
                    hallucinated.add(idx)
                i = run_end
            else:
                i += 1

        if hallucinated:
            kept = [s for idx, s in enumerate(segments) if idx not in hallucinated]
            logger.warning(
                f"Whisper hallucination detected: removed {len(hallucinated)} "
                f"repetitive segments (kept {len(kept)} of {len(segments)})"
            )
            return kept

        return segments

    # ═══════════════════════════════════════════════════════════
    # SPEAKER DIARIZATION
    # ═══════════════════════════════════════════════════════════

    # Diarization mode from environment
    DIARIZATION_MODE = os.getenv("DIARIZATION_MODE", "auto")  # auto|pyannote|embedding|kmeans

    # Speaker count defaults and ranges per meeting type
    # turn_gap_ms: gap threshold for that conversation style
    SPEAKER_DEFAULTS = {
        "sales_call":            {"default": 2, "min": 2, "max": 3, "turn_gap_ms": 400},
        "interview":             {"default": 2, "min": 2, "max": 4, "turn_gap_ms": 600},
        "internal":              {"default": 4, "min": 2, "max": 8, "turn_gap_ms": 800},
        "client_meeting":        {"default": 3, "min": 2, "max": 8, "turn_gap_ms": 600},
        "meeting":               {"default": 4, "min": 2, "max": 8, "turn_gap_ms": 800},
        "podcast":               {"default": 2, "min": 2, "max": 4, "turn_gap_ms": 600},
        "lecture":               {"default": 1, "min": 1, "max": 2, "turn_gap_ms": 1000},
        "presentation":          {"default": 1, "min": 1, "max": 3, "turn_gap_ms": 1000},
        "debate":                {"default": 2, "min": 2, "max": 4, "turn_gap_ms": 400},
        "casual_conversation":   {"default": 2, "min": 2, "max": 4, "turn_gap_ms": 400},
    }

    # ── Speaker Embedding Model (ECAPA-TDNN) ──

    def _load_embedding_model(self):
        """Lazy-load SpeechBrain ECAPA-TDNN speaker encoder (192-dim embeddings).

        MUST install from develop branch: pip install git+https://github.com/speechbrain/speechbrain.git@develop
        PyPI release has broken hf_hub compatibility. See Rule 2 in SpeechBrain guide.
        """
        if self._embedding_model is not None:
            return self._embedding_model

        try:
            from speechbrain.inference.speaker import EncoderClassifier

            savedir = os.getenv(
                "SPEECHBRAIN_CACHE_DIR",
                str(Path(__file__).parent / "pretrained_models" / "spkrec-ecapa-voxceleb"),
            )
            logger.info("Loading SpeechBrain ECAPA-TDNN speaker embedding model...")
            self._embedding_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=savedir,
                run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            )
            logger.info("SpeechBrain ECAPA-TDNN model loaded.")
            return self._embedding_model
        except ImportError:
            logger.warning(
                "speechbrain not installed — embedding diarization unavailable. "
                "Install with: pip install git+https://github.com/speechbrain/speechbrain.git@develop"
            )
            return None
        except Exception as e:
            logger.warning(f"Failed to load ECAPA-TDNN model: {e}")
            return None

    EMBEDDING_DIM = 192  # ECAPA-TDNN produces 192-dim speaker embeddings

    def _compute_speaker_embeddings(
        self,
        segments: list[dict],
        y: np.ndarray,
        sr: int,
    ) -> Optional[np.ndarray]:
        """
        Compute ECAPA-TDNN speaker embedding (192-dim) for each segment.

        Desplanques et al. 2020: ECAPA-TDNN achieves ~0.8% EER on VoxCeleb1.
        Audio must be 16kHz mono. Segments shorter than 0.5s get a zero vector.

        Returns (N_segments, 192) numpy array, or None if unavailable.
        """
        model = self._load_embedding_model()
        if model is None:
            return None

        total_samples = len(y)
        embeddings = []
        valid_mask = []
        MIN_DURATION_S = 0.5

        for seg in segments:
            start_sample = int(seg["start_ms"] / 1000 * sr)
            end_sample = int(seg["end_ms"] / 1000 * sr)
            start_sample = max(0, min(start_sample, total_samples - 1))
            end_sample = max(start_sample + 1, min(end_sample, total_samples))

            chunk = y[start_sample:end_sample]
            duration_s = len(chunk) / sr

            if duration_s < MIN_DURATION_S:
                embeddings.append(np.zeros(self.EMBEDDING_DIM))
                valid_mask.append(False)
                continue

            try:
                # ECAPA-TDNN expects (batch, samples) tensor at 16kHz
                waveform = torch.from_numpy(chunk).unsqueeze(0).float()
                # Resample if not 16kHz
                if sr != 16000:
                    waveform = _ta.functional.resample(waveform, sr, 16000)
                with torch.no_grad():
                    emb = model.encode_batch(waveform)
                emb_np = emb.squeeze().cpu().numpy()
                embeddings.append(emb_np)
                valid_mask.append(True)
            except Exception as e:
                logger.debug(f"Embedding failed for segment: {e}")
                embeddings.append(np.zeros(self.EMBEDDING_DIM))
                valid_mask.append(False)

        n_valid = sum(valid_mask)
        if n_valid < 2:
            logger.warning(f"Only {n_valid} valid embeddings — need at least 2")
            return None

        logger.info(f"Computed {n_valid}/{len(segments)} ECAPA-TDNN speaker embeddings")
        return np.array(embeddings, dtype=np.float32)

    def _estimate_speaker_count(self, segments: list[dict]) -> int:
        """
        Estimate number of speakers from turn patterns + meeting type.
        """
        if len(segments) <= 1:
            return 1

        meeting_type = getattr(self, "_meeting_type", "sales_call") or "sales_call"
        config = self.SPEAKER_DEFAULTS.get(meeting_type, {"default": 2, "min": 2, "max": 8})
        type_default = config["default"]
        type_min = config["min"]
        type_max = config["max"]

        turn_gap_ms = 1200
        turns = 1
        short_turns = 0
        long_turns = 0
        for i, seg in enumerate(segments):
            duration_ms = max(0, seg["end_ms"] - seg["start_ms"])
            if duration_ms < 2500:
                short_turns += 1
            elif duration_ms > 7000:
                long_turns += 1
            if i == 0:
                continue
            gap_ms = seg["start_ms"] - segments[i - 1]["end_ms"]
            if gap_ms > turn_gap_ms:
                turns += 1

        if turns <= 2 and len(segments) <= 3:
            return max(1, type_min)

        if turns <= 6:
            estimated = type_default
        elif turns <= 15:
            estimated = type_default + (1 if short_turns > long_turns else 0)
        elif turns <= 30:
            estimated = type_default + 1
        else:
            estimated = type_default + 2

        estimated = max(type_min, min(estimated, type_max))

        logger.info(
            f"Speaker estimate: {estimated} (type={meeting_type}, "
            f"turns={turns}, short={short_turns}, long={long_turns}, "
            f"range={type_min}-{type_max})"
        )
        return estimated

    def _compute_diarization_confidence(self, segments: list[dict]) -> float:
        """
        Estimate confidence in diarization quality (0.0 - 0.85).

        Factors:
          - Backend quality (pyannote > embedding > kmeans > gap_pitch > heuristic)
          - Speaker balance (more balanced = higher confidence)
          - Turn count (more alternation evidence = higher confidence)

        Max 0.85 per NEXUS confidence ceiling principle.
        """
        from collections import Counter

        backend = self._last_diarization_backend

        # Base confidence from backend quality
        backend_base = {
            "pyannote": 0.80,
            "embedding": 0.70,
            "kmeans": 0.50,
            "gap_pitch": 0.30,
            "heuristic_multi": 0.20,
            "single_speaker": 0.60,
            "uninitialized": 0.0,
        }
        confidence = backend_base.get(backend, 0.25)

        if not segments or len(segments) < 2:
            return min(confidence, 0.85)

        # Speaker balance bonus (0 to +0.10)
        speaker_counts = Counter(seg.get("speaker", "Speaker_0") for seg in segments)
        n_speakers = len(speaker_counts)
        if n_speakers >= 2:
            total = sum(speaker_counts.values())
            max_pct = max(speaker_counts.values()) / total
            # Perfect balance (50/50) → +0.10, 90/10 → +0.01
            balance_score = 1.0 - max_pct
            confidence += balance_score * 0.15

        # Turn alternation bonus (0 to +0.05)
        turns = 1
        for i in range(1, len(segments)):
            if segments[i].get("speaker") != segments[i - 1].get("speaker"):
                turns += 1
        # More turns relative to segments = better evidence
        turn_ratio = turns / len(segments) if len(segments) > 0 else 0
        confidence += turn_ratio * 0.05

        return min(round(confidence, 3), 0.85)

    def _diarize_simple(
        self,
        segments: list[dict],
        num_speakers: Optional[int] = None,
        audio_path: Optional[str] = None,
    ) -> list[dict]:
        """
        Speaker diarization cascade.

        Priority order:
          1. Pyannote Community-1 (frame-level, neural) — DEFAULT when HF_TOKEN set
          2. ECAPA-TDNN embedding clustering — fallback
          3. Acoustic KMeans (MFCCs + pitch + energy)
          4. Gap + pitch heuristic (2-speaker fallback)
          5. Heuristic multi-speaker (round-robin, last resort)

        DIARIZATION_MODE env var overrides: auto|pyannote|ecapa|kmeans
        """
        if not segments:
            return segments

        if num_speakers is None:
            estimated = self._estimate_speaker_count(segments)
            logger.info("No speaker-count hint provided; estimated %s speaker(s)", estimated)
            max_speakers = estimated
        else:
            max_speakers = min(max(1, num_speakers), 10)

        # Single speaker — assign all segments and skip clustering
        if max_speakers < 2:
            for seg in segments:
                seg["speaker"] = "Speaker_0"
            self._last_diarization_backend = "single_speaker"
            logger.info("Single-speaker mode: all segments assigned to Speaker_0")
            return segments

        mode = self.DIARIZATION_MODE

        # ── Tier 1: Pyannote Community-1 (frame-level, default) ──
        if mode in ("pyannote", "auto") and audio_path:
            try:
                result = self._diarize_pyannote_community(
                    segments, max_speakers, audio_path
                )
                if result is not None:
                    return result
            except Exception as e:
                logger.warning(f"Pyannote community-1 failed: {e}")

        # ── Tier 1b: Legacy Pyannote 3.1 (if USE_PYANNOTE=true) ──
        if mode in ("pyannote", "auto") and USE_PYANNOTE and audio_path:
            try:
                result = self._diarize_pyannote(audio_path, segments)
                if result is not None:
                    return result
            except Exception as e:
                logger.warning(f"Pyannote 3.1 failed: {e}")

        # ── Tier 2: Speaker embedding clustering (ECAPA-TDNN) ──
        if mode in ("ecapa", "embedding", "auto") and audio_path:
            try:
                result = self._diarize_embedding_clustering(
                    segments, max_speakers, audio_path
                )
                if result is not None:
                    return result
            except Exception as e:
                logger.warning(f"Embedding clustering failed: {e}")

        # ── Tier 3: Acoustic KMeans (enhanced with MFCCs) ──
        if mode in ("kmeans", "auto") and audio_path and max_speakers >= 2:
            try:
                clustered = self._diarize_acoustic_kmeans(
                    segments, max_speakers, audio_path
                )
                if clustered is not None:
                    return clustered
            except Exception as e:
                logger.warning(f"Acoustic KMeans diarization failed: {e}")

        # ── Tier 4/5: Gap+pitch or heuristic ──
        if max_speakers == 2:
            return self._diarize_gap_two_speaker(segments, audio_path)
        return self._diarize_simple_multi_speaker(segments, max_speakers)

    # ── Pyannote Community-1 Diarization (Tier 1) ──

    def _diarize_pyannote_community(
        self,
        segments: list[dict],
        num_speakers: int,
        audio_path: str,
    ) -> Optional[list[dict]]:
        """
        Frame-level speaker diarization using pyannote community-1.

        Operates at ~16ms frame resolution — detects speaker changes even with
        <300ms gaps. Solves the root cause: Whisper segments by silence, not
        by speaker identity.

        Uses exclusive_speaker_diarization for clean 1-speaker-at-a-time output.
        """
        from collections import Counter

        pipeline = get_pyannote_community_pipeline()
        if pipeline is None:
            return None

        logger.info(
            f"Running pyannote community-1 diarization: "
            f"{num_speakers} speakers, {len(segments)} segments"
        )

        # Build audio input — pass pre-loaded waveform to avoid torchcodec issues
        if self._audio_data is not None:
            y, sr = self._audio_data
            waveform = torch.from_numpy(y).unsqueeze(0).float()
            audio_input = {"waveform": waveform, "sample_rate": sr}
        else:
            audio_input = audio_path

        # Run diarization with explicit speaker count
        diarize_kwargs = {"num_speakers": min(max(1, num_speakers), 10)}

        diarization = pipeline(audio_input, **diarize_kwargs)

        # Extract speaker turns
        speaker_turns = []
        # Try exclusive_speaker_diarization first (community-1 feature)
        if hasattr(diarization, "exclusive_speaker_diarization"):
            for turn, speaker in diarization.exclusive_speaker_diarization:
                speaker_turns.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker,
                })
        else:
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_turns.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker,
                })

        if not speaker_turns:
            logger.warning("Pyannote returned no speaker turns")
            return None

        logger.info(f"Pyannote produced {len(speaker_turns)} speaker turns")

        # ── Assign speakers to Whisper segments ──
        has_words = any(seg.get("words") for seg in segments)

        for seg in segments:
            if has_words and seg.get("words"):
                # Word-level assignment (highest precision)
                word_speakers = []
                for word in seg["words"]:
                    w_start = word.get("start", seg["start_ms"] / 1000)
                    w_end = word.get("end", seg["end_ms"] / 1000)
                    spk = self._get_dominant_speaker(w_start, w_end, speaker_turns)
                    word_speakers.append(spk)

                # Segment speaker = majority among its words
                valid_speakers = [s for s in word_speakers if s is not None]
                if valid_speakers:
                    seg["speaker"] = Counter(valid_speakers).most_common(1)[0][0]
                else:
                    seg["speaker"] = self._get_dominant_speaker(
                        seg["start_ms"] / 1000, seg["end_ms"] / 1000, speaker_turns
                    ) or "SPEAKER_00"
            else:
                # Segment-level assignment via overlap
                seg["speaker"] = self._get_dominant_speaker(
                    seg["start_ms"] / 1000, seg["end_ms"] / 1000, speaker_turns
                ) or "SPEAKER_00"

        # ── Normalize labels to Speaker_0, Speaker_1, ... ──
        label_map = {}
        counter = 0
        for seg in segments:
            raw = seg.get("speaker", "")
            if raw not in label_map:
                label_map[raw] = f"Speaker_{counter}"
                counter += 1
            seg["speaker"] = label_map[raw]

        # ── Linguistic post-correction ──
        segments = self._linguistic_post_correction(segments, num_speakers)

        # ── Balance check ──
        counts = Counter(seg["speaker"] for seg in segments)
        total = len(segments)
        max_pct = max(counts.values()) / total if total > 0 else 1.0
        if max_pct > 0.95:
            logger.warning(
                f"Pyannote produced imbalanced split: {dict(counts)} "
                f"({max_pct:.0%}). Falling through."
            )
            return None

        self._last_diarization_backend = "pyannote_community"
        logger.info(
            f"Pyannote community-1 diarization complete: "
            f"{len(counts)} speakers — "
            + ", ".join(f"{k}: {v} segs" for k, v in sorted(counts.items()))
        )
        return segments

    @staticmethod
    def _get_dominant_speaker(
        start: float,
        end: float,
        speaker_turns: list[dict],
    ) -> Optional[str]:
        """Find the speaker with the most overlap in the given time range."""
        overlap_by_speaker: dict[str, float] = {}
        for turn in speaker_turns:
            overlap = max(0.0, min(end, turn["end"]) - max(start, turn["start"]))
            if overlap > 0:
                sp = turn["speaker"]
                overlap_by_speaker[sp] = overlap_by_speaker.get(sp, 0) + overlap

        if not overlap_by_speaker:
            return None
        return max(overlap_by_speaker, key=overlap_by_speaker.get)

    # ── Embedding-Based Diarization (Tier 2) ──

    def _diarize_embedding_clustering(
        self,
        segments: list[dict],
        num_speakers: int,
        audio_path: str,
    ) -> Optional[list[dict]]:
        """
        Speaker diarization using ECAPA-TDNN speaker embeddings + agglomerative
        clustering with cosine affinity.

        Desplanques et al. 2020: ECAPA-TDNN produces 192-dim vectors that encode
        speaker identity regardless of what is being said.
        """
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics.pairwise import cosine_distances

        logger.info(
            f"Running embedding diarization: {num_speakers} speakers, "
            f"{len(segments)} segments"
        )

        # Load audio
        if self._audio_data is not None:
            y, sr = self._audio_data
        else:
            import librosa
            try:
                from shared.utils.audio_loader import load_audio
                y, sr = load_audio(audio_path, sr=16000)
            except ImportError:
                y, sr = librosa.load(audio_path, sr=16000, mono=True)

        # Compute embeddings
        embeddings = self._compute_speaker_embeddings(segments, y, sr)
        if embeddings is None:
            return None

        # Build valid mask (non-zero embeddings)
        valid_mask = [np.linalg.norm(e) > 0.01 for e in embeddings]
        n_valid = sum(valid_mask)
        if n_valid < num_speakers:
            logger.warning(
                f"Only {n_valid} valid embeddings, need {num_speakers}. Falling back."
            )
            return None

        # Agglomerative clustering with cosine distance
        dist_matrix = cosine_distances(embeddings)
        clustering = AgglomerativeClustering(
            n_clusters=num_speakers,
            metric="precomputed",
            linkage="average",
        )
        labels = clustering.fit_predict(dist_matrix)

        # ── Temporal smoothing: reassign isolated micro-fragments ──
        labels = labels.tolist()
        for i in range(1, len(labels) - 1):
            prev_label, next_label = labels[i - 1], labels[i + 1]
            if prev_label == next_label and labels[i] != prev_label:
                seg_duration_ms = segments[i]["end_ms"] - segments[i]["start_ms"]
                gap_before = segments[i]["start_ms"] - segments[i - 1]["end_ms"]
                gap_after = segments[i + 1]["start_ms"] - segments[i]["end_ms"]
                if seg_duration_ms < 1500 and gap_before < 200 and gap_after < 200:
                    labels[i] = prev_label

        # ── Balance check (relaxed for embeddings: 1/N + 0.45) ──
        from collections import Counter
        counts = Counter(labels)
        total = len(labels)
        max_pct = max(counts.values()) / total if total > 0 else 1.0
        max_allowed_pct = (1.0 / num_speakers) + 0.45
        if max_pct > max_allowed_pct:
            logger.warning(
                f"Embedding clustering imbalanced: {dict(counts)} "
                f"({max_pct:.0%} in one cluster, limit={max_allowed_pct:.0%}). "
                f"Falling back."
            )
            return None

        # ── Assign labels by first-appearance order ──
        label_order = []
        for lbl in labels:
            if lbl not in label_order:
                label_order.append(lbl)
        cluster_to_speaker = {
            lbl: f"Speaker_{rank}" for rank, lbl in enumerate(label_order)
        }

        for i, seg in enumerate(segments):
            seg["speaker"] = cluster_to_speaker[labels[i]]

        # ── Assign invalid segments from nearest valid neighbor ──
        for i, (seg, valid) in enumerate(zip(segments, valid_mask)):
            if not valid:
                # Find nearest valid neighbor by time
                best_dist = float("inf")
                best_speaker = "Speaker_0"
                seg_mid = (seg["start_ms"] + seg["end_ms"]) / 2
                for j, (other_seg, other_valid) in enumerate(zip(segments, valid_mask)):
                    if other_valid and j != i:
                        other_mid = (other_seg["start_ms"] + other_seg["end_ms"]) / 2
                        dist = abs(seg_mid - other_mid)
                        if dist < best_dist:
                            best_dist = dist
                            best_speaker = other_seg["speaker"]
                seg["speaker"] = best_speaker

        # ── Linguistic post-correction ──
        segments = self._linguistic_post_correction(segments, num_speakers)

        # Log results
        speaker_counts = Counter(seg["speaker"] for seg in segments)
        self._last_diarization_backend = "embedding"
        logger.info(
            f"Embedding diarization complete: "
            f"{len(speaker_counts)} speakers — "
            + ", ".join(f"{k}: {v} segs" for k, v in sorted(speaker_counts.items()))
        )
        return segments

    def _linguistic_post_correction(
        self,
        segments: list[dict],
        num_speakers: int = 2,
    ) -> list[dict]:
        """
        Fix obvious diarization errors using linguistic patterns (conservative).

        Rule 1: Question → short answer: if a short segment ends with '?' and the
        next segment is short (<2s), same speaker, and separated by >200ms gap,
        flip the response to the other speaker.

        Rule 2: Isolated flip: if a single short (<1.5s) segment breaks an
        otherwise consistent speaker run on both sides, with tiny gaps (<150ms),
        re-assign it to match the surrounding speaker.
        """
        if num_speakers != 2 or len(segments) < 3:
            return segments

        speakers = set(seg.get("speaker", "") for seg in segments)
        if len(speakers) < 2:
            return segments
        speaker_list = sorted(speakers)

        def other_speaker(spk: str) -> str:
            return speaker_list[1] if spk == speaker_list[0] else speaker_list[0]

        corrections = 0

        # Rule 1: Question → response pattern (conservative)
        for i in range(len(segments) - 1):
            curr = segments[i]
            nxt = segments[i + 1]
            curr_text = curr.get("text", "").strip()
            nxt_duration = nxt["end_ms"] - nxt["start_ms"]
            curr_duration = curr["end_ms"] - curr["start_ms"]
            gap_ms = nxt["start_ms"] - curr["end_ms"]

            if (
                curr_text.endswith("?")
                and curr.get("speaker") == nxt.get("speaker")
                and nxt_duration < 2000
                and curr_duration < 3000
                and gap_ms > 200
            ):
                nxt["speaker"] = other_speaker(curr["speaker"])
                corrections += 1

        # Rule 2: Isolated single-segment flip
        for i in range(1, len(segments) - 1):
            prev_spk = segments[i - 1].get("speaker")
            curr_spk = segments[i].get("speaker")
            next_spk = segments[i + 1].get("speaker")
            seg_duration = segments[i]["end_ms"] - segments[i]["start_ms"]

            if (
                prev_spk == next_spk
                and curr_spk != prev_spk
                and seg_duration < 1500
            ):
                gap_before = segments[i]["start_ms"] - segments[i - 1]["end_ms"]
                gap_after = segments[i + 1]["start_ms"] - segments[i]["end_ms"]
                if gap_before < 150 and gap_after < 150:
                    segments[i]["speaker"] = prev_spk
                    corrections += 1

        # Rule 3: Greeting → query pattern
        # If segment N is a greeting/intro and segment N+1 is a short question
        # from the SAME speaker, flip N+1 (the response is from the other person).
        for i in range(len(segments) - 1):
            curr = segments[i]
            nxt = segments[i + 1]
            if curr.get("speaker") != nxt.get("speaker"):
                continue

            curr_text = curr.get("text", "").strip().lower()
            nxt_text = nxt.get("text", "").strip()
            nxt_duration = nxt["end_ms"] - nxt["start_ms"]
            gap_ms = nxt["start_ms"] - curr["end_ms"]

            greeting_patterns = [
                "good morning", "good afternoon", "good evening",
                "hi,", "hello,", "hey,", "this is ", "my name is ",
            ]
            is_greeting = any(curr_text.startswith(p) for p in greeting_patterns)
            is_short_question = (
                nxt_text.endswith("?")
                and nxt_duration < 2000
                and len(nxt_text.split()) <= 6
            )

            if is_greeting and is_short_question and gap_ms > 100:
                nxt["speaker"] = other_speaker(curr["speaker"])
                corrections += 1

        if corrections > 0:
            logger.info(f"Linguistic post-correction: {corrections} fixes applied")

        return segments

    # ── Acoustic KMeans Diarization (Tier 3) ──

    def _diarize_acoustic_kmeans(
        self,
        segments: list[dict],
        num_speakers: int,
        audio_path: str,
    ) -> Optional[list[dict]]:
        """
        Cluster transcript segments into speakers using MFCCs + pitch + energy.

        Enhanced feature set (16D per segment):
          - 13 mean MFCCs (vocal tract shape — Davis & Mermelstein 1980)
          - mean F0 pitch + pitch std (speaker-specific F0 range)
          - mean RMS energy (speaking style)
        """
        import librosa
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        logger.info(
            f"Running acoustic KMeans diarization: {num_speakers} speakers, "
            f"{len(segments)} segments"
        )

        # Use pre-loaded audio if available, otherwise load from disk
        if self._audio_data is not None:
            y, sr = self._audio_data
        else:
            try:
                from shared.utils.audio_loader import load_audio
                y, sr = load_audio(audio_path, sr=16000)
            except ImportError:
                y, sr = librosa.load(audio_path, sr=16000, mono=True)
            except ValueError:
                logger.warning("Could not decode audio — skipping KMeans diarization")
                return None
        total_samples = len(y)

        N_MFCC = 13
        N_FEATURES = N_MFCC + 3  # 13 MFCCs + pitch + pitch_std + energy
        features = []
        valid_mask = []

        for seg in segments:
            start_sample = int(seg["start_ms"] / 1000 * sr)
            end_sample = int(seg["end_ms"] / 1000 * sr)
            start_sample = max(0, min(start_sample, total_samples - 1))
            end_sample = max(start_sample + 1, min(end_sample, total_samples))

            chunk = y[start_sample:end_sample]

            if len(chunk) < sr * 0.1:  # Skip very short chunks (< 100ms)
                features.append([0.0] * N_FEATURES)
                valid_mask.append(False)
                continue

            # 13 MFCCs — vocal tract shape (most discriminative non-neural feature)
            try:
                mfccs = librosa.feature.mfcc(
                    y=chunk, sr=sr, n_mfcc=N_MFCC, n_fft=2048, hop_length=512
                )
                mean_mfccs = np.mean(mfccs, axis=1).tolist()  # (13,)
            except Exception:
                mean_mfccs = [0.0] * N_MFCC

            # Mean pitch + pitch std via pyin
            try:
                f0, voiced_flag, _ = librosa.pyin(
                    chunk, fmin=60, fmax=500, sr=sr,
                    frame_length=2048, hop_length=512,
                )
                voiced_f0 = f0[voiced_flag] if voiced_flag is not None else f0[~np.isnan(f0)]
                mean_pitch = float(np.nanmean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
                pitch_std = float(np.nanstd(voiced_f0)) if len(voiced_f0) > 1 else 0.0
            except Exception:
                mean_pitch = 0.0
                pitch_std = 0.0

            # Mean RMS energy
            rms = librosa.feature.rms(y=chunk, frame_length=2048, hop_length=512)[0]
            mean_energy = float(np.mean(rms)) if len(rms) > 0 else 0.0

            features.append(mean_mfccs + [mean_pitch, pitch_std, mean_energy])
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

        # StandardScaler (robust across different feature ranges)
        valid_indices = [i for i, v in enumerate(valid_mask) if v]
        scaler = StandardScaler()
        scaler.fit(features_arr[valid_indices])
        features_arr = scaler.transform(features_arr)

        # Zero out invalid segments
        for i, valid in enumerate(valid_mask):
            if not valid:
                features_arr[i, :] = 0.0

        # KMeans clustering
        kmeans = KMeans(
            n_clusters=num_speakers,
            n_init=10,
            random_state=42,
        )
        labels = kmeans.fit_predict(features_arr)

        # ── Temporal smoothing: reassign isolated micro-fragments ──
        labels = labels.tolist()
        for i in range(1, len(labels) - 1):
            prev_label = labels[i - 1]
            next_label = labels[i + 1]
            if prev_label == next_label and labels[i] != prev_label:
                seg_duration_ms = segments[i]["end_ms"] - segments[i]["start_ms"]
                gap_before = segments[i]["start_ms"] - segments[i - 1]["end_ms"]
                gap_after = segments[i + 1]["start_ms"] - segments[i]["end_ms"]
                if seg_duration_ms < 1500 and gap_before < 200 and gap_after < 200:
                    labels[i] = prev_label
        labels = np.array(labels)

        # ── Balance check: relaxed for short audio (<2 min) ──
        from collections import Counter
        counts = Counter(labels.tolist())
        total = len(labels)
        max_pct = max(counts.values()) / total if total > 0 else 1.0
        duration_s = (segments[-1]["end_ms"] - segments[0]["start_ms"]) / 1000 if segments else 0
        # Short audio genuinely can be dominated by one speaker (cold call pitch)
        balance_slack = 0.50 if duration_s < 120 else 0.40
        max_allowed_pct = (1.0 / num_speakers) + balance_slack
        if max_pct > max_allowed_pct:
            logger.warning(
                f"KMeans produced imbalanced split: {dict(counts)} "
                f"({max_pct:.0%} in one cluster, limit={max_allowed_pct:.0%}). "
                f"Falling to gap+pitch."
            )
            return None

        # Assign speaker labels; order clusters by mean pitch (lower pitch = Speaker_0)
        cluster_pitches = {}
        # Pitch is at index N_MFCC (after MFCCs)
        for cluster_id in range(num_speakers):
            mask = labels == cluster_id
            cluster_pitches[cluster_id] = float(np.mean(features_arr[mask, N_MFCC]))

        sorted_clusters = sorted(cluster_pitches, key=lambda c: cluster_pitches[c])
        cluster_to_speaker = {
            cluster_id: f"Speaker_{rank}"
            for rank, cluster_id in enumerate(sorted_clusters)
        }

        for i, seg in enumerate(segments):
            seg["speaker"] = cluster_to_speaker[labels[i]]

        # Log results
        speaker_counts = Counter(seg["speaker"] for seg in segments)
        self._last_diarization_backend = "kmeans"
        logger.info(
            f"Acoustic KMeans diarization complete: "
            f"{len(speaker_counts)} speakers — "
            + ", ".join(f"{k}: {v} segs" for k, v in sorted(speaker_counts.items()))
        )

        return segments

    def _diarize_gap_two_speaker(
        self,
        segments: list[dict],
        audio_path: Optional[str] = None,
    ) -> list[dict]:
        """
        Enhanced fallback: 2-speaker alternation based on gaps + pitch + MFCC.

        Speaker changes are detected when ANY of:
          - Silence gap > meeting-type threshold (400ms for sales_call, 800ms default)
          - Mean F0 pitch jump > 15Hz between adjacent segments
          - MFCC cosine distance > 0.3 between adjacent segments (timbre change)

        Correction pass: if same speaker talks >15s straight, force-check for
        missed turn boundaries.
        """
        import librosa
        from scipy.spatial.distance import cosine as cosine_dist

        current_speaker = "Speaker_0"

        # Meeting-type-aware thresholds
        meeting_type = getattr(self, "_meeting_type", "sales_call") or "sales_call"
        config = self.SPEAKER_DEFAULTS.get(meeting_type, {})
        TURN_GAP_MS = config.get("turn_gap_ms", 800)
        PITCH_JUMP_HZ = 15.0  # Lowered from 25Hz for phone-quality audio
        MFCC_DISTANCE_THRESHOLD = 0.3

        # Compute per-segment features from audio
        seg_pitches: list[Optional[float]] = [None] * len(segments)
        seg_mfccs: list[Optional[np.ndarray]] = [None] * len(segments)

        y, sr = None, 16000
        if self._audio_data is not None:
            y, sr = self._audio_data
        elif audio_path:
            try:
                from shared.utils.audio_loader import load_audio
                y, sr = load_audio(audio_path, sr=16000)
            except ImportError:
                try:
                    y, sr = librosa.load(audio_path, sr=16000, mono=True)
                except Exception:
                    pass
            except Exception:
                pass

        if y is not None:
            total_samples = len(y)
            for idx, seg in enumerate(segments):
                start_sample = int(seg["start_ms"] / 1000 * sr)
                end_sample = int(seg["end_ms"] / 1000 * sr)
                start_sample = max(0, min(start_sample, total_samples - 1))
                end_sample = max(start_sample + 1, min(end_sample, total_samples))
                chunk = y[start_sample:end_sample]

                if len(chunk) < sr * 0.2:
                    continue
                try:
                    f0, voiced, _ = librosa.pyin(
                        chunk, fmin=60, fmax=500, sr=sr,
                        frame_length=2048, hop_length=512,
                    )
                    voiced_f0 = f0[voiced] if voiced is not None else f0[~np.isnan(f0)]
                    if len(voiced_f0) > 0:
                        seg_pitches[idx] = float(np.nanmean(voiced_f0))
                except Exception:
                    pass

                # MFCC vector for timbre comparison
                try:
                    mfccs = librosa.feature.mfcc(
                        y=chunk, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512
                    )
                    seg_mfccs[idx] = np.mean(mfccs, axis=1)
                except Exception:
                    pass

        # Detect speaker changes
        for i, seg in enumerate(segments):
            if i > 0:
                gap_ms = seg["start_ms"] - segments[i - 1]["end_ms"]

                # Pitch shift check
                pitch_shift = False
                if seg_pitches[i] is not None and seg_pitches[i - 1] is not None:
                    pitch_shift = abs(seg_pitches[i] - seg_pitches[i - 1]) > PITCH_JUMP_HZ

                # MFCC timbre distance check
                mfcc_change = False
                if seg_mfccs[i] is not None and seg_mfccs[i - 1] is not None:
                    try:
                        dist = cosine_dist(seg_mfccs[i], seg_mfccs[i - 1])
                        mfcc_change = dist > MFCC_DISTANCE_THRESHOLD
                    except Exception:
                        pass

                if gap_ms > TURN_GAP_MS or pitch_shift or mfcc_change:
                    current_speaker = (
                        "Speaker_1" if current_speaker == "Speaker_0"
                        else "Speaker_0"
                    )
            seg["speaker"] = current_speaker

        # ── Correction pass: if same speaker talks >15s straight, look for
        # the largest gap within that run and force a turn boundary there ──
        MAX_MONO_MS = 15000
        i = 0
        while i < len(segments):
            run_start = i
            run_speaker = segments[i]["speaker"]
            while i < len(segments) and segments[i]["speaker"] == run_speaker:
                i += 1
            run_end = i  # exclusive

            run_duration = segments[run_end - 1]["end_ms"] - segments[run_start]["start_ms"]
            if run_duration > MAX_MONO_MS and (run_end - run_start) >= 3:
                # Find the largest gap within this run
                best_gap_idx = run_start + 1
                best_gap = 0
                for j in range(run_start + 1, run_end):
                    gap = segments[j]["start_ms"] - segments[j - 1]["end_ms"]
                    if gap > best_gap:
                        best_gap = gap
                        best_gap_idx = j

                # Force speaker change at the largest gap
                other_speaker = "Speaker_1" if run_speaker == "Speaker_0" else "Speaker_0"
                for j in range(best_gap_idx, run_end):
                    segments[j]["speaker"] = other_speaker

                # Re-scan from the split point
                i = best_gap_idx
                continue

        # Log balance
        from collections import Counter
        counts = Counter(seg["speaker"] for seg in segments)
        pitched = sum(1 for p in seg_pitches if p is not None)
        self._last_diarization_backend = "gap_pitch"
        logger.info(
            f"Gap+pitch diarization: {dict(counts)} "
            f"(gap={TURN_GAP_MS}ms, pitch={PITCH_JUMP_HZ}Hz, "
            f"{len(segments)} segments, {pitched} with pitch data)"
        )

        return segments

    def _diarize_simple_multi_speaker(
        self,
        segments: list[dict],
        max_speakers: int,
    ) -> list[dict]:
        """
        Last-resort multi-speaker heuristic diarization (3-10 speakers).

        Only called when acoustic KMeans has already failed (insufficient
        features or imbalanced clusters). Uses gap + duration heuristics.

        Strategy:
        1. Detect turn changes via gaps (> 1s gap = new turn)
        2. Group consecutive same-speaker segments into turns
        3. Cluster turns by duration pattern and assign speakers
           round-robin up to max_speakers
        """
        TURN_GAP_MS = 1000

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
        self._last_diarization_backend = "heuristic_multi"
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
                if not hasattr(_ta, "list_audio_backends"):
                    _ta.list_audio_backends = lambda: ["soundfile"]

                if not hasattr(_ta, "get_audio_backend"):
                    _ta.get_audio_backend = lambda: "soundfile"

                if not hasattr(_ta, "set_audio_backend"):
                    _ta.set_audio_backend = lambda backend: None
                from pyannote.audio import Pipeline
                from huggingface_hub import login
                import huggingface_hub as _hf_hub

                hf_token = os.getenv("HF_TOKEN", "")
                if not hf_token:
                    logger.warning("HF_TOKEN not set, falling back to simple diarization")
                    return self._diarize_simple(segments, self._num_speakers, audio_path)

                # Store token in HF cache so hf_hub finds it automatically
                login(token=hf_token, add_to_git_credential=False)

                # Monkey-patch all hf_hub functions that pyannote 3.x calls
                # with the deprecated use_auth_token kwarg
                _originals = {}
                _funcs_to_patch = ["hf_hub_download", "model_info", "snapshot_download"]
                for fn_name in _funcs_to_patch:
                    orig_fn = getattr(_hf_hub, fn_name, None)
                    if orig_fn is not None:
                        _originals[fn_name] = orig_fn
                        def _make_patched(orig):
                            def _patched(*args, **kwargs):
                                kwargs.pop("use_auth_token", None)
                                return orig(*args, **kwargs)
                            return _patched
                        setattr(_hf_hub, fn_name, _make_patched(orig_fn))

                logger.info("Loading pyannote diarization pipeline...")
                try:
                    self._diarization_pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1",
                    )
                finally:
                    for fn_name, orig_fn in _originals.items():
                        setattr(_hf_hub, fn_name, orig_fn)
                logger.info("Pyannote pipeline loaded.")

            if torch.cuda.is_available():
                self._diarization_pipeline.to(torch.device("cuda"))

            # Build speaker count hints for pyannote
            diarize_params = {}
            if self._num_speakers is not None:
                diarize_params["num_speakers"] = min(max(1, self._num_speakers), 10)
            else:
                meeting_type = getattr(self, "_meeting_type", "sales_call") or "sales_call"
                config = self.SPEAKER_DEFAULTS.get(meeting_type, {"default": 2, "min": 2, "max": 8})
                diarize_params["min_speakers"] = config["min"]
                diarize_params["max_speakers"] = config["max"]

            # Run diarization — pass waveform tensor if available (pyannote
            # can't read video containers like MP4/WebM directly, and
            # pre-loaded 16kHz mono avoids redundant resampling)
            if self._audio_data is not None:
                y, sr = self._audio_data
                waveform = torch.from_numpy(y).unsqueeze(0).float()  # (1, samples)
                audio_input = {"waveform": waveform, "sample_rate": sr}
            else:
                audio_input = audio_path

            logger.info(f"Running pyannote diarization: {diarize_params}")
            diarization = self._diarization_pipeline(audio_input, **diarize_params)

            # Create speaker timeline
            speaker_timeline = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_timeline.append({
                    "speaker": speaker,
                    "start_ms": int(turn.start * 1000),
                    "end_ms": int(turn.end * 1000),
                })

            # Assign speakers to transcript segments by overlap,
            # falling back to nearest speaker if no overlap found
            for seg in segments:
                best_speaker = None
                best_overlap = 0
                seg_mid = (seg["start_ms"] + seg["end_ms"]) / 2

                for turn in speaker_timeline:
                    overlap_start = max(seg["start_ms"], turn["start_ms"])
                    overlap_end = min(seg["end_ms"], turn["end_ms"])
                    overlap = max(0, overlap_end - overlap_start)

                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_speaker = turn["speaker"]

                # No overlap — find nearest speaker turn by time distance
                if best_speaker is None and speaker_timeline:
                    best_dist = float("inf")
                    for turn in speaker_timeline:
                        turn_mid = (turn["start_ms"] + turn["end_ms"]) / 2
                        dist = abs(seg_mid - turn_mid)
                        if dist < best_dist:
                            best_dist = dist
                            best_speaker = turn["speaker"]

                seg["speaker"] = best_speaker or "Speaker_0"

            speakers = set(seg["speaker"] for seg in segments)
            self._last_diarization_backend = "pyannote"
            logger.info(f"Pyannote diarization: {len(speakers)} speakers detected")

            return segments

        except Exception as e:
            logger.error(f"Pyannote diarization failed: {e}. Falling back to simple.")
            return self._diarize_simple(segments, self._num_speakers, audio_path)
