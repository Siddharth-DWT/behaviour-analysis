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

# External GPU Diarization API (pyannote on GPU)
EXTERNAL_DIARIZE_URL = os.getenv("EXTERNAL_DIARIZE_URL", "")

# Deepgram API (diarization fallback — Nova-3 model)
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")

# AssemblyAI API (preferred — transcription + diarization in one call)
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "")


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
        self._diarize_client = None
        self._deepgram_client = None
        self._assemblyai_client = None
        self._use_external = False
        self._use_external_diarize = False
        self._use_deepgram_diarize = False
        self._use_assemblyai = False
        self._num_speakers = None  # Speaker count hint (2-10), None = auto
        self._audio_data = None    # Pre-loaded (y, sr) tuple, avoids redundant disk reads
        self._last_diarization_backend = "uninitialized"

        # Priority for transcription + diarization (NOT feature extraction):
        #   1. AssemblyAI (transcribe+diarize in one call)
        #   2. Deepgram Nova-3 (transcribe+diarize in one call)
        #   3. External GPU Whisper + external diarize endpoint
        #   4. Local faster-whisper + KMeans/pyannote
        #
        # Feature extraction always runs separately via librosa (voice agent pipeline).
        if ASSEMBLYAI_API_KEY:
            self._init_assemblyai()

        if not self._use_assemblyai and DEEPGRAM_API_KEY:
            self._init_deepgram_diarize()

        if not self._use_assemblyai and not self._use_deepgram_diarize:
            # Fall back to separate Whisper + diarize
            if EXTERNAL_WHISPER_URL:
                self._init_external()

            if EXTERNAL_DIARIZE_URL:
                self._init_external_diarize()

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

    def _init_assemblyai(self):
        """Try to initialise AssemblyAI client (transcription + diarization)."""
        try:
            project_root = str(Path(__file__).parent.parent.parent)
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            from shared.utils.external_apis import AssemblyAIClient

            client = AssemblyAIClient(api_key=ASSEMBLYAI_API_KEY)

            if client.is_healthy():
                self._assemblyai_client = client
                self._use_assemblyai = True
                logger.info("Using ASSEMBLYAI Universal-3-Pro for transcription + diarization (preferred)")
            else:
                logger.warning("AssemblyAI API not reachable. Trying other backends.")
        except Exception as e:
            logger.warning(f"Could not initialise AssemblyAI client: {e}")

    def _init_deepgram_diarize(self):
        """Try to initialise Deepgram Nova-3 diarization client."""
        try:
            project_root = str(Path(__file__).parent.parent.parent)
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            from shared.utils.external_apis import DeepgramDiarizeClient

            client = DeepgramDiarizeClient(api_key=DEEPGRAM_API_KEY)

            if client.is_healthy():
                self._deepgram_client = client
                self._use_deepgram_diarize = True
                logger.info("Using DEEPGRAM Nova-3 for transcription + diarization (fallback)")
            else:
                logger.warning("Deepgram API not reachable. Trying other backends.")
        except Exception as e:
            logger.warning(f"Could not initialise Deepgram client: {e}")

    def _init_external_diarize(self):
        """Try to connect to the external GPU diarization API."""
        try:
            project_root = str(Path(__file__).parent.parent.parent)
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            from shared.utils.external_apis import DiarizeClient

            client = DiarizeClient(
                base_url=EXTERNAL_DIARIZE_URL,
                api_key=EXTERNAL_API_KEY,
            )

            if client.is_healthy():
                self._diarize_client = client
                self._use_external_diarize = True
                logger.info(f"Using EXTERNAL GPU diarization: {EXTERNAL_DIARIZE_URL}")
            else:
                logger.warning(
                    f"External diarize API at {EXTERNAL_DIARIZE_URL} is not healthy. "
                    f"Falling back to local diarization."
                )
        except Exception as e:
            logger.warning(
                f"Could not initialise external diarize client: {e}. "
                f"Falling back to local diarization."
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
            # Priority: AssemblyAI > Deepgram > Whisper+diarize > local
            if self._use_assemblyai:
                return self._transcribe_assemblyai(audio_path)
            elif self._use_deepgram_diarize:
                return self._transcribe_deepgram(audio_path)
            elif self._use_external:
                return self._transcribe_external(audio_path)
            else:
                return self._transcribe_local(audio_path)
        finally:
            # Clear per-request state to prevent leaking into next request
            self._num_speakers = None
            self._audio_data = None
            self._meeting_type = None

    def _transcribe_assemblyai(self, audio_path: str) -> dict:
        """
        Transcribe + diarize via AssemblyAI Universal-3-Pro (single API call).
        Falls back through the chain on failure.
        """
        try:
            result = self._assemblyai_client.transcribe_and_diarize(audio_path)
            self._last_diarization_backend = "assemblyai"
            return result
        except Exception as e:
            logger.error(f"AssemblyAI failed: {e}. Falling back.")
            if self._use_deepgram_diarize:
                return self._transcribe_deepgram(audio_path)
            elif self._use_external:
                return self._transcribe_external(audio_path)
            else:
                return self._transcribe_local(audio_path)

    def _transcribe_deepgram(self, audio_path: str) -> dict:
        """
        Transcribe + diarize via Deepgram Nova-3 (single API call).
        Falls back through the chain on failure.
        """
        try:
            result = self._deepgram_client.transcribe_and_diarize(audio_path)
            self._last_diarization_backend = "deepgram"
            return result
        except Exception as e:
            logger.error(f"Deepgram failed: {e}. Falling back.")
            if self._use_external:
                return self._transcribe_external(audio_path)
            else:
                return self._transcribe_local(audio_path)

    # ═══════════════════════════════════════════════════════════
    # EXTERNAL BACKEND (GPU Whisper API)
    # ═══════════════════════════════════════════════════════════

    def _transcribe_external(self, audio_path: str) -> dict:
        """
        Transcribe via the external GPU-accelerated Whisper API.

        When external GPU diarization is also available, runs both in
        parallel using ThreadPoolExecutor — cutting total time by ~40%.
        """
        import concurrent.futures

        meeting_type = getattr(self, "_meeting_type", "") or ""

        # ── PARALLEL PATH: both Whisper + diarization on GPU ──
        if self._use_external_diarize:
            logger.info(
                f"Transcribing + diarizing in PARALLEL via GPU APIs: {audio_path}"
            )

            whisper_result = None
            diarize_result = None
            whisper_error = None

            def run_whisper():
                return self._external_client.transcribe(
                    audio_path,
                    model=EXTERNAL_WHISPER_MODEL,
                    language="en",
                    word_timestamps=True,
                    vad_filter=True,
                    stream=True,
                )

            def run_diarize():
                config = self.SPEAKER_DEFAULTS.get(
                    meeting_type, {"default": 2, "min": 2, "max": 8}
                )
                # Deepgram handles MP4/MP3 natively — send original file
                # for best quality. Only fall back to pre-loaded WAV for
                # pyannote GPU which can't decode container formats.
                use_audio_data = (
                    None if self._use_deepgram_diarize
                    else self._audio_data
                )
                return self._diarize_client.diarize(
                    audio_path,
                    min_speakers=config["min"],
                    max_speakers=config["max"],
                    num_speakers=self._num_speakers or 0,
                    audio_data=use_audio_data,
                )

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                whisper_future = pool.submit(run_whisper)
                diarize_future = pool.submit(run_diarize)

                try:
                    whisper_result = whisper_future.result()
                except Exception as e:
                    whisper_error = e

                try:
                    diarize_result = diarize_future.result()
                except Exception as e:
                    logger.warning(f"Parallel GPU diarization failed: {e}")

            # If Whisper failed, fall back to local
            if whisper_error:
                logger.error(
                    f"External Whisper API failed: {whisper_error}. "
                    f"Falling back to local transcription."
                )
                self._use_external = False
                return self._transcribe_local(audio_path)

            # Parse Whisper result into segments
            result = whisper_result or {}
            segments = self._parse_external_segments(result)
            duration = result.get("duration", 0)
            model_used = result.get("model", EXTERNAL_WHISPER_MODEL)
            proc_time = result.get("processing_time", 0)

            # Strip hallucinations
            segments = self._strip_hallucinations(segments)

            logger.info(
                f"External transcription complete: {duration:.1f}s audio, "
                f"{len(segments)} segments, model={model_used}, "
                f"server_time={proc_time:.2f}s"
            )

            # Apply diarization from parallel GPU result
            if diarize_result and diarize_result.get("timeline"):
                segments = self._apply_diarize_timeline(
                    segments, diarize_result["timeline"]
                )
                self._last_diarization_backend = "external_gpu_parallel"
                logger.info(
                    f"Parallel GPU diarization applied: "
                    f"{diarize_result.get('num_speakers', '?')} speakers, "
                    f"diarize_time={diarize_result.get('processing_time', 0):.1f}s"
                )
            else:
                # Diarization failed — fall back to KMeans
                logger.warning("GPU diarization unavailable, falling back to KMeans")
                segments = self._diarize_simple(segments, self._num_speakers, audio_path)

            return {
                "duration_seconds": duration,
                "backend": "external",
                "model": model_used,
                "language": whisper_result.get("language", "en"),
                "language_probability": whisper_result.get("language_probability", 0),
                "processing_time": proc_time,
                "segments": segments,
            }

        # ── SEQUENTIAL PATH: Whisper only on GPU, diarize locally ──
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

        segments = self._parse_external_segments(result)
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

        # Apply speaker diarization — local fallback chain
        duration_min = duration / 60.0
        if USE_PYANNOTE and duration_min > 10:
            segments = self._diarize_pyannote(audio_path, segments)
        else:
            if USE_PYANNOTE and duration_min <= 10:
                logger.info(f"Audio is {duration_min:.1f}min — using KMeans instead of pyannote")
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
    # HELPERS — shared by external and local backends
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _parse_external_segments(result: dict) -> list[dict]:
        """Convert external Whisper API response into internal segment format."""
        segments = []
        for seg in result.get("segments", []):
            words = []
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
        return segments

    @staticmethod
    def _normalize_speaker_label(label: str) -> str:
        """Normalize GPU diarization speaker labels to internal format.

        The external pyannote API returns 'SPEAKER_00', 'SPEAKER_01', etc.
        Internally NEXUS uses 'Speaker_0', 'Speaker_1', etc.
        """
        import re
        m = re.match(r"SPEAKER_(\d+)", label)
        if m:
            return f"Speaker_{int(m.group(1))}"
        return label

    @staticmethod
    def _apply_diarize_timeline(
        segments: list[dict], timeline: list[dict]
    ) -> list[dict]:
        """Map a GPU diarization timeline onto transcript segments by max overlap."""
        for seg in segments:
            best_speaker = "Speaker_0"
            best_overlap = 0

            for turn in timeline:
                turn_start_ms = int(turn["start"] * 1000)
                turn_end_ms = int(turn["end"] * 1000)
                overlap_start = max(seg["start_ms"], turn_start_ms)
                overlap_end = min(seg["end_ms"], turn_end_ms)
                overlap = max(0, overlap_end - overlap_start)

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = Transcriber._normalize_speaker_label(turn["speaker"])

            # If no overlap, use nearest turn
            if best_overlap == 0 and timeline:
                seg_mid = (seg["start_ms"] + seg["end_ms"]) / 2
                nearest = min(
                    timeline,
                    key=lambda t: abs((t["start"] + t["end"]) / 2 * 1000 - seg_mid),
                )
                best_speaker = Transcriber._normalize_speaker_label(nearest["speaker"])

            seg["speaker"] = best_speaker

        return segments

    # ═══════════════════════════════════════════════════════════
    # LOCAL BACKEND (faster-whisper on CPU)
    # ═══════════════════════════════════════════════════════════

    def _run_whisper(self, audio_path: str, vad_filter: bool = True) -> tuple[list[dict], float]:
        """Run Whisper transcription and return (segments, duration_seconds)."""
        kwargs = dict(
            beam_size=5,
            word_timestamps=True,
            language="en",
        )
        if vad_filter:
            kwargs["vad_filter"] = True
            kwargs["vad_parameters"] = dict(
                min_silence_duration_ms=500,
                speech_pad_ms=200,
            )

        segments_raw, info = self._model.transcribe(audio_path, **kwargs)

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

        return segments, info.duration

    def _transcribe_local(self, audio_path: str, skip_vad: bool = False) -> dict:
        """Transcribe using local faster-whisper."""
        self._load_model()

        use_vad = not skip_vad
        logger.info(f"Transcribing via LOCAL faster-whisper: {audio_path} (vad={use_vad})")

        segments, duration = self._run_whisper(audio_path, vad_filter=use_vad)

        # ── VAD fallback: if VAD aggressively truncated, retry without it ──
        if use_vad and segments and duration > 0:
            max_seg_end_ms = max(s["end_ms"] for s in segments)
            coverage = max_seg_end_ms / (duration * 1000)
            if coverage < 0.50:
                logger.warning(
                    f"VAD filter covered only {coverage:.0%} of audio "
                    f"({max_seg_end_ms/1000:.0f}s of {duration:.0f}s). "
                    f"Retrying WITHOUT VAD..."
                )
                segments, duration = self._run_whisper(audio_path, vad_filter=False)

        # Strip Whisper hallucination loops
        segments = self._strip_hallucinations(segments)

        logger.info(f"Local transcription complete: {duration:.1f}s, {len(segments)} segments")

        # Apply speaker diarization — same priority as external backend
        duration_min = duration / 60.0
        meeting_type = getattr(self, "_meeting_type", "") or ""
        if self._use_external_diarize:
            segments = self._diarize_external(audio_path, segments, meeting_type)
        elif USE_PYANNOTE and duration_min > 10:
            segments = self._diarize_pyannote(audio_path, segments)
        else:
            if USE_PYANNOTE and duration_min <= 10:
                logger.info(f"Audio is {duration_min:.1f}min — using KMeans instead of pyannote")
            segments = self._diarize_simple(segments, self._num_speakers, audio_path)

        return {
            "duration_seconds": duration,
            "backend": "local",
            "model": WHISPER_MODEL,
            "segments": segments,
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

    # Speaker count defaults and ranges per meeting type
    SPEAKER_DEFAULTS = {
        "sales_call":            {"default": 2, "min": 2, "max": 3},
        "interview":             {"default": 2, "min": 2, "max": 4},
        "internal":              {"default": 4, "min": 2, "max": 8},
        "client_meeting":        {"default": 3, "min": 2, "max": 8},
        "meeting":               {"default": 4, "min": 2, "max": 8},
        "podcast":               {"default": 2, "min": 2, "max": 4},
        "lecture":               {"default": 1, "min": 1, "max": 2},
        "presentation":          {"default": 1, "min": 1, "max": 3},
        "debate":                {"default": 2, "min": 2, "max": 4},
        "casual_conversation":   {"default": 2, "min": 2, "max": 4},
    }

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

        # Fallback: gap-based heuristic (needs audio for real pitch)
        if max_speakers == 2:
            return self._diarize_gap_two_speaker(segments, audio_path)
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

        features = []  # 5 features per segment
        valid_mask = []  # Track which segments have valid features

        for seg in segments:
            start_sample = int(seg["start_ms"] / 1000 * sr)
            end_sample = int(seg["end_ms"] / 1000 * sr)
            # Clamp to audio bounds
            start_sample = max(0, min(start_sample, total_samples - 1))
            end_sample = max(start_sample + 1, min(end_sample, total_samples))

            chunk = y[start_sample:end_sample]

            if len(chunk) < sr * 0.1:  # Skip very short chunks (< 100ms)
                features.append([0.0, 0.0, 0.0, 0.0, 0.0])
                valid_mask.append(False)
                continue

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

            # Spectral centroid
            try:
                sc = librosa.feature.spectral_centroid(y=chunk, sr=sr)[0]
                spectral_centroid = float(np.mean(sc)) if len(sc) > 0 else 0.0
            except Exception:
                spectral_centroid = 0.0

            # Zero crossing rate
            try:
                zcr = librosa.feature.zero_crossing_rate(chunk)[0]
                zero_crossing = float(np.mean(zcr)) if len(zcr) > 0 else 0.0
            except Exception:
                zero_crossing = 0.0

            features.append([mean_pitch, mean_energy, spectral_centroid, zero_crossing, pitch_std])
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

        # Fill invalid segments with zeros so they don't bias clustering
        # toward the centroid (short/silent segments should be neutral)
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

        # ── Temporal smoothing: reassign truly isolated micro-fragments.
        # Only smooth if the segment is very short (<1.5s) AND gaps to both
        # neighbours are very small (<200ms). This catches Whisper artefacts
        # without destroying real A-B-A-B speaker alternation. ──
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

        # ── Cluster balance check: adaptive threshold based on speaker count.
        # For N speakers, dominant speaker should not exceed (1/N + 0.40).
        # e.g. 2 speakers: max 90%, 3 speakers: max 73%, 4 speakers: max 65%
        from collections import Counter
        counts = Counter(labels.tolist())
        total = len(labels)
        max_pct = max(counts.values()) / total if total > 0 else 1.0
        max_allowed_pct = (1.0 / num_speakers) + 0.40
        if max_pct > max_allowed_pct:
            logger.warning(
                f"KMeans produced imbalanced split: {dict(counts)} "
                f"({max_pct:.0%} in one cluster, limit={max_allowed_pct:.0%}). "
                f"Falling to gap+pitch."
            )
            return None

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
        Fallback: 2-speaker alternation based on gaps + real pitch shifts.

        Speaker changes are detected when EITHER:
          - Silence gap > 800ms between segments, OR
          - Mean F0 pitch jump > 25 Hz between adjacent segments
        """
        import librosa

        current_speaker = "Speaker_0"
        TURN_GAP_MS = 800
        PITCH_JUMP_HZ = 25.0

        # Compute real mean pitch (F0) per segment from audio
        seg_pitches: list[Optional[float]] = [None] * len(segments)

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
        else:
            y, sr = None, 16000

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

        # Detect speaker changes
        for i, seg in enumerate(segments):
            if i > 0:
                gap_ms = seg["start_ms"] - segments[i-1]["end_ms"]
                pitch_shift = False
                if seg_pitches[i] is not None and seg_pitches[i-1] is not None:
                    pitch_shift = abs(seg_pitches[i] - seg_pitches[i-1]) > PITCH_JUMP_HZ

                if gap_ms > TURN_GAP_MS or pitch_shift:
                    current_speaker = (
                        "Speaker_1" if current_speaker == "Speaker_0"
                        else "Speaker_0"
                    )
            seg["speaker"] = current_speaker

        # Log balance
        from collections import Counter
        counts = Counter(seg["speaker"] for seg in segments)
        pitched = sum(1 for p in seg_pitches if p is not None)
        self._last_diarization_backend = "gap_pitch"
        logger.info(
            f"Gap+pitch diarization: {dict(counts)} "
            f"({len(segments)} segments, {pitched} with pitch data)"
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

    def _diarize_external(
        self, audio_path: str, segments: list[dict], meeting_type: str = ""
    ) -> list[dict]:
        """
        Speaker diarization via external GPU API (pyannote on RTX 5090).
        Falls back to local diarization on failure.
        """
        if not self._diarize_client:
            logger.warning("External diarize client not available, falling back to local")
            return self._diarize_simple(segments, self._num_speakers, audio_path)

        try:
            config = self.SPEAKER_DEFAULTS.get(meeting_type, {"default": 2, "min": 2, "max": 8})
            num_speakers = self._num_speakers or 0

            result = self._diarize_client.diarize(
                audio_path,
                min_speakers=config["min"],
                max_speakers=config["max"],
                num_speakers=num_speakers,
                audio_data=self._audio_data,
            )

            speaker_timeline = result.get("timeline", [])
            if not speaker_timeline:
                logger.warning("External diarization returned empty timeline, falling back")
                return self._diarize_simple(segments, self._num_speakers, audio_path)

            # Map speaker timeline onto transcript segments by max overlap
            for seg in segments:
                best_speaker = "Speaker_0"
                best_overlap = 0

                for turn in speaker_timeline:
                    turn_start_ms = int(turn["start"] * 1000)
                    turn_end_ms = int(turn["end"] * 1000)
                    overlap_start = max(seg["start_ms"], turn_start_ms)
                    overlap_end = min(seg["end_ms"], turn_end_ms)
                    overlap = max(0, overlap_end - overlap_start)

                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_speaker = self._normalize_speaker_label(turn["speaker"])

                # If no overlap, use nearest turn
                if best_overlap == 0:
                    seg_mid = (seg["start_ms"] + seg["end_ms"]) / 2
                    nearest = min(
                        speaker_timeline,
                        key=lambda t: abs((t["start"] + t["end"]) / 2 * 1000 - seg_mid),
                    )
                    best_speaker = self._normalize_speaker_label(nearest["speaker"])

                seg["speaker"] = best_speaker

            detected = len(set(seg["speaker"] for seg in segments))
            self._last_diarization_backend = "external_gpu"
            logger.info(
                f"External GPU diarization complete: {detected} speakers, "
                f"{len(speaker_timeline)} turns, "
                f"API time={result.get('processing_time', 0):.1f}s"
            )
            return segments

        except Exception as e:
            logger.error(f"External diarization failed: {e}. Falling back to local.")
            return self._diarize_simple(segments, self._num_speakers, audio_path)

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

            # Create speaker timeline (normalize SPEAKER_00 → Speaker_0)
            speaker_timeline = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_timeline.append({
                    "speaker": self._normalize_speaker_label(speaker),
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
