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
import warnings
import logging
import tempfile
import numpy as np
import soundfile as sf

# Suppress torchcodec/libtorchcodec warnings from pyannote — we pre-convert to WAV
warnings.filterwarnings("ignore", message=".*torchcodec.*")
warnings.filterwarnings("ignore", message=".*libtorchcodec.*")

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

# GPU diarization clustering threshold (lower = more aggressive speaker merging)
# API default is 0.5. Use 0.3-0.4 if same-gender speakers are being over-split.
DIARIZE_CLUSTERING_THRESHOLD = float(os.getenv("DIARIZE_CLUSTERING_THRESHOLD", "0.5"))

# Transcription backend selection:
#   auto | assemblyai | deepgram | whisper-pyannote | parakeet | whisper | local
# auto = AssemblyAI > Deepgram > Whisper+Pyannote combined > Parakeet+diarize > Whisper+separate diarize > Local
TRANSCRIPTION_BACKEND = os.getenv("TRANSCRIPTION_BACKEND", "auto")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")

# Whisper+Pyannote combined endpoint (single call, no mapping needed)
EXTERNAL_TRANSCRIBE_DIARIZE_URL = os.getenv("EXTERNAL_DIARIZE_URL", "")

# Parakeet TDT (fast transcription with word timestamps, needs separate diarizer)
PARAKEET_URL = os.getenv("PARAKEET_URL", "")

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
            "Pyannote community-1 disabled. No HF_TOKEN set."
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
        logger.info("Falling back to local diarization")
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
        self._external_client = None
        self._deepgram_client = None
        self._assemblyai_client = None
        self._parakeet_client = None
        self._use_external = False
        self._use_deepgram = False
        self._use_deepgram_diarize = False
        self._use_assemblyai = False
        self._use_whisper_pyannote = False
        self._use_parakeet = False
        self._use_external_diarize = False
        self._num_speakers = None  # Speaker count hint (2-10), None = auto
        self._audio_data = None    # Pre-loaded (y, sr) tuple, avoids redundant disk reads
        self._last_diarization_backend = "uninitialized"

        backend = TRANSCRIPTION_BACKEND.lower()

        # Fallback chain:
        #   1. Whisper+NeMo combined (/transcribe-diarize — one GPU call, best quality)
        #   2. Parakeet + NeMo (Parakeet /transcribe + /diarize — fast transcription, separate diarize)
        #   3. AssemblyAI (one call, Universal-3 Pro)
        #   4. Deepgram (one call, Nova-3)
        #   5. Whisper transcribe + diarize cascade (two calls, mapping)
        #   6. Local faster-whisper + local diarize cascade (CPU)

        if backend in ("auto", "whisper-pyannote", "whisper-nemo") and EXTERNAL_TRANSCRIBE_DIARIZE_URL:
            self._init_whisper_pyannote()

        if backend in ("auto", "parakeet") and PARAKEET_URL:
            self._init_parakeet()

        # Parakeet needs a separate diarization backend (/diarize endpoint).
        # Also used by whisper separate mode.
        if backend in ("auto", "parakeet", "whisper") and EXTERNAL_TRANSCRIBE_DIARIZE_URL:
            if not self._use_external_diarize:
                self._init_external_diarize()

        if backend in ("auto", "assemblyai") and ASSEMBLYAI_API_KEY:
            self._init_assemblyai()

        if backend in ("auto", "deepgram") and DEEPGRAM_API_KEY:
            self._init_deepgram()

        if backend in ("auto", "whisper") and EXTERNAL_WHISPER_URL:
            self._init_external()

    def _init_whisper_pyannote(self):
        """Try to initialise Whisper+Pyannote combined endpoint (single call, GPU)."""
        if not EXTERNAL_TRANSCRIBE_DIARIZE_URL:
            return
        try:
            import httpx
            url = EXTERNAL_TRANSCRIBE_DIARIZE_URL.rstrip("/")
            # Derive health URL from base (strip /transcribe-diarize path)
            base = url.rsplit("/", 1)[0] if "/transcribe-diarize" in url else url
            headers = {"X-API-Key": EXTERNAL_API_KEY} if EXTERNAL_API_KEY else {}
            resp = httpx.get(f"{base}/health", headers=headers, timeout=30)
            if resp.status_code == 200:
                self._use_whisper_pyannote = True
                logger.info(f"Using Whisper+NEMO combined endpoint: {url}")
            else:
                logger.warning(f"Whisper+NEMO endpoint not healthy ({resp.status_code})")
        except Exception as e:
            logger.warning(f"Could not connect to Whisper+NEMO endpoint: {e}")

    def _init_parakeet(self):
        """Try to initialise Parakeet TDT client (fast transcription with word timestamps)."""
        try:
            project_root = str(Path(__file__).parent.parent.parent)
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            from shared.utils.parakeet_client import create_parakeet_client

            client = create_parakeet_client()
            if client is not None:
                self._parakeet_client = client
                self._use_parakeet = True
                logger.info("Using Parakeet backend (TDT 0.6B v2, transcription + separate diarize)")
            else:
                logger.warning("Parakeet URL set but client creation failed.")
        except Exception as e:
            logger.warning(f"Could not initialise Parakeet client: {e}")

    def _init_assemblyai(self):
        """Try to initialise AssemblyAI client (transcription + diarization in one call)."""
        try:
            project_root = str(Path(__file__).parent.parent.parent)
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            from shared.utils.assemblyai_client import create_assemblyai_client

            client = create_assemblyai_client()
            if client is not None:
                self._assemblyai_client = client
                self._use_assemblyai = True
                logger.info("Using AssemblyAI backend (Universal-3 Pro, transcription + diarization)")
            else:
                logger.warning("AssemblyAI API key set but client creation failed.")
        except Exception as e:
            logger.warning(f"Could not initialise AssemblyAI client: {e}")

    def _init_deepgram(self):
        """Try to initialise Deepgram client (transcription + diarization in one call)."""
        try:
            project_root = str(Path(__file__).parent.parent.parent)
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            from shared.utils.deepgram_client import create_deepgram_client

            client = create_deepgram_client()
            if client is not None:
                self._deepgram_client = client
                self._use_deepgram = True
                logger.info("Using Deepgram backend (Nova-3, transcription + diarization)")
            else:
                logger.warning("Deepgram API key set but client creation failed.")
        except Exception as e:
            logger.warning(f"Could not initialise Deepgram client: {e}")

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
                base_url=EXTERNAL_TRANSCRIBE_DIARIZE_URL,
                api_key=EXTERNAL_API_KEY,
            )

            if client.is_healthy():
                self._diarize_client = client
                self._use_external_diarize = True
                logger.info(f"Using EXTERNAL GPU diarization: {EXTERNAL_TRANSCRIBE_DIARIZE_URL}")
            else:
                logger.warning(
                    f"External diarize API at {EXTERNAL_TRANSCRIBE_DIARIZE_URL} is not healthy. "
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
        self._tmp_wav = None  # Track temp WAV for cleanup

        try:
            # Convert to 16kHz mono WAV upfront if input is a video/non-WAV container.
            # This ensures ALL backends get clean audio regardless of input format.
            effective_path = self._ensure_wav(audio_path)

            # Fallback chain:
            #   1. Whisper+NeMo combined (/transcribe-diarize — best quality)
            #   2. Parakeet + NeMo (/transcribe + /diarize — fast)
            #   3. AssemblyAI (one call, Universal-3 Pro)
            #   4. Deepgram (one call, Nova-3)
            #   5. Whisper + separate diarize cascade
            #   6. Local faster-whisper + local diarize (CPU)
            if self._use_whisper_pyannote:
                return self._transcribe_whisper_pyannote(effective_path)
            elif self._use_parakeet:
                return self._transcribe_parakeet(effective_path)
            elif self._use_assemblyai:
                return self._transcribe_assemblyai(effective_path)
            elif self._use_deepgram:
                return self._transcribe_deepgram(effective_path)
            elif self._use_external:
                return self._transcribe_external(effective_path)
            else:
                return self._transcribe_local(effective_path)
        finally:
            # Cleanup temp WAV if we created one
            if self._tmp_wav is not None:
                try:
                    os.unlink(self._tmp_wav)
                except OSError:
                    pass
                self._tmp_wav = None
            # Clear per-request state to prevent leaking into next request
            self._num_speakers = None
            self._audio_data = None
            self._meeting_type = None

    def _ensure_wav(self, audio_path: str) -> str:
        """
        Convert input file to 16kHz mono WAV if it's not already WAV.
        Uses pre-loaded audio_data if available, otherwise loads from file.
        Returns path to the WAV file (original if already WAV, temp file otherwise).
        """
        # If it's already a .wav, use as-is
        if audio_path.lower().endswith(".wav") and self._audio_data is None:
            return audio_path

        # If we have pre-loaded audio data, write it to a temp WAV
        if self._audio_data is not None:
            y, sr = self._audio_data
            if y.ndim > 1:
                y = np.mean(y, axis=0) if y.shape[0] < y.shape[1] else np.mean(y, axis=1)
            if sr != 16000:
                try:
                    import torchaudio.functional as _ta_fn
                    import torch
                    waveform = torch.from_numpy(y).float().unsqueeze(0)
                    waveform = _ta_fn.resample(waveform, sr, 16000)
                    y = waveform.squeeze(0).numpy()
                except ImportError:
                    import librosa
                    y = librosa.resample(y, orig_sr=sr, target_sr=16000)
                sr = 16000
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(tmp.name, y, sr)
            self._tmp_wav = tmp.name
            logger.info(f"Converted pre-loaded audio to 16kHz WAV: {tmp.name}")
            return tmp.name

        # Non-WAV file on disk (mp4, mp3, m4a, webm, ogg, flac) — load and convert
        if not audio_path.lower().endswith(".wav"):
            try:
                from shared.utils.audio_loader import load_audio
                y, sr = load_audio(audio_path, sr=16000)
            except ImportError:
                import librosa
                y, sr = librosa.load(audio_path, sr=16000, mono=True)
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(tmp.name, y, sr)
            self._tmp_wav = tmp.name
            logger.info(f"Converted {Path(audio_path).suffix} to 16kHz mono WAV: {tmp.name}")
            return tmp.name

        return audio_path

    # ═══════════════════════════════════════════════════════════
    # PARAKEET BACKEND (fast transcription + separate diarize cascade)
    # ═══════════════════════════════════════════════════════════

    def _transcribe_parakeet(self, audio_path: str) -> dict:
        """
        Transcribe via Parakeet TDT 0.6B v2 (174x realtime), then run
        diarize cascade for speaker labels.

        Fallback: Whisper separate -> Local
        """
        logger.info(f"Transcribing via Parakeet: {audio_path}")

        try:
            result = self._parakeet_client.transcribe(audio_path)
        except Exception as e:
            logger.error(f"Parakeet failed: {e}. Falling back to next backend.")
            self._use_parakeet = False
            if self._use_assemblyai:
                return self._transcribe_assemblyai(audio_path)
            elif self._use_deepgram:
                return self._transcribe_deepgram(audio_path)
            elif self._use_external:
                return self._transcribe_external(audio_path)
            return self._transcribe_local(audio_path)

        segments = result.get("segments", [])
        segments = self._strip_hallucinations(segments)

        # Parakeet returns very coarse segments (e.g., 2 segments for 20s audio).
        # Split them into per-sentence segments using word timestamps so the
        # diarizer has enough granularity to assign different speakers.
        segments = self._split_coarse_segments(segments)

        duration = result["duration_seconds"]
        proc_time = result["processing_time"]

        logger.info(
            f"Parakeet transcription complete: {duration:.1f}s audio, "
            f"{len(segments)} segments (after split), time={proc_time:.1f}s"
        )

        # Apply speaker diarization (same cascade as Whisper separate path)
        segments = self._diarize_simple(segments, self._num_speakers, audio_path)

        return {
            "duration_seconds": duration,
            "backend": "parakeet",
            "model": result.get("model", "parakeet-tdt-0.6b-v2"),
            "processing_time": proc_time,
            "segments": segments,
            "diarization_backend": self._last_diarization_backend,
            "diarization_confidence": self._compute_diarization_confidence(segments),
        }

    # ═══════════════════════════════════════════════════════════
    # WHISPER+PYANNOTE COMBINED (single GPU call, no mapping)
    # ═══════════════════════════════════════════════════════════

    def _transcribe_whisper_pyannote(self, audio_path: str) -> dict:
        """
        Transcribe + diarize via combined Whisper+Pyannote GPU endpoint.
        Single call returns segments with per-word speaker labels.
        No cross-provider mapping needed.

        Fallback: AssemblyAI -> Deepgram -> Whisper separate -> Local
        """
        url = EXTERNAL_TRANSCRIBE_DIARIZE_URL.rstrip("/")
        if not url.endswith("/transcribe-diarize"):
            url = f"{url}/transcribe-diarize"

        logger.info(f"Transcribing via Whisper+Pyannote combined: {audio_path}")

        # audio_path is already 16kHz mono WAV (converted by _ensure_wav)
        wav_path = audio_path

        headers = {"X-API-Key": EXTERNAL_API_KEY} if EXTERNAL_API_KEY else {}

        mt = self._meeting_type or "meeting"
        config = self.SPEAKER_DEFAULTS.get(mt, {"default": 3, "min": 2, "max": 10})

        data = {
            "model": EXTERNAL_WHISPER_MODEL,
            "min_speakers": str(config["min"]),
            "max_speakers": str(config["max"]),
            "backend": "nemo",
        }
        if self._num_speakers:
            data["num_speakers"] = str(self._num_speakers)

        try:
            import httpx
            with open(wav_path, "rb") as f:
                with httpx.Client(timeout=1800) as client:  # 30 min for large audio files
                    resp = client.post(
                        url,
                        files={"file": (Path(wav_path).name, f, "audio/wav")},
                        data=data,
                        headers=headers,
                    )
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            result = resp.json()
        except Exception as e:
            logger.error(f"Whisper+NeMo combined failed: {e}. Falling back to next backend.")
            self._use_whisper_pyannote = False
            if self._use_parakeet:
                return self._transcribe_parakeet(audio_path)
            elif self._use_assemblyai:
                return self._transcribe_assemblyai(audio_path)
            elif self._use_deepgram:
                return self._transcribe_deepgram(audio_path)
            elif self._use_external:
                return self._transcribe_external(audio_path)
            return self._transcribe_local(audio_path)

        # Convert response to NEXUS format
        # Response has: segments[{speaker, start, end, text, words}], speakers, duration
        raw_segments = result.get("segments", [])
        segments = []
        spk_map = {}
        for seg in raw_segments:
            raw_spk = seg.get("speaker", "SPEAKER_00")
            if raw_spk == "UNKNOWN":
                raw_spk = "SPEAKER_00"
            if raw_spk not in spk_map:
                spk_map[raw_spk] = f"Speaker_{len(spk_map)}"
            speaker = spk_map[raw_spk]

            words = []
            for w in seg.get("words", []):
                w_spk = w.get("speaker", raw_spk)
                if w_spk == "UNKNOWN":
                    w_spk = raw_spk
                words.append({
                    "word": w.get("word", ""),
                    "start": w.get("start", 0),
                    "end": w.get("end", 0),
                    "speaker": spk_map.get(w_spk, speaker),
                })

            text = seg.get("text", "").strip()
            if text:
                segments.append({
                    "speaker": speaker,
                    "start_ms": int(seg.get("start", 0) * 1000),
                    "end_ms": int(seg.get("end", 0) * 1000),
                    "text": text,
                    "words": words,
                })

        segments = self._strip_hallucinations(segments)
        duration = result.get("duration", 0)
        proc_time = result.get("processing_time", 0)
        num_speakers = result.get("num_speakers", len(spk_map))
        self._last_diarization_backend = "whisper_pyannote_combined"

        logger.info(
            f"Whisper+Pyannote combined complete: {duration:.1f}s audio, "
            f"{len(segments)} segments, {num_speakers} speakers, "
            f"time={proc_time:.1f}s"
        )

        return {
            "duration_seconds": duration,
            "backend": "whisper_pyannote_combined",
            "model": result.get("params_used", {}).get("model", EXTERNAL_WHISPER_MODEL),
            "language": result.get("language", "en"),
            "language_probability": result.get("language_probability", 0),
            "processing_time": proc_time,
            "segments": segments,
            "diarization_backend": "whisper_pyannote_combined",
            "diarization_confidence": 0.90,
        }

    # ═══════════════════════════════════════════════════════════
    # ASSEMBLYAI BACKEND (transcription + diarization in one call)
    # ═══════════════════════════════════════════════════════════

    def _transcribe_assemblyai(self, audio_path: str) -> dict:
        """
        Transcribe via AssemblyAI Universal-3 Pro — transcribes first,
        then diarizes, all in one API call.

        Fallback: Whisper transcribe + Deepgram diarize
                  -> Whisper transcribe + pyannote GPU diarize
                    -> Local
        """
        logger.info(f"Transcribing via AssemblyAI: {audio_path}")

        try:
            # AssemblyAIClient may come from assemblyai_client.py (.transcribe)
            # or external_apis.py (.transcribe_and_diarize) — try both
            if hasattr(self._assemblyai_client, 'transcribe'):
                result = self._assemblyai_client.transcribe(
                    audio_path,
                    speakers_expected=self._num_speakers,
                )
            else:
                result = self._assemblyai_client.transcribe_and_diarize(audio_path)
        except Exception as e:
            logger.error(f"AssemblyAI failed: {e}. Falling back to next backend.")
            self._use_assemblyai = False
            if self._use_deepgram:
                return self._transcribe_deepgram(audio_path)
            elif self._use_external:
                return self._transcribe_external(audio_path)
            return self._transcribe_local(audio_path)

        segments = result.get("segments", [])
        segments = self._strip_hallucinations(segments)

        self._last_diarization_backend = "assemblyai"

        logger.info(
            f"AssemblyAI complete: {result['duration_seconds']:.1f}s audio, "
            f"{len(segments)} segments, {len(result['speakers'])} speakers, "
            f"time={result['processing_time']:.1f}s"
        )

        return {
            "duration_seconds": result["duration_seconds"],
            "backend": "assemblyai",
            "model": "universal-3-pro",
            "segments": segments,
            "diarization_backend": "assemblyai",
            "diarization_confidence": 0.85,
        }

    # ═══════════════════════════════════════════════════════════
    # DEEPGRAM BACKEND (own transcription + diarization in one call)
    # ═══════════════════════════════════════════════════════════

    def _transcribe_deepgram(self, audio_path: str) -> dict:
        """
        Transcribe via Deepgram Nova-3 — returns its own transcript with
        speaker labels and word-level timestamps. No cross-provider mapping.

        Fallback: Whisper + pyannote GPU → Local
        """
        logger.info(f"Transcribing via Deepgram: {audio_path}")

        try:
            result = self._deepgram_client.transcribe(audio_path)
        except Exception as e:
            logger.error(f"Deepgram failed: {e}. Falling back to next backend.")
            self._use_deepgram = False
            if self._use_external:
                return self._transcribe_external(audio_path)
            return self._transcribe_local(audio_path)

        segments = result.get("segments", [])
        segments = self._strip_hallucinations(segments)

        self._last_diarization_backend = "deepgram"

        logger.info(
            f"Deepgram complete: {result['duration_seconds']:.1f}s audio, "
            f"{len(segments)} segments, {len(result['speakers'])} speakers, "
            f"time={result['processing_time']:.1f}s"
        )

        return {
            "duration_seconds": result["duration_seconds"],
            "backend": "deepgram",
            "model": result.get("model", "nova-3"),
            "segments": segments,
            "diarization_backend": "deepgram",
            "diarization_confidence": 0.85,
        }

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
                    clustering_threshold=DIARIZE_CLUSTERING_THRESHOLD,
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
    def _split_coarse_segments(segments: list[dict], max_seg_duration_ms: int = 5000) -> list[dict]:
        """
        Split coarse segments (e.g., from Parakeet) into finer per-sentence
        segments using word timestamps. This gives the diarizer enough
        granularity to assign different speakers within a long segment.

        A segment longer than max_seg_duration_ms is split at sentence
        boundaries (period, question mark, exclamation) using word timestamps.
        If no sentence boundaries exist, splits at natural pauses (gaps > 300ms).
        """
        result = []
        for seg in segments:
            duration_ms = seg.get("end_ms", 0) - seg.get("start_ms", 0)
            words = seg.get("words", [])

            # Short segment or no words — keep as-is
            if duration_ms <= max_seg_duration_ms or len(words) < 2:
                result.append(seg)
                continue

            # Find split points at sentence boundaries
            splits = []
            current_words = []
            for w in words:
                current_words.append(w)
                word_text = w.get("word", "")
                # Split at sentence-ending punctuation
                if word_text.rstrip().endswith((".", "?", "!")):
                    splits.append(current_words)
                    current_words = []

            # Remaining words
            if current_words:
                if splits:
                    splits.append(current_words)
                else:
                    # No sentence boundaries — split at pauses > 300ms
                    pause_splits = []
                    chunk = [words[0]]
                    for i in range(1, len(words)):
                        gap = (words[i].get("start", 0) - words[i - 1].get("end", 0)) * 1000
                        if gap > 300 and len(chunk) >= 2:
                            pause_splits.append(chunk)
                            chunk = []
                        chunk.append(words[i])
                    if chunk:
                        pause_splits.append(chunk)
                    splits = pause_splits if len(pause_splits) > 1 else [words]

            # Build new segments from splits
            for word_group in splits:
                if not word_group:
                    continue
                text = " ".join(w.get("word", "").strip() for w in word_group).strip()
                if not text:
                    continue
                result.append({
                    "start_ms": int(word_group[0].get("start", 0) * 1000),
                    "end_ms": int(word_group[-1].get("end", 0) * 1000),
                    "text": text,
                    "words": word_group,
                })

        if len(result) != len(segments):
            logger.info(f"Split {len(segments)} coarse segments into {len(result)} fine segments")

        return result

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
        return segments

    # ═══════════════════════════════════════════════════════════
    # SPEAKER DIARIZATION
    # ═══════════════════════════════════════════════════════════

    # Diarization mode from environment
    DIARIZATION_MODE = os.getenv("DIARIZATION_MODE", "auto")  # auto|pyannote|kmeans

    # Speaker count defaults and ranges per meeting type
    # turn_gap_ms: gap threshold for that conversation style
    SPEAKER_DEFAULTS = {
        "sales_call":            {"default": 2, "min": 2, "max": 3, "turn_gap_ms": 400},
        "interview":             {"default": 2, "min": 2, "max": 4, "turn_gap_ms": 600},
        "internal":              {"default": 4, "min": 2, "max": 10, "turn_gap_ms": 800},
        "client_meeting":        {"default": 3, "min": 2, "max": 10, "turn_gap_ms": 600},
        "meeting":               {"default": 4, "min": 2, "max": 10, "turn_gap_ms": 800},
        "podcast":               {"default": 2, "min": 2, "max": 4, "turn_gap_ms": 600},
        "lecture":               {"default": 1, "min": 1, "max": 2, "turn_gap_ms": 1000},
        "presentation":          {"default": 1, "min": 1, "max": 3, "turn_gap_ms": 1000},
        "debate":                {"default": 2, "min": 2, "max": 4, "turn_gap_ms": 400},
        "casual_conversation":   {"default": 2, "min": 2, "max": 4, "turn_gap_ms": 400},
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
          1. Deepgram diarization (when Whisper provides transcript)
          2. External GPU pyannote diarization
          3. Pyannote Community-1 (frame-level, neural)
          4. Legacy Pyannote 3.1
          5. Acoustic KMeans (MFCCs + pitch + energy)
          6. Gap + pitch heuristic (2-speaker fallback)
          7. Heuristic multi-speaker (round-robin, last resort)

        DIARIZATION_MODE env var overrides: auto|pyannote|kmeans
        """
        if not segments:
            return segments

        meeting_type = getattr(self, "_meeting_type", "sales_call") or "sales_call"
        config = self.SPEAKER_DEFAULTS.get(meeting_type, {"default": 3, "min": 2, "max": 10})

        if num_speakers is not None:
            max_speakers = min(max(1, num_speakers), 10)
        else:
            max_speakers = config["max"]  # Let GPU backends auto-detect within full range

        # Single speaker — assign all segments and skip clustering
        if num_speakers == 1:
            for seg in segments:
                seg["speaker"] = "Speaker_0"
            self._last_diarization_backend = "single_speaker"
            logger.info("Single-speaker mode: all segments assigned to Speaker_0")
            return segments

        mode = self.DIARIZATION_MODE
        logger.info(f"Diarization cascade: mode={mode}, audio_path={'yes' if audio_path else 'no'}, max_speakers={max_speakers}")

        # ── Tier 0: External GPU pyannote Diarization ──
        if mode in ("auto", "pyannote") and audio_path:
            try:
                result = self._diarize_external_gpu(segments, max_speakers, audio_path)
                if result is not None:
                    return result
            except Exception as e:
                logger.warning(f"External GPU diarization failed: {e}")

        # ── Tier 1: Pyannote Community-1 (frame-level fallback) ──
        if mode in ("pyannote", "auto") and audio_path:
            try:
                result = self._diarize_pyannote_community(
                    segments, max_speakers, audio_path
                )
                if result is not None:
                    return result
            except Exception as e:
                logger.warning(f"Pyannote community-1 failed: {e}")

        # ── Tier 2b: Legacy Pyannote 3.1 (if USE_PYANNOTE=true) ──
        if mode in ("pyannote", "auto") and USE_PYANNOTE and audio_path:
            try:
                result = self._diarize_pyannote(audio_path, segments)
                if result is not None:
                    return result
            except Exception as e:
                logger.warning(f"Pyannote 3.1 failed: {e}")

        # ── Local fallbacks need estimated speaker count ──
        # GPU backends above use full max_speakers range and auto-detect.
        # Local KMeans/gap need an explicit count to cluster.
        if num_speakers is None:
            estimated = self._estimate_speaker_count(segments)
            logger.info("Local fallback: estimated %s speaker(s) for clustering", estimated)
            local_max = estimated
        else:
            local_max = max_speakers

        # ── Tier 3: Acoustic KMeans (enhanced with MFCCs) ──
        if mode in ("kmeans", "auto") and audio_path and local_max >= 2:
            try:
                clustered = self._diarize_acoustic_kmeans(
                    segments, local_max, audio_path
                )
                if clustered is not None:
                    return clustered
            except Exception as e:
                logger.warning(f"Acoustic KMeans diarization failed: {e}")

        # ── Tier 4/5: Gap+pitch or heuristic ──
        if local_max == 2:
            return self._diarize_gap_two_speaker(segments, audio_path)
        return self._diarize_simple_multi_speaker(segments, local_max)

    # ── External GPU Diarization (Tier 0) ──

    def _diarize_external_gpu(
        self,
        segments: list[dict],
        max_speakers: int,
        audio_path: str,
    ) -> Optional[list[dict]]:
        """
        Send audio to external GPU server for diarization (NeMo/pyannote).
        Audio is already 16kHz mono WAV (converted by _ensure_wav upfront).

        Uses word-level timestamps (when available) for fine-grained speaker
        mapping instead of whole-segment overlap.

        Returns labeled segments or None if unavailable.
        """
        try:
            from shared.utils.external_apis import create_diarize_client
        except ImportError:
            logger.warning("External GPU diarization: import failed")
            return None

        client = create_diarize_client()
        if client is None:
            logger.warning("External GPU diarization: client creation failed (health check?)")
            return None

        mt = self._meeting_type or "meeting"
        config = self.SPEAKER_DEFAULTS.get(mt, {"default": 3, "min": 2, "max": 8})
        min_spk = config["min"]
        max_spk = max(max_speakers, min_spk)
        num_spk = self._num_speakers or 0
        clustering_threshold = DIARIZE_CLUSTERING_THRESHOLD

        logger.info(
            f"External GPU diarization: sending {Path(audio_path).name}, "
            f"speakers={min_spk}-{max_spk} (hint={num_spk or 'auto'}), "
            f"threshold={clustering_threshold}"
        )

        # audio_path is already WAV from _ensure_wav() — no conversion needed
        result = client.diarize(
            audio_path,
            min_speakers=min_spk,
            max_speakers=max_spk,
            num_speakers=num_spk,
            clustering_threshold=clustering_threshold,
        )

        if result is None:
            return None

        gpu_segments = result.get("segments", [])
        if not gpu_segments:
            logger.warning("GPU diarization returned 0 segments")
            return None

        # Filter out micro-fragments (<100ms) from diarize output — these are
        # pyannote noise at speaker boundaries and cause misassignment.
        MIN_DIARIZE_DURATION_S = 0.1
        filtered = [ds for ds in gpu_segments if (ds["end"] - ds["start"]) >= MIN_DIARIZE_DURATION_S]
        if filtered:
            dropped = len(gpu_segments) - len(filtered)
            if dropped:
                logger.info(f"Filtered {dropped} micro-fragments (<{MIN_DIARIZE_DURATION_S}s) from diarize output")
            gpu_segments = filtered

        # Normalize speaker names: SPEAKER_XX -> Speaker_X
        speaker_map = {}
        for ds in gpu_segments:
            raw = ds["speaker"]
            if raw not in speaker_map:
                speaker_map[raw] = f"Speaker_{len(speaker_map)}"
            ds["_norm_speaker"] = speaker_map[raw]

        # ── Word-level speaker mapping (preferred) ──
        # When Whisper provides word timestamps, map each WORD to the diarize
        # timeline individually, then assign each segment's speaker by majority
        # vote of its words. This is much more accurate at turn boundaries than
        # mapping whole segments.
        has_words = any(seg.get("words") for seg in segments)

        for seg in segments:
            words = seg.get("words", [])

            if has_words and words:
                # Map each word to a diarize speaker
                word_speakers = []
                for w in words:
                    w_mid_ms = ((w.get("start", 0) + w.get("end", 0)) / 2.0) * 1000
                    best_spk = None
                    best_overlap = 0
                    for ds in gpu_segments:
                        ds_start_ms = ds["start"] * 1000
                        ds_end_ms = ds["end"] * 1000
                        w_start_ms = w.get("start", 0) * 1000
                        w_end_ms = w.get("end", 0) * 1000
                        ov_start = max(w_start_ms, ds_start_ms)
                        ov_end = min(w_end_ms, ds_end_ms)
                        ov = max(0, ov_end - ov_start)
                        if ov > best_overlap:
                            best_overlap = ov
                            best_spk = ds["_norm_speaker"]
                    # Fallback: midpoint proximity
                    if best_spk is None:
                        best_dist = float("inf")
                        for ds in gpu_segments:
                            ds_mid_ms = (ds["start"] + ds["end"]) / 2.0 * 1000
                            d = abs(w_mid_ms - ds_mid_ms)
                            if d < best_dist:
                                best_dist = d
                                best_spk = ds["_norm_speaker"]
                    if best_spk:
                        word_speakers.append(best_spk)

                # Majority vote — assign segment to the speaker who owns most words
                if word_speakers:
                    from collections import Counter
                    vote = Counter(word_speakers).most_common(1)[0][0]
                    seg["speaker"] = vote
                else:
                    seg["speaker"] = "Speaker_0"
            else:
                # Fallback: segment-level overlap mapping (original approach)
                seg_mid = (seg["start_ms"] + seg["end_ms"]) / 2.0
                best_speaker = "Speaker_0"
                best_overlap = 0

                for ds in gpu_segments:
                    ds_start_ms = int(ds["start"] * 1000)
                    ds_end_ms = int(ds["end"] * 1000)
                    overlap_start = max(seg["start_ms"], ds_start_ms)
                    overlap_end = min(seg["end_ms"], ds_end_ms)
                    overlap = max(0, overlap_end - overlap_start)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_speaker = ds["_norm_speaker"]

                if best_overlap == 0:
                    best_dist = float("inf")
                    for ds in gpu_segments:
                        ds_mid = (ds["start"] + ds["end"]) / 2.0 * 1000
                        dist = abs(seg_mid - ds_mid)
                        if dist < best_dist:
                            best_dist = dist
                            best_speaker = ds["_norm_speaker"]

                seg["speaker"] = best_speaker

        # Clean up temp key
        for ds in gpu_segments:
            ds.pop("_norm_speaker", None)

        # Apply linguistic post-correction (Q→A, isolated flip, greeting→response).
        # This was previously missing for external GPU path — fixes cases where
        # pyannote merges two similar voices into one speaker but Q&A patterns
        # reveal they are different speakers.
        num_spk_detected = len(set(s["speaker"] for s in segments))
        segments = self._linguistic_post_correction(segments, num_spk_detected)

        speakers = sorted(set(s["speaker"] for s in segments))
        counts = {sp: sum(1 for s in segments if s["speaker"] == sp) for sp in speakers}
        self._last_diarization_backend = "external_gpu"
        logger.info(
            f"External GPU diarization mapped: {len(speakers)} speakers — "
            + ", ".join(f"{sp}: {counts[sp]} segs" for sp in speakers)
            + f" (clustering_threshold={clustering_threshold})"
        )
        return segments

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
            # Ensure waveform is mono
            if y.ndim > 1:
                y = np.mean(y, axis=0) if y.shape[0] < y.shape[1] else np.mean(y, axis=1)
            # Convert to torch tensor
            waveform = torch.from_numpy(y).float()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            # Resample to 16 kHz if needed
            if sr != 16000:
                import torchaudio as _ta
                waveform = _ta.functional.resample(waveform, sr, 16000)
                sr = 16000
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


        # --- Split transcript segments at diarization boundaries ---
        diar_boundaries = sorted(set([turn["start"] for turn in speaker_turns] + [turn["end"] for turn in speaker_turns]))
        diar_boundaries = [b for b in diar_boundaries if b is not None]
        new_segments = []
        for seg in segments:
            seg_start = seg["start_ms"] / 1000
            seg_end = seg["end_ms"] / 1000
            # Find all diarization boundaries within this segment
            split_points = [b for b in diar_boundaries if seg_start < b < seg_end]
            split_times = [seg_start] + split_points + [seg_end]
            for i in range(len(split_times) - 1):
                chunk_start = split_times[i]
                chunk_end = split_times[i+1]
                # Assign speaker for this chunk
                spk = self._get_dominant_speaker(chunk_start, chunk_end, speaker_turns) or "SPEAKER_00"
                # Split words if present
                chunk_words = []
                if seg.get("words"):
                    for word in seg["words"]:
                        w_start = word.get("start", seg_start)
                        w_end = word.get("end", seg_end)
                        if w_start < chunk_end and w_end > chunk_start:
                            chunk_words.append(word)
                new_segments.append({
                    "start_ms": int(chunk_start * 1000),
                    "end_ms": int(chunk_end * 1000),
                    "text": seg["text"],
                    "words": chunk_words,
                    "speaker": spk,
                })
        segments = new_segments

        # ── Normalize labels to Speaker_0, Speaker_1, ... ──
        label_map = {}
        counter = 0
        for seg in segments:
            raw = seg.get("speaker", "")
            if raw not in label_map:
                label_map[raw] = f"Speaker_{counter}"
                counter += 1
            seg["speaker"] = label_map[raw]

        # ── Layer 2: Linguistic post-correction ──
        segments = self._linguistic_post_correction(segments, num_speakers)

        # ── Layer 3: Pitch Kalman filter (optional) ──
        if os.getenv("DIARIZATION_KALMAN", "false").lower() == "true":
            segments = self._apply_pitch_kalman_correction(segments, audio_path)

        # ── Layer 4: Evidence-based LLM correction (optional) ──
        if os.getenv("DIARIZATION_LLM", "false").lower() == "true":
            evidence = self._collect_evidence(segments, audio_path, pyannote_output=diarization)
            meeting_type = getattr(self, "_meeting_type", "sales_call") or "sales_call"
            segments = self._apply_llm_correction(segments, meeting_type, evidence=evidence)

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

    def _linguistic_post_correction(
        self,
        segments: list[dict],
        num_speakers: int = 2,
    ) -> list[dict]:
        """
        Fix obvious diarization errors using linguistic patterns.
        Works for 2+ speakers (not limited to 2-speaker calls).

        Rule 1: Question → Answer — flip response to most recent different speaker
        Rule 2: Isolated flip — short segment breaking a consistent run
        Rule 3: Greeting → Query — intro followed by question from same speaker
        Rule 4: Greeting → Response — greeting followed by polite reply
        Rule 5: Self-introduction detection — "I'm X" assigns speaker identity
        Rule 6: Addressed-by-name — "X, can you..." → next speaker is X
        """
        if len(segments) < 3:
            return segments

        speakers = set(seg.get("speaker", "") for seg in segments)
        if len(speakers) < 2:
            return segments

        def recent_different_speaker(idx: int, curr_spk: str) -> str:
            """Find the most recent speaker before idx that isn't curr_spk."""
            for j in range(idx - 1, -1, -1):
                spk = segments[j].get("speaker", "")
                if spk and spk != curr_spk:
                    return spk
            # Fallback: pick any other speaker
            for spk in sorted(speakers):
                if spk != curr_spk:
                    return spk
            return curr_spk

        corrections = 0

        # Rule 1: Question → response pattern
        # Short question + short answer from same speaker with gap → flip answer
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
                nxt["speaker"] = recent_different_speaker(i + 1, curr["speaker"])
                corrections += 1

        # Rule 2: Isolated single-segment flip
        # A-B-A where B is short and tiny gaps → B should be A
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
                nxt["speaker"] = recent_different_speaker(i + 1, curr["speaker"])
                corrections += 1

        # Rule 4: Greeting → response pattern
        for i in range(len(segments) - 1):
            curr = segments[i]
            nxt = segments[i + 1]
            if curr.get("speaker") != nxt.get("speaker"):
                continue

            curr_text = curr.get("text", "").strip().lower()
            nxt_text = nxt.get("text", "").strip().lower()
            nxt_duration = nxt["end_ms"] - nxt["start_ms"]
            gap_ms = nxt["start_ms"] - curr["end_ms"]

            polite_responses = [
                "i'm good", "i'm fine", "i'm great", "i'm doing",
                "no problem", "no worries", "sure", "of course",
            ]
            is_greeting_or_q = (
                any(curr_text.startswith(p) for p in ["hi", "hello", "hey", "how are"])
                or curr_text.endswith("?")
            )
            is_polite = (
                any(nxt_text.startswith(p) for p in polite_responses)
                and nxt_duration < 2000
                and len(nxt_text.split()) <= 6
            )

            if is_greeting_or_q and is_polite and gap_ms < 500:
                nxt["speaker"] = recent_different_speaker(i + 1, curr["speaker"])
                corrections += 1

        # Rule 5: Addressed-by-name detection
        # "Lucy, can you..." or "Sue, what do you think?" → next segment is that person
        # Build name→speaker map from self-introductions first
        name_to_speaker: dict[str, str] = {}
        import re
        intro_pattern = re.compile(
            r"(?:i'm|i am|my name is|this is)\s+(\w+)",
            re.IGNORECASE,
        )
        for seg in segments:
            match = intro_pattern.search(seg.get("text", ""))
            if match:
                name = match.group(1).capitalize()
                name_to_speaker[name] = seg.get("speaker", "")

        # Now check for addressed-by-name patterns
        if name_to_speaker:
            address_pattern = re.compile(
                r"^(" + "|".join(re.escape(n) for n in name_to_speaker) + r")[,\s]",
                re.IGNORECASE,
            )
            for i in range(len(segments) - 1):
                curr = segments[i]
                nxt = segments[i + 1]
                curr_text = curr.get("text", "").strip()
                match = address_pattern.match(curr_text)
                if match:
                    addressed_name = match.group(1).capitalize()
                    expected_speaker = name_to_speaker.get(addressed_name)
                    if (
                        expected_speaker
                        and expected_speaker != curr.get("speaker")
                        and nxt.get("speaker") != expected_speaker
                    ):
                        gap_ms = nxt["start_ms"] - curr["end_ms"]
                        if gap_ms < 2000:
                            nxt["speaker"] = expected_speaker
                            corrections += 1

        if corrections > 0:
            logger.info(f"Linguistic post-correction: {corrections} fixes applied")

        return segments

    # ── Layer 3: Pitch Kalman Filter Post-Correction ──

    def _apply_pitch_kalman_correction(
        self,
        segments: list[dict],
        audio_path: str,
    ) -> list[dict]:
        """
        Scan diarized segments for pitch (F0) discontinuities that suggest
        a missed speaker change. Uses a Kalman filter to model pitch continuity
        and flags/corrects segments where F0 jumps unpredictably.

        Hogg et al. (IEEE ICASSP 2019): Kalman-based speaker change detection
        improved from 43.3% to 70.5% on AMI corpus.

        High confidence (>0.7): flip speaker directly.
        Medium confidence (0.4-0.7): flag for LLM layer.
        """
        try:
            from filterpy.kalman import KalmanFilter as KF
        except ImportError:
            logger.debug("filterpy not installed — skipping pitch Kalman correction")
            return segments

        if len(segments) < 3:
            return segments

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

        # Extract full pitch track
        pitch_times, pitch_values = self._extract_pitch_track(y, sr)
        if pitch_times is None or len(pitch_times) < 10:
            logger.debug("Insufficient pitch data — skipping Kalman correction")
            return segments

        segments = list(segments)
        corrections = 0
        speakers = set(seg.get("speaker", "") for seg in segments)
        if len(speakers) < 2:
            return segments
        speaker_list = sorted(speakers)

        def other_speaker(spk: str) -> str:
            return speaker_list[1] if spk == speaker_list[0] else speaker_list[0]

        for i in range(len(segments) - 1):
            curr = segments[i]
            nxt = segments[i + 1]

            # Only check consecutive same-speaker segments with small gaps
            if curr.get("speaker") != nxt.get("speaker"):
                continue
            gap_s = (nxt["start_ms"] - curr["end_ms"]) / 1000.0
            if gap_s > 0.5:
                continue

            boundary_s = curr["end_ms"] / 1000.0
            change_detected, confidence = self._detect_pitch_change_kalman(
                KF, pitch_times, pitch_values,
                boundary_s - 0.3, boundary_s + 0.3, boundary_s
            )

            if change_detected and confidence > 0.4:
                # Always FLAG, never auto-flip. Let LLM layer decide.
                # Auto-flipping on phone audio causes too many false positives.
                nxt["pitch_kalman_suspicious"] = True
                nxt["pitch_kalman_confidence"] = confidence
                corrections += 1
                logger.info(
                    f"Pitch Kalman: F0 discontinuity at {boundary_s:.2f}s "
                    f"(conf={confidence:.2f}), flagging '{nxt.get('text', '')[:40]}'"
                )

        if corrections > 0:
            logger.info(f"Pitch Kalman: corrected {corrections} segment(s)")
        return segments

    def _extract_pitch_track(
        self, y: np.ndarray, sr: int, frame_step: float = 0.01,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract F0 pitch track. Uses Praat if available, else librosa.pyin."""
        # Try Praat (parselmouth) — better on phone audio
        try:
            import parselmouth
            from parselmouth.praat import call
            snd = parselmouth.Sound(y, sampling_frequency=sr)
            pitch_obj = call(snd, "To Pitch", frame_step, 75.0, 500.0)
            n_frames = call(pitch_obj, "Get number of frames")
            times, values = [], []
            for i in range(1, n_frames + 1):
                t = call(pitch_obj, "Get time from frame number", i)
                f0 = call(pitch_obj, "Get value in frame", i, "Hertz")
                times.append(t)
                values.append(f0 if not np.isnan(f0) else 0.0)
            return np.array(times), np.array(values)
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Praat pitch extraction failed: {e}")

        # Fallback: librosa.pyin
        try:
            import librosa
            f0, _, _ = librosa.pyin(y, fmin=75, fmax=500, sr=sr, hop_length=int(sr * frame_step))
            times = np.arange(len(f0)) * frame_step
            f0 = np.where(np.isnan(f0), 0.0, f0)
            return times, f0
        except Exception as e:
            logger.warning(f"Pitch extraction failed: {e}")
            return None, None

    @staticmethod
    def _detect_pitch_change_kalman(
        KF, pitch_times, pitch_values,
        start_time, end_time, boundary_time,
    ) -> tuple[bool, float]:
        """
        Run Kalman filter across a segment boundary and detect F0 spike.
        Returns (change_detected, confidence).
        """
        mask = (pitch_times >= start_time) & (pitch_times <= end_time)
        window_f0 = pitch_values[mask]
        window_times = pitch_times[mask]

        # Filter to voiced frames
        voiced_mask = window_f0 > 0
        voiced_f0 = window_f0[voiced_mask]
        voiced_times = window_times[voiced_mask]

        if len(voiced_f0) < 4:
            return False, 0.0

        before = voiced_f0[voiced_times < boundary_time]
        after = voiced_f0[voiced_times >= boundary_time]
        if len(before) < 2 or len(after) < 2:
            return False, 0.0

        # Kalman filter: state = [pitch, pitch_velocity]
        dt = 0.01
        kf = KF(dim_x=2, dim_z=1)
        kf.F = np.array([[1.0, dt], [0.0, 1.0]])
        kf.H = np.array([[1.0, 0.0]])
        kf.Q = np.array([[10.0, 0.0], [0.0, 50.0]])
        kf.R = np.array([[25.0]])
        kf.x = np.array([[voiced_f0[0]], [0.0]])
        kf.P = np.array([[100.0, 0.0], [0.0, 100.0]])

        innovations = []
        inn_times = []
        for j in range(1, len(voiced_f0)):
            kf.predict()
            z = np.array([[voiced_f0[j]]])
            innovation = float(z - kf.H @ kf.x)
            innovations.append(abs(innovation))
            inn_times.append(voiced_times[j])
            kf.update(z)

        if not innovations:
            return False, 0.0

        innovations = np.array(innovations)
        inn_times = np.array(inn_times)

        before_inn = innovations[inn_times < boundary_time]
        after_inn = innovations[inn_times >= boundary_time]
        if len(before_inn) == 0 or len(after_inn) == 0:
            return False, 0.0

        boundary_innovation = after_inn[0]
        baseline_mean = np.mean(before_inn)
        baseline_std = max(np.std(before_inn), 3.0)
        z_score = (boundary_innovation - baseline_mean) / baseline_std

        # Raw pitch jump
        pitch_before = np.median(before[-3:])
        pitch_after = np.median(after[:3])
        raw_jump = abs(pitch_after - pitch_before)

        # Confidence scoring
        confidence = 0.0
        if z_score > 3.0:
            confidence += 0.5
        elif z_score > 2.0:
            confidence += 0.3
        elif z_score > 1.5:
            confidence += 0.15

        if raw_jump > 40:
            confidence += 0.4
        elif raw_jump > 25:
            confidence += 0.25
        elif raw_jump > 15:
            confidence += 0.1

        # Pitch ranges separate?
        if np.min(after[:3]) > np.max(before[-3:]) or np.min(before[-3:]) > np.max(after[:3]):
            confidence += 0.15

        confidence = min(confidence, 1.0)
        return confidence > 0.4, confidence

    # ── Layer 4: Evidence-Based LLM Post-Correction ──

    def _collect_evidence(
        self,
        segments: list[dict],
        audio_path: str,
        pyannote_output=None,
    ) -> list[dict]:
        """
        Build multi-signal evidence card per segment. Fuses:
        - Pyannote frame-level probability
        - Pitch median + Kalman flag
        - Linguistic patterns
        Then assigns confidence tier: HIGH / MEDIUM / LOW.
        """
        from collections import Counter

        # ── Load audio ──
        if self._audio_data is not None:
            y, sr = self._audio_data
        else:
            import librosa
            y, sr = librosa.load(audio_path, sr=16000, mono=True)

        # ── Pitch data per segment ──
        pitch_times, pitch_values = self._extract_pitch_track(y, sr)
        seg_pitch: dict[int, float] = {}
        if pitch_times is not None:
            for i, seg in enumerate(segments):
                start_s = seg["start_ms"] / 1000
                end_s = seg["end_ms"] / 1000
                mask = (pitch_times >= start_s) & (pitch_times <= end_s) & (pitch_values > 0)
                voiced = pitch_values[mask]
                seg_pitch[i] = float(np.median(voiced)) if len(voiced) > 2 else 0.0

        # ── Speaker pitch ranges (from long, high-confidence segments) ──
        spk_pitches: dict[str, list[float]] = {}
        for i, seg in enumerate(segments):
            f0 = seg_pitch.get(i, 0)
            dur = seg["end_ms"] - seg["start_ms"]
            if f0 > 0 and dur > 1000:
                spk = seg.get("speaker", "")
                spk_pitches.setdefault(spk, []).append(f0)
        spk_pitch_ranges = {}
        for spk, vals in spk_pitches.items():
            if len(vals) >= 2:
                spk_pitch_ranges[spk] = (float(np.percentile(vals, 20)), float(np.percentile(vals, 80)))

        # ── Build evidence cards ──
        evidence = []

        for i, seg in enumerate(segments):
            ev: dict[str, Any] = {
                "pyannote_speaker": seg.get("speaker", ""),
                "pyannote_probability": 0.5,
                "median_pitch_hz": seg_pitch.get(i, 0.0),
                "pitch_matches": "",
                "kalman_suspicious": seg.get("pitch_kalman_suspicious", False),
                "kalman_confidence": seg.get("pitch_kalman_confidence", 0.0),
                "duration_ms": seg["end_ms"] - seg["start_ms"],
                "gap_ms": (seg["start_ms"] - segments[i - 1]["end_ms"]) if i > 0 else 0,
                "word_count": len(seg.get("text", "").split()),
                "linguistic": [],
                "confidence_tier": "MEDIUM",
                "agreement": 0.0,
            }

            # Pyannote confidence from raw output
            if pyannote_output is not None:
                try:
                    start_s = seg["start_ms"] / 1000
                    end_s = seg["end_ms"] / 1000
                    dur = end_s - start_s
                    overlap = 0.0
                    for turn, _, spk_label in pyannote_output.itertracks(yield_label=True):
                        if spk_label != seg.get("_raw_pyannote_label", seg.get("speaker", "")):
                            continue
                        o = max(0, min(end_s, turn.end) - max(start_s, turn.start))
                        overlap += o
                    ev["pyannote_probability"] = round(min(overlap / max(dur, 0.001), 1.0), 2)
                except Exception:
                    pass

            # Pitch match
            f0 = ev["median_pitch_hz"]
            if f0 > 0 and spk_pitch_ranges:
                for spk, (lo, hi) in spk_pitch_ranges.items():
                    if lo <= f0 <= hi:
                        ev["pitch_matches"] = spk
                        break
                if not ev["pitch_matches"]:
                    best_spk, best_dist = "", 999
                    for spk, (lo, hi) in spk_pitch_ranges.items():
                        d = min(abs(f0 - lo), abs(f0 - hi))
                        if d < best_dist:
                            best_dist, best_spk = d, spk
                    if best_dist < 30:
                        ev["pitch_matches"] = best_spk

            # Linguistic patterns
            text = seg.get("text", "").strip().lower()
            if text.endswith("?"):
                ev["linguistic"].append("question")
            if any(g in text for g in ["good morning", "hi,", "hello,", "this is "]):
                ev["linguistic"].append("greeting")
            if any(o in text for o in ["not looking", "not interested", "busy", "don't need"]):
                ev["linguistic"].append("objection")
            if ev["word_count"] <= 4 and ev["duration_ms"] < 1500:
                ev["linguistic"].append("short_response")

            # ── Confidence tier ──
            votes = []
            if ev["pyannote_probability"] > 0.6:
                votes.append(ev["pyannote_speaker"])
            if ev["pitch_matches"]:
                votes.append(ev["pitch_matches"])

            agreement = 0.0
            if votes:
                majority = Counter(votes).most_common(1)[0][1]
                agreement = majority / len(votes)
            ev["agreement"] = round(agreement, 2)

            if ev["pyannote_probability"] >= 0.85 and not ev["kalman_suspicious"]:
                ev["confidence_tier"] = "HIGH"
            elif agreement >= 0.9 and len(votes) >= 2:
                ev["confidence_tier"] = "HIGH"
            elif ev["kalman_suspicious"] or ev["pyannote_probability"] < 0.6:
                ev["confidence_tier"] = "LOW"
            elif agreement < 0.5 and len(votes) >= 2:
                ev["confidence_tier"] = "LOW"
            else:
                ev["confidence_tier"] = "MEDIUM"

            # Short segments with tiny gaps are unreliable
            if ev["duration_ms"] < 600 and ev["gap_ms"] < 100 and ev["confidence_tier"] == "HIGH":
                ev["confidence_tier"] = "MEDIUM"

            # Short polite responses after greetings/questions — downgrade so LLM can review
            if i > 0 and ev["duration_ms"] < 2000 and ev["gap_ms"] < 500:
                prev_text = segments[i - 1].get("text", "").strip().lower()
                curr_text_low = seg.get("text", "").strip().lower()
                prev_is_greeting_or_q = (
                    any(prev_text.startswith(p) for p in ["hi", "hello", "hey", "how are", "good morning"])
                    or prev_text.endswith("?")
                )
                curr_is_polite = any(curr_text_low.startswith(p) for p in [
                    "i'm good", "i'm fine", "i'm great", "no problem", "no worries",
                    "sure", "of course",
                ])
                if prev_is_greeting_or_q and curr_is_polite and ev["confidence_tier"] == "HIGH":
                    ev["confidence_tier"] = "MEDIUM"

            evidence.append(ev)

        tiers = Counter(e["confidence_tier"] for e in evidence)
        logger.info(f"Evidence tiers: {dict(tiers)}")
        return evidence

    def _apply_llm_correction(
        self,
        segments: list[dict],
        meeting_type: str = "sales_call",
        evidence: Optional[list[dict]] = None,
    ) -> list[dict]:
        """
        Evidence-based LLM post-correction. The LLM sees structured evidence
        cards with confidence tiers and can ONLY flip LOW-tier segments.
        HIGH-tier flips are blocked at the code level even if the LLM tries.
        """
        import re

        openai_key = os.environ.get("OPENAI_API_KEY", "")
        if not openai_key or len(segments) < 2:
            return segments

        # Skip if no LOW-confidence segments
        if evidence:
            low_count = sum(1 for e in evidence if e["confidence_tier"] == "LOW")
            if low_count == 0:
                logger.info("LLM correction: no LOW-confidence segments — skipping")
                return segments

        try:
            import httpx
        except ImportError:
            return segments

        try:
            # Build evidence cards
            cards = []
            for i, seg in enumerate(segments):
                text = seg.get("text", "").strip()
                ev = evidence[i] if evidence and i < len(evidence) else {}

                tier = ev.get("confidence_tier", "MEDIUM")
                card = (
                    f"--- Segment {i+1} ---\n"
                    f"Speaker: {seg.get('speaker', '')}\n"
                    f'Text: "{text}"\n'
                    f"Confidence: {tier}\n"
                    f"Duration: {ev.get('duration_ms', 0)}ms | Gap: {ev.get('gap_ms', 0)}ms\n"
                    f"Acoustic signals:\n"
                    f"  pyannote: {ev.get('pyannote_speaker', '?')} (conf={ev.get('pyannote_probability', 0):.0%})\n"
                    f"  pitch: {ev.get('median_pitch_hz', 0):.0f}Hz -> matches {ev.get('pitch_matches', '?')}\n"
                    f"  kalman: {'FLAGGED' if ev.get('kalman_suspicious') else 'clean'}\n"
                    f"  agreement: {ev.get('agreement', 0):.0%}\n"
                    f"Linguistic: {', '.join(ev.get('linguistic', [])) or 'none'}"
                )
                cards.append(card)

            high_count = sum(1 for e in (evidence or []) if e.get("confidence_tier") == "HIGH")
            low_count = sum(1 for e in (evidence or []) if e.get("confidence_tier") == "LOW")

            prompt = (
                f"You are a speaker diarization correction system analyzing a 2-person phone call.\n\n"
                f"Below are {len(segments)} segments with MULTI-SIGNAL EVIDENCE CARDS.\n\n"
                f"Context: {meeting_type}. Speaker_0=caller, Speaker_1=prospect.\n\n"
                f"CONFIDENCE TIERS:\n"
                f"- HIGH ({high_count} segments): Multiple acoustic signals agree. Almost certainly correct.\n"
                f"- MEDIUM: Some agreement.\n"
                f"- LOW ({low_count} segments): Signals disagree. Candidates for correction.\n\n"
                f"HARD RULES:\n"
                f"1. NEVER flip a segment marked 'Confidence: HIGH'\n"
                f"2. ONLY consider flipping 'Confidence: LOW' segments\n"
                f"3. Use conversational logic for LOW segments only\n"
                f"4. NEVER change the text in quotes\n"
                f"5. When uncertain, DO NOT flip\n"
                f"6. Output ONLY numbered lines: 1. [Speaker_X] \"text\"\n\n"
                + "\n\n".join(cards)
                + "\n\nOutput the corrected segment list. No explanations."
            )

            # Call LLM (OpenAI)
            response = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"},
                json={"model": "gpt-4o", "temperature": 0, "max_tokens": 2048,
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=20,
            )
            if response.status_code != 200:
                logger.warning(f"LLM API error: {response.status_code}")
                return segments
            completion = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")

            if not completion.strip():
                return segments

            # Parse response
            pattern = re.compile(r"^\d+\.\s*\[(Speaker_\d+)\]")
            new_labels = []
            for line in completion.strip().split("\n"):
                m = pattern.match(line.strip())
                if m:
                    new_labels.append(m.group(1))

            if len(new_labels) != len(segments):
                logger.warning(f"LLM returned {len(new_labels)} labels for {len(segments)} segments")
                return segments

            # ── CODE-LEVEL ENFORCEMENT: block HIGH-tier flips ──
            changes = 0
            blocked = 0
            for i, (seg, label) in enumerate(zip(segments, new_labels)):
                if seg.get("speaker") == label:
                    continue
                ev = evidence[i] if evidence and i < len(evidence) else {}
                if ev.get("confidence_tier") == "HIGH":
                    logger.info(
                        f"LLM tried to flip HIGH segment {i+1} "
                        f"'{seg.get('text', '')[:30]}' — BLOCKED"
                    )
                    blocked += 1
                    continue  # do NOT apply this flip
                seg["speaker"] = label
                changes += 1

            # Clean up flags
            for seg in segments:
                seg.pop("pitch_kalman_suspicious", None)
                seg.pop("pitch_kalman_confidence", None)
                seg.pop("_raw_pyannote_label", None)

            if changes > 0 or blocked > 0:
                logger.info(f"LLM correction v2: flipped {changes}, blocked {blocked} HIGH-tier")
            return segments

        except Exception as e:
            logger.warning(f"LLM correction failed: {e}")
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
                clustering_threshold=DIARIZE_CLUSTERING_THRESHOLD,
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
