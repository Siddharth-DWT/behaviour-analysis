"""
NEXUS Shared — External API Clients
Wraps the GPU-accelerated Whisper STT and Coqui TTS services
running on the external server (110.227.200.12).

These are optional accelerators:
  - Whisper STT: GPU-powered transcription (replaces local faster-whisper)
  - Coqui TTS:  Neural TTS with voice cloning (replaces macOS `say`)

Environment variables:
  EXTERNAL_WHISPER_URL   e.g. http://110.227.200.12:8008
  EXTERNAL_TTS_URL       e.g. http://110.227.200.12:8009
  EXTERNAL_API_KEY       API key for both services
"""
import io
import os
import json
import time
import wave
import logging
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger("nexus.external_apis")

# ── Configuration ──────────────────────────────────────────────
WHISPER_URL = os.getenv("EXTERNAL_WHISPER_URL", "")
TTS_URL = os.getenv("EXTERNAL_TTS_URL", "")
API_KEY = os.getenv("EXTERNAL_API_KEY", "")

DEFAULT_WHISPER_MODEL = os.getenv("EXTERNAL_WHISPER_MODEL", "base")
DEFAULT_TIMEOUT = 120  # seconds


def is_whisper_available() -> bool:
    """Check if external Whisper STT API is configured."""
    return bool(WHISPER_URL and API_KEY)


def is_tts_available() -> bool:
    """Check if external Coqui TTS API is configured."""
    return bool(TTS_URL and API_KEY)


# ═══════════════════════════════════════════════════════════════
# WHISPER STT CLIENT
# ═══════════════════════════════════════════════════════════════

class WhisperClient:
    """
    Client for GPU-accelerated Whisper STT API.

    Usage:
        client = WhisperClient()
        if client.is_healthy():
            result = client.transcribe("path/to/audio.wav", model="large-v3")
            print(result["text"])
            print(result["segments"])  # with timestamps
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.base_url = (base_url or WHISPER_URL).rstrip("/")
        self.api_key = api_key or API_KEY
        self.model = model or DEFAULT_WHISPER_MODEL
        self.timeout = timeout

        if not self.base_url:
            raise ValueError(
                "Whisper URL not configured. Set EXTERNAL_WHISPER_URL env var."
            )
        if not self.api_key:
            raise ValueError(
                "API key not configured. Set EXTERNAL_API_KEY env var."
            )

    @property
    def _headers(self) -> dict:
        return {"X-API-Key": self.api_key}

    def is_healthy(self) -> bool:
        """Check if the Whisper service is running and GPU is available."""
        try:
            resp = httpx.get(
                f"{self.base_url}/health",
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                healthy = data.get("status") == "healthy"
                gpu = data.get("gpu_available", False)
                if healthy:
                    logger.info(
                        f"Whisper API healthy: GPU={data.get('gpu_name', '?')}, "
                        f"models={data.get('loaded_models', [])}"
                    )
                return healthy
        except Exception as e:
            logger.warning(f"Whisper health check failed: {e}")
        return False

    def get_models(self) -> list[dict]:
        """List available Whisper models."""
        resp = httpx.get(f"{self.base_url}/models", timeout=10)
        resp.raise_for_status()
        return resp.json().get("available_models", [])

    def transcribe(
        self,
        audio_path: str,
        model: Optional[str] = None,
        language: str = "en",
        word_timestamps: bool = True,
        vad_filter: bool = True,
        stream: bool = False,
    ) -> dict:
        """
        Transcribe an audio file via the external Whisper API.

        Args:
            audio_path: Path to audio file (WAV, MP3, etc.)
            model: Whisper model (tiny/base/small/medium/large-v2/large-v3)
            language: Language code or "auto" for auto-detect
            word_timestamps: Include per-word timestamps
            vad_filter: Enable voice activity detection
            stream: If True, returns segments as they complete (NDJSON)

        Returns:
            {
                "text": str,           # Full transcribed text
                "segments": [{         # Timestamped segments
                    "id": int,
                    "start": float,    # seconds
                    "end": float,
                    "text": str,
                    "words": [{        # if word_timestamps=True
                        "word": str,
                        "start": float,
                        "end": float,
                        "probability": float
                    }],
                    "avg_logprob": float,
                    "no_speech_prob": float,
                }],
                "language": str,
                "language_probability": float,
                "duration": float,
                "processing_time": float,
                "model": str,
            }
        """
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        use_model = model or self.model
        logger.info(
            f"Transcribing via external Whisper API: "
            f"{audio_file.name} (model={use_model}, lang={language})"
        )

        start_time = time.time()

        with open(audio_path, "rb") as f:
            files = {"file": (audio_file.name, f, "audio/wav")}
            data = {
                "model": use_model,
                "language": language if language != "auto" else "",
                "word_timestamps": str(word_timestamps).lower(),
                "vad_filter": str(vad_filter).lower(),
                "stream": str(stream).lower(),
            }

            if stream:
                return self._transcribe_stream(files, data)

            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    f"{self.base_url}/transcribe",
                    files=files,
                    data=data,
                    headers=self._headers,
                )

        elapsed = time.time() - start_time

        if resp.status_code != 200:
            raise RuntimeError(
                f"Whisper API error ({resp.status_code}): {resp.text[:500]}"
            )

        result = resp.json()
        result["_client_elapsed"] = elapsed

        seg_count = len(result.get("segments", []))
        logger.info(
            f"Transcription complete: {result.get('duration', 0):.1f}s audio, "
            f"{seg_count} segments, model={use_model}, "
            f"API time={result.get('processing_time', 0):.2f}s, "
            f"total={elapsed:.2f}s"
        )

        return result

    def _transcribe_stream(self, files: dict, data: dict) -> dict:
        """Streaming transcription — returns segments as they complete."""
        segments = []
        done_data = {}

        with httpx.Client(timeout=self.timeout) as client:
            with client.stream(
                "POST",
                f"{self.base_url}/transcribe",
                files=files,
                data=data,
                headers=self._headers,
            ) as resp:
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"Whisper stream error ({resp.status_code})"
                    )

                for line in resp.iter_lines():
                    if line.strip():
                        obj = json.loads(line)
                        if obj.get("done"):
                            done_data = obj
                        else:
                            segments.append(obj)
                            logger.debug(
                                f"Stream segment: [{obj.get('start', 0):.1f}s] "
                                f"{obj.get('text', '')[:60]}"
                            )

        # Reconstruct a response matching the non-stream format
        full_text = " ".join(seg.get("text", "").strip() for seg in segments)
        return {
            "text": full_text,
            "segments": segments,
            "language": done_data.get("language", "en"),
            "language_probability": done_data.get("language_probability", 0),
            "duration": done_data.get("duration", 0),
            "processing_time": done_data.get("processing_time", 0),
            "model": data.get("model", "base"),
        }

    def detect_language(self, audio_path: str) -> dict:
        """Detect the language of an audio file."""
        with open(audio_path, "rb") as f:
            files = {"file": (Path(audio_path).name, f, "audio/wav")}
            data = {"model": self.model}

            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    f"{self.base_url}/detect-language",
                    files=files,
                    data=data,
                    headers=self._headers,
                )

        resp.raise_for_status()
        return resp.json()

    def translate(self, audio_path: str) -> dict:
        """Transcribe audio from any language and translate to English."""
        with open(audio_path, "rb") as f:
            files = {"file": (Path(audio_path).name, f, "audio/wav")}
            data = {"model": self.model}

            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    f"{self.base_url}/translate",
                    files=files,
                    data=data,
                    headers=self._headers,
                )

        resp.raise_for_status()
        return resp.json()


# ═══════════════════════════════════════════════════════════════
# COQUI TTS CLIENT
# ═══════════════════════════════════════════════════════════════

class TTSClient:
    """
    Client for GPU-accelerated Coqui TTS API (XTTS v2).

    Usage:
        client = TTSClient()
        if client.is_healthy():
            # Default voice
            audio_bytes = client.synthesize("Hello world", language="en")

            # Voice cloning
            audio_bytes = client.synthesize_clone(
                "Hello world",
                speaker_wav_path="path/to/speaker_sample.wav",
                language="en",
            )
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.base_url = (base_url or TTS_URL).rstrip("/")
        self.api_key = api_key or API_KEY
        self.timeout = timeout

        if not self.base_url:
            raise ValueError(
                "TTS URL not configured. Set EXTERNAL_TTS_URL env var."
            )
        if not self.api_key:
            raise ValueError(
                "API key not configured. Set EXTERNAL_API_KEY env var."
            )

    @property
    def _headers(self) -> dict:
        return {"X-API-Key": self.api_key}

    def is_healthy(self) -> bool:
        """Check if the TTS service is running."""
        try:
            resp = httpx.get(f"{self.base_url}/health", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                healthy = data.get("status") == "healthy"
                if healthy:
                    logger.info(
                        f"TTS API healthy: GPU={data.get('gpu_name', '?')}, "
                        f"models={data.get('loaded_models', [])}"
                    )
                return healthy
        except Exception as e:
            logger.warning(f"TTS health check failed: {e}")
        return False

    def synthesize(
        self,
        text: str,
        language: str = "en",
        stream: bool = False,
    ) -> bytes:
        """
        Synthesize speech from text using the default voice.

        Args:
            text: Text to synthesize
            language: Language code (en, es, fr, de, etc.)
            stream: If True, returns chunked audio

        Returns:
            WAV audio bytes
        """
        if not text.strip():
            raise ValueError("Text cannot be empty")

        logger.info(f"TTS synthesize: '{text[:60]}...' (lang={language})")

        data = {
            "text": text,
            "language": language,
        }

        if stream:
            data["stream"] = "true"

        start_time = time.time()

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.base_url}/tts",
                data=data,
                headers=self._headers,
            )

        elapsed = time.time() - start_time

        if resp.status_code != 200:
            raise RuntimeError(
                f"TTS API error ({resp.status_code}): {resp.text[:500]}"
            )

        logger.info(
            f"TTS complete: {len(resp.content)} bytes, {elapsed:.2f}s"
        )

        return resp.content

    def synthesize_clone(
        self,
        text: str,
        speaker_wav_path: str,
        language: str = "en",
    ) -> bytes:
        """
        Synthesize speech using voice cloning from a reference speaker WAV.

        Args:
            text: Text to synthesize
            speaker_wav_path: Path to speaker reference WAV (6+ seconds ideal)
            language: Language code

        Returns:
            WAV audio bytes
        """
        if not text.strip():
            raise ValueError("Text cannot be empty")

        speaker_file = Path(speaker_wav_path)
        if not speaker_file.exists():
            raise FileNotFoundError(
                f"Speaker WAV not found: {speaker_wav_path}"
            )

        logger.info(
            f"TTS clone: '{text[:60]}...' using voice from {speaker_file.name}"
        )

        start_time = time.time()

        with open(speaker_wav_path, "rb") as f:
            files = {"speaker_wav": (speaker_file.name, f, "audio/wav")}
            data = {
                "text": text,
                "language": language,
            }

            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    f"{self.base_url}/tts-clone",
                    files=files,
                    data=data,
                    headers=self._headers,
                )

        elapsed = time.time() - start_time

        if resp.status_code != 200:
            raise RuntimeError(
                f"TTS clone error ({resp.status_code}): {resp.text[:500]}"
            )

        logger.info(
            f"TTS clone complete: {len(resp.content)} bytes, {elapsed:.2f}s"
        )

        return resp.content

    def synthesize_to_file(
        self,
        text: str,
        output_path: str,
        language: str = "en",
        speaker_wav_path: Optional[str] = None,
    ) -> Path:
        """
        Synthesize speech and save to a WAV file.

        Args:
            text: Text to synthesize
            output_path: Where to save the WAV file
            language: Language code
            speaker_wav_path: Optional speaker reference for voice cloning

        Returns:
            Path to the saved WAV file
        """
        if speaker_wav_path:
            audio_bytes = self.synthesize_clone(
                text, speaker_wav_path, language
            )
        else:
            audio_bytes = self.synthesize(text, language)

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            f.write(audio_bytes)

        logger.info(f"Saved TTS audio: {out} ({len(audio_bytes)} bytes)")
        return out


# ═══════════════════════════════════════════════════════════════
# CONVENIENCE — Quick access for common patterns
# ═══════════════════════════════════════════════════════════════

def get_whisper_client(**kwargs) -> Optional[WhisperClient]:
    """
    Get a WhisperClient if the external API is configured.
    Returns None if not configured.
    """
    if not is_whisper_available():
        return None
    try:
        return WhisperClient(**kwargs)
    except ValueError:
        return None


def get_tts_client(**kwargs) -> Optional[TTSClient]:
    """
    Get a TTSClient if the external API is configured.
    Returns None if not configured.
    """
    if not is_tts_available():
        return None
    try:
        return TTSClient(**kwargs)
    except ValueError:
        return None
