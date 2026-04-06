"""
NEXUS Shared — External API Clients
Wraps the GPU-accelerated Whisper STT and Coqui TTS services
running on the external GPU server.

These are optional accelerators:
  - Whisper STT:   GPU-powered transcription (replaces local faster-whisper)
  - Coqui TTS:     Neural TTS with voice cloning (replaces macOS `say`)
  - Diarization:   GPU-powered pyannote speaker diarization (replaces local CPU pyannote/KMeans)

Environment variables:
  EXTERNAL_WHISPER_URL   e.g. http://your-gpu-server:8008
  EXTERNAL_TTS_URL       e.g. http://your-gpu-server:8009
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
DIARIZE_URL = os.getenv("EXTERNAL_DIARIZE_URL", "")
API_KEY = os.getenv("EXTERNAL_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")

DEFAULT_WHISPER_MODEL = os.getenv("EXTERNAL_WHISPER_MODEL", "base")
DEFAULT_TIMEOUT = 600  # 10 min for transcription (large files need upload + processing time)
DIARIZE_TIMEOUT = 600  # 10 min for long audio diarization


def is_whisper_available() -> bool:
    """Check if external Whisper STT API is configured."""
    return bool(WHISPER_URL)


def is_tts_available() -> bool:
    """Check if external Coqui TTS API is configured."""
    return bool(TTS_URL)


def is_assemblyai_available() -> bool:
    """Check if AssemblyAI API is configured."""
    return bool(ASSEMBLYAI_API_KEY)


def is_deepgram_available() -> bool:
    """Check if Deepgram API is configured."""
    return bool(DEEPGRAM_API_KEY)


def is_diarize_available() -> bool:
    """Check if external GPU diarization API is configured."""
    return bool(DIARIZE_URL)


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
    @property
    def _headers(self) -> dict:
        if self.api_key:
            return {"X-API-Key": self.api_key}
        return {}

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
    @property
    def _headers(self) -> dict:
        if self.api_key:
            return {"X-API-Key": self.api_key}
        return {}

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
# GPU DIARIZATION CLIENT
# ═══════════════════════════════════════════════════════════════

class DiarizeClient:
    """
    Client for GPU-accelerated speaker diarization API (pyannote on GPU).

    Usage:
        client = DiarizeClient()
        if client.is_healthy():
            result = client.diarize("path/to/audio.wav", min_speakers=2, max_speakers=4)
            print(result["speakers"])     # ["SPEAKER_00", "SPEAKER_01"]
            print(result["timeline"])     # [{"speaker": ..., "start": ..., "end": ...}]
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = DIARIZE_TIMEOUT,
    ):
        self.base_url = (base_url or DIARIZE_URL).rstrip("/")
        self.api_key = api_key or API_KEY
        self.timeout = timeout

        if not self.base_url:
            raise ValueError(
                "Diarize URL not configured. Set EXTERNAL_DIARIZE_URL env var."
            )

    @property
    def _headers(self) -> dict:
        if self.api_key:
            return {"X-API-Key": self.api_key}
        return {}

    def is_healthy(self) -> bool:
        """Check if the diarization service is running and GPU is available."""
        try:
            resp = httpx.get(f"{self.base_url}/health", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                healthy = data.get("status") == "healthy"
                if healthy:
                    logger.info(
                        f"Diarize API healthy: GPU={data.get('gpu_name', '?')}, "
                        f"pipeline_loaded={data.get('pipeline_loaded', False)}"
                    )
                return healthy
        except Exception as e:
            logger.warning(f"Diarize health check failed: {e}")
        return False

    def diarize(
        self,
        audio_path: str,
        min_speakers: int = 2,
        max_speakers: int = 8,
        num_speakers: int = 0,
        audio_data: Optional[tuple] = None,
        clustering_threshold: float = 0.0,
        **kwargs,
    ) -> dict:
        """
        Diarize an audio file via the external GPU API.

        The server expects WAV audio. If audio_data (numpy_array, sr) is
        provided, it is exported as in-memory WAV and sent directly.

        Args:
            audio_path: Path to audio file (used when audio_data is None)
            min_speakers: Minimum expected speakers
            max_speakers: Maximum expected speakers
            num_speakers: Exact count (0 = auto-detect using min/max)
            audio_data: Optional (numpy_array, sample_rate) tuple
            clustering_threshold: NeMo/pyannote clustering threshold (0 = server default)
            **kwargs: Extra params forwarded to API (e.g. backend, segmentation_threshold)

        Returns:
            {
                "speakers": ["SPEAKER_00", ...],
                "timeline": [{"speaker": ..., "start": ..., "end": ..., "duration": ...}],
                "num_speakers": int,
                "duration": float,
                "processing_time": float,
            }
        """
        audio_file = Path(audio_path)
        file_label = audio_file.name

        logger.info(
            f"Diarizing via external GPU API: {file_label} "
            f"(min={min_speakers}, max={max_speakers}, exact={num_speakers})"
        )

        start_time = time.time()

        data = {
            "min_speakers": str(min_speakers or 1),
            "max_speakers": str(max_speakers or 20),
            "num_speakers": str(num_speakers or 0),
            "backend": "nemo",
        }
        if clustering_threshold and clustering_threshold > 0:
            data["clustering_threshold"] = str(clustering_threshold)
        # Forward any extra kwargs (e.g. backend="nemo", segmentation_threshold=0.5)
        for k, v in kwargs.items():
            if v is not None:
                data[k] = str(v)

        if audio_data is not None:
            # Export pre-loaded numpy audio to in-memory WAV bytes
            import numpy as np
            y, sr = audio_data
            wav_buf = io.BytesIO()
            y_int16 = np.clip(y * 32767, -32768, 32767).astype(np.int16)
            with wave.open(wav_buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sr)
                wf.writeframes(y_int16.tobytes())
            wav_buf.seek(0)
            files = {"file": ("audio.wav", wav_buf, "audio/wav")}
            logger.debug(f"Sending pre-loaded audio as WAV ({len(y_int16)} samples, {sr} Hz)")
        else:
            if not audio_file.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            files = {"file": (file_label, open(audio_path, "rb"), "audio/wav")}

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    f"{self.base_url}/diarize",
                    files=files,
                    data=data,
                    headers=self._headers,
                )
        finally:
            # Close the file handle if we opened one
            if audio_data is None and "file" in files:
                files["file"][1].close()

        elapsed = time.time() - start_time

        if resp.status_code != 200:
            raise RuntimeError(
                f"Diarize API error ({resp.status_code}): {resp.text[:500]}"
            )

        result = resp.json()

        logger.info(
            f"GPU diarization complete: {result.get('num_speakers', '?')} speakers, "
            f"{len(result.get('timeline', []))} turns, "
            f"API time={result.get('processing_time', 0):.1f}s, "
            f"total={elapsed:.1f}s"
        )

        return result


# ═══════════════════════════════════════════════════════════════
# ASSEMBLYAI CLIENT (transcription + diarization in one call)
# ═══════════════════════════════════════════════════════════════

class AssemblyAIClient:
    """
    Transcription + speaker diarization via AssemblyAI Universal-3-Pro.

    AssemblyAI provides both transcription and diarization in a single
    async API call, replacing both Whisper and Deepgram/pyannote.

    Flow: Upload audio → Submit job → Poll until complete → Return result.

    Usage:
        client = AssemblyAIClient(api_key="...")
        result = client.transcribe_and_diarize("path/to/audio.mp4")
        print(result["segments"])    # transcript segments with speaker labels
        print(result["num_speakers"])
    """

    BASE_URL = "https://api.assemblyai.com/v2"

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
        poll_interval: int = 5,
    ):
        self.api_key = api_key or ASSEMBLYAI_API_KEY
        self.timeout = timeout
        self.poll_interval = poll_interval

        if not self.api_key:
            raise ValueError(
                "AssemblyAI API key not configured. Set ASSEMBLYAI_API_KEY env var."
            )

    @property
    def _headers(self) -> dict:
        return {"Authorization": self.api_key}

    def is_healthy(self) -> bool:
        """Check if AssemblyAI API is reachable."""
        try:
            resp = httpx.get(
                f"{self.BASE_URL}/transcript",
                headers=self._headers,
                params={"limit": 1},
                timeout=10,
            )
            healthy = resp.status_code in (200, 401, 403)
            if healthy:
                logger.info("AssemblyAI API reachable")
            return healthy
        except Exception as e:
            logger.warning(f"AssemblyAI health check failed: {e}")
        return False

    def _upload(self, audio_path: str) -> str:
        """Upload audio file and return the CDN URL."""
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Uploading to AssemblyAI: {audio_file.name}")
        start = time.time()

        with open(audio_path, "rb") as f:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    f"{self.BASE_URL}/upload",
                    headers={**self._headers, "Content-Type": "application/octet-stream"},
                    content=f,
                )

        if resp.status_code != 200:
            raise RuntimeError(f"AssemblyAI upload error ({resp.status_code}): {resp.text[:500]}")

        upload_url = resp.json()["upload_url"]
        logger.info(f"Upload complete in {time.time() - start:.1f}s")
        return upload_url

    def _submit(self, audio_url: str) -> str:
        """Submit a transcription job and return the transcript ID."""
        payload = {
            "audio_url": audio_url,
            "speaker_labels": True,
            "speech_models": ["universal-3-pro"],
        }

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.BASE_URL}/transcript",
                headers={**self._headers, "Content-Type": "application/json"},
                json=payload,
            )

        if resp.status_code != 200:
            raise RuntimeError(f"AssemblyAI submit error ({resp.status_code}): {resp.text[:500]}")

        data = resp.json()
        tid = data.get("id")
        if not tid:
            raise RuntimeError(f"AssemblyAI submit returned no ID: {data}")

        logger.info(f"AssemblyAI job submitted: {tid}")
        return tid

    def _poll(self, transcript_id: str, max_wait: int = 600) -> dict:
        """Poll until transcription is complete. Returns full response."""
        url = f"{self.BASE_URL}/transcript/{transcript_id}"
        elapsed = 0

        while elapsed < max_wait:
            time.sleep(self.poll_interval)
            elapsed += self.poll_interval

            with httpx.Client(timeout=30) as client:
                resp = client.get(url, headers=self._headers)

            if resp.status_code != 200:
                raise RuntimeError(f"AssemblyAI poll error ({resp.status_code})")

            data = resp.json()
            status = data.get("status")

            if status == "completed":
                logger.info(f"AssemblyAI transcription completed in {elapsed}s")
                return data
            elif status == "error":
                raise RuntimeError(f"AssemblyAI transcription failed: {data.get('error', 'unknown')}")

        raise TimeoutError(f"AssemblyAI transcription timed out after {max_wait}s")

    def transcribe_and_diarize(self, audio_path: str) -> dict:
        """
        Upload, transcribe, and diarize audio in one flow.

        Returns a result dict compatible with the voice agent pipeline:
            {
                "duration_seconds": float,
                "backend": "assemblyai",
                "model": "universal-3-pro",
                "segments": [
                    {
                        "speaker": "Speaker_0",
                        "start_ms": int,
                        "end_ms": int,
                        "text": str,
                        "words": [{"word": str, "start": float, "end": float, "probability": float}],
                    }
                ],
                "num_speakers": int,
                "speakers": ["Speaker_0", ...],
            }
        """
        start_time = time.time()
        file_label = Path(audio_path).name

        logger.info(f"AssemblyAI transcribe+diarize: {file_label}")

        # Upload → Submit → Poll
        upload_url = self._upload(audio_path)
        transcript_id = self._submit(upload_url)
        result = self._poll(transcript_id)

        total_time = time.time() - start_time

        # Convert AssemblyAI response to NEXUS internal format
        # Speaker labels: A→Speaker_0, B→Speaker_1, etc.
        speaker_map = {}

        segments = []
        for utt in result.get("utterances", []):
            raw_speaker = utt["speaker"]
            if raw_speaker not in speaker_map:
                speaker_map[raw_speaker] = f"Speaker_{len(speaker_map)}"
            speaker_label = speaker_map[raw_speaker]

            # Collect words for this utterance
            words = []
            for w in utt.get("words", []):
                words.append({
                    "word": w["text"],
                    "start": w["start"] / 1000.0,
                    "end": w["end"] / 1000.0,
                    "probability": round(w.get("confidence", 0), 3),
                })

            segments.append({
                "speaker": speaker_label,
                "start_ms": utt["start"],
                "end_ms": utt["end"],
                "text": utt["text"],
                "words": words,
            })

        duration = result.get("audio_duration", 0)
        speakers = sorted(set(speaker_map.values()))

        logger.info(
            f"AssemblyAI complete: {duration}s audio, {len(segments)} segments, "
            f"{len(speakers)} speakers, total={total_time:.1f}s"
        )

        return {
            "duration_seconds": duration,
            "backend": "assemblyai",
            "model": "universal-3-pro",
            "segments": segments,
            "num_speakers": len(speakers),
            "speakers": speakers,
            "processing_time": total_time,
        }


# ═══════════════════════════════════════════════════════════════
# DEEPGRAM DIARIZATION CLIENT
# ═══════════════════════════════════════════════════════════════

class DeepgramDiarizeClient:
    """
    Speaker diarization via Deepgram Nova-3 API.

    Deepgram provides transcription + diarization in a single call.
    We use it here only for the diarization timeline — the transcript
    from Whisper large-v3 is kept as the primary source of truth.

    Usage:
        client = DeepgramDiarizeClient(api_key="...")
        result = client.diarize("path/to/audio.wav")
        print(result["num_speakers"])
        print(result["timeline"])
    """

    BASE_URL = "https://api.deepgram.com/v1/listen"

    # Map file extensions to MIME types Deepgram accepts
    MIME_TYPES = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
        ".aac": "audio/aac",
        ".opus": "audio/opus",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = DIARIZE_TIMEOUT,
    ):
        self.api_key = api_key or DEEPGRAM_API_KEY
        self.timeout = timeout

        if not self.api_key:
            raise ValueError(
                "Deepgram API key not configured. Set DEEPGRAM_API_KEY env var."
            )

    def is_healthy(self) -> bool:
        """Check if Deepgram API is reachable."""
        try:
            resp = httpx.get(
                "https://api.deepgram.com/v1/projects",
                headers={"Authorization": f"Token {self.api_key}"},
                timeout=10,
            )
            healthy = resp.status_code in (200, 401, 403)
            if healthy:
                logger.info("Deepgram API reachable")
            return healthy
        except Exception as e:
            logger.warning(f"Deepgram health check failed: {e}")
        return False

    def diarize(
        self,
        audio_path: str,
        min_speakers: int = 2,
        max_speakers: int = 8,
        num_speakers: int = 0,
        audio_data: Optional[tuple] = None,
        clustering_threshold: float = 0.0,
        **kwargs,
    ) -> dict:
        """
        Diarize audio via Deepgram Nova-3.

        Returns a result dict matching the same format as DiarizeClient
        so it can be used as a drop-in replacement.
        Note: clustering_threshold and kwargs are accepted for signature
        compatibility but not used by Deepgram API.

        Returns:
            {
                "speakers": ["Speaker_0", ...],
                "timeline": [{"speaker": str, "start": float, "end": float, "duration": float}, ...],
                "num_speakers": int,
                "duration": float,
                "processing_time": float,
            }
        """
        audio_file = Path(audio_path)
        file_label = audio_file.name

        logger.info(
            f"Diarizing via Deepgram Nova-3: {file_label} "
            f"(min={min_speakers}, max={max_speakers}, exact={num_speakers})"
        )

        start_time = time.time()

        # Build query params
        params = {
            "diarize": "true",
            "punctuate": "true",
            "utterances": "true",
            "model": "nova-3",
            "language": "en",
        }

        headers = {"Authorization": f"Token {self.api_key}"}

        if audio_data is not None:
            # Export pre-loaded numpy audio to WAV bytes
            import numpy as np
            y, sr = audio_data
            wav_buf = io.BytesIO()
            y_int16 = np.clip(y * 32767, -32768, 32767).astype(np.int16)
            with wave.open(wav_buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(y_int16.tobytes())
            content = wav_buf.getvalue()
            headers["Content-Type"] = "audio/wav"
        else:
            if not audio_file.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            suffix = audio_file.suffix.lower()
            headers["Content-Type"] = self.MIME_TYPES.get(suffix, "audio/wav")
            with open(audio_path, "rb") as f:
                content = f.read()

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                self.BASE_URL,
                params=params,
                headers=headers,
                content=content,
            )

        elapsed = time.time() - start_time

        if resp.status_code != 200:
            raise RuntimeError(
                f"Deepgram API error ({resp.status_code}): {resp.text[:500]}"
            )

        data = resp.json()
        results = data.get("results", {})
        utterances = results.get("utterances", [])
        metadata = data.get("metadata", {})
        audio_duration = metadata.get("duration", 0)

        # Convert Deepgram utterances to our standard timeline format
        timeline = []
        speakers_seen = set()
        for utt in utterances:
            spk = utt["speaker"]
            speaker_label = f"Speaker_{spk}"
            speakers_seen.add(speaker_label)
            timeline.append({
                "speaker": speaker_label,
                "start": utt["start"],
                "end": utt["end"],
                "duration": round(utt["end"] - utt["start"], 3),
            })

        num_detected = len(speakers_seen)

        logger.info(
            f"Deepgram diarization complete: {num_detected} speakers, "
            f"{len(timeline)} turns, "
            f"API time={elapsed:.1f}s"
        )

        return {
            "speakers": sorted(speakers_seen),
            "timeline": timeline,
            "num_speakers": num_detected,
            "duration": audio_duration,
            "processing_time": elapsed,
        }

    def transcribe_and_diarize(self, audio_path: str) -> dict:
        """
        Transcribe + diarize via Deepgram Nova-3 in a single API call.

        Returns a result dict compatible with the voice agent pipeline
        (same format as AssemblyAIClient.transcribe_and_diarize).
        """
        audio_file = Path(audio_path)
        file_label = audio_file.name

        logger.info(f"Deepgram transcribe+diarize: {file_label}")

        start_time = time.time()

        params = {
            "diarize": "true",
            "punctuate": "true",
            "utterances": "true",
            "model": "nova-3",
            "language": "en",
            "smart_format": "true",
        }

        headers = {"Authorization": f"Token {self.api_key}"}

        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        suffix = audio_file.suffix.lower()
        headers["Content-Type"] = self.MIME_TYPES.get(suffix, "audio/wav")
        with open(audio_path, "rb") as f:
            content = f.read()

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                self.BASE_URL,
                params=params,
                headers=headers,
                content=content,
            )

        elapsed = time.time() - start_time

        if resp.status_code != 200:
            raise RuntimeError(
                f"Deepgram API error ({resp.status_code}): {resp.text[:500]}"
            )

        data = resp.json()
        results = data.get("results", {})
        utterances = results.get("utterances", [])
        metadata = data.get("metadata", {})
        audio_duration = metadata.get("duration", 0)

        # Convert Deepgram utterances → NEXUS segment format
        segments = []
        speakers_seen = set()
        for utt in utterances:
            speaker_label = f"Speaker_{utt['speaker']}"
            speakers_seen.add(speaker_label)

            words = []
            for w in utt.get("words", []):
                words.append({
                    "word": w.get("punctuated_word", w.get("word", "")),
                    "start": w["start"],
                    "end": w["end"],
                    "probability": round(w.get("confidence", 0), 3),
                })

            segments.append({
                "speaker": speaker_label,
                "start_ms": int(utt["start"] * 1000),
                "end_ms": int(utt["end"] * 1000),
                "text": utt["transcript"],
                "words": words,
            })

        speakers = sorted(speakers_seen)

        logger.info(
            f"Deepgram complete: {audio_duration}s audio, {len(segments)} segments, "
            f"{len(speakers)} speakers, total={elapsed:.1f}s"
        )

        return {
            "duration_seconds": audio_duration,
            "backend": "deepgram",
            "model": "nova-3",
            "segments": segments,
            "num_speakers": len(speakers),
            "speakers": speakers,
            "processing_time": elapsed,
        }


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


def create_diarize_client(**kwargs) -> Optional[DiarizeClient]:
    """Factory: create a DiarizeClient if configured."""
    if not is_diarize_available():
        return None
    try:
        client = DiarizeClient(**kwargs)
        if client.is_healthy():
            return client
        logger.warning(f"External diarize API at {DIARIZE_URL} is not healthy")
        return None
    except ValueError:
        return None
