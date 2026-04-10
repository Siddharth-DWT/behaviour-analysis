"""
NEXUS Shared — AssemblyAI Client
Single API call for transcription + diarization (Universal-3 Pro).

AssemblyAI transcribes first, then diarizes — both in one request.
Returns transcript with speaker labels and word-level timestamps.

Environment variables:
  ASSEMBLYAI_API_KEY   Required
"""
import os
import time
import logging
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger("nexus.assemblyai")

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "")
BASE_URL = "https://api.assemblyai.com/v2"
POLL_INTERVAL = 5  # seconds
UPLOAD_TIMEOUT = 300  # 5 min for large file uploads
POLL_TIMEOUT = 30
MAX_POLL_WAIT = 1200  # 20 min cap for transcription job to terminate


def is_available() -> bool:
    """Check if AssemblyAI API key is configured."""
    return bool(ASSEMBLYAI_API_KEY)


class AssemblyAIClient:
    """
    Client for AssemblyAI transcription + diarization.

    Uploads audio, creates a transcript job with speaker_labels=True,
    polls until completion, and converts to NEXUS internal format.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or ASSEMBLYAI_API_KEY
        if not self.api_key:
            raise ValueError("ASSEMBLYAI_API_KEY not configured")
        self._headers = {
            "authorization": self.api_key,
            "content-type": "application/json",
        }

    def transcribe(
        self,
        audio_path: str,
        language_code: str = "en",
        speakers_expected: Optional[int] = None,
    ) -> dict:
        """
        Transcribe audio with speaker diarization.

        Returns:
            {
                "duration_seconds": float,
                "segments": [{speaker, start_ms, end_ms, text, words}],
                "speakers": [str],
                "backend": "assemblyai",
                "model": "universal-3-pro",
                "processing_time": float,
            }
        """
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        start_time = time.time()

        # Step 1: Upload
        logger.info(f"AssemblyAI: uploading {audio_file.name}...")
        upload_url = self._upload(audio_path)

        # Step 2: Create transcript request
        request_body = {
            "audio_url": upload_url,
            "speech_models": ["universal-3-pro"],
            "speaker_labels": True,
            "language_code": language_code,
        }
        if speakers_expected is not None:
            request_body["speakers_expected"] = speakers_expected

        logger.info("AssemblyAI: creating transcript...")
        with httpx.Client(timeout=POLL_TIMEOUT) as client:
            resp = client.post(
                f"{BASE_URL}/transcript",
                headers=self._headers,
                json=request_body,
            )
            resp.raise_for_status()
            transcript_id = resp.json()["id"]

        # Step 3: Poll for completion
        logger.info(f"AssemblyAI: polling transcript {transcript_id}...")
        data = self._poll(transcript_id)

        elapsed = time.time() - start_time
        duration = data.get("audio_duration", 0)

        # Step 4: Convert to NEXUS format
        result = self._convert(data, elapsed)

        logger.info(
            f"AssemblyAI complete: {duration}s audio, "
            f"{len(result['segments'])} segments, "
            f"{len(result['speakers'])} speakers, "
            f"time={elapsed:.1f}s"
        )
        return result

    def _upload(self, audio_path: str) -> str:
        """Upload audio file and return the upload URL."""
        with open(audio_path, "rb") as f:
            resp = httpx.post(
                f"{BASE_URL}/upload",
                headers={"authorization": self.api_key},
                content=f,
                timeout=UPLOAD_TIMEOUT,
            )
        resp.raise_for_status()
        return resp.json()["upload_url"]

    def _poll(self, transcript_id: str) -> dict:
        """Poll until transcript is completed or errored. Caps at MAX_POLL_WAIT."""
        deadline = time.time() + MAX_POLL_WAIT
        while True:
            if time.time() > deadline:
                raise TimeoutError(
                    f"AssemblyAI transcript {transcript_id} did not finish within {MAX_POLL_WAIT}s"
                )

            with httpx.Client(timeout=POLL_TIMEOUT) as client:
                resp = client.get(
                    f"{BASE_URL}/transcript/{transcript_id}",
                    headers=self._headers,
                )
            resp.raise_for_status()
            data = resp.json()
            status = data["status"]

            if status == "completed":
                return data
            elif status == "error":
                raise RuntimeError(f"AssemblyAI error: {data.get('error')}")

            logger.debug(f"AssemblyAI: status={status}, waiting...")
            time.sleep(POLL_INTERVAL)

    def _convert(self, data: dict, elapsed: float) -> dict:
        """Convert AssemblyAI response to NEXUS internal format."""

        # Utterances = speaker-labeled segments (text already merged per turn)
        utterances = data.get("utterances", [])

        segments = []
        for u in utterances:
            # Normalize speaker labels: A -> Speaker_0, B -> Speaker_1, etc.
            speaker_raw = u.get("speaker", "A")
            speaker_idx = ord(speaker_raw) - ord("A") if len(speaker_raw) == 1 and speaker_raw.isalpha() else 0
            speaker = f"Speaker_{speaker_idx}"

            words = []
            for w in u.get("words", []):
                words.append({
                    "word": w.get("text", ""),
                    "start": w.get("start", 0) / 1000.0,
                    "end": w.get("end", 0) / 1000.0,
                    "confidence": w.get("confidence", 0),
                })

            text = u.get("text", "").strip()
            if text:
                segments.append({
                    "speaker": speaker,
                    "start_ms": u.get("start", 0),
                    "end_ms": u.get("end", 0),
                    "text": text,
                    "words": words,
                })

        speakers = sorted(set(seg["speaker"] for seg in segments))

        return {
            "duration_seconds": data.get("audio_duration", 0),
            "segments": segments,
            "speakers": speakers,
            "backend": "assemblyai",
            "model": "universal-3-pro",
            "processing_time": elapsed,
        }


def create_assemblyai_client(**kwargs) -> Optional[AssemblyAIClient]:
    """Factory: create an AssemblyAIClient if API key is configured."""
    if not is_available():
        return None
    try:
        return AssemblyAIClient(**kwargs)
    except ValueError:
        return None
