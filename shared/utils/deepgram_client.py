"""
NEXUS Shared — Deepgram Client
Single API call for transcription + diarization (Nova-3).

Provides transcript with speaker labels and word-level timestamps
in one request — no separate diarize step needed.

Environment variables:
  DEEPGRAM_API_KEY   Required
"""
import os
import time
import logging
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger("nexus.deepgram")

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
BASE_URL = "https://api.deepgram.com/v1"
REQUEST_TIMEOUT = 300  # 5 min for long audio


def is_available() -> bool:
    """Check if Deepgram API key is configured."""
    return bool(DEEPGRAM_API_KEY)


class DeepgramClient:
    """
    Client for Deepgram Nova-3 transcription + diarization.

    Returns transcript with speaker labels and word-level timestamps
    from a single API call (no polling needed — synchronous response).
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or DEEPGRAM_API_KEY
        if not self.api_key:
            raise ValueError("DEEPGRAM_API_KEY not configured")

    def transcribe(
        self,
        audio_path: str,
        model: str = "nova-3",
        language: str = "en",
        diarize: bool = True,
        utterances: bool = True,
        smart_format: bool = True,
        punctuate: bool = True,
    ) -> dict:
        """
        Transcribe audio with speaker diarization.

        Returns:
            {
                "duration_seconds": float,
                "segments": [{speaker, start_ms, end_ms, text, words}],
                "speakers": [str],
                "backend": "deepgram",
                "model": str,
                "processing_time": float,
            }
        """
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        start_time = time.time()

        # Build query params
        params = {
            "model": model,
            "language": language,
            "diarize": str(diarize).lower(),
            "utterances": str(utterances).lower(),
            "smart_format": str(smart_format).lower(),
            "punctuate": str(punctuate).lower(),
        }

        # Detect content type
        suffix = audio_file.suffix.lower()
        content_types = {
            ".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4",
            ".mp4": "video/mp4", ".flac": "audio/flac", ".ogg": "audio/ogg",
            ".webm": "video/webm", ".aac": "audio/aac",
        }
        content_type = content_types.get(suffix, "audio/wav")

        logger.info(f"Deepgram: transcribing {audio_file.name} (model={model}, diarize={diarize})")

        with open(audio_path, "rb") as f:
            with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
                resp = client.post(
                    f"{BASE_URL}/listen",
                    params=params,
                    headers={
                        "Authorization": f"Token {self.api_key}",
                        "Content-Type": content_type,
                    },
                    content=f.read(),
                )

        if resp.status_code != 200:
            raise RuntimeError(f"Deepgram error ({resp.status_code}): {resp.text[:300]}")

        data = resp.json()
        elapsed = time.time() - start_time

        result = self._convert(data, elapsed, model)

        logger.info(
            f"Deepgram complete: {result['duration_seconds']:.1f}s audio, "
            f"{len(result['segments'])} segments, "
            f"{len(result['speakers'])} speakers, "
            f"time={elapsed:.1f}s"
        )
        return result

    def _convert(self, data: dict, elapsed: float, model: str) -> dict:
        """Convert Deepgram response to NEXUS internal format."""
        metadata = data.get("metadata", {})
        results = data.get("results", {})
        duration = metadata.get("duration", 0)

        # Use utterances (speaker-labeled segments) as primary output
        utterances = results.get("utterances", [])

        # Get word-level data from the channel for richer word info
        channel = (results.get("channels", [{}]) or [{}])[0]
        alt = (channel.get("alternatives", [{}]) or [{}])[0]
        all_words = alt.get("words", [])

        segments = []
        for u in utterances:
            speaker_idx = u.get("speaker", 0)
            speaker = f"Speaker_{speaker_idx}"
            start_ms = int(u.get("start", 0) * 1000)
            end_ms = int(u.get("end", 0) * 1000)
            text = u.get("transcript", "").strip()

            # Get word-level timestamps for this utterance
            words = []
            for w in u.get("words", []):
                words.append({
                    "word": w.get("punctuated_word", w.get("word", "")),
                    "start": w.get("start", 0),
                    "end": w.get("end", 0),
                    "confidence": w.get("confidence", 0),
                })

            if text:
                segments.append({
                    "speaker": speaker,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "text": text,
                    "words": words,
                })

        speakers = sorted(set(seg["speaker"] for seg in segments))

        return {
            "duration_seconds": duration,
            "segments": segments,
            "speakers": speakers,
            "backend": "deepgram",
            "model": model,
            "processing_time": elapsed,
        }


def create_deepgram_client(**kwargs) -> Optional[DeepgramClient]:
    """Factory: create a DeepgramClient if API key is configured."""
    if not is_available():
        return None
    try:
        return DeepgramClient(**kwargs)
    except ValueError:
        return None
