"""
NEXUS Shared — Parakeet TDT Client
NVIDIA Parakeet TDT 0.6B v2 for fast English transcription with word timestamps.

174x realtime speed. Returns segments + word-level timestamps.
No speaker diarization — needs separate diarizer (NeMo/pyannote).

Environment variables:
  PARAKEET_URL   e.g. http://your-gpu-server:8043
"""
import os
import time
import logging
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger("nexus.parakeet")

PARAKEET_URL = os.getenv("PARAKEET_URL", "")
API_KEY = os.getenv("EXTERNAL_API_KEY", "")
REQUEST_TIMEOUT = 600


def is_available() -> bool:
    return bool(PARAKEET_URL)


class ParakeetClient:
    """
    Client for NVIDIA Parakeet TDT 0.6B v2 transcription.

    Returns segments with word-level timestamps. No speaker labels —
    pair with NeMo MSDD or pyannote for diarization.
    """

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        self.base_url = (base_url or PARAKEET_URL).rstrip("/")
        self.api_key = api_key or API_KEY
        if not self.base_url:
            raise ValueError("PARAKEET_URL not configured")

    @property
    def _headers(self) -> dict:
        if self.api_key:
            return {"X-API-Key": self.api_key}
        return {}

    def is_healthy(self) -> bool:
        try:
            resp = httpx.get(f"{self.base_url}/health", headers=self._headers, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                healthy = data.get("status") == "healthy"
                if healthy:
                    logger.info(
                        f"Parakeet API healthy: GPU={data.get('gpu_name', '?')}, "
                        f"model={data.get('model', '?')}"
                    )
                return healthy
        except Exception as e:
            logger.warning(f"Parakeet health check failed: {e}")
        return False

    def transcribe(self, audio_path: str) -> dict:
        """
        Transcribe audio with word-level timestamps.

        Returns:
            {
                "duration_seconds": float,
                "segments": [{start_ms, end_ms, text, words}],
                "backend": "parakeet",
                "model": str,
                "processing_time": float,
            }
        """
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Parakeet: transcribing {audio_file.name}...")
        start_time = time.time()

        with open(audio_path, "rb") as f:
            with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
                resp = client.post(
                    f"{self.base_url}/transcribe",
                    files={"file": (audio_file.name, f, "audio/wav")},
                    data={"word_timestamps": "true"},
                    headers=self._headers,
                )

        if resp.status_code != 200:
            raise RuntimeError(f"Parakeet error ({resp.status_code}): {resp.text[:300]}")

        data = resp.json()
        elapsed = time.time() - start_time

        result = self._convert(data, elapsed)

        logger.info(
            f"Parakeet complete: {result['duration_seconds']:.1f}s audio, "
            f"{len(result['segments'])} segments, "
            f"time={elapsed:.1f}s"
        )
        return result

    def _convert(self, data: dict, elapsed: float) -> dict:
        """Convert Parakeet response to NEXUS format."""
        raw_segments = data.get("segments", [])
        top_words = data.get("words", [])
        duration = data.get("duration", 0)

        # Map top-level words into their segments by time overlap
        segments = []
        for seg in raw_segments:
            seg_start = seg.get("start", 0)
            seg_end = seg.get("end", 0)
            text = seg.get("text", "").strip()

            # Find words belonging to this segment
            seg_words = []
            for w in top_words:
                w_start = w.get("start", 0)
                w_end = w.get("end", 0)
                # Word belongs to segment if its midpoint falls within segment bounds
                w_mid = (w_start + w_end) / 2
                if seg_start <= w_mid <= seg_end:
                    seg_words.append({
                        "word": w.get("word", ""),
                        "start": w_start,
                        "end": w_end,
                    })

            if text:
                segments.append({
                    "start_ms": int(seg_start * 1000),
                    "end_ms": int(seg_end * 1000),
                    "text": text,
                    "words": seg_words,
                })

        return {
            "duration_seconds": duration,
            "segments": segments,
            "backend": "parakeet",
            "model": data.get("model", "parakeet-tdt-0.6b-v2"),
            "processing_time": elapsed,
        }


def create_parakeet_client(**kwargs) -> Optional[ParakeetClient]:
    if not is_available():
        return None
    try:
        client = ParakeetClient(**kwargs)
        if client.is_healthy():
            return client
        logger.warning(f"Parakeet API at {PARAKEET_URL} is not healthy")
        return None
    except ValueError:
        return None
