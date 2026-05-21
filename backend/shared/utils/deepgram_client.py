"""
NEXUS Shared — Deepgram Client
Single API call for transcription + diarization (Nova-3).

Features we use from Deepgram:
  - diarize           Speaker diarization (who said what)
  - detect_language   Auto-detect spoken language when not specified
  - language          Explicit language hint (overridden by detect_language)
  - smart_format      Formatting: currency, dates, phone, emails + punctuation
  - punctuate         Sentence punctuation (redundant if smart_format=true, sent anyway)
  - filler_words      Preserve "uh" / "um" in transcript
  - numerals          Convert spoken numbers to digits ("nine hundred" → "900")
  - multichannel      Each audio channel transcribed independently
  - keyterm           Boost domain-specific vocabulary (Nova-3 / Flux only)
  - utterances        Semantic turn segmentation (produces speaker-labeled turns)

Features intentionally excluded (NEXUS handles these better):
  - sentiment         → Language agent (DistilBERT + Claude)
  - intents           → NEXUS conversation dynamics
  - topics            → Entity extraction agent
  - detect_entities   → Entity extraction agent
  - summarize         → Report generation (Claude)
  - profanity_filter  → Not used
  - redact            → Not used

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
    from a single synchronous API call (no polling needed).
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or DEEPGRAM_API_KEY
        if not self.api_key:
            raise ValueError("DEEPGRAM_API_KEY not configured")

    def transcribe(
        self,
        audio_path: str,
        model: str = "nova-3",
        language: Optional[str] = None,   # None = auto-detect via detect_language
        diarize: bool = True,
        smart_format: bool = True,        # dates, currency, phone, email formatting
        punctuate: bool = True,           # sentence punctuation
        filler_words: bool = False,       # preserve "uh" / "um"
        numerals: bool = True,            # "nine hundred" → "900"
        multichannel: bool = False,
        key_terms: Optional[list] = None, # boost domain vocab (Nova-3 / Flux only)
        utterances: bool = True,          # semantic turn segmentation
        summarize: bool = True,           # Deepgram v2 summary of the full transcript
    ) -> dict:
        """
        Transcribe audio with speaker diarization.

        Features used: diarize, language detection, smart_format, punctuate,
        numerals, filler_words, utterances, summarize.

        Features intentionally excluded (NEXUS handles these):
          sentiment, intents, topics, detect_entities, summarize (NEXUS Claude report),
          profanity_filter, redact, paragraphs, replace.

        Returns:
            {
                "duration_seconds": float,
                "segments": [{speaker, start_ms, end_ms, text, words}],
                "speakers": [str],
                "summary": str | None,
                "backend": "deepgram",
                "model": str,
                "processing_time": float,
            }
        """
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        start_time = time.time()

        # detect_language=true auto-detects; language= gives a hint when known.
        # smart_format=true auto-enables punctuation but we also send punctuate
        # explicitly for predictable behaviour across model versions.
        params: dict = {
            "model": model,
            "diarize": "true" if diarize else "false",
            "utterances": "true" if utterances else "false",
            "smart_format": "true" if smart_format else "false",
            "punctuate": "true" if punctuate else "false",
            "numerals": "true" if numerals else "false",
            "summarize": "v2" if summarize else "false",
        }

        if language:
            params["language"] = language
        else:
            params["detect_language"] = "true"

        if filler_words:
            params["filler_words"] = "true"

        if multichannel:
            params["multichannel"] = "true"

        if key_terms:
            # keyterm replaces the legacy keywords field for Nova-3 / Flux models
            params["keyterm"] = key_terms   # httpx serialises list as repeated params

        # Detect content type from extension
        suffix = audio_file.suffix.lower()
        content_types = {
            ".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4",
            ".mp4": "video/mp4", ".flac": "audio/flac", ".ogg": "audio/ogg",
            ".webm": "video/webm", ".aac": "audio/aac",
        }
        content_type = content_types.get(suffix, "audio/wav")

        logger.info(
            f"Deepgram: transcribing {audio_file.name} "
            f"(model={model}, diarize={diarize}, lang={language or 'auto'})"
        )

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

        if not resp.is_success:
            logger.error(f"Deepgram request failed {resp.status_code}: {resp.text[:500]}")
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

        # Utterances give us speaker-labeled semantic turns.
        # Fall back to the first channel alternative if utterances are absent.
        utterances = results.get("utterances") or []
        if not utterances:
            channel = (results.get("channels", [{}]) or [{}])[0]
            alt = (channel.get("alternatives", [{}]) or [{}])[0]
            full_text = alt.get("transcript", "").strip()
            if full_text:
                all_words = alt.get("words", [])
                start_ms = int(all_words[0].get("start", 0) * 1000) if all_words else 0
                end_ms = int(all_words[-1].get("end", duration) * 1000) if all_words else int(duration * 1000)
                utterances = [{
                    "speaker": 0,
                    "start": start_ms / 1000,
                    "end": end_ms / 1000,
                    "transcript": full_text,
                    "words": all_words,
                }]

        segments = []
        for u in utterances:
            speaker_idx = u.get("speaker", 0)
            speaker = f"Speaker_{speaker_idx}"
            start_ms = int(u.get("start", 0) * 1000)
            end_ms = int(u.get("end", 0) * 1000)
            text = u.get("transcript", "").strip()

            words = [
                {
                    "word": w.get("punctuated_word", w.get("word", "")),
                    "start": w.get("start", 0),
                    "end": w.get("end", 0),
                    "confidence": w.get("confidence", 0),
                }
                for w in u.get("words", [])
            ]

            if text:
                segments.append({
                    "speaker": speaker,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "text": text,
                    "words": words,
                })

        speakers = sorted(set(seg["speaker"] for seg in segments))

        # Extract Deepgram v2 summary if present
        summary_obj = results.get("summary", {})
        summary = summary_obj.get("short") if isinstance(summary_obj, dict) else None

        return {
            "duration_seconds": duration,
            "segments": segments,
            "speakers": speakers,
            "summary": summary,
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
