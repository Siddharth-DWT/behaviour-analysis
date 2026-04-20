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
POLL_INTERVAL = 5    # seconds between status checks
UPLOAD_TIMEOUT = 300  # 5 min for large file uploads
POLL_TIMEOUT = 30    # per HTTP request
MAX_POLL_WAIT = 1200  # 20 min total — bail out if job doesn't finish


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
        language_code: Optional[str] = None,
        speakers_expected: Optional[int] = None,
        speaker_labels: bool = True,
        key_terms: Optional[list] = None,
        custom_prompt: Optional[str] = None,
        keep_filler_words: bool = False,
        auto_punctuation: bool = True,
        multichannel: bool = False,
        temperature: Optional[float] = None,
        translate_to: Optional[str] = None,
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

        # Step 2: Build transcript request for Universal-3 Pro.
        # Features we use from AssemblyAI:
        #   - speaker_labels: diarization (who said what)
        #   - speaker_identification: enhanced speaker labelling via speech_understanding
        #   - language_detection: auto-detect spoken language (used when no language_code given)
        #   - language_code: only when user explicitly picks a language
        #   - temperature, keyterms_prompt, prompt, format_text, punctuate, disfluencies
        #   - translation via speech_understanding (when translate_to is set)
        # Everything else (sentiment, IAB, PII, content_safety, profanity) is handled
        # better by NEXUS own agents and is NOT sent to AssemblyAI.
        # universal-3-pro does not support the format_text parameter — omit it.
        request_body: dict = {
            "audio_url": upload_url,
            "speech_models": ["universal-3-pro"],
            "speaker_labels": speaker_labels,
            "punctuate": auto_punctuation,
            "disfluencies": keep_filler_words,
        }

        # Language: explicit code takes priority; otherwise let AssemblyAI auto-detect
        if language_code:
            request_body["language_code"] = language_code
        else:
            request_body["language_detection"] = True

        if speakers_expected is not None:
            # Top-level integer — correct AssemblyAI v2 API field.
            # (speaker_options.min/max_speakers_expected is not a valid v2 field.)
            request_body["speakers_expected"] = speakers_expected
        if multichannel:
            request_body["multichannel"] = True
            del request_body["speaker_labels"]     # mutually exclusive with multichannel
        if key_terms:
            request_body["keyterms_prompt"] = key_terms
        if custom_prompt:
            request_body["prompt"] = custom_prompt
        if temperature is not None:
            request_body["temperature"] = max(0.0, min(1.0, temperature))

        # Build speech_understanding block — translation only.
        # speaker_identification requires pre-enrolled voice profiles (known_values).
        # Sending speaker_identification with known_values=[] overrides speaker_labels
        # diarization and collapses all audio to a single speaker — do NOT include it.
        if translate_to and not multichannel:
            request_body["speech_understanding"] = {
                "translation": {
                    "target_languages": [translate_to],
                    "match_original_utterance": True,
                }
            }

        logger.info("AssemblyAI: creating transcript...")
        with httpx.Client(timeout=POLL_TIMEOUT) as client:
            resp = client.post(
                f"{BASE_URL}/transcript",
                headers=self._headers,
                json=request_body,
            )
            if not resp.is_success:
                logger.error(f"AssemblyAI transcript request failed {resp.status_code}: {resp.text}")
            resp.raise_for_status()
            transcript_id = resp.json()["id"]

        # Step 3: Poll for completion
        logger.info(f"AssemblyAI: polling transcript {transcript_id}...")
        data = self._poll(transcript_id)

        elapsed = time.time() - start_time
        duration = data.get("audio_duration", 0)

        # Step 4: Convert to NEXUS format
        result = self._convert(data, elapsed, translate_to=translate_to)

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
        """Poll until transcript is completed or errored. Raises on timeout."""
        deadline = time.time() + MAX_POLL_WAIT
        while True:
            if time.time() > deadline:
                raise TimeoutError(
                    f"AssemblyAI transcript {transcript_id} did not finish "
                    f"within {MAX_POLL_WAIT}s"
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

    def _convert(self, data: dict, elapsed: float, translate_to: Optional[str] = None) -> dict:
        """Convert AssemblyAI response to NEXUS internal format."""

        # Utterances = speaker-labeled segments (text already merged per turn).
        # When speaker_labels=False, utterances is absent — fall back to a single
        # segment for the whole transcript labelled Speaker_0.
        utterances = data.get("utterances") or []
        if not utterances and data.get("text"):
            utterances = [{
                "speaker": "A",
                "start": data.get("words", [{}])[0].get("start", 0) if data.get("words") else 0,
                "end": data.get("words", [{}])[-1].get("end", int(data.get("audio_duration", 0) * 1000)) if data.get("words") else int(data.get("audio_duration", 0) * 1000),
                "text": data["text"],
                "words": data.get("words", []),
            }]

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

            # Use translated text if translation was requested and available
            if translate_to and u.get("translated_texts"):
                text = u["translated_texts"].get(translate_to) or u.get("text", "")
            else:
                text = u.get("text", "")
            text = text.strip()

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
