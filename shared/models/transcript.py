"""
Shared Transcript Models
TranscriptSegment crosses the Voice → Language agent boundary.
Voice Agent produces these; Language Agent consumes them.
"""
from typing import Optional
from pydantic import BaseModel


class TranscriptSegment(BaseModel):
    """
    A single diarised transcript segment from the Voice Agent / Whisper.
    Passed verbatim to the Language Agent POST /analyse.
    """
    speaker: str = "Speaker_0"         # Diarisation label, e.g. "SPEAKER_00"
    start_ms: int = 0
    end_ms: int = 0
    text: str = ""
    words: Optional[list[dict]] = None  # Word-level timestamps from Whisper (optional)
