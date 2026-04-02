"""
Shared Request / Response Models per Agent
These are the Pydantic models for each service's HTTP API.
Import from here instead of redefining per service.
"""
from typing import Optional
from pydantic import BaseModel

from shared.models.signals import FusionSignalInput
from shared.models.transcript import TranscriptSegment


# ─────────────────────────────────────────────────────────
# VOICE AGENT  (port 8001)
# ─────────────────────────────────────────────────────────

class VoiceAnalysisRequest(BaseModel):
    """POST /analyse — process an audio file already on disk."""
    file_path: str
    session_id: Optional[str] = None
    meeting_type: Optional[str] = "sales_call"
    num_speakers: Optional[int] = None  # Diarisation hint (2–10)


class VoiceAnalysisResponse(BaseModel):
    """Response from Voice Agent POST /analyse."""
    session_id: str
    duration_seconds: float
    speakers: list[dict]                        # [{speaker_id, baseline, signal_count}]
    signals: list[dict]                         # List of Signal.to_dict()
    summary: dict                               # Per-speaker stress / filler / tone stats
    transcript_segments: Optional[list[dict]] = None  # Diarised segments for Language Agent


# ─────────────────────────────────────────────────────────
# LANGUAGE AGENT  (port 8002)
# ─────────────────────────────────────────────────────────

class LanguageAnalysisRequest(BaseModel):
    """POST /analyse — analyse transcript segments."""
    segments: list[TranscriptSegment]
    session_id: Optional[str] = None
    meeting_type: Optional[str] = "sales_call"
    content_type: Optional[str] = None         # Auto-detected if omitted
    run_intent_classification: Optional[bool] = True


class LanguageAnalysisResponse(BaseModel):
    """Response from Language Agent POST /analyse."""
    session_id: str
    segment_count: int
    speakers: list[str]
    signals: list[dict]                         # List of Signal.to_dict()
    summary: dict                               # Per-speaker sentiment / buying / objection stats


# ─────────────────────────────────────────────────────────
# FUSION AGENT  (port 8007)
# ─────────────────────────────────────────────────────────

class FusionAnalyseRequest(BaseModel):
    """POST /analyse — run fusion on pre-collected signals."""
    voice_signals: list[FusionSignalInput] = []
    language_signals: list[FusionSignalInput] = []
    session_id: Optional[str] = None
    meeting_type: Optional[str] = "sales_call"
    content_type: Optional[str] = None
    generate_report: Optional[bool] = True
    voice_summary: Optional[dict] = None
    language_summary: Optional[dict] = None


class FusionSessionAnalyseRequest(BaseModel):
    """POST /analyse/session — run fusion by reading from Redis Streams."""
    session_id: str
    meeting_type: Optional[str] = "sales_call"
    content_type: Optional[str] = None
    generate_report: Optional[bool] = True
    voice_summary: Optional[dict] = None
    language_summary: Optional[dict] = None


class ReportRequest(BaseModel):
    """POST /report — generate a narrative report for a completed session."""
    session_id: str
    duration_seconds: float
    speakers: list[str]
    voice_summary: dict
    language_summary: dict
    fusion_signals: list[dict] = []
    unified_states: list[dict] = []
    meeting_type: Optional[str] = "sales_call"


class FusionAnalyseResponse(BaseModel):
    """Response from Fusion Agent POST /analyse."""
    session_id: str
    speakers: list[str]
    fusion_signals: list[dict]
    unified_states: list[dict]
    alerts: list[dict]
    report: Optional[dict] = None
    summary: dict


# ─────────────────────────────────────────────────────────
# API GATEWAY  (port 8000)
# ─────────────────────────────────────────────────────────

class SessionCreateResponse(BaseModel):
    """Response from POST /sessions — upload + full pipeline result."""
    session_id: str
    status: str
    title: str
    meeting_type: str
    duration_seconds: Optional[float] = None
    speaker_count: Optional[int] = None
    voice_signal_count: int = 0
    language_signal_count: int = 0
    fusion_signal_count: int = 0
    alert_count: int = 0
    report_generated: bool = False
    agent_status: Optional[dict[str, str]] = None


class SessionListResponse(BaseModel):
    """Response from GET /sessions — paginated session list."""
    sessions: list[dict]
    total: int
    limit: int
    offset: int
