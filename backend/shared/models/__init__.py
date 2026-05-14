"""
shared.models — NEXUS shared Pydantic models

Quick import guide:
    from shared.models import Signal, FusionSignalInput
    from shared.models import TranscriptSegment
    from shared.models import Alert
    from shared.models.requests import (
        VoiceAnalysisRequest, VoiceAnalysisResponse,
        LanguageAnalysisRequest, LanguageAnalysisResponse,
        FusionAnalyseRequest, FusionAnalyseResponse,
        ReportRequest, SessionCreateResponse, SessionListResponse,
    )
"""
from shared.models.signals import Signal, FusionSignalInput, SpeakerBaseline, UnifiedSpeakerState
from shared.models.transcript import TranscriptSegment
from shared.models.alerts import Alert

__all__ = [
    "Signal",
    "FusionSignalInput",
    "SpeakerBaseline",
    "UnifiedSpeakerState",
    "TranscriptSegment",
    "Alert",
]
