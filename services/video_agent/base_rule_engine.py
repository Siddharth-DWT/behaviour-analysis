"""
Base class for all Video Agent rule engines.
Provides the Signal factory (_make_signal) shared by Facial, Gaze, and Body engines.
"""
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from shared.models.signals import Signal
except ImportError:
    from models.signals import Signal  # type: ignore


class BaseVideoRuleEngine(ABC):
    """
    Abstract base for Facial / Gaze / Body rule engines.

    OOP: abstract method enforces evaluate() contract; _make_signal is the
    single factory for all Signal creation — keeps confidence cap (0.85)
    and agent name injection in one place.
    """

    AGENT_NAME: str = "video"

    @abstractmethod
    def evaluate(
        self,
        windows_by_speaker: dict,
        baselines: dict,
        session_id: str = "",
        meeting_type: str = "general",
    ) -> list[dict]:
        """
        Run all rules for every speaker.

        Args:
            windows_by_speaker: {speaker_id: list[WindowFeatures]}
            baselines:          {speaker_id: (FacialBaseline, BodyBaseline, GazeBaseline)}
            session_id:         for logging
            meeting_type:       "sales" | "interview" | "general"

        Returns:
            list of Signal.to_dict()
        """

    def _make_signal(
        self,
        rule_id: str,
        signal_type: str,
        speaker_id: str,
        value: float,
        value_text: str,
        confidence: float,
        window_start_ms: int,
        window_end_ms: int,
        metadata: Optional[dict] = None,
    ) -> dict:
        """Create a Signal dict with confidence hard-capped at 0.85."""
        return Signal(
            agent=self.AGENT_NAME,
            speaker_id=speaker_id,
            signal_type=signal_type,
            value=round(value, 4),
            value_text=value_text,
            confidence=round(min(confidence, 0.85), 4),
            window_start_ms=window_start_ms,
            window_end_ms=window_end_ms,
            metadata={"rule_id": rule_id, **(metadata or {})},
        ).to_dict()
