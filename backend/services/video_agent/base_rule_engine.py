"""
Base class for all Video Agent rule engines.
Provides the Signal factory (_make_signal) shared by Facial, Gaze, and Body engines.
"""
from abc import ABC, abstractmethod
from typing import Optional  # noqa: F401 — kept for callers that import it

from shared.models.signals import Signal


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
        video_quality: "Optional[object]" = None,
    ) -> list[dict]:
        """
        Run all rules for every speaker.

        Args:
            windows_by_speaker: {speaker_id: list[WindowFeatures]}
            baselines:          {speaker_id: (FacialBaseline, BodyBaseline, GazeBaseline)}
            session_id:         for logging
            meeting_type:       "sales" | "interview" | "general"
            video_quality:      VideoQuality descriptor from feature_extractor, or None

        Returns:
            list of Signal.to_dict()
        """

    @staticmethod
    def _face_quality_mult(face_box_area: float, video_quality: "Optional[object]") -> float:
        """
        Per-face confidence multiplier for quality-adaptive tracking.

        Only fires in PRISTINE tier (FACE_QUALITY_ADAPTIVE=1, ≥1440p sharp video)
        where the tracker admits faces down to 90px vs the normal 153px floor.
        Faces in the newly-admitted 90-153px band are penalized because MediaPipe
        blendshapes and gaze predictions are less reliable at those sizes.

        Ramp: 0.6 at min_track_area (90px in PRISTINE, admission gate)
              1.0 at quality_mult_ceiling_area (153px equivalent, standard-reliable)
        For all non-PRISTINE tiers ceiling == min_track_area → range is empty
        → always returns 1.0.  Returns 1.0 when video_quality is None (static mode).
        """
        if video_quality is None:
            return 1.0
        min_area = getattr(video_quality, "min_track_area", 0.0)
        ceiling  = getattr(video_quality, "quality_mult_ceiling_area", min_area)
        # Ceiling equals min_area for STANDARD/DEGRADED/POOR — no penalty range.
        if ceiling <= min_area or face_box_area >= ceiling:
            return 1.0
        if face_box_area <= min_area:
            return 0.6
        # Linear ramp: 0.6 at the tracker admission floor, 1.0 at the standard floor.
        t = (face_box_area - min_area) / (ceiling - min_area)
        return round(0.6 + 0.4 * max(0.0, min(1.0, t)), 4)

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
        w = getattr(self, "_current_w", None)
        area = (getattr(w, "face_box_area_mean", 0.0) or 0.0) if w is not None else 0.0
        rate = (getattr(w, "face_detection_rate", 0.0) or 0.0) if w is not None else 0.0

        base_meta: dict = {"rule_id": rule_id}
        if metadata:
            base_meta.update(metadata)
        # Auto-inject face grid position from the current window if available.
        # Set self._current_w = w in each rule engine's evaluate loop.
        if w is not None:
            cx = round(getattr(w, "face_centre_x", 0.0), 3)
            cy = round(getattr(w, "face_centre_y", 0.0), 3)
            if cx > 0 or cy > 0:
                base_meta.setdefault("face_centre_x", cx)
                base_meta.setdefault("face_centre_y", cy)
            if area > 0:
                base_meta.setdefault("face_box_area", round(area, 4))
            if rate > 0:
                base_meta.setdefault("face_detection_rate", round(rate, 3))
        capped_confidence = round(min(confidence, 0.85), 4)

        return Signal(
            agent=self.AGENT_NAME,
            speaker_id=speaker_id,
            signal_type=signal_type,
            value=round(value, 4),
            value_text=value_text,
            confidence=capped_confidence,
            window_start_ms=window_start_ms,
            window_end_ms=window_end_ms,
            metadata=base_meta,
        ).to_dict()
