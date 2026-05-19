# services/video_agent/handcuff_detector.py
"""
HandcuffDetector — multi-method handcuff detection for interrogation video sessions.

Two complementary detection paths:
  1. Visual      — constant high arms_crossed_pct with low variance (WindowFeatures)
  2. Contextual  — Miranda warning / explicit restraint mention in transcript

Results feed into InterrogationVideoRules.evaluate(handcuffed=True) which
suppresses motor_inhibition, adjusts freezing_response confidence, and
switches barrier_behavior to lean-proxy measurement.

get_suppression_rules() provides structured audit metadata for report generation.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger("nexus.video.handcuff_detection")


class HandcuffDetector:
    """
    Multi-method handcuff detection with rule suppression metadata.

    Detection methods are independent — a positive from either path is
    sufficient.  Combined confidence is the max of both methods.

    DSA: O(S × W) visual detection where S = speakers, W = sample windows.
    Both are small (< 10 speakers, 10 sample windows) so cost is negligible.
    """

    # ── Transcript patterns ─────────────────────────────────────────────────

    _MIRANDA_PATTERNS = re.compile(
        r"\b("
        r"you have the right to remain silent|"
        r"anything you say can and will be used against you|"
        r"you have the right to an attorney"
        r")\b",
        re.IGNORECASE,
    )

    _CUFF_MENTIONS = re.compile(
        r"\b("
        r"handcuff(ed|s)?|"
        r"cuff(ed|s)?|"
        r"restrain(ed|t|ts)?|"
        r"take the cuffs off|remove the cuffs|"
        r"we need to cuff you|in custody"
        r")\b",
        re.IGNORECASE,
    )

    # ── Visual detection thresholds ─────────────────────────────────────────

    # arms_crossed_pct above this for all early windows → structural posture (cuffed)
    _VISUAL_CROSSING_THRESHOLD: float = 0.85
    # Variance below this → constant posture, not behavioural
    _VISUAL_VARIANCE_THRESHOLD: float = 0.03
    # Minimum windows before visual assessment is reliable
    _VISUAL_MIN_WINDOWS: int = 6
    # How many early windows to sample
    _VISUAL_SAMPLE_WINDOWS: int = 10

    # ── Rules suppressed when handcuffed ───────────────────────────────────

    _SUPPRESSED_RULES: list[str] = [
        "arms_crossed",
        "barrier_behavior",
        "motor_inhibition",
        "self_adaptors",
        "hand_illustrators",
        "fidgeting_hands",
        "BODY-GEST-01",
        "BODY-TOUCH-01",
        "BODY-TOUCH-02",
    ]

    _CONFIDENCE_PENALTIES: dict[str, float] = {
        "freezing_response": 0.5,
        "posture_shift":     0.3,
        "BODY-LEAN-01":      0.3,
    }

    _ALTERNATIVE_MEASUREMENTS: dict[str, str] = {
        "barrier_behavior": "torso_lean_away",
    }

    # ── Public API ──────────────────────────────────────────────────────────

    def detect_visual(
        self,
        windows_by_speaker: dict,
        session_id: str = "",
    ) -> dict:
        """
        Visual handcuff detection from WindowFeatures pose data.

        Uses arms_crossed_pct (derived from wrist/elbow proximity by MediaPipe)
        rather than raw pose landmarks — avoids an extra compute pass.

        Heuristic: hands locked in front (standard interrogation table posture)
        appear as constant crossed-arm posture to MediaPipe.  Structural posture
        has arms_crossed_pct > 0.85 AND variance < 0.03 across first N windows,
        distinguishing restraint from voluntary behavioural crossing.

        Returns:
            {
                "handcuffs_detected": bool,
                "confidence": float,
                "method": "visual_pose",
                "evidence": dict
            }
        """
        for spk, windows in windows_by_speaker.items():
            if len(windows) < self._VISUAL_MIN_WINDOWS:
                continue

            sample = [
                w.arms_crossed_pct
                for w in windows[: self._VISUAL_SAMPLE_WINDOWS]
                if hasattr(w, "arms_crossed_pct")
            ]
            if not sample:
                continue

            mean_cross = sum(sample) / len(sample)
            if mean_cross < self._VISUAL_CROSSING_THRESHOLD:
                continue

            variance = sum((x - mean_cross) ** 2 for x in sample) / len(sample)
            if variance >= self._VISUAL_VARIANCE_THRESHOLD:
                continue

            logger.info(
                "[%s] HandcuffDetector visual: %s arms_crossed_pct=%.2f var=%.4f → HANDCUFFED",
                session_id, spk, mean_cross, variance,
            )
            return {
                "handcuffs_detected": True,
                "confidence": 0.75,
                "method": "visual_pose",
                "evidence": {
                    "speaker_id":          spk,
                    "mean_arms_crossed":   round(mean_cross, 4),
                    "variance":            round(variance, 4),
                    "sample_windows":      len(sample),
                    "interpretation": (
                        "Consistently high arms-crossing posture with low variance "
                        "indicates structural wrist restraint rather than behavioural "
                        "crossing."
                    ),
                },
            }

        return {
            "handcuffs_detected": False,
            "confidence": 0.0,
            "method": "visual_pose",
            "evidence": {},
        }

    def detect_contextual(
        self,
        transcript: str,
    ) -> dict:
        """
        Contextual handcuff detection from session transcript text.

        Miranda warning recitation implies formal arrest → subject likely
        restrained.  Explicit restraint mentions provide direct evidence.

        Returns:
            {
                "handcuffs_detected": bool,
                "confidence": float,
                "method": "contextual_transcript",
                "evidence": dict
            }
        """
        if not transcript:
            return {
                "handcuffs_detected": False,
                "confidence": 0.0,
                "method": "contextual_transcript",
                "evidence": {},
            }

        miranda_match = self._MIRANDA_PATTERNS.search(transcript)
        cuff_match    = self._CUFF_MENTIONS.search(transcript)

        if not miranda_match and not cuff_match:
            return {
                "handcuffs_detected": False,
                "confidence": 0.0,
                "method": "contextual_transcript",
                "evidence": {},
            }

        # Miranda is stronger evidence than an incidental cuff mention
        confidence = 0.80 if miranda_match else 0.65

        evidence: dict = {}
        if miranda_match:
            evidence["miranda_detected"] = True
            evidence["miranda_excerpt"]  = miranda_match.group(0)
        if cuff_match:
            evidence["cuff_mention"]     = True
            evidence["cuff_excerpt"]     = cuff_match.group(0)

        logger.info(
            "[HandcuffDetector] Contextual: miranda=%s cuff_mention=%s conf=%.2f",
            bool(miranda_match), bool(cuff_match), confidence,
        )
        return {
            "handcuffs_detected": True,
            "confidence":         confidence,
            "method":             "contextual_transcript",
            "evidence":           evidence,
        }

    def detect(
        self,
        windows_by_speaker: dict,
        transcript: str = "",
        session_id: str = "",
    ) -> dict:
        """
        Combined detection — runs visual AND contextual paths.

        Either path returning True is sufficient.
        Confidence = max(visual_confidence, contextual_confidence).

        Returns:
            {
                "handcuffs_detected": bool,
                "confidence": float,
                "method": str,   # comma-joined method names if both triggered
                "evidence": dict
            }
        """
        visual     = self.detect_visual(windows_by_speaker, session_id=session_id)
        contextual = self.detect_contextual(transcript)

        if visual["handcuffs_detected"] or contextual["handcuffs_detected"]:
            confidence = max(visual["confidence"], contextual["confidence"])

            if visual["handcuffs_detected"] and contextual["handcuffs_detected"]:
                method = f"{visual['method']}+{contextual['method']}"
            elif visual["handcuffs_detected"]:
                method = visual["method"]
            else:
                method = contextual["method"]

            return {
                "handcuffs_detected": True,
                "confidence":         confidence,
                "method":             method,
                "evidence":           {**visual.get("evidence", {}), **contextual.get("evidence", {})},
            }

        return {
            "handcuffs_detected": False,
            "confidence":         0.0,
            "method":             "visual_pose+contextual_transcript",
            "evidence":           {},
        }

    def get_suppression_rules(self, handcuff_result: dict) -> dict:
        """
        Structured audit metadata for report generation when handcuffs are detected.

        Note: InterrogationVideoRules.evaluate(handcuffed=True) handles the
        actual signal-level suppression internally.  This method provides the
        human-readable audit trail for the session report.

        Returns:
            {
                "suppressed_rules": list[str],
                "confidence_penalties": dict[str, float],
                "alternative_measurements": dict[str, str],
                "alert": str | None
            }
        """
        if not handcuff_result.get("handcuffs_detected"):
            return {
                "suppressed_rules":        [],
                "confidence_penalties":    {},
                "alternative_measurements": {},
                "alert":                   None,
            }

        confidence = handcuff_result["confidence"]
        return {
            "suppressed_rules":         self._SUPPRESSED_RULES,
            "confidence_penalties":     self._CONFIDENCE_PENALTIES,
            "alternative_measurements": self._ALTERNATIVE_MEASUREMENTS,
            "alert": (
                f"Subject appears handcuffed (confidence: {confidence:.2f}). "
                f"Upper-body gesture analysis disabled. "
                f"{len(self._SUPPRESSED_RULES)} rules suppressed."
            ),
        }
