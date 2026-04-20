"""
Gaze Rule Engine — Phase 2C
Implements GAZE-DIR-01, GAZE-CONTACT-01, GAZE-BLINK-01, GAZE-ATT-01, GAZE-DIST-01.
GAZE-SYNC-01 is experimental (requires multi-speaker simultaneous gaze data).

Research anchors:
  Argyle & Cook 1976       — 60-70% mutual gaze is normal in conversations
  Sellen et al. 1992       — Video call gaze offset from screen vs. camera
  Bentivoglio et al. 1997  — Normal blink rate: 12-20 blinks/min; stress raises it
  Rayner 1998              — Sustained off-screen gaze > 8s = attention break
"""
import logging

from base_rule_engine import BaseVideoRuleEngine

try:
    from feature_extractor import WindowFeatures
    from calibration import FacialBaseline, BodyBaseline, GazeBaseline
except ImportError:
    from services.video_agent.feature_extractor import WindowFeatures
    from services.video_agent.calibration import FacialBaseline, BodyBaseline, GazeBaseline

logger = logging.getLogger("nexus.video.gaze_rules")


class GazeRuleEngine(BaseVideoRuleEngine):
    """
    Runs gaze behaviour rules across all speakers.

    DSA: runs in two passes —
      Pass 1 (per-window): GAZE-DIR-01, GAZE-BLINK-01, GAZE-ATT-01 fire immediately.
      Pass 2 (session-level): GAZE-CONTACT-01 (30s blocks), GAZE-DIST-01 (run-length).
    This avoids O(W²) nested comparisons; both passes are O(W).
    """

    AGENT_NAME = "video"

    # Thresholds
    BLINK_FAST_THRESHOLD   = 1.5   # blink rate multiplier above baseline → stress
    BLINK_SLOW_THRESHOLD   = 0.5   # blink rate multiplier below baseline → fatigue/focus
    GAZE_OFF_THRESHOLD     = 0.30  # gaze_on_screen_pct below this = off-screen window
    DISTRACT_RUN_WINDOWS   = 4     # consecutive off-screen windows → GAZE-DIST-01 (≈8s @ 2s)
    ATTENTION_HIGH_PCT     = 0.80  # on-screen % above this = high attention
    ATTENTION_LOW_PCT      = 0.50  # on-screen % below this (but not distraction) = reduced
    CONTACT_BLOCK_WINDOWS  = 15    # 30s block (15 × 2s windows)
    MIN_GAZE_RATE          = 0.30  # skip window if face < 30%

    def evaluate(
        self,
        windows_by_speaker: dict,
        baselines: dict,
        session_id: str = "",
        meeting_type: str = "general",
    ) -> list[dict]:
        signals: list[dict] = []
        for speaker_id, windows in windows_by_speaker.items():
            _, _, gaze_bl = baselines.get(
                speaker_id,
                (None, None, GazeBaseline(speaker_id=speaker_id)),
            )
            signals += self._per_window_rules(windows, gaze_bl, speaker_id)
            signals += self._rule_screen_contact(windows, gaze_bl, speaker_id)
            signals += self._rule_distraction(windows, gaze_bl, speaker_id)

        logger.info(f"[{session_id}] GazeRuleEngine: {len(signals)} signals")
        return signals

    # ── GAZE-DIR-01 + GAZE-BLINK-01 + GAZE-ATT-01 ────────────────────────────
    def _per_window_rules(
        self,
        windows: list[WindowFeatures],
        bl: GazeBaseline,
        speaker_id: str,
    ) -> list[dict]:
        signals: list[dict] = []
        for w in windows:
            if w.face_detection_rate < self.MIN_GAZE_RATE:
                continue
            conf_mult = bl.screen_engagement_rate * w.face_detection_rate

            signals += self._rule_gaze_direction(w, bl, speaker_id, conf_mult)
            signals += self._rule_blink_rate(w, bl, speaker_id, conf_mult)
            signals += self._rule_attention(w, bl, speaker_id, conf_mult)

        return signals

    def _rule_gaze_direction(
        self,
        w: WindowFeatures,
        bl: GazeBaseline,
        speaker_id: str,
        conf_mult: float,
    ) -> list[dict]:
        """
        GAZE-DIR-01: significant horizontal or vertical gaze shift from baseline offset.
        Iris offset normalised by eye width; baseline corrects for camera-above-screen.
        """
        x_delta = abs(w.gaze_x_mean - bl.gaze_x_offset)
        y_delta = abs(w.gaze_y_mean - bl.gaze_y_offset)

        # Only fire if shift is substantial (>2σ from baseline gaze std)
        x_sigma = x_delta / max(bl.gaze_x_std_mean, 0.01)
        y_sigma = y_delta / max(bl.gaze_y_std_mean, 0.01)

        if x_sigma < 2.0 and y_sigma < 2.0:
            return []

        dominant_axis = "horizontal" if x_sigma > y_sigma else "vertical"
        magnitude = max(x_sigma, y_sigma)
        confidence = min(magnitude * 0.15 * conf_mult, 0.60)

        return [self._make_signal(
            rule_id="GAZE-DIR-01",
            signal_type="gaze_direction_shift",
            speaker_id=speaker_id,
            value=round(magnitude, 4),
            value_text=f"gaze_shift_{dominant_axis}",
            confidence=confidence,
            window_start_ms=w.window_start_ms,
            window_end_ms=w.window_end_ms,
            metadata={
                "x_sigma": round(x_sigma, 2),
                "y_sigma": round(y_sigma, 2),
                "gaze_x_mean": round(w.gaze_x_mean, 4),
                "gaze_y_mean": round(w.gaze_y_mean, 4),
            },
        )]

    def _rule_blink_rate(
        self,
        w: WindowFeatures,
        bl: GazeBaseline,
        speaker_id: str,
        conf_mult: float,
    ) -> list[dict]:
        """
        GAZE-BLINK-01: blink rate deviation from per-speaker baseline.
        Fast blinks > 1.5× baseline → autonomic stress response.
        Slow blinks < 0.5× baseline → fatigue or intense focus.
        """
        baseline_bpm = max(bl.blink_rate_bpm, 5.0)
        ratio = w.blink_rate_bpm / baseline_bpm

        if ratio >= self.BLINK_FAST_THRESHOLD:
            label = "elevated_blink_rate"
            confidence = min((ratio - 1.0) * 0.3 * conf_mult, 0.60)
        elif ratio <= self.BLINK_SLOW_THRESHOLD and w.blink_rate_bpm > 0:
            label = "suppressed_blink_rate"
            confidence = min((1.0 - ratio) * 0.3 * conf_mult, 0.50)
        else:
            return []

        return [self._make_signal(
            rule_id="GAZE-BLINK-01",
            signal_type="blink_rate_anomaly",
            speaker_id=speaker_id,
            value=round(w.blink_rate_bpm, 2),
            value_text=label,
            confidence=confidence,
            window_start_ms=w.window_start_ms,
            window_end_ms=w.window_end_ms,
            metadata={
                "blink_rate_bpm": round(w.blink_rate_bpm, 2),
                "baseline_bpm": round(baseline_bpm, 2),
                "ratio": round(ratio, 3),
            },
        )]

    def _rule_attention(
        self,
        w: WindowFeatures,
        bl: GazeBaseline,
        speaker_id: str,
        conf_mult: float,
    ) -> list[dict]:
        """
        GAZE-ATT-01: per-window screen engagement quality.
        High attention: gaze on-screen > 80% + low gaze std.
        Reduced attention: on-screen 30-50%.
        """
        on_screen = w.gaze_on_screen_pct
        gaze_stability = 1.0 - min(w.gaze_x_std + w.gaze_y_std, 1.0)

        if on_screen >= self.ATTENTION_HIGH_PCT:
            label = "high_attention"
            value = on_screen * gaze_stability
            confidence = min(value * conf_mult * 0.8, 0.70)
        elif on_screen < self.ATTENTION_LOW_PCT and on_screen >= self.GAZE_OFF_THRESHOLD:
            label = "reduced_attention"
            value = 1.0 - on_screen
            confidence = min(value * conf_mult * 0.5, 0.55)
        else:
            return []

        return [self._make_signal(
            rule_id="GAZE-ATT-01",
            signal_type="attention_level",
            speaker_id=speaker_id,
            value=round(on_screen, 4),
            value_text=label,
            confidence=confidence,
            window_start_ms=w.window_start_ms,
            window_end_ms=w.window_end_ms,
            metadata={
                "gaze_on_screen_pct": round(on_screen, 4),
                "gaze_stability": round(gaze_stability, 4),
            },
        )]

    # ── GAZE-CONTACT-01 ───────────────────────────────────────────────────────
    def _rule_screen_contact(
        self,
        windows: list[WindowFeatures],
        bl: GazeBaseline,
        speaker_id: str,
    ) -> list[dict]:
        """
        GAZE-CONTACT-01: screen engagement aggregated over 30s blocks.
        Argyle & Cook 1976: 60-70% mutual gaze normal; below 40% = low contact.
        """
        signals: list[dict] = []
        block_size = self.CONTACT_BLOCK_WINDOWS
        valid_windows = [w for w in windows if w.face_detection_rate >= self.MIN_GAZE_RATE]

        for i in range(0, len(valid_windows), block_size):
            block = valid_windows[i: i + block_size]
            if not block:
                continue

            avg_on_screen = sum(w.gaze_on_screen_pct for w in block) / len(block)
            delta = avg_on_screen - bl.screen_engagement_rate
            block_start = block[0].window_start_ms
            block_end   = block[-1].window_end_ms

            if avg_on_screen < 0.40:
                label = "low_screen_contact"
                confidence = min(abs(delta) * 0.8, 0.65)
                signals.append(self._make_signal(
                    rule_id="GAZE-CONTACT-01",
                    signal_type="screen_contact",
                    speaker_id=speaker_id,
                    value=round(avg_on_screen, 4),
                    value_text=label,
                    confidence=confidence,
                    window_start_ms=block_start,
                    window_end_ms=block_end,
                    metadata={
                        "avg_on_screen_pct": round(avg_on_screen, 4),
                        "baseline_engagement": round(bl.screen_engagement_rate, 4),
                        "block_windows": len(block),
                    },
                ))
            elif avg_on_screen > 0.85:
                label = "sustained_eye_contact"
                confidence = min(avg_on_screen * 0.6, 0.60)
                signals.append(self._make_signal(
                    rule_id="GAZE-CONTACT-01",
                    signal_type="screen_contact",
                    speaker_id=speaker_id,
                    value=round(avg_on_screen, 4),
                    value_text=label,
                    confidence=confidence,
                    window_start_ms=block_start,
                    window_end_ms=block_end,
                    metadata={
                        "avg_on_screen_pct": round(avg_on_screen, 4),
                        "baseline_engagement": round(bl.screen_engagement_rate, 4),
                        "block_windows": len(block),
                    },
                ))

        return signals

    # ── GAZE-DIST-01 ─────────────────────────────────────────────────────────
    def _rule_distraction(
        self,
        windows: list[WindowFeatures],
        bl: GazeBaseline,
        speaker_id: str,
    ) -> list[dict]:
        """
        GAZE-DIST-01: sustained off-screen gaze using run-length encoding.
        4 consecutive windows (≈8s) with gaze_on_screen < 30% = distraction event.

        DSA: single O(W) pass with run counter — avoids nested loops.
        """
        signals: list[dict] = []
        run_count = 0
        run_start_ms: int = 0
        run_end_ms: int = 0

        def _flush_run() -> None:
            if run_count >= self.DISTRACT_RUN_WINDOWS:
                duration_s = (run_end_ms - run_start_ms) / 1000.0
                confidence = min(run_count * 0.12, 0.65)
                signals.append(self._make_signal(
                    rule_id="GAZE-DIST-01",
                    signal_type="sustained_distraction",
                    speaker_id=speaker_id,
                    value=round(duration_s, 2),
                    value_text="sustained_off_screen",
                    confidence=confidence,
                    window_start_ms=run_start_ms,
                    window_end_ms=run_end_ms,
                    metadata={
                        "consecutive_windows": run_count,
                        "duration_seconds": round(duration_s, 2),
                    },
                ))

        for w in windows:
            if w.face_detection_rate < self.MIN_GAZE_RATE:
                continue
            if w.gaze_on_screen_pct < self.GAZE_OFF_THRESHOLD:
                if run_count == 0:
                    run_start_ms = w.window_start_ms
                run_count += 1
                run_end_ms = w.window_end_ms
            else:
                _flush_run()
                run_count = 0

        _flush_run()
        return signals
