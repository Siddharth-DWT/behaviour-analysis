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
import math

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

    # GAZE-SYNC-01 thresholds
    SYNC_ALIGN_MS   = 3000  # two gaze breaks within 3s = "simultaneous"
    SYNC_MIN_BREAKS = 3     # minimum synchronized breaks before flagging

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

        # Cross-speaker rule (needs all speakers simultaneously)
        signals += self._rule_gaze_sync(windows_by_speaker)

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
            # face_detection_rate is the per-window reliability gate (face visible this window).
            # calibration_confidence is already enforced by MIN_GAZE_RATE above; using it as
            # a direct multiplier here permanently halves all gaze confidence when baselines
            # are sparse, pushing signals below the 0.30 display threshold.
            conf_mult = w.face_detection_rate

            signals += self._rule_gaze_direction(w, bl, speaker_id, conf_mult)
            signals += self._rule_blink_rate(w, bl, speaker_id, conf_mult)
            signals += self._rule_attention(w, bl, speaker_id, conf_mult)

        return signals

    # Approximate conversion: iris offset (normalised by eye width) → degrees.
    # At a typical 60cm viewing distance, 1.0 normalised ≈ 45°. Rough but interpretable.
    _GAZE_DEG_SCALE = 45.0
    _DEAD_ZONE_DEG  = 5.0   # Angular error margin — genuinely ambiguous (Cai 2021: 3.11°)
    _CLEAR_ZONE_DEG = 10.0  # Above this = clearly off-screen

    def _rule_gaze_direction(
        self,
        w: WindowFeatures,
        bl: GazeBaseline,
        speaker_id: str,
        conf_mult: float,
    ) -> list[dict]:
        """
        GAZE-DIR-01: Named zone classification with dead zone (G-2).
        Zones: CAMERA (< 5°) → no signal, DEAD_ZONE (5-10°) → uncertain,
               LEFT/RIGHT/DOWN/UP (> 10°) → confident directional shift.
        Dead zone accounts for MediaPipe's 3-5° angular error.
        Sigma guard still applied first to suppress noise below 2× baseline std.
        """
        x_delta_raw = w.gaze_x_mean - bl.gaze_x_offset
        y_delta_raw = w.gaze_y_mean - bl.gaze_y_offset

        # Sigma guard: suppress sub-noise shifts (keeps per-speaker calibration benefit)
        x_sigma = abs(x_delta_raw) / max(bl.gaze_x_std_mean, 0.01)
        y_sigma = abs(y_delta_raw) / max(bl.gaze_y_std_mean, 0.01)
        if x_sigma < 2.0 and y_sigma < 2.0:
            return []

        # Convert to approximate degrees for interpretable zone classification
        x_deg = abs(x_delta_raw) * self._GAZE_DEG_SCALE
        y_deg = abs(y_delta_raw) * self._GAZE_DEG_SCALE
        max_deg = max(x_deg, y_deg)

        if max_deg < self._DEAD_ZONE_DEG:
            # Below angular error margin — do not emit (suppressed by sigma guard anyway)
            return []
        elif max_deg < self._CLEAR_ZONE_DEG:
            zone = "gaze_shift_uncertain"
            confidence = min(0.25 * conf_mult, 0.30)  # Dead zone: very low confidence
        else:
            # Clear shift — classify direction
            if x_deg >= y_deg:
                zone = "gaze_shift_left" if x_delta_raw < 0 else "gaze_shift_right"
            else:
                zone = "gaze_shift_down" if y_delta_raw > 0 else "gaze_shift_up"
            confidence = min((max_deg / 30.0) * conf_mult, 0.65)

        return [self._make_signal(
            rule_id="GAZE-DIR-01",
            signal_type="gaze_direction_shift",
            speaker_id=speaker_id,
            value=round(max_deg, 2),
            value_text=zone,
            confidence=confidence,
            window_start_ms=w.window_start_ms,
            window_end_ms=w.window_end_ms,
            metadata={
                "x_deg":       round(x_deg, 2),
                "y_deg":       round(y_deg, 2),
                "x_sigma":     round(x_sigma, 2),
                "y_sigma":     round(y_sigma, 2),
                "zone":        zone,
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
        baseline_bpm = max(bl.blink_rate_bpm, 15.0)
        ratio = w.blink_rate_bpm / baseline_bpm

        if ratio >= self.BLINK_FAST_THRESHOLD:
            label = "elevated_blink_rate"
            confidence = min((ratio - 1.0) * 0.3 * conf_mult, 0.60)
            deviation = min(ratio - 1.0, 1.0)  # 0–1: how far above baseline
        elif ratio <= self.BLINK_SLOW_THRESHOLD and w.blink_rate_bpm > 0:
            label = "suppressed_blink_rate"
            confidence = min((1.0 - ratio) * 0.3 * conf_mult, 0.50)
            deviation = min(1.0 - ratio, 1.0)  # 0–1: how far below baseline
        else:
            return []

        return [self._make_signal(
            rule_id="GAZE-BLINK-01",
            signal_type="blink_rate_anomaly",
            speaker_id=speaker_id,
            value=round(deviation, 4),  # normalized 0-1 deviation from baseline
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
        GAZE-ATT-01: 4-factor composite attention score (G-4, Rayner 1998).
          F1 screen_engagement (0.35) — on-screen gaze %
          F2 gaze_stability    (0.20) — low variance = focused
          F3 blink_normalcy    (0.20) — deviation from baseline penalises
          F4 no_gaze_shift     (0.25) — inverse of directional shift magnitude
        """
        # F1: screen engagement
        f1_screen = w.gaze_on_screen_pct

        # F2: gaze stability — Euclidean norm of std components (not sum, avoids double-counting)
        f2_stability = 1.0 - min(math.sqrt(w.gaze_x_std ** 2 + w.gaze_y_std ** 2), 1.0)

        # F3: blink rate normalcy — deviation from baseline penalises
        baseline_bpm = max(bl.blink_rate_bpm, 15.0)
        blink_ratio  = w.blink_rate_bpm / baseline_bpm if baseline_bpm > 0 else 1.0
        f3_blink     = max(0.0, 1.0 - abs(blink_ratio - 1.0))

        # F4: inverse gaze shift magnitude (less deviation = more on-task)
        x_shift = abs(w.gaze_x_mean - bl.gaze_x_offset)
        y_shift = abs(w.gaze_y_mean - bl.gaze_y_offset)
        shift   = min(math.sqrt(x_shift ** 2 + y_shift ** 2) * 5.0, 1.0)
        f4_no_shift = 1.0 - shift

        composite = (
            0.35 * f1_screen
            + 0.20 * f2_stability
            + 0.20 * f3_blink
            + 0.25 * f4_no_shift
        )

        if composite >= self.ATTENTION_HIGH_PCT:
            label = "high_attention"
            confidence = min(composite * conf_mult * 0.8, 0.70)
        elif composite < self.ATTENTION_LOW_PCT:
            label = "reduced_attention"
            confidence = min((1.0 - composite) * conf_mult * 0.5, 0.55)
        else:
            return []

        return [self._make_signal(
            rule_id="GAZE-ATT-01",
            signal_type="attention_level",
            speaker_id=speaker_id,
            value=round(composite, 4),
            value_text=label,
            confidence=confidence,
            window_start_ms=w.window_start_ms,
            window_end_ms=w.window_end_ms,
            metadata={
                "f1_screen_engagement": round(f1_screen, 4),
                "f2_gaze_stability":    round(f2_stability, 4),
                "f3_blink_normalcy":    round(f3_blink, 4),
                "f4_no_gaze_shift":     round(f4_no_shift, 4),
                "composite":            round(composite, 4),
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
        GAZE-CONTACT-01: screen engagement with speaking/listening split (G-1).
        Argyle (1972): people gaze ~75% while listening, ~40% while speaking.
        Thresholds are adjusted so normal speaking gaze-away is not flagged.

        In the current single-camera architecture, SpeakerFaceMapper assigns each
        window to the dominant speaker (the one talking), so is_speaking is always
        True for windows in windows_by_speaker. The split is kept in the code to
        support future multi-camera / full-session-face-tracking.

        Speaking norms: 30-60% screen time normal → flag below 25%
        Listening norms: 55-80% screen time normal → flag below 40%
        """
        signals: list[dict] = []
        block_size = self.CONTACT_BLOCK_WINDOWS
        valid_windows = [w for w in windows if w.face_detection_rate >= self.MIN_GAZE_RATE]

        for i in range(0, len(valid_windows), block_size):
            block = valid_windows[i: i + block_size]
            if not block:
                continue

            avg_on_screen = sum(w.gaze_on_screen_pct for w in block) / len(block)
            delta         = avg_on_screen - bl.screen_engagement_rate
            block_start   = block[0].window_start_ms
            block_end     = block[-1].window_end_ms

            # Speaking/listening context from WindowFeatures.is_speaking
            speaking_frames  = sum(1 for w in block if getattr(w, "is_speaking", True))
            speaking_ratio   = speaking_frames / len(block)
            is_speaking      = speaking_ratio > 0.5

            if is_speaking:
                low_threshold  = 0.25   # < 25% while speaking = very low
                high_threshold = 0.85
            else:
                low_threshold  = 0.40   # < 40% while listening = disengaged
                high_threshold = 0.90

            if avg_on_screen < low_threshold:
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
                        "avg_on_screen_pct":    round(avg_on_screen, 4),
                        "baseline_engagement":  round(bl.screen_engagement_rate, 4),
                        "context":              "speaking" if is_speaking else "listening",
                        "speaking_ratio":       round(speaking_ratio, 2),
                        "threshold_used":       low_threshold,
                        "block_windows":        len(block),
                    },
                ))
            elif avg_on_screen > high_threshold:
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
                        "avg_on_screen_pct":    round(avg_on_screen, 4),
                        "baseline_engagement":  round(bl.screen_engagement_rate, 4),
                        "context":              "speaking" if is_speaking else "listening",
                        "speaking_ratio":       round(speaking_ratio, 2),
                        "block_windows":        len(block),
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

    # ── GAZE-SYNC-01 (EXPERIMENTAL) ──────────────────────────────────────────
    def _rule_gaze_sync(self, windows_by_speaker: dict) -> list[dict]:
        """
        GAZE-SYNC-01 (EXPERIMENTAL): synchronized gaze breaks across speakers.
        When multiple participants look away simultaneously it suggests shared
        distraction or mutual disengagement (Sellen et al. 1992).
        Cap 0.40 — experimental, too many confounders.

        DSA: sorted windows + advancing pointer → O(A + B) per pair.
        """
        speaker_ids = list(windows_by_speaker.keys())
        if len(speaker_ids) < 2:
            return []

        signals: list[dict] = []

        for i in range(len(speaker_ids)):
            for j in range(i + 1, len(speaker_ids)):
                sp_a = speaker_ids[i]
                sp_b = speaker_ids[j]

                wins_a = sorted(
                    [w for w in windows_by_speaker[sp_a]
                     if w.face_detection_rate >= self.MIN_GAZE_RATE],
                    key=lambda w: w.window_start_ms,
                )
                wins_b = sorted(
                    [w for w in windows_by_speaker[sp_b]
                     if w.face_detection_rate >= self.MIN_GAZE_RATE],
                    key=lambda w: w.window_start_ms,
                )
                if not wins_a or not wins_b:
                    continue

                sync_events: list[tuple[int, int]] = []
                ptr_b = 0
                for wa in wins_a:
                    if wa.gaze_on_screen_pct >= self.GAZE_OFF_THRESHOLD:
                        continue
                    # Advance pointer past windows that ended too early
                    while (ptr_b < len(wins_b)
                           and wins_b[ptr_b].window_end_ms
                           < wa.window_start_ms - self.SYNC_ALIGN_MS):
                        ptr_b += 1
                    for wb in wins_b[ptr_b:]:
                        if wb.window_start_ms > wa.window_end_ms + self.SYNC_ALIGN_MS:
                            break
                        if wb.gaze_on_screen_pct < self.GAZE_OFF_THRESHOLD:
                            sync_events.append((
                                min(wa.window_start_ms, wb.window_start_ms),
                                max(wa.window_end_ms,   wb.window_end_ms),
                            ))
                            break

                if len(sync_events) < self.SYNC_MIN_BREAKS:
                    continue

                all_wins = wins_a + wins_b
                session_start = min(w.window_start_ms for w in all_wins)
                session_end   = max(w.window_end_ms   for w in all_wins)
                sync_rate  = len(sync_events) / max(len(wins_a), 1)
                confidence = min(sync_rate * 1.5, 0.40)  # hard cap 0.40

                for sp in (sp_a, sp_b):
                    signals.append(self._make_signal(
                        rule_id="GAZE-SYNC-01",
                        signal_type="gaze_synchrony",
                        speaker_id=sp,
                        value=round(sync_rate, 4),
                        value_text="synchronized_gaze_break",
                        confidence=confidence,
                        window_start_ms=session_start,
                        window_end_ms=session_end,
                        metadata={
                            "pair": f"{sp_a}+{sp_b}",
                            "sync_break_count": len(sync_events),
                            "sync_rate": round(sync_rate, 4),
                            "experimental": True,
                        },
                    ))

        return signals
