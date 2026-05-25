# services/video_agent/interrogation_rules.py
"""
Interrogation-specific video rules (NEXUS INTERROGATION_IMPLEMENTATION.MD v2.0).

All rules operate on existing WindowFeatures fields — no new data extraction needed.
Confidence caps follow research-validated accuracy rates per the spec.

Rules implemented:
  BLINK-PATTERN-01        Blink Suppression→Spike Pattern  (conf 0.70 — Leal & Vrij 2008)
  MOTOR-INHIBIT-01        Motor Inhibition Pattern          (conf 0.35 — DePaulo 2003 meta)
  INAPPROPRIATE-AFFECT-01 Smile in Severe-Negative Context  (conf 0.45 — PMC 2024)
  GAZE-SACCADES-01        Erratic Gaze Saccades             (conf 0.40 — VPS 2025)
  INTERROG-BODY-02        Freezing Response                 (conf 0.55 — Navarro 2008)
  INTERROG-BODY-03        Self-Adaptor Increase             (conf 0.35 — Li 2024, d=0.10)

Removed rules (unreliable, per prompt.md evidence review):
  INTERROG-BODY-01  barrier_behavior — no meta-analytic effect size; arms crossing
                    is habitual/thermal in majority of cases; not in DePaulo 2003.
  LOW-AUTONOMIC     low_autonomic_reactivity — NEXUS-original; zero published research;
                    absence of stress is not a validated deception cue.

Quality-adaptive thresholds (INTERROGATION_UPDATES1.MD §3 & §5):
  HIGH_QUALITY  fps ≥ 60 — lab/broadcast quality, full spec confidences
  STANDARD_FPS  fps 30–59 — webcam/phone; blink conf reduced 0.70→0.55
  CCTV_QUALITY  fps 15–29 — relaxed thresholds, significantly reduced confidences
  SUPPRESSED    fps < 15  — all video rules return [] (video unusable)

Room camera gate (all interrogation_video sessions):
  Interrogation rooms use corner/ceiling-mounted cameras.  The oblique angle
  makes gaze direction tracking unreliable — eye contact cannot be measured.
  Suppressed: erratic_gaze_pattern.
  Body and face signals (blink, smile, motor, freeze, self-adaptor) remain active.

CRITICAL DESIGN PRINCIPLE:
  None of these rules claim deception.  Every signal carries multiple
  interpretations.  Maximum confidence is 0.70 (blink pattern, ML-validated).
"""
from __future__ import annotations

import re
import logging
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .feature_extractor import WindowFeatures
    from .calibration import FacialBaseline, BodyBaseline, GazeBaseline

logger = logging.getLogger("nexus.video.interrogation")


# ── Quality-adaptive tiers (INTERROGATION_UPDATES1.MD Section 5) ─────────────

class VideoQualityTier(Enum):
    HIGH_QUALITY = "HIGH_QUALITY"   # fps >= 60 — lab/broadcast, full confidence
    STANDARD_FPS = "STANDARD_FPS"  # fps 30-59 — webcam/phone, blink conf reduced
    CCTV_QUALITY = "CCTV_QUALITY"  # fps 15-29 — CCTV, relaxed thresholds, reduced confidence
    SUPPRESSED   = "SUPPRESSED"    # fps < 15  — suppress all video rules


def _tier_from_fps(fps: float) -> VideoQualityTier:
    if fps >= 60.0:
        return VideoQualityTier.HIGH_QUALITY
    elif fps >= 30.0:
        return VideoQualityTier.STANDARD_FPS
    elif fps >= 15.0:
        return VideoQualityTier.CCTV_QUALITY
    return VideoQualityTier.SUPPRESSED


# Signal types that require reliable frontal gaze tracking.
# Interrogation rooms use corner/ceiling-mounted cameras — oblique angle makes
# gaze direction unreliable regardless of fps. Suppressed for ALL interrogation sessions.
_ROOM_CAMERA_GATED: frozenset[str] = frozenset({
    "erratic_gaze_pattern",  # gaze direction/saccades invalid from room camera angle
    "sustained_distraction", # requires subject to be looking at a screen — no screen in interrogation room
    "screen_contact",        # "eye contact with camera/screen" has no meaning in CCTV angle
    "attention_level",       # derived from screen_contact + gaze direction — both unreliable here
})

# Master confidence table — source: INTERROGATION_UPDATES1.MD §5.
# HIGH_QUALITY = 60fps+.  STANDARD_FPS = 30-59fps (only blink degrades per §3).
# CCTV_QUALITY = 15-29fps (all visual signals degraded).
_QUALITY_CONF: dict[str, dict[VideoQualityTier, float]] = {
    # §3: 30fps misses ~20% of blinks → 0.70 → 0.55; 15fps misses ~50% → 0.35
    "blink_suppression_spike": {
        VideoQualityTier.HIGH_QUALITY: 0.70,
        VideoQualityTier.STANDARD_FPS: 0.55,
        VideoQualityTier.CCTV_QUALITY: 0.35,
    },
    "motor_inhibition": {
        VideoQualityTier.HIGH_QUALITY: 0.35,
        VideoQualityTier.STANDARD_FPS: 0.35,
        VideoQualityTier.CCTV_QUALITY: 0.15,
    },
    "smile_context_incongruence": {
        VideoQualityTier.HIGH_QUALITY: 0.45,
        VideoQualityTier.STANDARD_FPS: 0.45,
        VideoQualityTier.CCTV_QUALITY: 0.30,
    },
    "erratic_gaze_pattern": {
        VideoQualityTier.HIGH_QUALITY: 0.40,
        VideoQualityTier.STANDARD_FPS: 0.40,
        VideoQualityTier.CCTV_QUALITY: 0.15,
    },
    "freezing_response": {
        VideoQualityTier.HIGH_QUALITY: 0.55,
        VideoQualityTier.STANDARD_FPS: 0.55,
        VideoQualityTier.CCTV_QUALITY: 0.35,
    },
    # d=0.10 (DePaulo 2003); temporal-trend concept unvalidated → cap 0.35
    "self_adaptor_increase": {
        VideoQualityTier.HIGH_QUALITY: 0.35,
        VideoQualityTier.STANDARD_FPS: 0.35,
        VideoQualityTier.CCTV_QUALITY: 0.20,
    },
}


# ── Thresholds ────────────────────────────────────────────────────────────────
# BLINK-PATTERN-01: suppression during → spike after (§3 threshold sets)
BLINK_SUPPRESSION_RATIO      = 0.70  # HIGH_QUALITY (60fps+) — 30% drop from baseline
BLINK_SPIKE_RATIO            = 1.40  # HIGH_QUALITY (60fps+) — 40% spike above baseline
BLINK_SUPPRESSION_RATIO_STD  = 0.60  # STANDARD_FPS (30-59fps) — 30fps misses ~20% of blinks
BLINK_SPIKE_RATIO_STD        = 1.60  # STANDARD_FPS (30-59fps) — need larger change
BLINK_SUPPRESSION_RATIO_CCTV = 0.50  # CCTV_QUALITY (15-29fps) — misses ~50% of blinks
BLINK_SPIKE_RATIO_CCTV       = 2.00  # CCTV_QUALITY (15-29fps) — requires very large change

# BLINK-PATTERN-01: minimum face size for CCTV tier (§3 spec: face > 120px at 480p)
# head_shoulder_dist_mean is nose-to-shoulder normalized by frame height.
# At 480p, face_h/frame_h = 120/480 = 0.25 → head_shoulder_dist ≈ face_h → threshold 0.10
# (conservative: catches truly tiny faces where blink detection is unreliable)
BLINK_FACE_HS_MIN = 0.10

# MOTOR-INHIBIT-01: hand/gesture velocity reduction
MOTOR_INHIBIT_THRESHOLD = 0.45   # current < baseline * 0.45 → inhibition (55%+ drop); reduced from 0.60 for ceiling-camera geometry where baseline readings are naturally depressed

# GAZE-SACCADES-01: elevated gaze std
GAZE_SACCADE_RATIO      = 1.50   # (gaze_x_std + gaze_y_std) > baseline * 1.50

# INTERROG-BODY-02: freeze — near-zero body movement
FREEZE_MOVEMENT_RATIO   = 0.35   # < 35% of baseline body_movement_mean; reduced from 0.20 for ceiling/corner camera geometry
FREEZE_MIN_WINDOWS      = 3      # must persist for at least 3 consecutive windows

# INTERROG-BODY-03: self-adaptor increase — session-thirds self-touch rate trend
SELF_ADAPTOR_TREND_RATIO = 1.5   # last-third rate / first-third rate >= 1.5 → fire
SELF_ADAPTOR_MIN_WINDOWS = 6     # need at least 2 windows per third for reliable trend

_ACCUSATORY_RE = re.compile(
    r"\b(you (did|killed|were there|lied|took|stole|shot|stabbed|hurt|committed)|"
    r"we (know|have proof|can prove)|the evidence (shows|indicates|proves) (you|that)|"
    r"I know you|you'?re (guilty|lying|involved)|you can'?t (deny|explain)|"
    r"you were (there|with|seen)|we (found|have) (your|evidence)|"
    r"(your|the) (DNA|fingerprints?|blood) (was|were|matched)|"
    r"witnesses? (saw|identified|confirmed) you)\b",
    re.IGNORECASE,
)

# Severe negative keywords for inappropriate affect (smile context incongruence)
_SEVERE_NEGATIVE_RE = re.compile(
    r"\b(murder|victim|blood|kill|killed|dead|corpse|stab|stabbed|shoot|shot|"
    r"assault|rape|rapist|torture|weapon|crime scene|body|death|deceased|"
    r"homicide|attacked|strangled|beaten|manslaughter|bloody)\b",
    re.IGNORECASE,
)

# MediaPipe blendshape keys for smile and Duchenne detection
_SMILE_BLEND_KEYS = ("mouthSmileLeft", "mouthSmileRight")
_DUCHENNE_KEYS    = ("cheekSquintLeft", "cheekSquintRight")

# Threshold for smile blendshape (0-1 normalized MediaPipe score)
_SMILE_THRESHOLD    = 0.40
_DUCHENNE_THRESHOLD = 0.25

# Minimum calibration confidence before interrogation rules fire
_MIN_CAL_CONF = 0.20


def _get_smile_score(wf: "WindowFeatures") -> float:
    bs = wf.blendshapes_mean
    return (bs.get("mouthSmileLeft", 0.0) + bs.get("mouthSmileRight", 0.0)) / 2.0


def _is_duchenne(wf: "WindowFeatures") -> bool:
    bs = wf.blendshapes_mean
    cheek = (bs.get("cheekSquintLeft", 0.0) + bs.get("cheekSquintRight", 0.0)) / 2.0
    return cheek >= _DUCHENNE_THRESHOLD


def _segment_text_near_window(
    diar_segments: list[dict],
    window_start_ms: int,
    window_end_ms: int,
    context_ms: int = 10_000,
) -> str:
    """Return concatenated transcript text within ±context_ms of the window."""
    texts = []
    lo = window_start_ms - context_ms
    hi = window_end_ms   + context_ms
    for seg in diar_segments:
        seg_start = int(seg.get("start", 0) * 1000)
        seg_end   = int(seg.get("end",   0) * 1000)
        if seg_end >= lo and seg_start <= hi:
            texts.append(seg.get("text", "") or "")
    return " ".join(texts)


class InterrogationVideoRules:
    """
    Stateless per-call video interrogation rules.
    evaluate() is the single entry point.
    """

    def evaluate(
        self,
        windows_by_speaker: dict[str, list["WindowFeatures"]],
        baselines: dict[str, tuple["FacialBaseline", "BodyBaseline", "GazeBaseline"]],
        diar_segments: list[dict],
        session_id: str = "",
        video_fps: float = 30.0,
        handcuffed: bool = False,
    ) -> list[dict]:
        """
        Run all video interrogation rules for all speakers.
        Returns a flat list of signal dicts tagged agent='video'.

        video_fps is used to select the quality tier (HIGH_QUALITY / CCTV_QUALITY /
        SUPPRESSED).  Default 30.0 → HIGH_QUALITY → identical behaviour to pre-tier
        implementation when caller does not supply fps.

        handcuffed=True activates the HANDCUFFED tier:
          - motor_inhibition suppressed (arms restrained → gesture velocity invalid)
          - self_adaptor_increase suppressed (wrist restraint blocks self-touch)
          - freezing_response confidence capped at 0.40 (body can still freeze)
          - blink and smile signals unchanged (face is unrestrained)
        """
        tier = _tier_from_fps(video_fps)

        if tier == VideoQualityTier.SUPPRESSED:
            logger.info(
                "[%s] Interrogation video rules suppressed (fps=%.1f < 15 — video unusable)",
                session_id, video_fps,
            )
            return []

        if tier == VideoQualityTier.STANDARD_FPS:
            logger.info(
                "[%s] Interrogation video rules: STANDARD_FPS tier (fps=%.1f) — blink conf 0.70→0.55",
                session_id, video_fps,
            )
        elif tier == VideoQualityTier.CCTV_QUALITY:
            logger.info(
                "[%s] Interrogation video rules: CCTV quality tier (fps=%.1f)",
                session_id, video_fps,
            )

        if handcuffed:
            logger.info(
                "[%s] HANDCUFFED tier active — motor_inhibition suppressed, "
                "self_adaptor_increase suppressed, "
                "freezing_response capped at conf=0.40",
                session_id,
            )

        signals: list[dict] = []
        for spk, windows in windows_by_speaker.items():
            if not windows:
                continue
            bl = baselines.get(spk)
            if bl is None:
                continue
            facial_bl, body_bl, gaze_bl = bl

            # Skip if calibration confidence too low
            if facial_bl.calibration_confidence < _MIN_CAL_CONF:
                continue

            signals.extend(self._blink_pattern(spk, windows, facial_bl, tier))
            signals.extend(self._motor_inhibition(spk, windows, body_bl, tier, handcuffed=handcuffed))
            signals.extend(self._inappropriate_affect(spk, windows, diar_segments, tier))
            signals.extend(self._gaze_saccades(spk, windows, gaze_bl, tier))
            signals.extend(self._freezing_response(spk, windows, body_bl, tier, handcuffed=handcuffed))
            signals.extend(self._self_adaptor_increase(spk, windows, tier, handcuffed=handcuffed))

        # Room camera gate: oblique angle makes gaze-based signals unreliable
        gated = [s for s in signals if s["signal_type"] in _ROOM_CAMERA_GATED]
        if gated:
            signals = [s for s in signals if s["signal_type"] not in _ROOM_CAMERA_GATED]
            logger.info(
                "[%s] Room camera gate: suppressed %d signal(s) (%s) — "
                "gaze direction unreliable from corner-mounted interrogation room camera",
                session_id, len(gated),
                ", ".join(sorted({s["signal_type"] for s in gated})),
            )

        return signals

    # ── BLINK-PATTERN-01 ─────────────────────────────────────────────────────

    def _blink_pattern(
        self,
        spk: str,
        windows: list["WindowFeatures"],
        bl: "FacialBaseline",
        tier: VideoQualityTier,
    ) -> list[dict]:
        """
        Leal & Vrij (2008, 2010), Monaro et al. (2020) — 70% ML accuracy.
        Pattern: blink SUPPRESSION during response → SPIKE immediately after.
        Mechanism: cognitive load + freeze suppress blinking; tension release spikes.
        Confidence: 0.70 (HIGH_QUALITY) / 0.35 (CCTV — 15fps misses ~50% of blinks).
        """
        baseline_bpm = bl.blink_rate_bpm
        if baseline_bpm <= 0:
            return []

        # §3 spec: CCTV tier requires face > 120px (≈ head_shoulder_dist > 0.10 normalized).
        # Face too small → miss rate > 50% even with relaxed thresholds → suppress to AUDIO_ONLY.
        if tier == VideoQualityTier.CCTV_QUALITY and windows:
            avg_hs = sum(w.head_shoulder_dist_mean for w in windows) / len(windows)
            if avg_hs < BLINK_FACE_HS_MIN:
                logger.debug(
                    "Blink suppressed for %s — face too small at CCTV quality "
                    "(avg_head_shoulder_dist=%.3f < %.3f)",
                    spk, avg_hs, BLINK_FACE_HS_MIN,
                )
                return []

        conf = _QUALITY_CONF["blink_suppression_spike"][tier]
        if tier == VideoQualityTier.CCTV_QUALITY:
            sup_ratio = BLINK_SUPPRESSION_RATIO_CCTV
            spk_ratio = BLINK_SPIKE_RATIO_CCTV
        elif tier == VideoQualityTier.STANDARD_FPS:
            sup_ratio = BLINK_SUPPRESSION_RATIO_STD
            spk_ratio = BLINK_SPIKE_RATIO_STD
        else:
            sup_ratio = BLINK_SUPPRESSION_RATIO
            spk_ratio = BLINK_SPIKE_RATIO

        signals: list[dict] = []
        for i in range(len(windows) - 1):
            curr = windows[i]
            nxt  = windows[i + 1]

            during_rate = curr.blink_rate_bpm
            after_rate  = nxt.blink_rate_bpm

            if (during_rate <= baseline_bpm * sup_ratio and
                    after_rate  >= baseline_bpm * spk_ratio):
                suppression_pct = round((1 - during_rate / baseline_bpm) * 100, 1)
                spike_pct       = round((after_rate / baseline_bpm - 1) * 100, 1)
                meta: dict = {
                    "rule_id":           "BLINK-PATTERN-01",
                    "quality_tier":      tier.value,
                    "baseline_bpm":      round(baseline_bpm, 1),
                    "during_bpm":        round(during_rate, 1),
                    "after_bpm":         round(after_rate, 1),
                    "suppression_pct":   suppression_pct,
                    "spike_pct":         spike_pct,
                    "interpretation":    "Blink suppression during response followed by compensatory spike. Indicates high cognitive load.",
                    "context":           "Occurs during: (1) Deceptive response requiring fabrication, (2) Complex truthful explanation, (3) High-stakes truthful response with anxiety.",
                    "recommendation":    "Pair with response latency and speech hesitation signals.",
                }
                if tier == VideoQualityTier.STANDARD_FPS:
                    meta["quality_disclaimer"] = (
                        "Confidence reduced from 0.70 to 0.55 — 30fps misses ~20% of blinks. "
                        "Relaxed thresholds applied (suppression 0.70→0.60, spike 1.40→1.60)."
                    )
                elif tier == VideoQualityTier.CCTV_QUALITY:
                    meta["quality_disclaimer"] = (
                        "Confidence reduced from 0.70 to 0.35 — 15fps CCTV misses ~50% of blinks. "
                        "Relaxed thresholds applied to compensate."
                    )
                signals.append({
                    "agent":            "video",
                    "speaker_id":       spk,
                    "signal_type":      "blink_suppression_spike",
                    "value":            round(suppression_pct / 100, 3),
                    "value_text":       "suppression_then_spike",
                    "confidence":       conf,
                    "window_start_ms":  curr.window_start_ms,
                    "window_end_ms":    nxt.window_end_ms,
                    "metadata":         meta,
                })
        return signals

    # ── MOTOR-INHIBIT-01 ─────────────────────────────────────────────────────

    def _motor_inhibition(
        self,
        spk: str,
        windows: list["WindowFeatures"],
        bl: "BodyBaseline",
        tier: VideoQualityTier,
        handcuffed: bool = False,
    ) -> list[dict]:
        """
        Vrij et al. (1996), DePaulo et al. (2003) meta — d = -0.10.
        Liars show DECREASED movement (opposite of folk wisdom).
        Mechanism: inhibitory control + cognitive load reduce motor activity.
        Confidence: 0.35 (HIGH_QUALITY) / 0.15 (CCTV — coarse movement only).
        HANDCUFFED: suppressed — gesture velocity is invalid when arms are restrained.
        """
        if handcuffed:
            return []

        baseline_movement = bl.body_movement_mean
        if baseline_movement <= 0:
            return []

        conf = _QUALITY_CONF["motor_inhibition"][tier]

        signals: list[dict] = []
        for w in windows:
            # Use gesture_velocity_mean as proxy for hand/gesture movement
            current = w.gesture_velocity_mean
            if current < baseline_movement * MOTOR_INHIBIT_THRESHOLD:
                reduction_pct = round((1 - current / baseline_movement) * 100, 1)
                meta: dict = {
                    "rule_id":         "MOTOR-INHIBIT-01",
                    "quality_tier":    tier.value,
                    "baseline_mean":   round(baseline_movement, 4),
                    "current_mean":    round(current, 4),
                    "reduction_pct":   reduction_pct,
                    "research_note":   "OPPOSITE of folk wisdom. Liars show DECREASED fidgeting (DePaulo 2003: d = -0.10). Mechanism: inhibitory control + cognitive load.",
                    "context":         "Indicates cognitive effort — not necessarily deception. Occurs during any complex mental task.",
                    "recommendation":  "Pair with blink pattern and response latency.",
                }
                if tier == VideoQualityTier.CCTV_QUALITY:
                    meta["quality_disclaimer"] = (
                        "Confidence reduced from 0.35 to 0.15 — CCTV resolution limits "
                        "gesture velocity accuracy. Only gross movement changes are reliable."
                    )
                signals.append({
                    "agent":            "video",
                    "speaker_id":       spk,
                    "signal_type":      "motor_inhibition",
                    "value":            round(reduction_pct / 100, 3),
                    "value_text":       "reduced_movement",
                    "confidence":       conf,
                    "window_start_ms":  w.window_start_ms,
                    "window_end_ms":    w.window_end_ms,
                    "metadata":         meta,
                })
        return signals

    # ── INAPPROPRIATE-AFFECT-01 ───────────────────────────────────────────────

    def _inappropriate_affect(
        self,
        spk: str,
        windows: list["WindowFeatures"],
        diar_segments: list[dict],
        tier: VideoQualityTier,
    ) -> list[dict]:
        """
        FBI (2017): inappropriate affect in psychopaths during crime discussion.
        PMC (2024): duping delight exists but manifests as micro-expression/smirk.
        Confidence: 0.45 (HIGH_QUALITY) / 0.30 (CCTV — coarse smile detection only).
        """
        conf = _QUALITY_CONF["smile_context_incongruence"][tier]

        signals: list[dict] = []
        for w in windows:
            smile_score = _get_smile_score(w)
            if smile_score < _SMILE_THRESHOLD:
                continue

            is_duchenne = _is_duchenne(w)
            # Non-Duchenne smiles are less diagnostic (social masking is common)
            if not is_duchenne:
                continue

            context_text = _segment_text_near_window(
                diar_segments, w.window_start_ms, w.window_end_ms
            )
            if not _SEVERE_NEGATIVE_RE.search(context_text):
                continue

            keywords_found = _SEVERE_NEGATIVE_RE.findall(context_text.lower())[:5]
            meta: dict = {
                "rule_id":          "INAPPROPRIATE-AFFECT-01",
                "quality_tier":     tier.value,
                "smile_score":      round(smile_score, 3),
                "is_duchenne":      True,
                "context_keywords": list(set(keywords_found)),
                "severity":         "HIGH",
                "interpretations": [
                    "Nervous laughter: anxiety-related inappropriate smiling",
                    "Emotional dysregulation: inability to modulate affect",
                    "Contempt or superiority toward victim/severity",
                    "Inappropriate affect: possible psychopathic traits (requires PCL-R)",
                ],
                "recommendation":   "Flag for psychological evaluation. Cannot diagnose psychopathy from behavior alone — requires PCL-R assessment.",
            }
            if tier == VideoQualityTier.CCTV_QUALITY:
                meta["quality_disclaimer"] = (
                    "Confidence reduced from 0.45 to 0.30 — CCTV resolution limits "
                    "facial AU precision. Only broad smile patterns are reliable."
                )
            signals.append({
                "agent":            "video",
                "speaker_id":       spk,
                "signal_type":      "smile_context_incongruence",
                "value":            round(smile_score, 3),
                "value_text":       "duchenne_in_negative_context",
                "confidence":       conf,
                "window_start_ms":  w.window_start_ms,
                "window_end_ms":    w.window_end_ms,
                "metadata":         meta,
            })
        return signals

    # ── GAZE-SACCADES-01 ─────────────────────────────────────────────────────

    def _gaze_saccades(
        self,
        spk: str,
        windows: list["WindowFeatures"],
        bl: "GazeBaseline",
        tier: VideoQualityTier,
    ) -> list[dict]:
        """
        VPS (2025), PMC (2017 Aviation): anxiety correlates with RANDOMNESS of
        eye movements, NOT direction. NLP directional cues (Wiseman 2012) = debunked.
        Confidence: 0.40 (HIGH_QUALITY) / 0.15 (CCTV — frequency only, no direction).
        """
        baseline_gaze_spread = bl.gaze_x_std_mean + bl.gaze_y_std_mean
        if baseline_gaze_spread <= 0:
            return []

        conf = _QUALITY_CONF["erratic_gaze_pattern"][tier]

        signals: list[dict] = []
        for w in windows:
            current_spread = w.gaze_x_std + w.gaze_y_std
            if current_spread >= baseline_gaze_spread * GAZE_SACCADE_RATIO:
                ratio = round(current_spread / max(baseline_gaze_spread, 1e-6), 2)
                meta: dict = {
                    "rule_id":            "GAZE-SACCADES-01",
                    "quality_tier":       tier.value,
                    "baseline_spread":    round(baseline_gaze_spread, 4),
                    "current_spread":     round(current_spread, 4),
                    "elevation_ratio":    ratio,
                    "nlp_warning":        "NLP eye-accessing cues (looking left = lying, right = truth) are COMPLETELY DEBUNKED (Wiseman 2012). Only frequency/randomness matter, NOT direction.",
                    "interpretations": [
                        "High cognitive load — processing complex information",
                        "Severe anxiety — stress response, hypervigilance",
                        "Environmental scanning — assessing surroundings",
                        "Avoidance — discomfort with sustained eye contact",
                        "Fatigue — extended interrogation causing attention lapses",
                        "Cultural communication style",
                    ],
                    "recommendation":     "Pair with other cognitive load indicators. Low confidence due to multiple confounds.",
                }
                if tier == VideoQualityTier.CCTV_QUALITY:
                    meta["quality_disclaimer"] = (
                        "Confidence reduced from 0.40 to 0.15 — CCTV frame rate limits "
                        "gaze tracking precision. Only gross frequency changes are reliable."
                    )
                signals.append({
                    "agent":            "video",
                    "speaker_id":       spk,
                    "signal_type":      "erratic_gaze_pattern",
                    "value":            round(current_spread, 4),
                    "value_text":       "elevated_gaze_randomness",
                    "confidence":       conf,
                    "window_start_ms":  w.window_start_ms,
                    "window_end_ms":    w.window_end_ms,
                    "metadata":         meta,
                })
        return signals

    # ── INTERROG-BODY-02: Freezing Response ──────────────────────────────────

    def _freezing_response(
        self,
        spk: str,
        windows: list["WindowFeatures"],
        bl: "BodyBaseline",
        tier: VideoQualityTier,
        handcuffed: bool = False,
    ) -> list[dict]:
        """
        Navarro (2008), Hagenaars et al. (2014): PAG-mediated freeze response.
        Breath-holding + movement cessation after accusatory statements.
        Confidence: 0.55 (HIGH_QUALITY) / 0.35 (CCTV — coarse movement only).
        HANDCUFFED: capped at 0.40 — body can still freeze, but baseline movement
        is already artificially suppressed by restraints, reducing diagnostic value.
        """
        baseline_movement = bl.body_movement_mean
        if baseline_movement <= 0:
            return []

        conf = _QUALITY_CONF["freezing_response"][tier]
        if handcuffed:
            conf = min(conf, 0.40)
        disclaimer = (
            "Confidence reduced from 0.55 to 0.35 — CCTV resolution limits "
            "fine movement detection. Only prolonged gross-movement freezes are reliable."
        ) if tier == VideoQualityTier.CCTV_QUALITY else None

        signals: list[dict] = []
        freeze_start_idx: int | None = None

        for i, w in enumerate(windows):
            is_frozen = w.body_movement_mean < baseline_movement * FREEZE_MOVEMENT_RATIO

            if is_frozen:
                if freeze_start_idx is None:
                    freeze_start_idx = i
            else:
                if freeze_start_idx is not None:
                    freeze_len = i - freeze_start_idx
                    if freeze_len >= FREEZE_MIN_WINDOWS:
                        start_w = windows[freeze_start_idx]
                        end_w   = windows[i - 1]
                        duration_ms = end_w.window_end_ms - start_w.window_start_ms
                        meta: dict = {
                            "rule_id":           "INTERROG-BODY-02",
                            "quality_tier":      tier.value,
                            "duration_ms":       duration_ms,
                            "window_count":      freeze_len,
                            "baseline_movement": round(baseline_movement, 4),
                            "interpretation":    "Limbic freeze response detected. Indicates threat perception, extreme stress, or fear.",
                            "context":           "NOT a deception marker — common in BOTH guilty and innocent suspects under accusation. First-stage defense response to perceived inescapable threat.",
                        }
                        if disclaimer:
                            meta["quality_disclaimer"] = disclaimer
                        signals.append({
                            "agent":            "video",
                            "speaker_id":       spk,
                            "signal_type":      "freezing_response",
                            "value":            round(duration_ms / 1000, 2),
                            "value_text":       f"{duration_ms // 1000}s_freeze",
                            "confidence":       conf,
                            "window_start_ms":  start_w.window_start_ms,
                            "window_end_ms":    end_w.window_end_ms,
                            "metadata":         meta,
                        })
                    freeze_start_idx = None

        # Catch freeze at end of session
        if freeze_start_idx is not None:
            freeze_len = len(windows) - freeze_start_idx
            if freeze_len >= FREEZE_MIN_WINDOWS:
                start_w = windows[freeze_start_idx]
                end_w   = windows[-1]
                duration_ms = end_w.window_end_ms - start_w.window_start_ms
                meta = {
                    "rule_id":           "INTERROG-BODY-02",
                    "quality_tier":      tier.value,
                    "duration_ms":       duration_ms,
                    "window_count":      freeze_len,
                    "baseline_movement": round(baseline_movement, 4),
                    "interpretation":    "Limbic freeze response detected.",
                    "context":           "NOT a deception marker — common in both guilty and innocent under accusation.",
                }
                if disclaimer:
                    meta["quality_disclaimer"] = disclaimer
                signals.append({
                    "agent":            "video",
                    "speaker_id":       spk,
                    "signal_type":      "freezing_response",
                    "value":            round(duration_ms / 1000, 2),
                    "value_text":       f"{duration_ms // 1000}s_freeze",
                    "confidence":       conf,
                    "window_start_ms":  start_w.window_start_ms,
                    "window_end_ms":    end_w.window_end_ms,
                    "metadata":         meta,
                })
        return signals

    # ── INTERROG-BODY-03: Self-Adaptor Increase ──────────────────────────────

    def _self_adaptor_increase(
        self,
        spk: str,
        windows: list["WindowFeatures"],
        tier: VideoQualityTier,
        handcuffed: bool = False,
    ) -> list[dict]:
        """
        Li et al. (2024), DePaulo et al. (2003) — d=0.10 (small effect).
        Liars exhibit higher frequency of self-adaptors (face/hair/arm touching).
        Temporal trend: last-session-third rate / first-session-third rate >= 1.5.

        HANDCUFFED: suppressed — wrist restraint physically prevents self-touch.
        Confidence: 0.35 (HIGH_QUALITY/STANDARD_FPS) / 0.20 (CCTV).
        Temporal-trend concept is an engineering application; no paper validates
        the specific effect size for trend-based self-adaptor detection.
        """
        if handcuffed:
            return []

        if len(windows) < SELF_ADAPTOR_MIN_WINDOWS:
            return []

        n = len(windows)
        third = max(1, n // 3)
        first_third = windows[:third]
        last_third  = windows[n - third:]

        def _mean_self_touch(ws: list) -> float:
            rates = [getattr(w, "self_touch_pct", 0.0) for w in ws]
            return sum(rates) / len(rates) if rates else 0.0

        first_rate = _mean_self_touch(first_third)
        last_rate  = _mean_self_touch(last_third)

        denominator = max(first_rate, 0.01)
        ratio = last_rate / denominator

        if ratio < SELF_ADAPTOR_TREND_RATIO:
            return []

        conf  = _QUALITY_CONF["self_adaptor_increase"][tier]
        value = round(min(1.0, (ratio - SELF_ADAPTOR_TREND_RATIO) / SELF_ADAPTOR_TREND_RATIO), 4)

        logger.info(
            "[%s] INTERROG-BODY-03: %s self_adaptor_increase "
            "first_rate=%.3f last_rate=%.3f ratio=%.2f conf=%.2f",
            spk, spk, first_rate, last_rate, ratio, conf,
        )
        return [{
            "agent":            "video",
            "speaker_id":       spk,
            "signal_type":      "self_adaptor_increase",
            "value":            value,
            "value_text":       "increasing_self_touch_trend",
            "confidence":       conf,
            "window_start_ms":  last_third[0].window_start_ms,
            "window_end_ms":    last_third[-1].window_end_ms,
            "metadata": {
                "rule_id":           "INTERROG-BODY-03",
                "quality_tier":      tier.value,
                "first_third_rate":  round(first_rate, 4),
                "last_third_rate":   round(last_rate, 4),
                "trend_ratio":       round(ratio, 3),
                "threshold":         SELF_ADAPTOR_TREND_RATIO,
                "windows_analyzed":  n,
                "research": (
                    "Li et al. 2024 Frontiers in Psychology 15 "
                    "doi:10.3389/fpsyg.2024.1331653 — 'liars exhibited higher "
                    "frequency of self-adaptors'. DePaulo et al. 2003 Psychological "
                    "Bulletin 129(1):74-118 — fidgeting d=0.10 (small effect)."
                ),
                "effect_note": (
                    "Direction supported (d=0.10 for absolute fidgeting rate). "
                    "The 'increasing temporal trend' concept is an engineering "
                    "application — no study quantifies the effect size of "
                    "self-adaptor rate change over session duration."
                ),
                "interpretation": (
                    "Rate of face/hair/arm touching increased across the session. "
                    "Equally present in innocent suspects experiencing mounting "
                    "pressure, fatigue, or discomfort from prolonged sitting."
                ),
            },
        }]
