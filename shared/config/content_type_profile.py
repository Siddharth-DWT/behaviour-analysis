"""
NEXUS Content-Type Profile System

Single class loaded per session. Every rule calls it for:
  - Gating (should this rule fire for this content type?)
  - Threshold adjustment (different thresholds per type)
  - Confidence multiplier (scale confidence up/down)
  - Signal renaming (buying_signal -> candidate_interest)
  - Prompt template selection (different LLM prompts per type)

Usage:
    profile = ContentTypeProfile("interview")
    profile.is_gated("LANG-BUY-01")           # False (fires with rename)
    profile.is_gated("LANG-PERS-01")           # True (suppressed)
    profile.get_threshold("VOICE-FILLER-01", "spike_delta", 0.50)  # 1.0
    profile.get_confidence_multiplier("VOICE-TONE-03")  # 0.6
    profile.rename_signal("buying_signal")      # "candidate_interest"
"""
import logging
from typing import Optional
from pydantic import BaseModel, Field

logger = logging.getLogger("nexus.content_profile")


# ═══════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════


class RuleProfile(BaseModel):
    """Per-rule configuration for a specific content type."""
    gated: bool = False
    confidence_multiplier: float = 1.0
    thresholds: dict[str, float] = Field(default_factory=dict)
    renames: dict[str, str] = Field(default_factory=dict)
    interpretation: Optional[str] = None


class ContentTypeConfig(BaseModel):
    """Full configuration for a single content type across all rules."""
    label: str
    rules: dict[str, RuleProfile] = Field(default_factory=dict)
    intent_categories: list[str] = Field(default_factory=list)
    signal_renames: dict[str, str] = Field(default_factory=dict)
    interpretations: dict[str, str] = Field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# CONTENT TYPE CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════

PROFILES: dict[str, ContentTypeConfig] = {
    "sales_call": ContentTypeConfig(
        label="Sales Call",
        intent_categories=[
            "INFORM", "QUESTION", "REQUEST", "PROPOSE", "AGREE", "DISAGREE",
            "ACKNOWLEDGE", "NEGOTIATE", "COMMIT", "DEFLECT", "RAPPORT",
            "CLOSE", "OBJECTION",
        ],
        rules={
            # Matrix: only 1 threshold change for sales_call (Gong: top performers 43%, underperformers 64%)
            "VOICE-TALK-01": RuleProfile(thresholds={"significant_pct": 60.0}),
        },
        signal_renames={
            "normal": "deliberative",
        },
    ),

    "client_meeting": ContentTypeConfig(
        label="Client Meeting",
        intent_categories=[
            "INFORM", "QUESTION", "REQUEST", "PROPOSE", "AGREE", "DISAGREE",
            "ACKNOWLEDGE", "COMMIT", "PRESENT", "FOLLOW_UP", "GREET",
            "ASSIGN_ACTION", "ESCALATE",
        ],
        rules={
            # Matrix: LANG-PERS-01 confidence reduced — influence attempts less diagnostic in client context
            "LANG-PERS-01": RuleProfile(confidence_multiplier=0.7),
            # Matrix: FUSION-07 FIRE for client_meeting — do NOT cap below 0.65
        },
        signal_renames={
            # Matrix: same detection, different labels for client context
            "buying_signal":              "engagement_signal",       # LANG-BUY-01
            "objection_signal":           "concern",                 # LANG-OBJ-01
            "persuasion_technique":       "influence_tactic",        # LANG-PERS-01
            "purchase_intent_validation": "engagement_verification", # FUSION-05 (Phase 2)
            "decision_readiness":         "commitment_signals",      # COMPOUND-04 (Phase 3)
            "objection_formation":        "concern_formation",       # TEMPORAL-04 (Phase 3)
            "buying_decision_sequence":   "commitment_sequence",     # TEMPORAL-06 (Phase 3)
            "normal": "deliberative",
        },
    ),

    "internal": ContentTypeConfig(
        label="Internal Meeting",
        intent_categories=[
            "INFORM", "QUESTION", "REQUEST", "PROPOSE", "AGREE", "DISAGREE",
            "ACKNOWLEDGE", "COMMIT", "PRESENT", "FOLLOW_UP", "GREET",
            "ASSIGN_ACTION", "FACILITATE",
        ],
        rules={
            # Matrix: GATE — buying signals have no meaning in peer/internal context
            "LANG-BUY-01": RuleProfile(gated=True),
            # Matrix: GATE — urgency/persuasion detection irrelevant (no persuasion context)
            "FUSION-13": RuleProfile(gated=True),
            # Matrix: GATE — no buying/decision progression in peer meetings
            "TEMPORAL-06": RuleProfile(gated=True),
            # Matrix: ADAPT — noticeable threshold +0.5% (Bortfeld 2001: informal speech, 82% more fillers)
            "VOICE-FILLER-02": RuleProfile(thresholds={"noticeable_pct": 3.0}),
            # Matrix: ADAPT — >50% any peer = significant in flat-hierarchy meetings
            "VOICE-TALK-01": RuleProfile(thresholds={"significant_pct": 50.0}),
            # Matrix: ADAPT — professional speech runs lower emotional density (Tausczik 2010, avg ~3-3.5%)
            "LANG-SENT-02": RuleProfile(thresholds={"high_pct": 0.06, "suppressed_pct": 0.015}),
            # Note: VOICE-FILLER-01 is FIRE for internal (matrix row 3) — baseline absorbs variation
            # Note: LANG-PERS-01 is RENAME "influence_attempt", NOT gated (matrix row 23)
        },
        signal_renames={
            # Matrix: same detection, reframed for peer context
            "objection_signal":           "disagreement",            # LANG-OBJ-01
            "persuasion_technique":       "influence_attempt",       # LANG-PERS-01 (Cortina: boundary-setting)
            "stonewalling":               "disengagement",           # LANG-NEG-01 (Cortina: may be adaptive)
            "decision_readiness":         "consensus_signals",       # COMPOUND-04 (Phase 3)
            "objection_formation":        "disagreement_formation",  # TEMPORAL-04 (Phase 3)
            "normal": "deliberative",
        },
    ),

    "interview": ContentTypeConfig(
        label="Interview",
        intent_categories=[
            "INFORM", "QUESTION", "ANSWER", "RESPOND", "ELABORATE",
            "CLARIFY", "PROBE", "AGREE", "ACKNOWLEDGE",
            "INTEREST", "HESITATION", "GREET", "CLOSE",
        ],
        rules={
            # Matrix: GATE — filler credibility unfair in interview (cognitive load, not credibility)
            "VOICE-FILLER-02": RuleProfile(gated=True),
            # Matrix: ADAPT — extended hesitation >3000ms (Stivers 2009: complex answers need formulation time)
            "VOICE-PAUSE-01": RuleProfile(thresholds={"extended_pause_ms": 3000.0}),
            # Matrix: ADAPT — flag only if candidate <30% or >70% (BarRaiser: optimal 40-55%)
            "VOICE-TALK-01": RuleProfile(thresholds={"significant_pct": 70.0}),
            # Matrix: GATE — urgency authenticity inappropriate for interviews
            "FUSION-13": RuleProfile(gated=True),
            # Matrix: ADAPT — expected Gini 0.20-0.40 (candidate talks more than symmetric)
            "CONVO-BAL-01": RuleProfile(thresholds={"expected_gini_low": 0.20, "expected_gini_high": 0.40}),
            # Matrix: structural dominance expected; flag only extreme deviation
            "CONVO-DOM-01": RuleProfile(thresholds={"expected_dominant_pct": 70.0}),
            # Matrix: GATE — deception detection produces unacceptable false positives
            # (DePaulo 2003: stress and deception share overlapping profiles in interviews)
            "COMPOUND-12": RuleProfile(gated=True),
            # All other rules FIRE with universal thresholds — per-speaker baseline absorbs
            # interview-elevated stress, pitch, filler rate (matrix rows 2-13, 30-37)
        },
        signal_renames={
            # Matrix: same detection, candidate-appropriate labels
            "buying_signal":              "interest_signal",         # LANG-BUY-01
            "objection_signal":           "hesitation",              # LANG-OBJ-01
            "persuasion_technique":       "impression_management",   # LANG-PERS-01
            "stonewalling":               "disengagement",           # LANG-NEG-01 (legitimate in interview)
            "defensiveness":              "resistance",              # LANG-NEG-01
            "purchase_intent_validation": "interest_verification",   # FUSION-05 (Phase 2)
            "decision_readiness":         "decision_signals",        # COMPOUND-04 (Phase 3)
            "objection_formation":        "hesitation_formation",    # TEMPORAL-04 (Phase 3)
            "buying_decision_sequence":   "decision_sequence",       # TEMPORAL-06 (Phase 3)
            "aggressive":                 "assertive",
            "cold":                       "low_energy",
            "normal":                     "deliberative",
        },
        interpretations={
            "vocal_stress_score":   "Interview stress is expected and less diagnostic",
            "filler_detection":     "Cognitive load during complex answers",
            "interruption_event":   "Interviewer interruptions = topic control",
            "power_language_score": "Powerful = competent; hedging may be appropriate",
            "empathy_language":     "Soft skill signal for candidate assessment",
        },
    ),

    "podcast": ContentTypeConfig(
        label="Podcast",
        intent_categories=[
            # Matrix: NARRATE replaces ANSWER, add JOKE + ACKNOWLEDGE, remove RAPPORT/SUMMARIZE
            "INFORM", "QUESTION", "NARRATE", "ELABORATE", "AGREE",
            "DISAGREE", "JOKE", "TRANSITION", "ACKNOWLEDGE",
        ],
        rules={
            # Matrix: GATE — sales/persuasion signals don't apply to podcasts
            "LANG-BUY-01":    RuleProfile(gated=True),
            "LANG-OBJ-01":    RuleProfile(gated=True),
            "LANG-PERS-01":   RuleProfile(gated=True),
            "LANG-NEG-01":    RuleProfile(gated=True),
            "FUSION-02":      RuleProfile(gated=True),
            "FUSION-05":      RuleProfile(gated=True),   # Matrix: GATE (audio-visual sales rule)
            "FUSION-07":      RuleProfile(gated=True),   # N/A — no video track for head-nod detection
            "FUSION-13":      RuleProfile(gated=True),
            "FUSION-GRAPH-03": RuleProfile(gated=True),
            "CONVO-CONF-01":  RuleProfile(gated=True),
            "TEMPORAL-06":    RuleProfile(gated=True),   # Matrix: GATE — no buying/decision progression
            # Matrix: ADAPT — podcast speech has more filler (informal context)
            "VOICE-FILLER-01": RuleProfile(thresholds={"spike_delta": 0.30}),
            # Matrix: ADAPT — 400ms overlap threshold (podcast crosstalk is style, not interruption)
            "VOICE-INT-01":    RuleProfile(thresholds={"overlap_ms": 400.0}),
            # Matrix: ADAPT — flag host if >50% (guest should dominate 60-80%)
            "VOICE-TALK-01":   RuleProfile(thresholds={"significant_pct": 50.0}),
            "CONVO-TURN-01":   RuleProfile(thresholds={"monologue_per_min": 0.5}),
            "CONVO-LAT-01":    RuleProfile(thresholds={"delayed_ms": 3000.0}),
            # Matrix: ADAPT — expected Gini 0.30-0.50 (guest dominates)
            "CONVO-BAL-01":    RuleProfile(thresholds={"expected_gini_low": 0.30, "expected_gini_high": 0.50}),
            "LANG-CLAR-01":    RuleProfile(confidence_multiplier=1.2),
        },
        signal_renames={
            # Matrix: 0 renames for podcast
            "normal": "deliberative",
        },
        interpretations={
            "excited":      "Positive engagement signal",
            "energy_level": "Elevated energy is positive for podcasts",
        },
    ),

    "debate": ContentTypeConfig(
        label="Debate",
        intent_categories=[
            "INFORM", "QUESTION", "REQUEST", "PROPOSE", "AGREE", "DISAGREE",
            "ACKNOWLEDGE", "NEGOTIATE", "COMMIT", "DEFLECT", "RAPPORT",
            "CLOSE", "OBJECTION",
        ],
        signal_renames={
            "normal": "deliberative",
        },
    ),
}


# ═══════════════════════════════════════════════════════════════
# PROFILE CLASS
# ═══════════════════════════════════════════════════════════════


class ContentTypeProfile:
    """
    Content-type aware rule configuration.
    Instantiate once per session, pass to all agents/rules.
    Backed by Pydantic models for type safety and validation.
    """

    def __init__(self, content_type: str = "sales_call"):
        self.content_type = content_type
        self._config = PROFILES.get(content_type, PROFILES["sales_call"])

    @property
    def label(self) -> str:
        return self._config.label

    def _get_rule(self, rule_id: str) -> RuleProfile:
        return self._config.rules.get(rule_id, RuleProfile())

    def is_gated(self, rule_id: str) -> bool:
        """Returns True if this rule should NOT fire for the current content type."""
        return self._get_rule(rule_id).gated

    def get_threshold(self, rule_id: str, param: str, default: float) -> float:
        """Get a threshold override for the current content type, or return default."""
        return self._get_rule(rule_id).thresholds.get(param, default)

    def get_confidence_multiplier(self, rule_id: str) -> float:
        """Get confidence multiplier (1.0 = no change) for the current content type."""
        return self._get_rule(rule_id).confidence_multiplier

    def rename_signal(self, signal_type: str) -> str:
        """Rename a signal type for the current content type, or return original."""
        return self._config.signal_renames.get(signal_type, signal_type)

    def get_intent_categories(self) -> list[str]:
        """Get intent classification categories for the current content type."""
        return self._config.intent_categories or PROFILES["sales_call"].intent_categories

    def get_interpretation(self, signal_type: str) -> Optional[str]:
        """Get interpretation note for a signal in the current content type."""
        return self._config.interpretations.get(signal_type)

    def apply_confidence(self, rule_id: str, confidence: float) -> float:
        """Apply confidence multiplier, clamped to [0, 0.85]."""
        mult = self.get_confidence_multiplier(rule_id)
        return min(confidence * mult, 0.85)

    def __repr__(self):
        return f"ContentTypeProfile({self.content_type!r})"
