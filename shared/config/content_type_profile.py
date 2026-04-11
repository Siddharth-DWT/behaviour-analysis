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
            "VOICE-TALK-01": RuleProfile(thresholds={"significant_pct": 60.0}),
            "CONVO-DOM-01": RuleProfile(thresholds={"expected_dominant_pct": 65.0}),
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
            "LANG-PERS-01": RuleProfile(confidence_multiplier=0.7),
            "FUSION-07": RuleProfile(thresholds={"max_confidence": 0.60}),
        },
        signal_renames={
            # Matrix: RENAME buying_signal → engagement_signal, objection → concern,
            #         persuasion_technique → influence_tactic
            "buying_signal": "engagement_signal",
            "objection_signal": "concern",
            "persuasion_technique": "influence_tactic",
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
            "LANG-BUY-01": RuleProfile(gated=True),
            # LANG-PERS-01: matrix says RENAME "influence_attempt", NOT gate
            "FUSION-13": RuleProfile(gated=True),
            # FUSION-02 removed: matrix says FIRE for internal meetings
            # FUSION-07 removed: matrix says FIRE for internal meetings
            "VOICE-FILLER-01": RuleProfile(thresholds={"spike_delta": 0.75}),
            # noticeable threshold +0.5% from default 2.5% (Bortfeld 2001: informal speech)
            "VOICE-FILLER-02": RuleProfile(thresholds={"noticeable_pct": 3.0}),
            # >50% any peer = significant in peer/internal meetings
            "VOICE-TALK-01": RuleProfile(thresholds={"significant_pct": 50.0}),
            # Professional speech runs lower emotional density (Tausczik 2010)
            "LANG-SENT-02": RuleProfile(thresholds={"high_pct": 0.06, "suppressed_pct": 0.015}),
        },
        signal_renames={
            # Matrix: disagreement (not concern_raised), influence_attempt (not gated),
            #         stonewalling → disengagement (Cortina: may be adaptive boundary-setting)
            "objection_signal": "disagreement",
            "persuasion_technique": "influence_attempt",
            "stonewalling": "disengagement",
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
            "VOICE-STRESS-01": RuleProfile(thresholds={"stress_offset": 0.15}),
            "VOICE-FILLER-01": RuleProfile(thresholds={"spike_delta": 1.00}),
            "VOICE-FILLER-02": RuleProfile(gated=True),
            "VOICE-PITCH-01": RuleProfile(thresholds={"mild_pct": 12.0}),
            "VOICE-PITCH-02": RuleProfile(thresholds={"variance_drop_pct": 50.0}),
            "VOICE-RATE-01": RuleProfile(thresholds={"anomaly_pct": 35.0}),
            "VOICE-TONE-03": RuleProfile(confidence_multiplier=0.6),
            "VOICE-TONE-04": RuleProfile(confidence_multiplier=1.2),
            "VOICE-PAUSE-01": RuleProfile(thresholds={"extended_pause_ms": 3000.0}),
            # LANG-PERS-01: matrix says RENAME "impression_management", NOT gate
            "LANG-NEG-01": RuleProfile(confidence_multiplier=0.7),
            "LANG-CLAR-01": RuleProfile(confidence_multiplier=1.2),
            "FUSION-13": RuleProfile(gated=True),
            "FUSION-02": RuleProfile(thresholds={"max_confidence": 0.40}),
            "FUSION-07": RuleProfile(thresholds={"confidence_floor": 0.35, "max_confidence": 0.50}),
            "FUSION-GRAPH-01": RuleProfile(thresholds={"min_signals": 4.0}),
            "FUSION-GRAPH-03": RuleProfile(thresholds={"threshold_bonus": 3.0}),
            "CONVO-TURN-01": RuleProfile(thresholds={"monologue_per_min": 1.0}),
            "CONVO-LAT-01": RuleProfile(thresholds={"delayed_ms": 2500.0}),
            "CONVO-DOM-01": RuleProfile(thresholds={"expected_dominant_pct": 70.0}),
            "CONVO-BAL-01": RuleProfile(thresholds={"expected_gini_low": 0.20, "expected_gini_high": 0.40}),
            "CONVO-CONF-01": RuleProfile(thresholds={"min_indicators": 3.0}),
            "VOICE-TALK-01": RuleProfile(thresholds={"significant_pct": 70.0}),
        },
        signal_renames={
            # Matrix: interest_signal, hesitation, impression_management (not gated),
            #         stonewalling → disengagement, defensiveness → resistance
            "buying_signal": "interest_signal",
            "objection_signal": "hesitation",
            "persuasion_technique": "impression_management",
            "stonewalling": "disengagement",
            "defensiveness": "resistance",
            "aggressive": "assertive",
            "cold": "low_energy",
            "normal": "deliberative",
        },
        interpretations={
            "vocal_stress_score": "Interview stress is expected and less diagnostic",
            "filler_detection": "Cognitive load during complex answers",
            "interruption_event": "Interviewer interruptions = topic control",
            "power_language_score": "Powerful = competent, hedging may be appropriate",
            "empathy_language": "Soft skill signal for candidate assessment",
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
            "LANG-BUY-01": RuleProfile(gated=True),
            "LANG-OBJ-01": RuleProfile(gated=True),
            "LANG-PERS-01": RuleProfile(gated=True),
            "LANG-NEG-01": RuleProfile(gated=True),
            "FUSION-02": RuleProfile(gated=True),
            "FUSION-07": RuleProfile(gated=True),
            "FUSION-13": RuleProfile(gated=True),
            "FUSION-GRAPH-03": RuleProfile(gated=True),
            "CONVO-CONF-01": RuleProfile(gated=True),
            "VOICE-FILLER-01": RuleProfile(thresholds={"spike_delta": 0.30}),
            "VOICE-INT-01": RuleProfile(thresholds={"overlap_ms": 400.0}),
            # Matrix: flag host if >50% (guest should dominate 60-80%)
            "VOICE-TALK-01": RuleProfile(thresholds={"significant_pct": 50.0}),
            "CONVO-TURN-01": RuleProfile(thresholds={"monologue_per_min": 0.5}),
            "CONVO-LAT-01": RuleProfile(thresholds={"delayed_ms": 3000.0}),
            "CONVO-BAL-01": RuleProfile(thresholds={"expected_gini_low": 0.30, "expected_gini_high": 0.50}),
            "LANG-CLAR-01": RuleProfile(confidence_multiplier=1.2),
        },
        signal_renames={
            "aggressive": "assertive",
            "normal": "deliberative",
        },
        interpretations={
            "excited": "Positive engagement signal",
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
