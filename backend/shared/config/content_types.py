"""
NEXUS Content Type Configuration

Defines roles, entity fields, and speaker stats for each supported
content type. Used by agents to adapt extraction and analysis.
"""

CONTENT_TYPE_CONFIG = {
    "sales_call": {
        "label": "Sales Call",
        "roles": ["Seller", "Prospect"],
        "entity_fields": [
            "objections", "buying_signals", "commitments", "sales_stages",
        ],
        "speaker_stats": {
            "Seller": ["persuasion_count"],
            "Prospect": ["buying_signal_count"],
        },
    },
    "client_meeting": {
        "label": "Client Meeting",
        "roles": ["Account Manager", "Client"],
        "entity_fields": [
            "action_items", "decisions", "satisfaction_indicators", "risk_flags",
        ],
        "speaker_stats": {
            "Client": ["satisfaction"],
            "Account Manager": ["action_items_owned"],
        },
    },
    "internal": {
        "label": "Internal Meeting",
        "roles": ["Facilitator", "Participant"],
        "entity_fields": ["action_items", "decisions"],
        "speaker_stats": {"all": ["talk_time_pct"]},
    },
    "interview": {
        "label": "Interview",
        "roles": ["Interviewer", "Candidate"],
        "entity_fields": [
            "questions_asked", "candidate_strengths", "candidate_concerns",
        ],
        "speaker_stats": {
            "Candidate": ["confidence"],
            "Interviewer": ["questions_asked_count"],
        },
    },
    "podcast": {
        "label": "Podcast",
        "roles": ["Host", "Guest"],
        "entity_fields": ["topics", "key_terms"],
        "speaker_stats": {"all": ["talk_time_pct"]},
    },
    "debate": {
        "label": "Debate",
        "roles": ["Speaker A", "Speaker B"],
        "entity_fields": ["topics", "key_terms"],
        "speaker_stats": {"all": ["talk_time_pct", "stress_avg"]},
    },
    "interrogation_video": {
        "label": "Interrogation Video",
        "roles": ["Interrogator", "Suspect"],
        "entity_fields": [
            "denial_statements", "contradiction_markers",
            "case_facts_introduced", "false_confession_indicators",
        ],
        "speaker_stats": {
            "Suspect":      ["denial_strength", "contamination_count", "stress_avg"],
            "Interrogator": ["tactic_type", "evidence_disclosure_count"],
        },
    },
}


# Speaker count defaults and ranges per meeting type.
# Single source of truth — imported by voiceAgent/transcriber.py and
# video_agent/feature_extractor.py so both services stay aligned.
SPEAKER_DEFAULTS: dict[str, dict] = {
    "sales_call":            {"default": 2, "min": 2, "max": 3,  "turn_gap_ms": 400},
    "interview":             {"default": 2, "min": 2, "max": 4,  "turn_gap_ms": 600},
    "internal":              {"default": 4, "min": 2, "max": 10, "turn_gap_ms": 800},
    "client_meeting":        {"default": 3, "min": 2, "max": 10, "turn_gap_ms": 600},
    "meeting":               {"default": 4, "min": 2, "max": 10, "turn_gap_ms": 800},
    "podcast":               {"default": 2, "min": 2, "max": 4,  "turn_gap_ms": 600},
    "lecture":               {"default": 1, "min": 1, "max": 2,  "turn_gap_ms": 1000},
    "presentation":          {"default": 1, "min": 1, "max": 3,  "turn_gap_ms": 1000},
    "debate":                {"default": 2, "min": 2, "max": 4,  "turn_gap_ms": 400},
    "casual_conversation":   {"default": 2, "min": 2, "max": 4,  "turn_gap_ms": 400},
    "interrogation_video":   {"default": 2, "min": 2, "max": 4,  "turn_gap_ms": 800},
}

_SPEAKER_DEFAULTS_FALLBACK = {"default": 2, "min": 2, "max": 8, "turn_gap_ms": 600}


def get_speaker_defaults(meeting_type: str) -> dict:
    """Return speaker count defaults for a meeting type."""
    return SPEAKER_DEFAULTS.get(meeting_type, _SPEAKER_DEFAULTS_FALLBACK)


def get_config(content_type: str) -> dict:
    """Return config for a content type, falling back to sales_call."""
    return CONTENT_TYPE_CONFIG.get(content_type, CONTENT_TYPE_CONFIG["sales_call"])


def get_roles(content_type: str) -> list[str]:
    """Return expected roles for a content type."""
    return get_config(content_type)["roles"]
