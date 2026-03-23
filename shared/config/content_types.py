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
}


def get_config(content_type: str) -> dict:
    """Return config for a content type, falling back to sales_call."""
    return CONTENT_TYPE_CONFIG.get(content_type, CONTENT_TYPE_CONFIG["sales_call"])


def get_roles(content_type: str) -> list[str]:
    """Return expected roles for a content type."""
    return get_config(content_type)["roles"]
