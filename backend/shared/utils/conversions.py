"""
Shared type-conversion helpers for NEXUS agents.

These handle values that may arrive as strings from Redis, None from
optional fields, or empty strings from JSON deserialization.
"""


def to_float(v) -> float:
    """Safely convert a value (possibly string from Redis) to float."""
    if v is None or v == "":
        return 0.0
    try:
        return float(v)
    except (ValueError, TypeError):
        return 0.0


def to_int(v) -> int:
    """Safely convert a value (possibly string from Redis) to int."""
    if v is None or v == "":
        return 0
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return 0
