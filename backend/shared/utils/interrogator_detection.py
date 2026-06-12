# shared/utils/interrogator_detection.py
"""
Shared interrogator-detection helper used by the language, conversation, and
fusion agents.

A speaker is classified as an interrogator when they contribute ≥ threshold
of all detected questions across the session.  A segment counts as a question
if it ends with "?" OR matches the question-starter patterns — making detection
robust to ASR transcripts that omit punctuation.

DSA: O(S) single pass through segments.
"""
from __future__ import annotations

import re
from collections import defaultdict

# Question-starter patterns (mirrors conversation_agent/feature_extractor.py).
# A segment is a question when "?" is present OR the text matches this regex.
_QUESTION_PATTERNS = re.compile(
    r"^\s*("
    r"do you|did you|have you|has he|has she|will you|would you|could you|can you|"
    r"are you|is it|is that|is there|was it|were you|"
    r"what about|what do|what did|what is|what are|what was|what will|"
    r"how does|how do|how did|how is|how are|how was|how will|how would|how can|"
    r"where do|where is|where are|where did|"
    r"when do|when is|when did|when will|"
    r"why do|why is|why did|why would|"
    r"who is|who are|who did|who will|"
    r"don't you|doesn't it|isn't it|aren't you|won't you|"
    r"shall we|should we|shouldn't we"
    r")",
    re.IGNORECASE,
)


def _is_question(text: str) -> bool:
    """True when a segment is a question — punctuation-independent."""
    return "?" in text or bool(_QUESTION_PATTERNS.match(text))


def detect_interrogators(
    segments: list[dict],
    threshold: float = 0.15,
) -> tuple[set[str], str]:
    """
    Detect ALL interrogators by question proportion (punctuation-independent).

    Any speaker whose question count constitutes ≥ threshold of all questions
    is an interrogator.  Question detection uses both "?" presence and regex
    pattern matching so ASR transcripts without punctuation still work.

    Returns:
        (set of all interrogator IDs, primary interrogator ID)
        Primary = speaker with the most questions (used for contamination).
        The full set is used to exclude all interrogators from suspect-only rules.

    Edge case — no questions found: returns (empty set, "").
    """
    question_counts: dict[str, int] = defaultdict(int)
    for seg in segments:
        text = seg.get("text", "") or ""
        spk  = seg.get("speaker", seg.get("speaker_id", ""))
        if spk and _is_question(text):
            # Count "?" marks when present, otherwise treat the segment as one question
            count = text.count("?") if "?" in text else 1
            question_counts[spk] += count

    if not question_counts:
        return set(), ""

    total_questions = sum(question_counts.values())
    interrogators = {
        spk for spk, cnt in question_counts.items()
        if cnt / max(total_questions, 1) >= threshold
    }
    primary = max(question_counts, key=question_counts.get)
    interrogators.add(primary)
    return interrogators, primary
