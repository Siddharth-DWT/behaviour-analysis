"""
NEXUS Shared — Content Type Auto-Classifier
Uses the LLM (via shared/utils/llm_client.py) to classify content type
from the first ~2 minutes of transcript.

Content types determine which rules to emphasize and which report
template to use in the Language Agent and Fusion Agent.

Supported types:
  - sales_call:            Buying signals, objections, close probability
  - podcast:               Speaker dynamics, topic flow, engagement peaks
  - interview:             Confidence trajectory, question quality, rapport
  - lecture:               Clarity, pacing, energy, audience engagement
  - debate:                Argument strength, dominance, persuasion
  - meeting:               Participation balance, decisions, action items
  - presentation:          Clarity, persuasion, audience engagement
  - casual_conversation:   General dynamics, rapport, engagement
  - other:                 Default general communication analysis

Usage:
    from shared.utils.content_classifier import classify_content_type

    # Sync
    content_type = classify_content_type_sync(transcript_segments)

    # Async
    content_type = await classify_content_type(transcript_segments)
"""
import os
import sys
import json
import logging
import asyncio
from pathlib import Path
from typing import Optional

logger = logging.getLogger("nexus.content_classifier")

# Add shared to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ── Valid content types ──
CONTENT_TYPES = [
    "sales_call",
    "podcast",
    "interview",
    "lecture",
    "debate",
    "meeting",
    "presentation",
    "casual_conversation",
    "other",
]

# Default type when classification fails
DEFAULT_CONTENT_TYPE = "other"

# How much transcript to use for classification (in characters, ~2 minutes)
MAX_TRANSCRIPT_CHARS = 4000
MAX_SEGMENTS_FOR_CLASSIFICATION = 30


def _extract_classification_text(segments: list[dict]) -> str:
    """
    Extract the first ~2 minutes of transcript text for classification.
    Includes speaker labels for context.
    """
    lines = []
    total_chars = 0

    for seg in segments[:MAX_SEGMENTS_FOR_CLASSIFICATION]:
        speaker = seg.get("speaker", "Unknown")
        text = seg.get("text", "").strip()
        if not text:
            continue

        line = f"[{speaker}]: {text}"
        lines.append(line)
        total_chars += len(line)

        if total_chars >= MAX_TRANSCRIPT_CHARS:
            break

    return "\n".join(lines)


def _build_classification_prompt(transcript_text: str) -> tuple[str, str]:
    """Build the system and user prompts for content type classification."""
    system_prompt = (
        "You are a content type classifier for a behavioural analysis system. "
        "You classify conversations and recordings into exactly one category "
        "based on the transcript content. Return only structured JSON."
    )

    user_prompt = f"""Classify this transcript into exactly ONE content type.

Available types:
- sales_call: A sales conversation (pitch, negotiation, demo, cold call, qualification)
- podcast: A podcast episode (interviews, discussions, monologues with host/guests)
- interview: A job interview, journalistic interview, or Q&A session
- lecture: An educational presentation, class, or training session
- debate: A structured or unstructured debate, argument, or adversarial discussion
- meeting: A business meeting (team sync, planning, standup, retrospective)
- presentation: A formal presentation, keynote, or pitch deck walkthrough
- casual_conversation: Informal chat, social conversation, or unstructured talk
- other: None of the above categories fit

Analyze the transcript below and respond with ONLY a JSON object:
{{"content_type": "one_of_the_types_above", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}

Transcript (first ~2 minutes):
{transcript_text}"""

    return system_prompt, user_prompt


def classify_content_type_sync(
    segments: list[dict],
    fallback: str = DEFAULT_CONTENT_TYPE,
) -> dict:
    """
    Synchronously classify content type from transcript segments.

    Args:
        segments: List of transcript segment dicts with 'speaker' and 'text'
        fallback: Content type to return if classification fails

    Returns:
        {
            "content_type": str,
            "confidence": float,
            "reasoning": str,
            "method": "llm" | "fallback" | "heuristic"
        }
    """
    if not segments:
        return {
            "content_type": fallback,
            "confidence": 0.0,
            "reasoning": "No transcript segments provided",
            "method": "fallback",
        }

    # Try LLM classification first
    try:
        from shared.utils.llm_client import complete, is_configured

        if not is_configured():
            logger.warning("LLM not configured, using heuristic classification")
            return _heuristic_classify(segments)

        transcript_text = _extract_classification_text(segments)
        if not transcript_text.strip():
            return {
                "content_type": fallback,
                "confidence": 0.0,
                "reasoning": "Empty transcript",
                "method": "fallback",
            }

        system_prompt, user_prompt = _build_classification_prompt(transcript_text)

        response_text = complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=256,
            temperature=0.0,
        )

        # Parse response
        text = response_text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            text = text.rsplit("```", 1)[0].strip()

        result = json.loads(text)

        content_type = result.get("content_type", fallback)
        if content_type not in CONTENT_TYPES:
            logger.warning(f"LLM returned unknown type '{content_type}', using fallback")
            content_type = fallback

        return {
            "content_type": content_type,
            "confidence": min(float(result.get("confidence", 0.5)), 0.95),
            "reasoning": result.get("reasoning", ""),
            "method": "llm",
        }

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM classification response: {e}")
        return _heuristic_classify(segments)
    except ImportError:
        logger.warning("LLM client not available, using heuristic classification")
        return _heuristic_classify(segments)
    except Exception as e:
        logger.warning(f"LLM classification failed: {e}")
        return _heuristic_classify(segments)


async def classify_content_type(
    segments: list[dict],
    fallback: str = DEFAULT_CONTENT_TYPE,
) -> dict:
    """
    Async version of classify_content_type_sync.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        classify_content_type_sync,
        segments,
        fallback,
    )


def _heuristic_classify(segments: list[dict]) -> dict:
    """
    Heuristic content classification when LLM is unavailable.
    Uses keyword matching and structural patterns.
    """
    all_text = " ".join(seg.get("text", "") for seg in segments).lower()
    speakers = set(seg.get("speaker", "") for seg in segments)
    num_speakers = len(speakers)

    # Sales call indicators
    sales_keywords = [
        "pricing", "price", "cost", "budget", "roi", "demo",
        "trial", "pilot", "contract", "enterprise", "plan",
        "onboarding", "implementation", "sign up", "subscribe",
        "proposal", "quote", "discount", "deal",
    ]
    sales_score = sum(1 for kw in sales_keywords if kw in all_text)

    # Interview indicators
    interview_keywords = [
        "tell me about yourself", "experience", "resume",
        "why do you want", "strength", "weakness",
        "previous role", "qualification", "team",
        "your background", "walk me through",
    ]
    interview_score = sum(1 for kw in interview_keywords if kw in all_text)

    # Lecture indicators
    lecture_keywords = [
        "today we'll", "let's learn", "as you can see",
        "slide", "next topic", "in conclusion", "chapter",
        "research shows", "study", "theory", "concept",
        "example of this", "fundamental",
    ]
    lecture_score = sum(1 for kw in lecture_keywords if kw in all_text)

    # Meeting indicators
    meeting_keywords = [
        "action item", "agenda", "standup", "blocker",
        "update on", "status", "next steps", "follow up",
        "sprint", "deadline", "milestone", "priority",
    ]
    meeting_score = sum(1 for kw in meeting_keywords if kw in all_text)

    # Podcast indicators
    podcast_keywords = [
        "welcome to", "today's guest", "episode",
        "listeners", "audience", "subscribe", "show notes",
        "podcast", "tune in",
    ]
    podcast_score = sum(1 for kw in podcast_keywords if kw in all_text)

    # Score mapping
    scores = {
        "sales_call": sales_score,
        "interview": interview_score,
        "lecture": lecture_score,
        "meeting": meeting_score,
        "podcast": podcast_score,
    }

    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]

    if best_score >= 3:
        confidence = min(0.3 + best_score * 0.1, 0.70)
        return {
            "content_type": best_type,
            "confidence": confidence,
            "reasoning": f"Heuristic: {best_score} keyword matches for {best_type}",
            "method": "heuristic",
        }

    # Default based on speaker count
    if num_speakers == 1:
        return {
            "content_type": "lecture",
            "confidence": 0.30,
            "reasoning": "Single speaker detected, defaulting to lecture",
            "method": "heuristic",
        }
    elif num_speakers == 2:
        return {
            "content_type": "sales_call",
            "confidence": 0.25,
            "reasoning": "Two speakers detected, defaulting to sales_call",
            "method": "heuristic",
        }
    else:
        return {
            "content_type": "meeting",
            "confidence": 0.25,
            "reasoning": f"{num_speakers} speakers detected, defaulting to meeting",
            "method": "heuristic",
        }
