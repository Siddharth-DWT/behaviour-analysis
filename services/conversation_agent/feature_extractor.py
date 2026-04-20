# services/conversation_agent/feature_extractor.py
"""
NEXUS Conversation Agent - Feature Extractor
Extracts dialogue dynamics features from diarised transcript segments.

Computes per-speaker, per-pair, and session-level conversation metrics:
  - Talk time distribution and turn-taking patterns
  - Response latency and overlap detection
  - Interruption counting (overlap > 200ms)
  - Back-channel detection (short affirmative utterances)
  - Monologue detection (3+ consecutive same-speaker segments)
  - Dominance index (Gini coefficient of talk time)
  - Question detection (punctuation + pattern matching)

All timing is in milliseconds. Segments must have: speaker, start_ms, end_ms, text.

Research references:
  - Sacks, Schegloff & Jefferson 1974 (turn-taking organisation)
  - Tannen 1994 (conversational style — overlap vs. interruption)
  - Gravano & Hirschberg 2011 (turn-taking cues in dialogue)
  - Heldner & Edlund 2010 (pauses, gaps, overlaps in conversation)
"""
import re
import logging
from collections import defaultdict
from typing import Optional

logger = logging.getLogger("nexus.conversation.features")

# Back-channel words/phrases (short affirmative responses)
BACK_CHANNEL_WORDS = {
    "yeah", "yes", "uh huh", "uh-huh", "right", "okay", "ok",
    "sure", "mmm", "hmm", "mm-hmm", "mm hmm", "mhm", "yep",
    "got it", "i see", "exactly", "absolutely", "true",
}

# Question patterns (when "?" is not present)
QUESTION_PATTERNS = re.compile(
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

# Continuation gap: same speaker consecutive segments within this gap = same turn
CONTINUATION_GAP_MS = 2000

# Interruption threshold: overlap must exceed this to count as interruption
INTERRUPTION_OVERLAP_MS = 200

# Back-channel constraints
BACK_CHANNEL_MAX_WORDS = 3
BACK_CHANNEL_MAX_DURATION_MS = 1500

# Monologue: 3+ consecutive segments from same speaker
MONOLOGUE_MIN_SEGMENTS = 3


class ConversationFeatureExtractor:
    """
    Extracts conversation dynamics features from diarised transcript segments.
    """

    def __init__(self):
        logger.info("ConversationFeatureExtractor initialised")

    def extract_all(
        self,
        segments: list[dict],
        speakers: list[str] = None,
    ) -> dict:
        """
        Extract all conversation features.

        Args:
            segments: List of {speaker, start_ms, end_ms, text} dicts.
            speakers: Optional explicit speaker list. Auto-detected if omitted.

        Returns:
            {per_speaker: {...}, per_pair: {...}, session: {...}}
        """
        if not segments:
            return self._empty_features()

        # Normalise and sort segments by start time
        segs = self._normalise_segments(segments)
        if not segs:
            return self._empty_features()

        # Detect speakers
        if not speakers:
            speakers = sorted(set(s["speaker"] for s in segs))

        # Build turns (merge consecutive same-speaker segments within gap)
        turns = self._build_turns(segs)

        # Compute features
        per_speaker = self._compute_per_speaker(segs, turns, speakers)
        per_pair = self._compute_per_pair(turns, speakers)
        session = self._compute_session(segs, turns, per_speaker, speakers)

        return {
            "per_speaker": per_speaker,
            "per_pair": per_pair,
            "session": session,
        }

    # ──────────────────────────────────────────────────────
    # Normalisation & Turn Building
    # ──────────────────────────────────────────────────────

    def _normalise_segments(self, segments: list[dict]) -> list[dict]:
        """Normalise segment format and sort by start_ms."""
        normalised = []
        for seg in segments:
            speaker = seg.get("speaker", "unknown")
            start_ms = int(seg.get("start_ms", 0))
            end_ms = int(seg.get("end_ms", 0))
            text = str(seg.get("text", "")).strip()

            if end_ms <= start_ms:
                continue  # Skip zero/negative duration segments

            normalised.append({
                "speaker": speaker,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "text": text,
                "word_count": len(text.split()) if text else 0,
            })

        normalised.sort(key=lambda s: s["start_ms"])
        return normalised

    def _build_turns(self, segs: list[dict]) -> list[dict]:
        """
        Build turns from segments. A turn is a contiguous period of speech
        by one speaker. Consecutive segments from the same speaker with
        a gap < CONTINUATION_GAP_MS are merged into a single turn.
        """
        if not segs:
            return []

        turns = []
        current = {
            "speaker": segs[0]["speaker"],
            "start_ms": segs[0]["start_ms"],
            "end_ms": segs[0]["end_ms"],
            "texts": [segs[0]["text"]],
            "word_count": segs[0]["word_count"],
            "segment_count": 1,
        }

        for seg in segs[1:]:
            gap = seg["start_ms"] - current["end_ms"]
            same_speaker = seg["speaker"] == current["speaker"]

            if same_speaker and gap < CONTINUATION_GAP_MS:
                # Continuation of current turn
                current["end_ms"] = max(current["end_ms"], seg["end_ms"])
                current["texts"].append(seg["text"])
                current["word_count"] += seg["word_count"]
                current["segment_count"] += 1
            else:
                # New turn
                current["text"] = " ".join(current["texts"])
                turns.append(current)
                current = {
                    "speaker": seg["speaker"],
                    "start_ms": seg["start_ms"],
                    "end_ms": seg["end_ms"],
                    "texts": [seg["text"]],
                    "word_count": seg["word_count"],
                    "segment_count": 1,
                }

        # Don't forget the last turn
        current["text"] = " ".join(current["texts"])
        turns.append(current)

        return turns

    # ──────────────────────────────────────────────────────
    # Per-Speaker Features
    # ──────────────────────────────────────────────────────

    def _compute_per_speaker(
        self,
        segs: list[dict],
        turns: list[dict],
        speakers: list[str],
    ) -> dict:
        """Compute per-speaker conversation features."""
        result = {}

        for spk in speakers:
            spk_segs = [s for s in segs if s["speaker"] == spk]
            spk_turns = [t for t in turns if t["speaker"] == spk]

            # Talk time
            talk_time_ms = sum(s["end_ms"] - s["start_ms"] for s in spk_segs)
            total_talk = sum(s["end_ms"] - s["start_ms"] for s in segs)
            talk_time_pct = (talk_time_ms / total_talk * 100) if total_talk > 0 else 0

            # Segments
            segment_count = len(spk_segs)
            durations = [s["end_ms"] - s["start_ms"] for s in spk_segs]
            avg_seg_duration_ms = sum(durations) / len(durations) if durations else 0
            longest_seg_ms = max(durations) if durations else 0

            # Words
            word_count = sum(s["word_count"] for s in spk_segs)
            avg_words_per_turn = word_count / len(spk_turns) if spk_turns else 0

            # Questions
            questions_asked = self._count_questions(spk_segs)

            # Interruptions
            interruption_count, was_interrupted_count = self._count_interruptions_for_speaker(
                spk, turns,
            )

            # Back-channels
            back_channel_count = self._count_back_channels(spk_segs)

            # Monologues
            monologue_count = self._count_monologues(spk, segs)

            # Average silence after this speaker's segments
            silence_after_values = []
            for i, seg in enumerate(segs):
                if seg["speaker"] == spk and i + 1 < len(segs):
                    gap = segs[i + 1]["start_ms"] - seg["end_ms"]
                    if gap > 0:
                        silence_after_values.append(gap)
            silence_after_ms_avg = (
                sum(silence_after_values) / len(silence_after_values)
                if silence_after_values else 0
            )

            result[spk] = {
                "talk_time_ms": talk_time_ms,
                "talk_time_pct": round(talk_time_pct, 2),
                "segment_count": segment_count,
                "avg_segment_duration_ms": round(avg_seg_duration_ms, 1),
                "longest_segment_ms": longest_seg_ms,
                "word_count": word_count,
                "avg_words_per_turn": round(avg_words_per_turn, 1),
                "questions_asked": questions_asked,
                "interruption_count": interruption_count,
                "was_interrupted_count": was_interrupted_count,
                "back_channel_count": back_channel_count,
                "monologue_count": monologue_count,
                "silence_after_ms_avg": round(silence_after_ms_avg, 1),
            }

        return result

    # ──────────────────────────────────────────────────────
    # Per-Pair Features
    # ──────────────────────────────────────────────────────

    def _compute_per_pair(
        self,
        turns: list[dict],
        speakers: list[str],
    ) -> dict:
        """Compute per-speaker-pair features (response latency, overlap, etc.)."""
        result = {}

        for i, spk_a in enumerate(speakers):
            for spk_b in speakers[i + 1:]:
                pair_key = f"{spk_a}__{spk_b}"

                latencies = []
                overlaps = []
                turn_exchanges = 0
                qa_pairs = 0

                for idx in range(1, len(turns)):
                    prev = turns[idx - 1]
                    curr = turns[idx]

                    # Only look at transitions between this pair
                    if not (
                        (prev["speaker"] == spk_a and curr["speaker"] == spk_b) or
                        (prev["speaker"] == spk_b and curr["speaker"] == spk_a)
                    ):
                        continue

                    turn_exchanges += 1
                    latency = curr["start_ms"] - prev["end_ms"]
                    latencies.append(latency)

                    if latency < 0:
                        overlaps.append(abs(latency))

                    # Q→A pair: previous turn ends with question, next turn responds
                    if self._is_question(prev.get("text", "")):
                        qa_pairs += 1

                # Compute stats
                latencies_positive = [l for l in latencies if l >= 0]
                avg_latency = (
                    sum(latencies) / len(latencies) if latencies else 0
                )
                median_latency = self._median(latencies) if latencies else 0

                result[pair_key] = {
                    "response_latency_ms_avg": round(avg_latency, 1),
                    "response_latency_ms_median": round(median_latency, 1),
                    "turn_exchanges": turn_exchanges,
                    "overlap_count": len(overlaps),
                    "overlap_total_ms": sum(overlaps),
                    "question_answer_pairs": qa_pairs,
                }

        return result

    # ──────────────────────────────────────────────────────
    # Session-Level Features
    # ──────────────────────────────────────────────────────

    def _compute_session(
        self,
        segs: list[dict],
        turns: list[dict],
        per_speaker: dict,
        speakers: list[str],
    ) -> dict:
        """Compute session-level conversation features."""
        if not segs:
            return self._empty_session()

        total_duration_ms = segs[-1]["end_ms"] - segs[0]["start_ms"]
        if total_duration_ms <= 0:
            total_duration_ms = 1  # Avoid division by zero

        # Total speech time
        total_speech_ms = sum(s["end_ms"] - s["start_ms"] for s in segs)
        total_silence_ms = max(0, total_duration_ms - total_speech_ms)
        silence_pct = (total_silence_ms / total_duration_ms * 100) if total_duration_ms > 0 else 0

        # Turns
        total_turns = len(turns)
        turn_durations = [t["end_ms"] - t["start_ms"] for t in turns]
        avg_turn_duration_ms = sum(turn_durations) / len(turn_durations) if turn_durations else 0
        duration_minutes = total_duration_ms / 60000.0
        turn_rate_per_minute = total_turns / duration_minutes if duration_minutes > 0 else 0

        # Longest monologue
        longest_monologue_ms = 0
        longest_monologue_speaker = ""
        for spk in speakers:
            spk_turns = [t for t in turns if t["speaker"] == spk]
            for t in spk_turns:
                dur = t["end_ms"] - t["start_ms"]
                if dur > longest_monologue_ms:
                    longest_monologue_ms = dur
                    longest_monologue_speaker = spk

        # Dominance index (Gini coefficient of talk time distribution)
        talk_times = [per_speaker[spk]["talk_time_ms"] for spk in speakers]
        dominance_index = self._gini_coefficient(talk_times)

        # Interruption rate
        total_interruptions = sum(
            per_speaker[spk]["interruption_count"] for spk in speakers
        )
        interruption_rate = total_interruptions / duration_minutes if duration_minutes > 0 else 0

        return {
            "total_duration_ms": total_duration_ms,
            "total_silence_ms": total_silence_ms,
            "silence_pct": round(silence_pct, 2),
            "speaker_count": len(speakers),
            "total_turns": total_turns,
            "avg_turn_duration_ms": round(avg_turn_duration_ms, 1),
            "turn_rate_per_minute": round(turn_rate_per_minute, 2),
            "longest_monologue_ms": longest_monologue_ms,
            "longest_monologue_speaker": longest_monologue_speaker,
            "dominance_index": round(dominance_index, 4),
            "interruption_rate": round(interruption_rate, 2),
        }

    # ──────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────

    def _count_questions(self, spk_segs: list[dict]) -> int:
        """Count questions in a speaker's segments."""
        count = 0
        for seg in spk_segs:
            if self._is_question(seg.get("text", "")):
                count += 1
        return count

    def _is_question(self, text: str) -> bool:
        """Detect if text is a question (punctuation or pattern)."""
        text = text.strip()
        if not text:
            return False
        if text.endswith("?"):
            return True
        if QUESTION_PATTERNS.search(text):
            return True
        return False

    def _count_interruptions_for_speaker(
        self, speaker: str, turns: list[dict],
    ) -> tuple[int, int]:
        """
        Count how many times this speaker interrupted others, and how many
        times others interrupted this speaker.

        Interruption: B starts before A ends, with overlap > INTERRUPTION_OVERLAP_MS.
        """
        interrupted_others = 0
        was_interrupted = 0

        for i in range(1, len(turns)):
            prev = turns[i - 1]
            curr = turns[i]

            overlap = prev["end_ms"] - curr["start_ms"]
            if overlap <= INTERRUPTION_OVERLAP_MS:
                continue  # Not an interruption

            if curr["speaker"] == speaker and prev["speaker"] != speaker:
                interrupted_others += 1
            elif prev["speaker"] == speaker and curr["speaker"] != speaker:
                was_interrupted += 1

        return interrupted_others, was_interrupted

    def _count_back_channels(self, spk_segs: list[dict]) -> int:
        """Count back-channel utterances (short affirmative responses)."""
        count = 0
        for seg in spk_segs:
            text = seg.get("text", "").strip().lower()
            duration = seg["end_ms"] - seg["start_ms"]
            word_count = seg.get("word_count", len(text.split()))

            if word_count > BACK_CHANNEL_MAX_WORDS:
                continue
            if duration > BACK_CHANNEL_MAX_DURATION_MS:
                continue

            # Check if text matches any back-channel phrase
            # Strip trailing punctuation for matching
            clean = re.sub(r"[.!?,;:\-]+$", "", text).strip()
            if clean in BACK_CHANNEL_WORDS:
                count += 1

        return count

    def _count_monologues(self, speaker: str, segs: list[dict]) -> int:
        """Count monologue runs (3+ consecutive segments from the same speaker)."""
        count = 0
        consecutive = 0

        for seg in segs:
            if seg["speaker"] == speaker:
                consecutive += 1
                if consecutive == MONOLOGUE_MIN_SEGMENTS:
                    count += 1
            else:
                consecutive = 0

        return count

    def _gini_coefficient(self, values: list[float]) -> float:
        """
        Compute Gini coefficient (0 = perfect equality, 1 = total inequality).
        Used as dominance index for talk time distribution.
        """
        if not values or all(v == 0 for v in values):
            return 0.0

        n = len(values)
        if n == 1:
            return 0.0

        sorted_vals = sorted(values)
        total = sum(sorted_vals)
        if total == 0:
            return 0.0

        cumulative = 0.0
        weighted_sum = 0.0
        for i, val in enumerate(sorted_vals):
            cumulative += val
            weighted_sum += (2 * (i + 1) - n - 1) * val

        gini = weighted_sum / (n * total)
        return max(0.0, min(1.0, gini))

    def _median(self, values: list[float]) -> float:
        """Compute the median of a list of values."""
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        mid = n // 2
        if n % 2 == 0:
            return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
        return sorted_vals[mid]

    def _empty_features(self) -> dict:
        """Return empty feature structure."""
        return {
            "per_speaker": {},
            "per_pair": {},
            "session": self._empty_session(),
        }

    def _empty_session(self) -> dict:
        """Return empty session-level features."""
        return {
            "total_duration_ms": 0,
            "total_silence_ms": 0,
            "silence_pct": 0,
            "speaker_count": 0,
            "total_turns": 0,
            "avg_turn_duration_ms": 0,
            "turn_rate_per_minute": 0,
            "longest_monologue_ms": 0,
            "longest_monologue_speaker": "",
            "dominance_index": 0,
            "interruption_rate": 0,
        }
