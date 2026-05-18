# services/language_agent/interrogation_rules.py
"""
Interrogation-specific language rules (NEXUS INTERROGATION_IMPLEMENTATION.MD v2.0).

Stateful across segments — instantiate once per session, call evaluate_all().

Rules implemented:
  INTERROG-LANG-01  Pronoun Distancing           (conf 0.45 — Newman 2003)
  INTERROG-LANG-02  Tense Shift Detection        (conf 0.35 — Vrij 2005)
  INTERROG-LANG-03  Contamination Detection      (conf 0.80 — Garrett 2011: 97.5%)
  INTERROG-LANG-04  Denial Evolution Tracker     (conf 0.60 — Kassin 2010)

CRITICAL DESIGN PRINCIPLE:
  None of these rules claim "deception detected".
  Every signal carries multiple possible interpretations.
  Maximum confidence 0.80 (contamination only — research-validated).
"""
from __future__ import annotations

import re
import logging
from collections import defaultdict
from typing import Optional

logger = logging.getLogger("nexus.language.interrogation")

# ── Pronoun patterns ──────────────────────────────────────────────────────────
_FIRST_PERSON_RE = re.compile(
    r"\b(I|me|my|mine|myself)\b", re.IGNORECASE
)
_THIRD_PERSON_RE = re.compile(
    r"\b(he|him|his|she|her|hers|they|them|their|theirs|it|its)\b", re.IGNORECASE
)

# ── Verb tense patterns ───────────────────────────────────────────────────────
# Past tense: strong irregular past + regular -ed endings
_PAST_TENSE_RE = re.compile(
    r"\b(was|were|had|went|said|did|saw|came|made|took|told|knew|thought|"
    r"got|gave|found|left|called|tried|asked|needed|seemed|felt|became|"
    r"\w+ed)\b",
    re.IGNORECASE,
)
# Present tense: common present forms
_PRESENT_TENSE_RE = re.compile(
    r"\b(am|is|are|have|has|go|goes|say|says|do|does|see|sees|come|comes|"
    r"make|makes|take|takes|tell|tells|know|knows|think|thinks|get|gets|"
    r"give|gives|find|finds)\b",
    re.IGNORECASE,
)
# Indicators that a segment is describing a past event
_PAST_EVENT_TRIGGERS = re.compile(
    r"\b(when|that night|that day|that morning|that evening|yesterday|"
    r"last (week|month|night|year)|at the time|back then|earlier|before)\b",
    re.IGNORECASE,
)

# ── Denial strength patterns ──────────────────────────────────────────────────
_DENIAL_CATEGORICAL = re.compile(
    r"\b(I did not|I didn'?t|I never|I have not|I haven'?t|I was not|"
    r"I wasn'?t|I would not|I wouldn'?t|that'?s? (is )?not true|"
    r"that'?s? false|not guilty|I had nothing to do|I am innocent|"
    r"I'?m innocent|absolutely not|of course not)\b",
    re.IGNORECASE,
)
_DENIAL_STRONG = re.compile(
    r"\b(didn'?t do|never did|have nothing to do|know nothing about|"
    r"wasn'?t there|wasn'?t involved)\b",
    re.IGNORECASE,
)
_DENIAL_WEAK = re.compile(
    r"\b(I don'?t think I|I don'?t remember|I don'?t recall|"
    r"I'?m not sure (I|if)|I'?m not certain|I don'?t know if I)\b",
    re.IGNORECASE,
)
_DENIAL_ACQUIESCENCE = re.compile(
    r"\b(maybe I did|perhaps I|I suppose I|I guess I|if you say so|"
    r"I might have (done)?|possibly I|I could have)\b",
    re.IGNORECASE,
)

# ── Contamination: stopwords to exclude from term tracking ───────────────────
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "have", "has", "had", "do", "did", "will", "would", "could", "should",
    "may", "might", "can", "that", "this", "it", "he", "she", "they", "we",
    "you", "i", "my", "his", "her", "their", "our", "your", "what", "when",
    "where", "who", "how", "why", "yes", "no", "not", "so", "then", "than",
    "just", "also", "about", "like", "there", "here", "which", "if", "said",
    "going", "get", "got", "know", "think", "very", "really", "just", "even",
})

# Minimum word length and frequency to track for contamination
_MIN_TERM_LENGTH = 4
_CONTAMINATION_MIN_MATCHES = 2

# Denial strength levels
_DENIAL_LEVELS: dict[str, float] = {
    "categorical":   1.0,
    "strong":        0.8,
    "weak":          0.3,
    "acquiescence":  0.1,
}

# Baseline window: first N segments per speaker establish baseline
_BASELINE_SEGMENTS = 5


def _extract_content_words(text: str) -> set[str]:
    """Extract meaningful words (non-stopwords, min length 4)."""
    words = re.findall(r"\b[a-z]{%d,}\b" % _MIN_TERM_LENGTH, text.lower())
    return {w for w in words if w not in _STOPWORDS}


def _seg_ms(seg: dict) -> tuple[int, int]:
    """Return (start_ms, end_ms) from a segment regardless of format.

    Handles both TranscriptSegment format (start_ms/end_ms as int milliseconds)
    and legacy float-seconds format (start/end as float seconds).
    """
    if "start_ms" in seg:
        return int(seg["start_ms"]), int(seg["end_ms"])
    return int(float(seg.get("start", 0)) * 1_000), int(float(seg.get("end", 0)) * 1_000)


def _classify_denial(text: str) -> Optional[tuple[str, float]]:
    """Return (label, strength) for the strongest denial pattern found, or None."""
    if _DENIAL_CATEGORICAL.search(text):
        return ("categorical", _DENIAL_LEVELS["categorical"])
    if _DENIAL_STRONG.search(text):
        return ("strong", _DENIAL_LEVELS["strong"])
    if _DENIAL_WEAK.search(text):
        return ("weak", _DENIAL_LEVELS["weak"])
    if _DENIAL_ACQUIESCENCE.search(text):
        return ("acquiescence", _DENIAL_LEVELS["acquiescence"])
    return None


class InterrogationLanguageRules:
    """
    Stateful interrogation language rules.
    Instantiate once per session; call evaluate_all() with all segments at once.
    """

    def evaluate_all(
        self,
        segments: list[dict],
        features_list: list[dict],
        profile=None,
    ) -> list[dict]:
        """
        Run all 4 interrogation language rules over the full segment list.
        Returns a flat list of signal dicts tagged agent='language'.
        """
        if not segments:
            return []

        # Identify the probable interrogator: speaker with the most question marks
        question_counts: dict[str, int] = defaultdict(int)
        for seg in segments:
            text = seg.get("text", "") or ""
            spk  = seg.get("speaker", seg.get("speaker_id", ""))
            if "?" in text:
                question_counts[spk] += text.count("?")

        interrogator_id = max(question_counts, key=question_counts.get) if question_counts else None

        signals: list[dict] = []
        signals.extend(self._pronoun_distancing(segments, interrogator_id))
        signals.extend(self._tense_shift(segments, interrogator_id))
        signals.extend(self._contamination_detection(segments, interrogator_id))
        signals.extend(self._denial_evolution(segments, interrogator_id))
        return signals

    # ── INTERROG-LANG-01: Pronoun Distancing ─────────────────────────────────

    def _pronoun_distancing(
        self, segments: list[dict], interrogator_id: Optional[str]
    ) -> list[dict]:
        """
        Newman et al. (2003): deceptive narratives use fewer first-person
        singular pronouns. Confidence 0.45 — culture/language dependent.
        """
        # Build per-speaker first-person ratio baseline from first N segments
        speaker_fp_ratios: dict[str, list[float]] = defaultdict(list)
        for seg in segments:
            text = seg.get("text", "") or ""
            spk  = seg.get("speaker", seg.get("speaker_id", ""))
            words = len(text.split())
            if words < 5:
                continue
            fp = len(_FIRST_PERSON_RE.findall(text))
            speaker_fp_ratios[spk].append(fp / words)

        # Baseline = mean of first _BASELINE_SEGMENTS observations per speaker
        baselines: dict[str, float] = {}
        for spk, ratios in speaker_fp_ratios.items():
            if len(ratios) >= _BASELINE_SEGMENTS:
                baselines[spk] = sum(ratios[:_BASELINE_SEGMENTS]) / _BASELINE_SEGMENTS

        signals: list[dict] = []
        for seg in segments:
            text  = seg.get("text", "") or ""
            spk   = seg.get("speaker", seg.get("speaker_id", ""))
            start, end = _seg_ms(seg)
            words = len(text.split())

            if words < 10 or spk not in baselines:
                continue
            baseline = baselines[spk]
            if baseline < 0.01:   # speaker almost never uses first-person — not meaningful
                continue

            fp = len(_FIRST_PERSON_RE.findall(text))
            current_ratio = fp / words

            # Flag if >40% reduction from that speaker's own baseline
            if current_ratio < baseline * 0.60:
                conf = min(0.45, 0.30 + (baseline - current_ratio) * 5)
                signals.append({
                    "agent":            "language",
                    "speaker_id":       spk,
                    "signal_type":      "pronoun_distancing",
                    "value":            round(1.0 - (current_ratio / max(baseline, 0.001)), 3),
                    "value_text":       "reduced_first_person",
                    "confidence":       round(conf, 3),
                    "window_start_ms":  start,
                    "window_end_ms":    end,
                    "metadata": {
                        "rule_id":       "INTERROG-LANG-01",
                        "baseline_ratio": round(baseline, 4),
                        "current_ratio":  round(current_ratio, 4),
                        "reduction_pct":  round((1 - current_ratio / max(baseline, 0.001)) * 100, 1),
                        "interpretations": [
                            "Psychological distancing from described events",
                            "Cultural or individual communication style",
                            "Narrative construction requiring less self-reference",
                        ],
                        "note": "Confidence low-moderate. Culture and language dependent. Never interpret alone.",
                    },
                })
        return signals

    # ── INTERROG-LANG-02: Tense Shift Detection ───────────────────────────────

    def _tense_shift(
        self, segments: list[dict], interrogator_id: Optional[str]
    ) -> list[dict]:
        """
        Vrij (2005, 2008): truthful past-event narratives show consistent
        past tense. Present tense intrusions may indicate real-time construction.
        Confidence 0.35 — common naturally in storytelling.
        """
        signals: list[dict] = []
        for seg in segments:
            text  = seg.get("text", "") or ""
            spk   = seg.get("speaker", seg.get("speaker_id", ""))
            start, end = _seg_ms(seg)

            # Only analyse segments that appear to be past-event narrations
            if not _PAST_EVENT_TRIGGERS.search(text):
                continue
            if len(text.split()) < 15:
                continue

            past_verbs    = len(_PAST_TENSE_RE.findall(text))
            present_verbs = len(_PRESENT_TENSE_RE.findall(text))
            total_verbs   = past_verbs + present_verbs
            if total_verbs < 3:
                continue

            present_ratio = present_verbs / total_verbs
            # Flag if >30% present tense during past-event narration
            if present_ratio > 0.30:
                signals.append({
                    "agent":            "language",
                    "speaker_id":       spk,
                    "signal_type":      "tense_inconsistency",
                    "value":            round(present_ratio, 3),
                    "value_text":       "present_tense_intrusion",
                    "confidence":       0.35,
                    "window_start_ms":  start,
                    "window_end_ms":    end,
                    "metadata": {
                        "rule_id":         "INTERROG-LANG-02",
                        "present_verb_pct": round(present_ratio * 100, 1),
                        "past_verb_count":  past_verbs,
                        "present_verb_count": present_verbs,
                        "interpretations": [
                            "Real-time narrative construction rather than memory retrieval",
                            "Storytelling or dramatic present style",
                            "Second-language interference with tense selection",
                            "Trauma recall causing temporal disorientation",
                        ],
                        "note": "Only meaningful when question clearly asked for past-event narration.",
                    },
                })
        return signals

    # ── INTERROG-LANG-03: Contamination Detection ─────────────────────────────

    def _contamination_detection(
        self, segments: list[dict], interrogator_id: Optional[str]
    ) -> list[dict]:
        """
        Garrett (2011): 97.5% of proven false confessions contained details
        introduced by interrogator. Confidence 0.80 — very high research support.
        """
        if not interrogator_id:
            return []

        # Pass 1: build timeline of interrogator-introduced terms with timestamps
        # term → earliest timestamp_ms it was introduced by interrogator
        interrogator_terms: dict[str, int] = {}
        for seg in segments:
            spk   = seg.get("speaker", seg.get("speaker_id", ""))
            text  = seg.get("text", "") or ""
            start, _ = _seg_ms(seg)
            if spk != interrogator_id:
                continue
            for word in _extract_content_words(text):
                if word not in interrogator_terms:
                    interrogator_terms[word] = start

        if not interrogator_terms:
            return []

        # Pass 2: for each suspect segment, check how many interrogator-introduced
        # terms appear that were NOT already used by the suspect beforehand
        suspect_vocabulary: set[str] = set()  # cumulative suspect vocabulary
        signals: list[dict] = []

        for seg in segments:
            spk   = seg.get("speaker", seg.get("speaker_id", ""))
            text  = seg.get("text", "") or ""
            start, end = _seg_ms(seg)

            if spk == interrogator_id:
                continue   # skip interrogator's own speech

            seg_words = _extract_content_words(text)

            # Contaminated: interrogator introduced it BEFORE this segment
            contaminated = [
                w for w in seg_words
                if w in interrogator_terms
                and interrogator_terms[w] < start   # interrogator said it first
                and w not in suspect_vocabulary       # suspect didn't use it before
            ]

            # Update suspect vocabulary
            suspect_vocabulary.update(seg_words)

            if len(contaminated) >= _CONTAMINATION_MIN_MATCHES:
                signals.append({
                    "agent":            "language",
                    "speaker_id":       spk,
                    "signal_type":      "statement_contamination",
                    "value":            min(len(contaminated) / 10.0, 1.0),
                    "value_text":       "interrogator_terms_adopted",
                    "confidence":       0.80,
                    "window_start_ms":  start,
                    "window_end_ms":    end,
                    "metadata": {
                        "rule_id":             "INTERROG-LANG-03",
                        "contaminated_terms":  contaminated[:10],
                        "contaminated_count":  len(contaminated),
                        "severity":            "HIGH" if len(contaminated) >= 5 else "MODERATE",
                        "interpretation":      "Suspect adopting case-specific terminology introduced by interrogator. Strong false confession risk marker per Garrett (2011).",
                        "recommendation":      "Document all evidence disclosure timing. Cross-reference with denial evolution and session duration.",
                        "false_confession_risk": True,
                    },
                })
        return signals

    # ── INTERROG-LANG-04: Denial Evolution Tracker ────────────────────────────

    def _denial_evolution(
        self, segments: list[dict], interrogator_id: Optional[str]
    ) -> list[dict]:
        """
        Kassin et al. (2010): False confession trajectory — Strong denial →
        Weak denial → Acquiescence → Admission. Mean 16.3h vs 1.6h for true.
        Confidence 0.60 — must consider session duration and vulnerability.
        """
        # Per-speaker denial history: list of (timestamp_ms, strength_value, label)
        denial_history: dict[str, list[tuple[int, float, str]]] = defaultdict(list)

        for seg in segments:
            spk   = seg.get("speaker", seg.get("speaker_id", ""))
            text  = seg.get("text", "") or ""
            start, _ = _seg_ms(seg)

            if spk == interrogator_id:
                continue

            result = _classify_denial(text)
            if result:
                label, strength = result
                denial_history[spk].append((start, strength, label))

        signals: list[dict] = []
        for spk, history in denial_history.items():
            if len(history) < 3:
                continue

            timestamps = [h[0] for h in history]
            strengths  = [h[1] for h in history]

            # Compute slope via simple linear regression
            n = len(timestamps)
            t_mean = sum(timestamps) / n
            s_mean = sum(strengths) / n
            numerator   = sum((t - t_mean) * (s - s_mean) for t, s in zip(timestamps, strengths))
            denominator = sum((t - t_mean) ** 2 for t in timestamps)
            slope = numerator / denominator if denominator > 0 else 0.0

            # Negative slope = weakening denials (from strong → weak over time)
            # Scale: -0.00005 per ms → -0.003 per minute → significant weakening
            if slope < -4e-5:
                duration_ms = timestamps[-1] - timestamps[0]
                first_label = history[0][2]
                last_label  = history[-1][2]
                signals.append({
                    "agent":            "language",
                    "speaker_id":       spk,
                    "signal_type":      "denial_weakening",
                    "value":            round(abs(slope) * 60000, 4),   # strength loss per minute
                    "value_text":       f"{first_label}_to_{last_label}",
                    "confidence":       0.60,
                    "window_start_ms":  timestamps[0],
                    "window_end_ms":    timestamps[-1],
                    "metadata": {
                        "rule_id":          "INTERROG-LANG-04",
                        "denial_count":     len(history),
                        "first_label":      first_label,
                        "last_label":       last_label,
                        "first_strength":   strengths[0],
                        "last_strength":    strengths[-1],
                        "duration_ms":      duration_ms,
                        "slope_per_min":    round(slope * 60000, 6),
                        "interpretation":   "Denial strength declining over interrogation. May indicate genuine guilt OR psychological breakdown under prolonged pressure.",
                        "recommendation":   "If duration >4h + vulnerable suspect + contamination present → HIGH false confession risk. Cross-reference with duration and contamination signals.",
                        "false_confession_risk": last_label in ("weak", "acquiescence"),
                    },
                })
        return signals
