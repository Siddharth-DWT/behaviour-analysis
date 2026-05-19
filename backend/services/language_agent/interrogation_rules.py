# services/language_agent/interrogation_rules.py
"""
Interrogation-specific language rules (NEXUS INTERROGATION_IMPLEMENTATION.MD v2.1).

Stateful across segments — instantiate once per session, call evaluate_all().

Rules implemented:
  INTERROG-LANG-01  Pronoun Distancing           (conf 0.45 — Newman 2003)
  INTERROG-LANG-02  Tense Shift Detection        (conf 0.35 — Vrij 2005)
  INTERROG-LANG-03  Contamination Detection      (conf 0.80 — Garrett 2011: 97.5%)
  INTERROG-LANG-04  Denial Evolution Tracker     (conf 0.45-0.65 adaptive — Kassin 2010)

CRITICAL DESIGN PRINCIPLE:
  None of these rules claim "deception detected".
  Every signal carries multiple possible interpretations.
  Maximum confidence 0.80 (contamination only — research-validated).

v2.1 changes:
  - ContaminationDetector: NER + TF-IDF + validated stopwords (replaces handcrafted list)
  - _denial_evolution: duration-adaptive slope threshold + windowed comparison fallback
"""
from __future__ import annotations

import re
import logging
from collections import defaultdict
from typing import Optional

logger = logging.getLogger("nexus.language.interrogation")

# ── Validated stopwords — import chain per spec §1.2 ─────────────────────────
# Source priority: NLTK (179 words, Snowball project) → spaCy (326 words,
# Stone/Denis/Kwantes 2010) → sklearn → empty (regex-only fallback).
# DO NOT replace with a handcrafted list — these are peer-validated corpora.
try:
    from nltk.corpus import stopwords as _nltk_sw
    _VALIDATED_STOPWORDS: frozenset[str] = frozenset(_nltk_sw.words("english"))
    logger.debug("ContaminationDetector: using NLTK stopwords (%d words)", len(_VALIDATED_STOPWORDS))
except Exception:
    try:
        from spacy.lang.en.stop_words import STOP_WORDS as _SPACY_SW
        _VALIDATED_STOPWORDS = frozenset(w.lower() for w in _SPACY_SW)
        logger.debug("ContaminationDetector: using spaCy stopwords (%d words)", len(_VALIDATED_STOPWORDS))
    except Exception:
        try:
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as _SKL_SW
            _VALIDATED_STOPWORDS = frozenset(_SKL_SW)
            logger.debug("ContaminationDetector: using sklearn stopwords (%d words)", len(_VALIDATED_STOPWORDS))
        except Exception:
            _VALIDATED_STOPWORDS = frozenset()
            logger.warning("ContaminationDetector: no validated stopwords available — precision may be reduced")

# Domain additions from confirmed session false positives ONLY (not fabricated)
_INTERROGATION_DOMAIN_STOPWORDS: frozenset[str] = frozenset({
    "yeah", "okay", "alright", "gonna", "wanna", "gotta",
    "stuff", "kinda", "sorta", "nah", "nope", "yep",
})
_ALL_STOPWORDS: frozenset[str] = _VALIDATED_STOPWORDS | _INTERROGATION_DOMAIN_STOPWORDS

# ── Pronoun patterns ──────────────────────────────────────────────────────────
_FIRST_PERSON_RE = re.compile(
    r"\b(I|me|my|mine|myself)\b", re.IGNORECASE
)
_THIRD_PERSON_RE = re.compile(
    r"\b(he|him|his|she|her|hers|they|them|their|theirs|it|its)\b", re.IGNORECASE
)

# ── Verb tense patterns ───────────────────────────────────────────────────────
_PAST_TENSE_RE = re.compile(
    r"\b(was|were|had|went|said|did|saw|came|made|took|told|knew|thought|"
    r"got|gave|found|left|called|tried|asked|needed|seemed|felt|became|"
    r"\w+ed)\b",
    re.IGNORECASE,
)
_PRESENT_TENSE_RE = re.compile(
    r"\b(am|is|are|have|has|go|goes|say|says|do|does|see|sees|come|comes|"
    r"make|makes|take|takes|tell|tells|know|knows|think|thinks|get|gets|"
    r"give|gives|find|finds)\b",
    re.IGNORECASE,
)
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

_DENIAL_LEVELS: dict[str, float] = {
    "categorical":   1.0,
    "strong":        0.8,
    "weak":          0.3,
    "acquiescence":  0.1,
}

_BASELINE_SEGMENTS = 5


# ── Helpers ───────────────────────────────────────────────────────────────────

def _seg_ms(seg: dict) -> tuple[int, int]:
    """Return (start_ms, end_ms) regardless of segment format (ms ints or float seconds)."""
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


# ── ContaminationDetector ─────────────────────────────────────────────────────

class ContaminationDetector:
    """
    3-stage hybrid contamination detection (spec §1.1).

    Stage 1 — SpaCy NER: proper nouns and named entities (PERSON, ORG, GPE ...)
    Stage 2 — TF-IDF significance: high-IDF words across interrogator segments
    Stage 3 — Temporal ordering: suspect must adopt term AFTER interrogator

    Graceful degradation:
      SpaCy + sklearn → full hybrid  (highest precision)
      SpaCy only      → NER mode     (good precision)
      Neither         → validated stopwords filter (lowest precision, no crash)

    DSA: O(n + m) total — hash table for entity timeline, O(1) lookup per suspect word.
    """

    _CASE_ENTITY_LABELS: frozenset[str] = frozenset({
        "PERSON", "ORG", "GPE", "PRODUCT", "FAC", "LOC",
        "NORP", "EVENT", "LAW", "WORK_OF_ART",
    })

    def __init__(self) -> None:
        self._nlp = None
        self._tfidf_cls = None

        try:
            import spacy as _spacy
            self._nlp = _spacy.load("en_core_web_sm")
        except Exception:
            logger.debug("ContaminationDetector: SpaCy unavailable — using fallback mode")

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._tfidf_cls = TfidfVectorizer
        except ImportError:
            pass

    @property
    def detection_method(self) -> str:
        if self._nlp and self._tfidf_cls:
            return "ner+tfidf+stopwords"
        if self._nlp:
            return "ner+stopwords"
        return "validated_stopwords"

    def _ner_entities(self, text: str) -> set[str]:
        """Stage 1: extract ONLY named entities via SpaCy — no generic nouns."""
        doc = self._nlp(text)  # type: ignore[union-attr]
        terms: set[str] = set()
        for ent in doc.ents:
            if ent.label_ in self._CASE_ENTITY_LABELS:
                norm = ent.text.lower().strip()
                if len(norm) >= 3 and norm not in _ALL_STOPWORDS:
                    terms.add(norm)
        return terms

    def _tfidf_significant_terms(self, interrogator_texts: list[str]) -> set[str]:
        """
        Stage 2: words with mean TF-IDF > (mean + 2σ) across interrogator segments.
        min_df=2 requires the term to appear in ≥2 distinct interrogator segments,
        ruling out one-off mentions. Cutoff raised to 2σ for higher precision.
        """
        if len(interrogator_texts) < 3:
            return set()
        try:
            vec = self._tfidf_cls(  # type: ignore[operator]
                stop_words=list(_ALL_STOPWORDS),
                min_df=2,
                max_df=0.70,
                token_pattern=r"\b[a-z]{5,}\b",
                sublinear_tf=True,
            )
            matrix = vec.fit_transform(interrogator_texts)
            names = vec.get_feature_names_out()
            avg = matrix.mean(axis=0).A1
            cutoff = avg.mean() + 2.0 * avg.std()
            return {names[i] for i, v in enumerate(avg) if v > cutoff}
        except Exception:
            return set()

    def _fallback_terms(self, text: str) -> set[str]:
        """No SpaCy/sklearn: validated stopwords filter only."""
        words = re.findall(r"\b[a-z]{5,}\b", text.lower())
        return {w for w in words if w not in _ALL_STOPWORDS}

    def _extract_terms(self, text: str, tfidf_vocab: set[str]) -> set[str]:
        """Extract significant terms: NER named entities union high-TF-IDF vocab."""
        if self._nlp:
            terms = self._ner_entities(text)
            raw = set(re.findall(r"\b[a-z]{5,}\b", text.lower()))
            terms.update(raw & tfidf_vocab)
            return terms
        return self._fallback_terms(text)

    def detect(
        self,
        segments: list[dict],
        interrogator_id: str,
        min_matches: int = 2,
    ) -> list[dict]:
        """
        Full contamination detection. O(n + m) — n interrogator segs, m suspect segs.

        Pass 1: build interrogator term timeline — {term: first_mention_ms}
        Pass 2: scan suspect segments for terms adopted AFTER interrogator introduced them
        """
        if not interrogator_id:
            return []

        # Pre-compute TF-IDF vocabulary across all interrogator texts (Stage 2)
        interrogator_texts = [
            seg.get("text", "") or ""
            for seg in segments
            if seg.get("speaker", seg.get("speaker_id", "")) == interrogator_id
        ]
        tfidf_vocab = self._tfidf_significant_terms(interrogator_texts) if self._tfidf_cls else set()

        # Pass 1: build {term: earliest_ms} for interrogator-introduced terms
        interrogator_timeline: dict[str, int] = {}
        for seg in segments:
            if seg.get("speaker", seg.get("speaker_id", "")) != interrogator_id:
                continue
            text = seg.get("text", "") or ""
            start, _ = _seg_ms(seg)
            for term in self._extract_terms(text, tfidf_vocab):
                if term not in interrogator_timeline:
                    interrogator_timeline[term] = start

        if not interrogator_timeline:
            return []

        # Pass 2: detect adoption in suspect segments
        suspect_vocabulary: set[str] = set()
        signals: list[dict] = []

        for seg in segments:
            spk = seg.get("speaker", seg.get("speaker_id", ""))
            text = seg.get("text", "") or ""
            start, end = _seg_ms(seg)

            if spk == interrogator_id:
                continue

            seg_terms = self._extract_terms(text, tfidf_vocab)

            contaminated = [
                t for t in seg_terms
                if t in interrogator_timeline
                and interrogator_timeline[t] < start
                and t not in suspect_vocabulary
            ]
            suspect_vocabulary.update(seg_terms)

            if len(contaminated) >= min_matches:
                signals.append({
                    "agent":           "language",
                    "speaker_id":      spk,
                    "signal_type":     "statement_contamination",
                    "value":           min(len(contaminated) / 10.0, 1.0),
                    "value_text":      "interrogator_terms_adopted",
                    "confidence":      0.80,
                    "window_start_ms": start,
                    "window_end_ms":   end,
                    "metadata": {
                        "rule_id":              "INTERROG-LANG-03",
                        "contaminated_terms":   contaminated[:10],
                        "contaminated_count":   len(contaminated),
                        "severity":             "HIGH" if len(contaminated) >= 5 else "MODERATE",
                        "detection_method":     self.detection_method,
                        "interpretation":       (
                            "Suspect adopting case-specific terminology introduced by interrogator. "
                            "Strong false confession risk marker per Garrett (2011)."
                        ),
                        "recommendation":       (
                            "Document all evidence disclosure timing. "
                            "Cross-reference with denial evolution and session duration."
                        ),
                        "false_confession_risk": True,
                    },
                })

        return signals


# ── InterrogationLanguageRules ────────────────────────────────────────────────

class InterrogationLanguageRules:
    """
    Stateful interrogation language rules.
    Instantiate once per session; call evaluate_all() with all segments at once.
    """

    def __init__(self) -> None:
        self._contamination_detector = ContaminationDetector()

    def evaluate_all(
        self,
        segments: list[dict],
        features_list: list[dict],
        profile=None,
    ) -> list[dict]:
        """Run all 4 interrogation language rules. Returns flat list of signal dicts."""
        if not segments:
            return []

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
        speaker_fp_ratios: dict[str, list[float]] = defaultdict(list)
        for seg in segments:
            text = seg.get("text", "") or ""
            spk  = seg.get("speaker", seg.get("speaker_id", ""))
            words = len(text.split())
            if words < 5:
                continue
            fp = len(_FIRST_PERSON_RE.findall(text))
            speaker_fp_ratios[spk].append(fp / words)

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
            if baseline < 0.01:
                continue

            fp = len(_FIRST_PERSON_RE.findall(text))
            current_ratio = fp / words

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
                        "rule_id":        "INTERROG-LANG-01",
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
                        "rule_id":            "INTERROG-LANG-02",
                        "present_verb_pct":   round(present_ratio * 100, 1),
                        "past_verb_count":    past_verbs,
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
        """Delegates to ContaminationDetector (composition pattern)."""
        if not interrogator_id:
            return []
        return self._contamination_detector.detect(segments, interrogator_id)

    # ── INTERROG-LANG-04: Denial Evolution Tracker ────────────────────────────

    @staticmethod
    def _calculate_adaptive_denial_threshold(session_duration_ms: int) -> float:
        """
        Duration-aware slope threshold for denial weakening detection.
        Short sessions (15 min): -8e-5 (strict). Long sessions (60 min+): -2e-5 (relaxed).
        Formula: -4e-5 / scale, scale = min(2.0, max(0.5, hours / 0.5)).
        """
        hours = session_duration_ms / 3_600_000
        scale = min(2.0, max(0.5, hours / 0.5))
        return -4e-5 / scale

    def _denial_evolution(
        self, segments: list[dict], interrogator_id: Optional[str]
    ) -> list[dict]:
        """
        Kassin et al. (2010): False confession trajectory — Strong denial →
        Weak denial → Acquiescence → Admission. Mean 16.3h vs 1.6h for true.

        Dual trigger (either fires the signal):
          A) Slope trigger:    regression slope < duration-adaptive threshold
          B) Windowed trigger: first-third mean − last-third mean > min_drop

        Confidence: 0.45 base + duration bonus (max 0.15) + data bonus (max 0.10).
        """
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
            duration_ms = timestamps[-1] - timestamps[0]

            # ── Linear regression slope ───────────────────────────────────────
            n = len(timestamps)
            t_mean = sum(timestamps) / n
            s_mean = sum(strengths) / n
            numerator   = sum((t - t_mean) * (s - s_mean) for t, s in zip(timestamps, strengths))
            denominator = sum((t - t_mean) ** 2 for t in timestamps)
            slope = numerator / denominator if denominator > 0 else 0.0

            # ── Duration-adaptive thresholds ──────────────────────────────────
            slope_threshold = self._calculate_adaptive_denial_threshold(duration_ms)

            duration_min = duration_ms / 60_000
            if duration_min <= 15:
                min_windowed_drop = 0.20
            elif duration_min <= 45:
                min_windowed_drop = 0.15
            elif duration_min <= 90:
                min_windowed_drop = 0.10
            else:
                min_windowed_drop = 0.08

            # ── Windowed comparison: first third vs last third ─────────────────
            third = max(1, n // 3)
            first_third_mean = sum(strengths[:third]) / third
            last_third_mean  = sum(strengths[-third:]) / third
            windowed_drop    = first_third_mean - last_third_mean

            slope_triggered   = slope < slope_threshold
            windowed_triggered = windowed_drop > min_windowed_drop

            if not slope_triggered and not windowed_triggered:
                continue

            # ── Duration-aware confidence ──────────────────────────────────────
            duration_bonus = min(0.15, duration_min / 300)
            data_bonus     = min(0.10, n / 50)
            confidence     = round(min(0.65, 0.45 + duration_bonus + data_bonus), 3)

            # Value: strength loss per minute (windowed drop if that's what triggered)
            if windowed_triggered:
                value = round(windowed_drop, 4)
            else:
                value = round(abs(slope) * 60_000, 4)

            first_label = history[0][2]
            last_label  = history[-1][2]

            signals.append({
                "agent":            "language",
                "speaker_id":       spk,
                "signal_type":      "denial_weakening",
                "value":            value,
                "value_text":       f"{first_label}_to_{last_label}",
                "confidence":       confidence,
                "window_start_ms":  timestamps[0],
                "window_end_ms":    timestamps[-1],
                "metadata": {
                    "rule_id":              "INTERROG-LANG-04",
                    "denial_count":         n,
                    "first_label":          first_label,
                    "last_label":           last_label,
                    "first_strength":       strengths[0],
                    "last_strength":        strengths[-1],
                    "duration_ms":          duration_ms,
                    "slope_per_min":        round(slope * 60_000, 8),
                    "windowed_drop":        round(windowed_drop, 4),
                    "trigger":              (
                        "slope+window" if slope_triggered and windowed_triggered
                        else "slope" if slope_triggered
                        else "window"
                    ),
                    "slope_threshold":      round(slope_threshold, 8),
                    "min_windowed_drop":    min_windowed_drop,
                    "interpretation":       (
                        "Denial strength declining over interrogation. May indicate genuine guilt "
                        "OR psychological breakdown under prolonged pressure."
                    ),
                    "recommendation":       (
                        "If duration >4h + vulnerable suspect + contamination present → "
                        "HIGH false confession risk. Cross-reference with contamination signals."
                    ),
                    "false_confession_risk": last_label in ("weak", "acquiescence"),
                },
            })

        return signals

