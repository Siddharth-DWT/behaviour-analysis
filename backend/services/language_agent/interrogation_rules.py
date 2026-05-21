# services/language_agent/interrogation_rules.py
"""
Interrogation-specific language rules (NEXUS INTERROGATION_IMPLEMENTATION.MD v2.2).

Stateful across segments — instantiate once per session, call evaluate_all().

Rules implemented:
  INTERROG-LANG-01  Detail Reduction             (conf 0.50 — Vrij et al. 2017)
  INTERROG-LANG-02  Narrative Consistency Drift  (conf 0.40 — Fisher & Geiselman 1992)
  INTERROG-LANG-03  Contamination Detection      (conf 0.80 — Garrett 2011: 97.5%)
  INTERROG-LANG-04  Denial Evolution Tracker     (conf 0.45-0.65 adaptive — Kassin 2010)

CRITICAL DESIGN PRINCIPLE:
  None of these rules claim "deception detected".
  Every signal carries multiple possible interpretations.
  Maximum confidence 0.80 (contamination only — research-validated).

v2.2 changes:
  - INTERROG-LANG-01: replaced unreliable pronoun_distancing with detail_reduction
    (sensory-word density drop ≥40% first→second half; Vrij 2017)
  - INTERROG-LANG-02: replaced unreliable tense_inconsistency with narrative_consistency_drift
    (TF-IDF cosine / Jaccard fallback on retelling pairs >5 min apart; Fisher & Geiselman 1992)
  - ContaminationDetector: NER + TF-IDF + validated stopwords (replaces handcrafted list)
  - _denial_evolution: duration-adaptive slope threshold + windowed comparison fallback
"""
from __future__ import annotations

import re
import logging
from collections import defaultdict
from itertools import combinations
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

# ── Sensory vocabulary for detail_reduction (Vrij 2017) ──────────────────────
# Words that encode episodic/perceptual grounding in a recalled scene.
# Visual, auditory, tactile, olfactory, spatial, and embodied-action categories.
_SENSORY_WORDS: frozenset[str] = frozenset({
    # visual perception
    "saw", "seen", "look", "looked", "noticed", "watched", "observed", "glimpsed",
    "color", "colour", "bright", "dark", "light", "shadow", "shape",
    # auditory
    "heard", "hear", "sound", "sounds", "noise", "voice", "voices",
    "loud", "quiet", "yelled", "screamed", "whispered", "bang",
    # tactile / kinaesthetic
    "felt", "feel", "touch", "touched", "grabbed", "held", "pushed", "pulled",
    "smooth", "rough", "hard", "soft", "warm", "cold", "heavy",
    # olfactory / gustatory
    "smell", "smelled", "smelt", "odor", "odour", "taste", "tasted",
    # spatial grounding
    "left", "right", "behind", "front", "beside", "near", "close", "far",
    "inside", "outside", "above", "below", "corner", "edge",
    # embodied scene actions (imply specific imagery)
    "walked", "ran", "running", "stood", "sitting", "lying",
    "entered", "opened", "closed", "turned", "reached",
})

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
        signals.extend(self._detail_reduction(segments, interrogator_id))
        signals.extend(self._narrative_consistency_drift(segments, interrogator_id))
        signals.extend(self._contamination_detection(segments, interrogator_id))
        signals.extend(self._denial_evolution(segments, interrogator_id))
        return signals

    # ── INTERROG-LANG-01: Detail Reduction ───────────────────────────────────

    def _detail_reduction(
        self, segments: list[dict], interrogator_id: Optional[str]
    ) -> list[dict]:
        """
        Vrij et al. (2017): genuine episodic memories contain rich sensory and
        contextual detail. A ≥40% drop in sensory-word density from the first
        half to the second half of a suspect's testimony flags degrading recall
        richness — consistent with rehearsal exhaustion or fabrication under load.
        Confidence cap 0.50 — also observed in truthful speakers under fatigue.
        """
        suspect_segs: dict[str, list[dict]] = defaultdict(list)
        for seg in segments:
            spk  = seg.get("speaker", seg.get("speaker_id", ""))
            text = seg.get("text", "") or ""
            if spk and spk != interrogator_id and len(text.split()) >= 5:
                suspect_segs[spk].append(seg)

        def _sensory_density(seg_list: list[dict]) -> float:
            total = sensory = 0
            for s in seg_list:
                words = (s.get("text", "") or "").lower().split()
                total   += len(words)
                sensory += sum(1 for w in words if w in _SENSORY_WORDS)
            return sensory / total if total > 0 else 0.0

        signals: list[dict] = []
        for spk, segs in suspect_segs.items():
            if len(segs) < 6:
                continue
            mid = len(segs) // 2
            first_density  = _sensory_density(segs[:mid])
            second_density = _sensory_density(segs[mid:])

            if first_density < 0.01:
                continue
            drop = (first_density - second_density) / first_density
            if drop < 0.40:
                continue

            _, end_ms   = _seg_ms(segs[-1])
            start_ms, _ = _seg_ms(segs[mid])
            conf = round(min(0.50, 0.28 + drop * 0.30), 3)
            signals.append({
                "agent":            "language",
                "speaker_id":       spk,
                "signal_type":      "detail_reduction",
                "value":            round(drop, 3),
                "value_text":       "sensory_density_drop",
                "confidence":       conf,
                "window_start_ms":  start_ms,
                "window_end_ms":    end_ms,
                "metadata": {
                    "rule_id":              "INTERROG-LANG-01",
                    "first_half_density":   round(first_density, 4),
                    "second_half_density":  round(second_density, 4),
                    "drop_pct":             round(drop * 100, 1),
                    "interpretations": [
                        "Rehearsed narrative exhausted; maintaining spontaneous detail harder",
                        "Increasing cognitive load reducing perceptual grounding",
                        "Truthful speaker under fatigue — detail naturally degrades over long sessions",
                    ],
                    "note": "Requires cross-reference with denial evolution and session duration.",
                },
            })
        return signals

    # ── INTERROG-LANG-02: Narrative Consistency Drift ─────────────────────────

    def _narrative_consistency_drift(
        self, segments: list[dict], interrogator_id: Optional[str]
    ) -> list[dict]:
        """
        Fisher & Geiselman (1992): genuine episodic memories produce consistent
        retellings. Pairs of suspect segments >5 min apart that share ≥3 content
        words are retelling candidates. Cosine similarity <0.70 (TF-IDF) or
        Jaccard <0.25 (fallback) flags inconsistent retelling.
        Confidence cap 0.40 — natural memory decay also causes variation.

        DSA: O(n²) over suspect segments, bounded in practice by session length.
        Content-word set pre-computed per segment; shared-word check is O(1) via
        set intersection before the heavier similarity computation.
        """
        def _content_words(text: str) -> set[str]:
            return {
                w for w in re.findall(r"\b[a-z]{4,}\b", text.lower())
                if w not in _ALL_STOPWORDS
            }

        suspect_seg_words: list[tuple[dict, set[str]]] = [
            (seg, _content_words(seg.get("text", "") or ""))
            for seg in segments
            if seg.get("speaker", seg.get("speaker_id", "")) != interrogator_id
            and len((seg.get("text", "") or "").split()) >= 10
        ]

        signals: list[dict] = []
        seen: set[tuple[int, int]] = set()

        for i, (seg_a, words_a) in enumerate(suspect_seg_words):
            for j, (seg_b, words_b) in enumerate(suspect_seg_words):
                if j <= i:
                    continue
                spk_a = seg_a.get("speaker", seg_a.get("speaker_id", ""))
                spk_b = seg_b.get("speaker", seg_b.get("speaker_id", ""))
                if spk_a != spk_b:
                    continue

                start_a, end_a   = _seg_ms(seg_a)
                start_b, end_b   = _seg_ms(seg_b)
                if start_b - end_a < 300_000:   # < 5 min gap
                    continue
                if len(words_a & words_b) < 3:  # not a retelling candidate
                    continue

                pair_key = (i, j)
                if pair_key in seen:
                    continue
                seen.add(pair_key)

                sim, method = self._text_similarity(
                    seg_a.get("text", "") or "",
                    seg_b.get("text", "") or "",
                    words_a, words_b,
                )
                threshold = 0.70 if method == "cosine" else 0.25
                if sim >= threshold:
                    continue

                conf = round(min(0.40, 0.22 + (threshold - sim) * 0.35), 3)
                signals.append({
                    "agent":            "language",
                    "speaker_id":       spk_a,
                    "signal_type":      "narrative_consistency_drift",
                    "value":            round(1.0 - sim, 3),
                    "value_text":       "retelling_inconsistency",
                    "confidence":       conf,
                    "window_start_ms":  start_b,
                    "window_end_ms":    end_b,
                    "metadata": {
                        "rule_id":       "INTERROG-LANG-02",
                        "similarity":    round(sim, 3),
                        "method":        method,
                        "gap_minutes":   round((start_b - end_a) / 60_000, 1),
                        "shared_terms":  sorted(words_a & words_b)[:10],
                        "interpretations": [
                            "Same event described differently — possible confabulation or coached narrative",
                            "Natural memory reconstruction over extended session duration",
                            "Emotional state change causing different framing of same event",
                        ],
                        "note": "Only meaningful for retellings of the same event. Gap ≥5 min required.",
                    },
                })
        return signals

    def _text_similarity(
        self,
        text_a: str,
        text_b: str,
        words_a: set[str],
        words_b: set[str],
    ) -> tuple[float, str]:
        """TF-IDF cosine similarity with Jaccard fallback. Returns (score, method)."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity as _cos_sim
            vec = TfidfVectorizer(stop_words=list(_ALL_STOPWORDS), min_df=1)
            mat = vec.fit_transform([text_a, text_b])
            return float(_cos_sim(mat[0:1], mat[1:2])[0][0]), "cosine"
        except Exception:
            union = words_a | words_b
            if not union:
                return 0.0, "jaccard"
            return len(words_a & words_b) / len(union), "jaccard"

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

