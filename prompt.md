# NEXUS — Fix Contamination False Positives + Adaptive Denial Threshold

## Confirmed Issues from 54-Minute Interrogation Session

statement_contamination: 26 hits, 0.800 confidence — ~77% FALSE POSITIVES
  FALSE: ["yeah", "driving", "license", "anything", "didn", "told", "while", "guess", "talk", "other", "some", "back"]
  TRUE:  ["driving", "license", "locked", "straight"]
  Current precision: ~40% (10 true / 26 total). Target: >90%

denial_weakening: 0 hits — NEVER FIRED in 54-minute interrogation
  Despite 88 pronoun_distancing signals indicating clear defensive shift.
  Threshold -4e-5 too strict for gradual weakening patterns.

## IMPLEMENTATION REQUIREMENTS

File Locations:
  backend/services/language_agent/interrogation_rules.py
  backend/services/language_agent/tests/test_interrogation_rules.py (create if missing)

Architecture Principles — STRICT:
- OOP: Composition pattern — ContaminationDetector as separate class with Single Responsibility
- DSA: O(n) single-pass algorithms, hash tables for entity lookup, NO nested O(n²) loops
- Error Handling: Graceful degradation if SpaCy/sklearn not available — regex+validated stopwords fallback
- Testing: Unit tests with real data samples, measure precision/recall
- Stopwords: Use VALIDATED source (NLTK 179 words from Snowball project OR spaCy 326 words from Stone/Denis/Kwantes 2010). DO NOT fabricate a stopword list. Add ONLY domain-specific interrogation words from confirmed false positives.

Read these files before making changes:
- backend/services/language_agent/interrogation_rules.py — current _contamination_detection() and _denial_evolution()
- backend/services/language_agent/tests/ — existing test structure

---

## Part 1: Hybrid Contamination Detection

### 1.1 ContaminationDetector Class (NEW — composition, not inheritance)

3-stage hybrid detection:
  Stage 1: SpaCy NER — proper nouns (PERSON, ORG, GPE, PRODUCT, FAC, LOC, NORP, EVENT, LAW, WORK_OF_ART)
  Stage 2: TF-IDF significance — unusual frequency vs NLTK Brown corpus reference
  Stage 3: Temporal ordering — only flag if suspect mentions AFTER interrogator

Graceful degradation:
  SpaCy + sklearn → full hybrid (NER + TF-IDF + temporal)
  SpaCy only → entity-only mode
  Neither → regex with VALIDATED stopword list (NLTK/spaCy/sklearn fallback chain)

DSA: Entity timeline O(n) → HashMap. Suspect check O(m) → O(1) lookup. Total O(n+m+e), NOT O(n×m).
MIN_CONTAMINATION_MATCHES = 2 (hybrid is precise enough for lower threshold)

### 1.2 Validated Stopword Import Chain

```python
# Validated stopword source — NOT fabricated
try:
    from nltk.corpus import stopwords as _nltk_sw
    _VALIDATED_STOPWORDS = frozenset(_nltk_sw.words("english"))
except ImportError:
    try:
        from spacy.lang.en.stop_words import STOP_WORDS
        _VALIDATED_STOPWORDS = frozenset(w.lower() for w in STOP_WORDS)
    except ImportError:
        try:
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
            _VALIDATED_STOPWORDS = frozenset(ENGLISH_STOP_WORDS)
        except ImportError:
            _VALIDATED_STOPWORDS = frozenset()

# Domain additions from confirmed session false positives ONLY
_INTERROGATION_DOMAIN_STOPWORDS = frozenset({
    "yeah", "okay", "alright", "gonna", "wanna", "gotta",
    "stuff", "kinda", "sorta", "nah", "nope", "yep",
})
_ALL_STOPWORDS = _VALIDATED_STOPWORDS | _INTERROGATION_DOMAIN_STOPWORDS
```

---

## Part 2: Adaptive Denial Threshold

### 2.1 Duration-Adaptive Threshold

Fixed slope threshold misses gradual weakening in long sessions.
Threshold scales with duration:

  15 min → -4e-5 (strict, rapid breakdown expected if real)
  30 min → -2.7e-5 (moderate)
  54 min → -2e-5 (relaxed, gradual erosion meaningful)
  90 min → -2e-5 (floor)

Formula: threshold = -4e-5 / scale, where scale = min(2.0, max(0.5, hours / 0.5))

### 2.2 Windowed Comparison (Secondary Trigger)

Linear regression slope is noisy. Add first-third vs last-third mean comparison.
Duration-adaptive minimum drop:
  ≤15 min → 0.20 (need categorical→strong evidence)
  ≤45 min → 0.15 (moderate evidence)
  ≤90 min → 0.10 (gradual weakening meaningful)
  >90 min → 0.08 (even subtle drift matters)

Fire if EITHER slope OR windowed drop crosses threshold.

### 2.3 Duration-Aware Confidence

```python
base_conf = 0.45
duration_bonus = min(0.15, duration_min / 300)
data_bonus = min(0.10, len(denial_timeline) / 50)
conf = min(0.65, base_conf + duration_bonus + data_bonus)
```

---

## Part 3: Validation Tests

File: backend/services/language_agent/tests/test_interrogation_rules.py

### TestContaminationDetector
- test_filters_common_words: "yeah"/"guess"/"told" NOT flagged
- test_detects_case_specific_entities: "Walmart"/"knife" flagged via NER
- test_respects_temporal_ordering: suspect-first NOT flagged
- test_requires_minimum_matches: 1 match → no signal
- test_graceful_degradation_no_spacy: falls back, no crash

### TestAdaptiveDenialThreshold
- test_short_session_strict: 15 min → threshold ≈ -4e-5
- test_long_session_relaxed: 60 min → threshold ≈ -2e-5
- test_fires_on_gradual_weakening: 54 min categorical→acquiescence → fires
- test_does_not_fire_stable: identical denials → no signal
- test_windowed_comparison_fires: slope borderline but drop=0.25 → fires

### TestIntegration
- test_full_54min_interrogation: realistic segments → contamination 1-3 hits + denial 1 hit

---

## Expected Results After Fix

| Signal | Before | After |
|--------|:------:|:-----:|
| statement_contamination | 26 (77% false positive) | 4-8 (< 10% false positive) |
| denial_weakening | 0 (never fired) | 1-3 (if denials weakened) |

## Files Modified:
1. backend/services/language_agent/interrogation_rules.py — ContaminationDetector class + adaptive denial + validated stopwords (~200 lines changed)
2. backend/services/language_agent/tests/test_interrogation_rules.py — 10 tests (~200 lines created)