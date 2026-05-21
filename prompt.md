# NEXUS — Remove Unreliable Signals + Add Research-Verified Replacements

## What's Being Removed (4 signals)

| Signal | Why Remove |
|--------|-----------|
| `barrier_behavior` | No meta-analytic effect size. Arms crossing is habitual/thermal in majority of cases. Not in DePaulo 2003. |
| `low_autonomic_reactivity` | NEXUS-original. Zero published research. Absence of stress is not a validated cue. |
| `pronoun_distancing` | DePaulo et al. 2003 meta-analysis: first-person pronoun use d=0.03 (near zero). Adams 2002 found the effect reverses under stress. |
| `tense_inconsistency` | No controlled effect size. Present-tense intrusions occur equally in genuine trauma recall and fabrication. |

## What's Being Added (5 signals — verified citations only)

### Signal 1: `detail_reduction` (Language Agent)

**Research:**
- DePaulo et al. (2003) "Cues to Deception" Psychological Bulletin 129(1):74-118.
  Finding: liars are "less forthcoming" and "suspiciously bereft of ordinary
  imperfections and unusual details" (p.104). Of 14 significant cues across 158
  analyzed, detail-related cues were among the largest effects.
  Average d across all 14 significant cues: 0.25 (cited in Vrij 2012).
  Detail reduction is at or above this average.
- Vrij, A. (2008) "Detecting Lies and Deceit" 2nd ed. Wiley. Chapter on
  Criteria-Based Content Analysis (CBCA) confirms sensory details (visual,
  auditory, spatial, temporal) discriminate truthful from fabricated accounts.
  CBCA is an established forensic interview assessment tool.

**What is verified:** Direction (fewer details in deception). Effect size
approximately d=0.25-0.35 (among the largest in DePaulo's meta-analysis but
exact per-cue d not independently verified from the table).

**What is novel (honest label):** The specific sensory word list categories
and the density-comparison operationalization are engineering decisions, not
from any paper.

**Confidence cap:** 0.50 (one of the strongest verbal cues but still ~35% FP).

### Signal 2: `vocal_hesitation_cluster` (Voice Agent)

**Research:**
- Sporer, S.L. & Schwandt, B. (2006) "Paraverbal indicators of deception:
  A meta-analytic synthesis." Applied Cognitive Psychology 20:421-446.
  Finding: "speech errors were positively related to deception" in their
  meta-analysis of 9 paraverbal behaviors. Effect sizes described as "small."
  Filled pauses (um, uh, er) are a category of speech errors.
- Vrij, A. (2008) "Detecting Lies and Deceit" synthesizes: cognitive load
  during deception increases speech disfluencies.

**What is verified:** Direction (more filled pauses during deception). Effect
size is "small" per Sporer & Schwandt — exact d not extracted from their paper.

**What is novel (honest label):** The "3+ fillers in 10 seconds" cluster
threshold is an engineering heuristic. No paper defines a cluster threshold.
The concept of temporal clustering (vs individual fillers) is a reasonable
engineering application but NOT from research.

**Confidence cap:** 0.40 (direction valid but effect is small and clusters
also indicate general nervousness, not specifically deception).

### Signal 3: `speech_rate_change` (Voice Agent)

**Research:**
- DePaulo et al. (2003) meta-analysis includes speech rate as a cue.
- Sporer & Schwandt (2006): "speech rate was slightly positively related
  to deception after short preparation (r=.082), and unrelated after
  medium preparation time (r=.034)."

**CRITICAL: Direction is MIXED.** Liars speak FASTER with short preparation
(r=+0.082) and show no change with longer preparation. The relationship
"varied as a function of content, preparation, motivation, sanctioning."
This signal must detect significant change in EITHER direction, not
specifically deceleration.

**What is verified:** Speech rate changes during deception. Effect is small
(r≈0.08, approximately d≈0.16). Direction depends on context.

**What is novel (honest label):** The ">30% deviation from baseline" threshold
and "consecutive window" requirement are engineering decisions.

**Confidence cap:** 0.40 (small effect, direction unpredictable, also changes
with fatigue/topic difficulty).

### Signal 4: `narrative_consistency_drift` (Language Agent)

**Research:**
- Granhag, P.A. & Strömwall, L.A. (1999) "Repeated interrogations:
  stretching the deception detection paradigm." Expert Evidence 7:163-174.
  Finding: repeated questioning reveals inconsistency in deceptive accounts.
- Fisher, R.P. & Geiselman, R.E. (1992) "Memory-Enhancing Techniques for
  Investigative Interviewing: The Cognitive Interview." Charles C Thomas.
  The Cognitive Interview technique relies on retelling to detect inconsistency.
- Vrij, A., Mann, S., Fisher, R., Leal, S., Milne, R. & Bull, R. (2009)
  "Increasing cognitive load to facilitate lie detection." Psychology, Crime
  & Law 15(2-3):97-109. Consistency decreases across repeated accounts for liars.

**What is verified:** Direction (liars' accounts drift more across retellings).
Granhag & Strömwall 1999 and Vrij et al. 2009 support this direction.

**What is NOT verified:** No quantified effect size (d value) from these papers.
The d≈0.25 I previously cited was fabricated. Cite as: "direction supported;
no meta-analytic effect size available for this specific operationalization."

**What is novel (honest label):** TF-IDF cosine similarity between retelling
pairs is an engineering operationalization. Jaccard fallback is engineering.
The "3+ shared content words AND >5 minutes apart" pairing criterion is
an engineering heuristic.

**Confidence cap:** 0.40 (no quantified effect size; inconsistency also occurs
from genuine memory degradation, fatigue, or different question framing).

### Signal 5: `self_adaptor_increase` (Video Agent)

**Research:**
- Li, Y., Song, Y., Li, J. & Li, X. (2024) "Nonverbal cues to deception:
  insights from a mock crime scenario in a Chinese sample." Frontiers in
  Psychology 15. doi:10.3389/fpsyg.2024.1331653.
  Finding: "liars exhibited a higher frequency of self-adaptors."
- DePaulo et al. (2003) meta-analysis: fidgeting d=0.10 (small effect).
- Vrij et al. (1996) "instructed subjects to undergo interrogation in both
  truthful and deceptive scenarios and observed a significant decrease in leg
  and foot movements as well as hand and finger movements during deception."

**What is verified:** Direction (self-adaptors increase during deception).
Effect size: d=0.10 per DePaulo 2003 meta-analysis for fidgeting.

**What is NOT verified:** The "increasing temporal trend" concept (comparing
first-third vs last-third rate) is NOT from any paper. No study measures the
effect size of a temporal trend in self-adaptors specifically. The d=0.10 is
for absolute fidgeting rate, not for the trend.

**What is novel (honest label):** The temporal trend operationalization
(session thirds comparison) is engineering. The 1.5× ratio threshold is
an engineering heuristic.

**Confidence cap:** 0.35 (d=0.10 is very small; the trend concept is
unvalidated; mounting anxiety is equally present in innocent suspects
under prolonged questioning).

---

## CRITICAL INSTRUCTIONS

**Before writing ANY code:**

1. Read EVERY file in the dependency trace. Understand what each reference
   does before removing or modifying.

2. Use proper OOP and DSA:
   - **Sliding Window** (deque): hesitation clusters, speech rate consecutive windows
   - **HashMap** (frozenset): O(1) sensory word lookup for detail_reduction
   - **Cosine Similarity** (numpy/sklearn): narrative consistency comparison
   - **IntervalIndex** (bisect): O(log N) diar segment overlap for speech_rate_change
   - **Strategy Pattern**: each signal is a separate method in its agent's rule class

3. Do NOT break any existing functionality. Every removal has a corresponding
   update in every consumer listed in the dependency trace.

4. ALL confidence caps reflect actual research strength:
   - detail_reduction: 0.50 (strongest, d≈0.25-0.35)
   - vocal_hesitation_cluster: 0.40 (direction valid, small effect)
   - speech_rate_change: 0.40 (small effect, direction mixed)
   - narrative_consistency_drift: 0.40 (no quantified effect size)
   - self_adaptor_increase: 0.35 (d=0.10, trend concept unvalidated)

5. ALL signal metadata.interpretation MUST include multiple possible
   explanations. NEVER claim deception from any single signal.

6. ALL signal metadata MUST cite the actual paper with correct year
   and the specific finding (not an approximate effect size).

---

## Dependency Trace — Removals

### barrier_behavior → Replace with `self_adaptor_increase`

| File | Action |
|------|--------|
| `interrogation_patterns.py` line 54 — `_RESISTANCE_TYPES` | Replace |
| `interrogation_patterns.py` line 375 — `_RESISTANCE_BUILDING` | Replace |
| `handcuff_detector.py` line 71 — suppressed rules | Remove barrier_behavior, ADD self_adaptor_increase (restricted hand range when cuffed) |
| `handcuff_detector.py` line 88 — alternative measurements | Remove barrier_behavior→torso_lean_away entry |
| `VideoSignalPlayer.tsx` line 613 — SIGNAL_CONFIG | Remove, add self_adaptor_increase |
| `signalDisplayConfig.ts` — 2 entries (base + torso_lean_away) | Remove both, add self_adaptor_increase |
| `sessions.py` — `_VIDEO_OVERLAY_TYPES` | Replace |

### low_autonomic_reactivity → Remove entirely (no replacement)

| File | Action |
|------|--------|
| `VideoSignalPlayer.tsx` line 633 — SIGNAL_CONFIG | Remove |
| `signalDisplayConfig.ts` — entry | Remove |
| `sessions.py` — `_VIDEO_OVERLAY_TYPES` | Remove |

### pronoun_distancing → Replace with `detail_reduction`

| File | Action |
|------|--------|
| `interrogation_patterns.py` line 55 — `_RESISTANCE_TYPES` | Replace |
| `interrogation_patterns.py` line 339 — recommendation text | Update text |
| `interrogation_patterns.py` line 375 — `_RESISTANCE_BUILDING` | Replace |
| `narrative.py` line 307 — `_INTERROG_TYPES` | Replace |
| `InterrogationSummaryPanel.tsx` line 20 — INTERROGATION_TYPES | Replace |
| `VideoSignalPlayer.tsx` line 641 — SIGNAL_CONFIG | Remove, add detail_reduction |
| `signalDisplayConfig.ts` — entry | Remove, add detail_reduction |
| `sessions.py` — `_LANGUAGE_OVERLAY_TYPES` | Replace |

### tense_inconsistency → Replace with `vocal_hesitation_cluster` in patterns, `narrative_consistency_drift` in frontend/narrative

| File | Action |
|------|--------|
| `interrogation_patterns.py` line 56 — `_RESISTANCE_TYPES` | Replace with `vocal_hesitation_cluster` |
| `interrogation_patterns.py` line 376 — `_RESISTANCE_BUILDING` | Replace with `vocal_hesitation_cluster` |
| `narrative.py` line 307 — `_INTERROG_TYPES` | Replace with `narrative_consistency_drift` |
| `InterrogationSummaryPanel.tsx` line 21 — INTERROGATION_TYPES | Replace with `narrative_consistency_drift` |
| `VideoSignalPlayer.tsx` line 647 — SIGNAL_CONFIG | Remove, add narrative_consistency_drift + vocal_hesitation_cluster |
| `signalDisplayConfig.ts` — entry | Remove, add narrative_consistency_drift + vocal_hesitation_cluster |
| `sessions.py` — `_LANGUAGE_OVERLAY_TYPES` | Replace with `narrative_consistency_drift` |

---

## Implementation Specs

### detail_reduction (Language Agent)

**File:** `services/language_agent/interrogation_rules.py`

Sensory word categories — frozenset per category, O(1) lookup:
```
VISUAL: saw, looked, bright, dark, color, light, shadow, watched, noticed, appeared, visible, clear
AUDITORY: heard, sound, loud, quiet, noise, voice, bang, click, ring, whisper, screamed, yelled
SPATIAL: left, right, above, below, behind, front, inside, outside, corner, across, beside, near
TEMPORAL: before, after, during, moment, suddenly, immediately, already, earlier, later
TACTILE: cold, warm, hot, rough, smooth, soft, hard, wet, sharp, heavy, tight
```

Algorithm: O(S × W) per speaker. For each segment > 20 words, count
sensory words / total words = detail_density. Compare first-half mean
vs second-half mean. If drop > 40% → fire signal.

Metadata must include:
```python
"research": "DePaulo et al. 2003 Psychological Bulletin 129(1):74-118 — liars 'less forthcoming' and 'bereft of unusual details'. Vrij 2008 CBCA validates sensory detail discrimination.",
"effect_note": "Among the largest effects in DePaulo meta-analysis (average d=0.25 across significant cues). Exact per-cue d not independently verified.",
"interpretation": "Narrative lacks sensory richness compared to earlier accounts. Also occurs in genuine memory gaps, fatigue, or topics the speaker finds unimportant."
```

### vocal_hesitation_cluster (Voice Agent)

**File:** `services/voiceAgent/interrogation_rules.py`

Algorithm: Sliding window of 5 consecutive 2s voice windows (10s total).
Count fillers (um, uh, er, ah, like-as-filler) across the window. Compare
to speaker baseline filler_rate from calibration. If cluster ≥ 3 AND
ratio ≥ 2.0× baseline → fire.

Metadata must include:
```python
"research": "Sporer & Schwandt 2006 Applied Cognitive Psychology 20:421-446 — 'speech errors positively related to deception'. Effect sizes described as 'small'.",
"effect_note": "Direction validated. Exact effect size for filled pause clusters not available. Cluster threshold (3+ in 10s) is an engineering heuristic, not from research.",
"interpretation": "Burst of speech disfluencies indicating cognitive load spike. Equally occurs during genuine confusion, word-finding difficulty, or high emotional arousal."
```

### speech_rate_change (Voice Agent)

**File:** `services/voiceAgent/interrogation_rules.py`

Algorithm: Compare window speech_rate to calibrated baseline. If
abs(rate - baseline) / baseline > 0.30 for 2+ consecutive windows → fire.
Record direction (faster/slower) in value_text.

Metadata must include:
```python
"research": "Sporer & Schwandt 2006 Applied Cognitive Psychology 20:421-446 — 'speech rate slightly positively related to deception after short preparation (r=.082), unrelated after medium preparation'. DePaulo et al. 2003 includes speech rate.",
"effect_note": "Direction is MIXED — liars may speak faster OR slower depending on preparation time and context. r≈0.08 (approximately d≈0.16). Signal detects significant change in EITHER direction.",
"interpretation": "Significant speech rate shift from baseline. Acceleration may indicate rehearsed delivery. Deceleration may indicate careful word selection. Both also occur from fatigue, topic change, or emotional arousal."
```

### narrative_consistency_drift (Language Agent)

**File:** `services/language_agent/interrogation_rules.py`

Algorithm: Identify retelling pairs — segments from same speaker sharing
3+ content nouns/verbs AND > 5 minutes apart. Compute TF-IDF cosine
similarity (sklearn if available, Jaccard fallback). If cosine < 0.70 → drift.

Metadata must include:
```python
"research": "Granhag & Strömwall 1999 Expert Evidence 7:163-174 — repeated questioning reveals inconsistency in deceptive accounts. Vrij et al. 2009 Psychology Crime & Law 15(2-3):97-109 — consistency decreases across repeated accounts for liars.",
"effect_note": "Direction supported by multiple studies. No meta-analytic effect size (d value) available for this specific operationalization. TF-IDF cosine threshold (0.70) is an engineering heuristic.",
"interpretation": "Same event described differently at different session times. Also occurs from genuine memory degradation over a long session, different questioning frames, or progressive detail addition."
```

### self_adaptor_increase (Video Agent)

**File:** `services/video_agent/interrogation_rules.py`

Algorithm: Partition session into thirds. Count self_touch +
face_region_touch signals per third. If third_3_rate / max(third_1_rate, 1) ≥ 1.5 → fire. When handcuffed, SUPPRESS entirely.

Metadata must include:
```python
"research": "Li et al. 2024 Frontiers in Psychology 15 doi:10.3389/fpsyg.2024.1331653 — 'liars exhibited higher frequency of self-adaptors'. DePaulo et al. 2003 Psychological Bulletin 129(1):74-118 — fidgeting d=0.10 (small effect).",
"effect_note": "Direction supported (d=0.10 for absolute fidgeting rate). The 'increasing temporal trend' concept is an engineering application — no study quantifies the effect size of self-adaptor rate change over session duration.",
"interpretation": "Rate of face/hair/arm touching increased across the session. Equally present in innocent suspects experiencing mounting pressure, fatigue, or discomfort from prolonged sitting."
```

---

## Updated Frozensets

```python
_RESISTANCE_TYPES = frozenset({
    "self_adaptor_increase",      # Li 2024, DePaulo 2003 d=0.10
    "detail_reduction",           # DePaulo 2003 d≈0.25-0.35
    "vocal_hesitation_cluster",   # Sporer & Schwandt 2006, small effect
    "motor_inhibition",           # Vrij 1996 (kept — validated)
    "blink_suppression_spike",    # Frosina 2018 (kept — validated)
    "facial_emotion:contempt",    # Ekman 1991 (kept)
})

_RESISTANCE_BUILDING = frozenset({
    "self_adaptor_increase",
    "detail_reduction",
    "vocal_hesitation_cluster",
    "motor_inhibition",
})
```

---

## Frontend Display Config

```typescript
'detail_reduction': {
  label: 'Low Detail',
  description: 'Narrative lacks sensory details compared to earlier accounts. Strongest verbal deception cue (DePaulo 2003, d≈0.25-0.35). Also occurs in genuine memory gaps or fatigue.',
  icon: '📝', color: '#A78BFA', category: 'pattern', priority: 1,
},
'vocal_hesitation_cluster': {
  label: 'Hesitation Burst',
  description: 'Cluster of 3+ speech disfluencies within 10 seconds. Cognitive load indicator (Sporer & Schwandt 2006). Equally occurs during genuine confusion or high emotional arousal.',
  icon: '💬', color: '#8B5CF6', category: 'pattern', priority: 2,
},
'speech_rate_change': {
  label: 'Speech Rate Shift',
  description: 'Significant speech rate change (>30%) from baseline. Direction is context-dependent (Sporer & Schwandt 2006, r=0.08). Also changes with fatigue or topic difficulty.',
  icon: '⏩', color: '#8B5CF6', category: 'pattern', priority: 2,
},
'narrative_consistency_drift': {
  label: 'Story Drift',
  description: 'Same event described differently at different times (Granhag & Strömwall 1999). Also occurs from memory degradation or different questioning frames. No quantified effect size.',
  icon: '🔀', color: '#A78BFA', category: 'pattern', priority: 2,
},
'self_adaptor_increase': {
  label: 'Increasing Self-Touch',
  description: 'Self-touch rate increased across the session (Li et al. 2024, DePaulo 2003 d=0.10). Equally present in innocent suspects under prolonged pressure.',
  icon: '✋', color: '#8B5CF6', category: 'body', priority: 2,
},
```

---

## sessions.py Whitelist Updates

```python
# _VIDEO_OVERLAY_TYPES:
#   REMOVE: "barrier_behavior", "low_autonomic_reactivity"
#   ADD: "self_adaptor_increase"

# _LANGUAGE_OVERLAY_TYPES:
#   REMOVE: "pronoun_distancing", "tense_inconsistency"
#   ADD: "detail_reduction", "narrative_consistency_drift"

# _VOICE_INTERROG_TYPES:
#   ADD: "vocal_hesitation_cluster", "speech_rate_change"
```

---

## Verification After Implementation

1. Grep entire codebase for removed signal names — zero results expected:
   `grep -rn "barrier_behavior\|low_autonomic_reactivity\|pronoun_distancing\|tense_inconsistency"`

2. Run 54-minute interrogation video:
   - detail_reduction: should fire if suspect's narratives lose sensory detail
   - vocal_hesitation_cluster: should fire during high-pressure questioning segments
   - speech_rate_change: should fire when speech rate shifts >30% from baseline
   - narrative_consistency_drift: should fire if suspect retells events differently
   - self_adaptor_increase: should fire if self-touch rate increases across session

3. Check no existing functionality broken:
   - ResistanceHardening still fires (new signals in _RESISTANCE_TYPES)
   - FalseConfessionRiskAssessor still fires (new signals in _RESISTANCE_BUILDING)
   - Handcuff suppression works (self_adaptor_increase suppressed when cuffed)
   - Narrative interrogation section renders (updated _INTERROG_TYPES)
   - Frontend displays all new signals (updated SIGNAL_CONFIG + signalDisplayConfig)

## Files Modified (10 total):

### Backend (5):
1. `services/voiceAgent/interrogation_rules.py` — 2 new methods: vocal_hesitation_cluster, speech_rate_change (~60 lines)
2. `services/language_agent/interrogation_rules.py` — 2 new methods: detail_reduction, narrative_consistency_drift. Remove: _pronoun_distancing, _tense_shift (~70 lines net)
3. `services/video_agent/interrogation_rules.py` — 1 new method: self_adaptor_increase. Remove: _barrier_behavior, _low_autonomic_reactivity (~30 lines net)
4. `services/fusion_agent/interrogation_patterns.py` — update _RESISTANCE_TYPES, _RESISTANCE_BUILDING frozensets + recommendation text (~8 lines)
5. `services/fusion_agent/narrative.py` — update _INTERROG_TYPES (~2 lines)
6. `services/video_agent/handcuff_detector.py` — remove barrier_behavior, add self_adaptor_increase to suppression (~3 lines)
7. `backend/api/sessions.py` — update 3 whitelist arrays (~5 lines)

### Frontend (3):
8. `signalDisplayConfig.ts` — remove 5 entries, add 5 entries (~30 lines net)
9. `VideoSignalPlayer.tsx` — update SIGNAL_CONFIG (~8 lines)
10. `InterrogationSummaryPanel.tsx` — update INTERROGATION_TYPES (~2 lines)