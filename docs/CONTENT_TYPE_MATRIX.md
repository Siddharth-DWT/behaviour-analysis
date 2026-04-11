# NEXUS Content-Type Adaptation Matrix

**For each of the 97 rules: what changes per content type, based on research findings.**

Legend:
- **FIRE** = Rule runs with universal threshold, no change needed for this type
- **ADAPT** = Threshold/detection changes for this type (with specific values)
- **GATE** = Rule suppressed for this type
- **RENAME** = Same detection, different output label
- **REINTERPRET** = Same detection, same label, different narrative interpretation

The research finding: **85% of rules run universally across all types.** The per-speaker baseline handles most variation. Only the cells marked ADAPT/GATE/RENAME/REINTERPRET differ from the default.

---

## Phase 1: 42 Implemented Rules

### Voice Agent — 16 Rules

| # | Rule | Signal | Sales Call | Client Meeting | Internal | Interview | Podcast |
|---|------|--------|-----------|----------------|----------|-----------|---------|
| 1 | VOICE-CAL-01 | speaker_baseline | FIRE | FIRE | FIRE | FIRE (extend to 5+ min stabilization per Kappen 2024 — social-evaluative stress differs from cognitive stress) | FIRE |
| 2 | VOICE-STRESS-01 | vocal_stress_score | FIRE (weights: F0 0.30, Jitter 0.20, Rate 0.15, Filler 0.10, Pause 0.10, HNR 0.10, Shimmer 0.05) | FIRE | FIRE | FIRE — baseline absorbs interview-elevated stress automatically | FIRE |
| 3 | VOICE-FILLER-01 | filler_detection | FIRE (+50% spike from baseline) | FIRE | FIRE | FIRE — baseline absorbs higher interview filler rate | FIRE |
| 4 | VOICE-FILLER-02 | filler_credibility | FIRE (2.5% / 4.0% / 6.0%) | FIRE | ADAPT: noticeable threshold +0.5% (informal speech has more fillers, per Bortfeld 2001 — 82% increase from role switching alone) | FIRE — credibility impact applies equally to interview candidates | FIRE |
| 5 | VOICE-PITCH-01 | pitch_elevation | FIRE (7% / 12% / 20%) | FIRE | FIRE | FIRE — baseline absorbs elevated interview pitch | FIRE |
| 6 | VOICE-PITCH-02 | monotone_flag | FIRE (<40% of baseline F0 range) | FIRE | FIRE | FIRE | FIRE |
| 7 | VOICE-RATE-01 | speech_rate_anomaly | FIRE (±20%) | FIRE | FIRE | FIRE | FIRE |
| 8 | VOICE-TONE-01 | warm | FIRE | FIRE | FIRE | FIRE | FIRE |
| 9 | VOICE-TONE-02 | cold | FIRE | FIRE | FIRE | FIRE | FIRE |
| 10 | VOICE-TONE-03 | nervous | FIRE | FIRE | FIRE | FIRE — nervousness is equally diagnostic in interviews; baseline handles elevated arousal | FIRE |
| 11 | VOICE-TONE-04 | confident | FIRE | FIRE | FIRE | FIRE | FIRE |
| 12 | VOICE-TONE-05 | aggressive | FIRE | FIRE | FIRE | FIRE | FIRE |
| 13 | VOICE-TONE-06 | excited | FIRE | FIRE | FIRE | FIRE | FIRE |
| 14 | VOICE-ENERGY-01 | energy_level | FIRE (±6dB) | FIRE | FIRE | FIRE | FIRE |
| 15 | VOICE-PAUSE-01 | pause_classification | FIRE (hesitation >200ms, extended >2000ms) | FIRE | FIRE | ADAPT: extended_hesitation threshold **>3000ms** (complex answers require thinking time — not diagnostic in interviews per Stivers 2009 dispreference latency research) | FIRE |
| 16 | VOICE-INT-01 | interruption_event | FIRE (>500ms, ≤3 words backchannel) | FIRE | FIRE | FIRE — interruption detection is universal; interpretation changes in narrative | FIRE |

**Voice Talk Time — Special case (content-type detection-layer change):**

| # | Rule | Signal | Sales Call | Client Meeting | Internal | Interview | Podcast |
|---|------|--------|-----------|----------------|----------|-----------|---------|
| 17 | VOICE-TALK-01 | talk_time_ratio | ADAPT: **>60% seller = significant** (Gong: top performers 43%, underperformers 64%). Keep >70% extreme. | FIRE (>70%) | ADAPT: **>50% any peer = significant** in peer meetings | ADAPT: Flag only if **candidate <30% or >70%** (BarRaiser: optimal candidate talk 40-55%) | ADAPT: Flag only if **host >50%** (guest should dominate 60-80%) |

### Language Agent — 12 Rules

| # | Rule | Signal | Sales Call | Client Meeting | Internal | Interview | Podcast |
|---|------|--------|-----------|----------------|----------|-----------|---------|
| 18 | LANG-SENT-01 | sentiment_score | FIRE (model 0.85-0.90 / LIWC 0.10-0.15) | FIRE | FIRE | FIRE | FIRE |
| 19 | LANG-SENT-02 | emotional_intensity | FIRE (>8% high, <2% suppressed) | FIRE | ADAPT: **>6% high, <1.5% suppressed** (professional speech runs lower per Tausczik 2010 — avg 4%, professional ~3-3.5%) | FIRE | FIRE |
| 20 | LANG-BUY-01 | buying_signal | FIRE | RENAME: **"engagement_signal"** — same pricing/timeline/implementation question detection | GATE — not applicable | RENAME: **"interest_signal"** — candidate asking about start dates, team structure, comp = strong interest | GATE — not applicable |
| 21 | LANG-OBJ-01 | objection_signal | FIRE | RENAME: **"concern"** | RENAME: **"disagreement"** | RENAME: **"hesitation"** | GATE — podcast disagreement is content, not objection |
| 22 | LANG-PWR-01 | power_language | FIRE | FIRE | FIRE | FIRE — powerful language = competence signal (same detection, narrative interprets positively for candidates) | FIRE |
| 23 | LANG-PERS-01 | persuasion_technique | FIRE | RENAME: **"influence_tactic"** | RENAME: **"influence_attempt"** | RENAME: **"impression_management"** | GATE — persuasion detection irrelevant for podcasts |
| 24 | LANG-QUES-01 | question_type | FIRE (open/closed/tag + SPIN) | FIRE (open/closed/tag only; **GATE SPIN**) | FIRE (open/closed/tag only; **GATE SPIN**) | FIRE (open/closed/tag only; **GATE SPIN**) | FIRE (open/closed/tag only; **GATE SPIN**) |
| 25 | LANG-TOPIC-01 | topic_shift | FIRE (adaptive cosine ~0.3) | FIRE | FIRE | FIRE | FIRE |
| 26 | LANG-NEG-01 | gottman_horsemen | FIRE (criticism, contempt, defensiveness, stonewalling) | FIRE | FIRE but **REINTERPRET**: stonewalling → "disengagement" (may be adaptive boundary-setting per Cortina) | FIRE but **REINTERPRET**: stonewalling → "disengagement", defensiveness → "resistance" (legitimate in interview context) | GATE — podcast disagreement patterns differ fundamentally |
| 27 | LANG-EMP-01 | empathy_language | FIRE | FIRE | FIRE | FIRE — empathy from interviewer = good practice; from candidate = soft skill signal (narrative interprets per speaker role) | FIRE |
| 28 | LANG-CLAR-01 | clarity_score | FIRE (>18 words, >10% passive) | FIRE | FIRE | FIRE | FIRE |
| 29 | LANG-INTENT-01 | intent | FIRE (sales prompt: INFORM, QUESTION, PROPOSE, COMMIT, AGREE, DISAGREE, NEGOTIATE, DEFLECT, ACKNOWLEDGE, GREET, CLOSE) | ADAPT: **meeting prompt** (INFORM, QUESTION, PROPOSE, COMMIT, AGREE, DISAGREE, PRESENT, FOLLOW_UP, ACKNOWLEDGE, GREET, CLOSE) | ADAPT: **meeting prompt** (same as client) | ADAPT: **interview prompt** (INFORM, QUESTION, RESPOND, ELABORATE, CLARIFY, ACKNOWLEDGE, GREET, CLOSE — focus on question quality + answer depth) | ADAPT: **podcast prompt** (INFORM, QUESTION, NARRATE, ELABORATE, AGREE, DISAGREE, JOKE, TRANSITION, ACKNOWLEDGE) |

### Conversation Agent — 8 Rules

| # | Rule | Signal | Sales Call | Client Meeting | Internal | Interview | Podcast |
|---|------|--------|-----------|----------------|----------|-----------|---------|
| 30 | CONVO-TURN-01 | turn_taking | FIRE (>10/min rapid, <2/min monologue) | FIRE | FIRE | FIRE — lower turn rate is structurally expected but baseline handles this | FIRE |
| 31 | CONVO-LAT-01 | response_latency | FIRE (<200ms overlapping, 200-600ms engaged, 600-1500ms deliberative, >1500ms delayed) | FIRE | FIRE | FIRE — 600-1500ms band carries **neutral** interpretation for interview candidates (per Stivers 2009: complex answers need formulation time) | FIRE |
| 32 | CONVO-DOM-01 | dominance_score | FIRE (weights: talk 0.50, interrupt 0.15, monologue 0.25, question 0.10) | FIRE | FIRE | FIRE — interviewer structural dominance is expected; flag only if candidate <0.30 or >0.70 | FIRE |
| 33 | CONVO-INT-01 | interruption_pattern | FIRE (>3/min frequent) | FIRE | FIRE | FIRE | FIRE |
| 34 | CONVO-RAP-01 | rapport_indicator | FIRE | FIRE | FIRE | FIRE | FIRE |
| 35 | CONVO-ENG-01 | engagement | FIRE | FIRE | FIRE | FIRE | FIRE |
| 36 | CONVO-BAL-01 | balance | FIRE (Gini <0.15 balanced, >0.35 imbalanced) | FIRE | FIRE | ADAPT: Compare to **expected Gini 0.20-0.40** (candidate talks more). Flag deviation from expected, not from symmetry. | ADAPT: Compare to **expected Gini 0.30-0.50** (guest talks more). |
| 37 | CONVO-CONF-01 | conflict_score | FIRE (min 2 indicators) | FIRE | FIRE | FIRE | GATE — podcast disagreement is content, not conflict |

### Fusion Agent — 6 Audio-Only Rules

| # | Rule | Signal | Sales Call | Client Meeting | Internal | Interview | Podcast |
|---|------|--------|-----------|----------------|----------|-----------|---------|
| 38 | FUSION-02 | stress_sentiment_incongruence (renamed) | FIRE (cap 0.65) | FIRE | FIRE | FIRE — baseline absorbs interview stress; only genuine mismatch triggers | FIRE |
| 39 | FUSION-07 | verbal_incongruence | FIRE (cap 0.70) | FIRE | FIRE | FIRE | FIRE |
| 40 | FUSION-13 | urgency_authenticity | FIRE (cap 0.60) | FIRE (pitch/presentation segments) | GATE — urgency detection irrelevant for internal meetings (no persuasion context) | GATE — urgency detection inappropriate for interviews | GATE — not applicable |
| 41 | FUSION-GRAPH-01 | tension_cluster | FIRE (min 3 signals) | FIRE | FIRE | FIRE | FIRE |
| 42 | FUSION-GRAPH-02 | momentum_shift | FIRE | FIRE | FIRE | FIRE | FIRE |
| 43 | FUSION-GRAPH-03 | persistent_incongruence | FIRE | FIRE | FIRE | FIRE | FIRE |

---

## Phase 2: Video Agents — 22 Rules

### Facial Agent — 7 Rules

| # | Rule | Signal | Sales Call | Client Meeting | Internal | Interview | Podcast (video) |
|---|------|--------|-----------|----------------|----------|-----------|-----------------|
| 44 | FACE-CAL-01 | facial_baseline | FIRE (90-120s for formal) | FIRE (90-120s) | FIRE (60s OK) | FIRE (90-120s — greeting behaviors longer in interviews) | N/A audio podcast; FIRE if video |
| 45 | FACE-EMO-01 | primary_emotion | FIRE (per-emotion confidence: Happy 0.70, Angry/Sad 0.55, Disgust 0.35) | FIRE | FIRE | FIRE — all emotions equally diagnostic for candidates | N/A |
| 46 | FACE-SMILE-01 | smile_quality | FIRE (cap 0.45-0.55, temporal onset features) | FIRE | FIRE | FIRE — social smiling in interviews is normal, baseline handles it | N/A |
| 47 | FACE-MICRO-01 | rapid_expression_change | **DISABLED** at ≤30fps across ALL types | **DISABLED** | **DISABLED** | **DISABLED** | N/A |
| 48 | FACE-ENG-01 | engagement_visual | FIRE (cap 0.50) | FIRE | FIRE | FIRE | N/A |
| 49 | FACE-STRESS-01 | stress_visual | FIRE (cap 0.35-0.40) | FIRE | FIRE | FIRE — baseline-relative; interview facial stress absorbed by baseline | N/A |
| 50 | FACE-VA-01 | valence_arousal | FIRE (valence 0.55, arousal 0.40) | FIRE | FIRE | FIRE | N/A |

### Gaze Agent — 7 Rules

| # | Rule | Signal | Sales Call | Client Meeting | Internal | Interview | Podcast (video) |
|---|------|--------|-----------|----------------|----------|-----------|-----------------|
| 51 | GAZE-CAL-01 | gaze_baseline | FIRE (60-120s, camera position estimation) | FIRE | FIRE | FIRE | N/A |
| 52 | GAZE-DIR-01 | gaze_direction | FIRE (binary screen-directed vs away) | FIRE | FIRE | FIRE | N/A |
| 53 | GAZE-CONTACT-01 | screen_engagement | FIRE (speaking 40-60%, listening 60-75%) | FIRE | FIRE — but note Argyle norms NOT validated for video calls; use baseline-relative deviation | FIRE | N/A |
| 54 | GAZE-BLINK-01 | blink_rate | FIRE (stress >30-35 bpm, baseline-relative) | FIRE | FIRE | FIRE | N/A |
| 55 | GAZE-ATT-01 | attention_score | FIRE (cap 0.55, composite) | FIRE | FIRE | FIRE | N/A |
| 56 | GAZE-DIST-01 | distraction_count | FIRE (sustained break >8-10s) | FIRE | FIRE | FIRE — gaze aversion during candidate thinking is normal (Glenberg 1998: avg 6s cognitive aversion) | N/A |
| 57 | GAZE-SYNC-01 | gaze_alignment | FIRE (cap 0.40, EXPERIMENTAL) | FIRE | FIRE | FIRE | N/A |

### Body Agent — 8 Rules

| # | Rule | Signal | Sales Call | Client Meeting | Internal | Interview | Podcast (video) |
|---|------|--------|-----------|----------------|----------|-----------|-----------------|
| 58 | BODY-CAL-01 | body_baseline | FIRE (90-120s, discard first 30s) | FIRE | FIRE (60s OK for casual) | FIRE (90-120s) | N/A |
| 59 | BODY-POST-01 | posture_score | FIRE (cap 0.40-0.55, penalty when arms not visible) | FIRE | FIRE | FIRE | N/A |
| 60 | BODY-HEAD-01 | head_nod_shake | FIRE (cap 0.55-0.75, highest-reliability body signal) | FIRE | FIRE | FIRE — universal signal across all types | N/A |
| 61 | BODY-LEAN-01 | leaning_direction | FIRE (cap 0.30-0.40, min 8-10% head-size change) | FIRE | FIRE | FIRE | N/A |
| 62 | BODY-GEST-01 | gesture_type | FIRE (cap 0.45-0.80) | FIRE | FIRE | FIRE | N/A |
| 63 | BODY-FIDG-01 | fidget_rate | FIRE (cap 0.35-0.45, label "elevated_movement") | FIRE | FIRE | FIRE — fidgeting is ambiguous in interviews (anxiety, excitement, ADHD). Require co-occurring stress signals before interpreting as anxiety. | N/A |
| 64 | BODY-TOUCH-01 | self_touch | FIRE (cap 0.35-0.50) | FIRE | FIRE | FIRE | N/A |
| 65 | BODY-MIRROR-01 | body_mirroring | FIRE (cap 0.20-0.30, **EXPERIMENTAL**) | FIRE | FIRE | FIRE | N/A |

---

## Phase 2+: Remaining Fusion Pairwise — 13 Rules

| # | Rule | Cross-Modal | Sales Call | Client Meeting | Internal | Interview | Podcast |
|---|------|-------------|-----------|----------------|----------|-----------|---------|
| 66 | FUSION-01 | Tone × Face → Masking | FIRE (cap 0.75) | FIRE | FIRE | FIRE — incongruence detection is universal; narrative adds disclaimer that interview stress can cause voice-face mismatch | N/A |
| 67 | FUSION-03 | Posture × Energy → Enthusiasm | FIRE (cap 0.65) | FIRE | FIRE | FIRE | N/A |
| 68 | FUSION-04 | Gaze × Filler → Uncertainty | FIRE (cap 0.70) | FIRE | FIRE | FIRE — relabel as "cognitive_effort" in narrative for interview context (thinking, not uncertain) | N/A |
| 69 | FUSION-05 | Buy × Body → Purchase Intent | FIRE (cap 0.55-0.60) | RENAME: **"engagement_verification"** | GATE | RENAME: **"interest_verification"** | GATE |
| 70 | FUSION-06 | Micro × Lang → Leakage | **DISABLED** at ≤30fps across ALL types | **DISABLED** | **DISABLED** | **DISABLED** | N/A |
| 71 | FUSION-07* | Head Nod × Speech → Disagree | FIRE (cap 0.65) | FIRE | FIRE | FIRE — universal, highest-value fusion signal | N/A |
| 72 | FUSION-08 | Eye × Hedge → False Confidence | FIRE (cap 0.55) | FIRE | FIRE | FIRE | N/A |
| 73 | FUSION-09 | Smile × Sentiment → Sarcasm | FIRE (cap 0.60) | FIRE | FIRE | FIRE | N/A |
| 74 | FUSION-10 | Latency × Face → Cog Load | FIRE (cap 0.60) | FIRE | FIRE | REINTERPRET: cognitive load in interview = question difficulty signal (neutral, not negative) | N/A |
| 75 | FUSION-11 | Dominance × Gaze → Anxious | FIRE (cap 0.65) | FIRE | FIRE | FIRE | N/A |
| 76 | FUSION-12 | Interrupt × Body → Intent | FIRE (cap 0.55) | FIRE | FIRE | REINTERPRET: interviewer forward lean + interrupt = engagement (positive), not aggressive | N/A |
| 77 | FUSION-14 | Empathy × Nod → Rapport | FIRE (cap 0.70) | FIRE | FIRE | FIRE — strongest fusion rule, universal | N/A |
| 78 | FUSION-15 | Filler × Gaze → Uncertainty | **MERGE with FUSION-04** — redundant signals, inconsistent caps | — | — | — | — |

---

## Phase 3: Compound Patterns — 12 Rules

| # | Rule | Pattern | Sales Call | Client Meeting | Internal | Interview | Podcast |
|---|------|---------|-----------|----------------|----------|-----------|---------|
| 79 | COMPOUND-01 | Genuine Engagement | FIRE (cap 0.80) | FIRE | FIRE | FIRE — primary positive candidate signal | N/A (audio: use audio-only subset) |
| 80 | COMPOUND-02 | Active Disengagement | FIRE (cap 0.75) | FIRE | FIRE | FIRE — candidate disengagement = role mismatch (high value signal) | N/A |
| 81 | COMPOUND-03 | Emotional Suppression | FIRE (cap 0.70) | FIRE | FIRE | FIRE — professional emotional control in interviews is expected; narrative frames neutrally | N/A |
| 82 | COMPOUND-04 | Decision Engagement (renamed) | FIRE (cap 0.65, EXPERIMENTAL) | RENAME: **"commitment_signals"** | RENAME: **"consensus_signals"** | RENAME: **"decision_signals"** | N/A |
| 83 | COMPOUND-05 | Cognitive Overload | FIRE (cap 0.75) | FIRE | FIRE | REINTERPRET: overload on technical questions = question difficulty signal, not candidate weakness | N/A |
| 84 | COMPOUND-06 | Conflict Escalation | FIRE (cap 0.80) | FIRE | FIRE | FIRE | N/A |
| 85 | COMPOUND-07 | Verbal-Nonverbal Discordance (renamed) | FIRE (cap 0.65) | FIRE | FIRE | FIRE — add cultural sensitivity documentation | N/A |
| 86 | COMPOUND-08 | Rapport Peak | FIRE (cap 0.80) | FIRE | FIRE | FIRE — interviewer-candidate chemistry signal | N/A |
| 87 | COMPOUND-09 | Topic Avoidance | FIRE (cap 0.70) | FIRE | FIRE | REINTERPRET: candidate avoiding topic = possible weakness area (coaching note, not red flag) | N/A |
| 88 | COMPOUND-10 | Authentic Confidence | FIRE (cap 0.75) | FIRE | FIRE | FIRE — candidate assessment metric, high value | N/A |
| 89 | COMPOUND-11 | Anxiety Performance | FIRE (cap 0.65) | FIRE | FIRE | FIRE — **CORE interview diagnostic**: anxious but performing well = coachable | N/A |
| 90 | COMPOUND-12 | Deception Risk | FIRE (cap 0.55, best-calibrated) | FIRE | FIRE | GATE — deception detection in interviews produces unacceptable false positives (DePaulo 2003: stress and deception share overlapping profiles; Bogaard 2025: both innocent and guilty show similar arousal in interviews) | N/A |

---

## Phase 3: Temporal Sequences — 8 Rules

| # | Rule | Sequence | Sales Call | Client Meeting | Internal | Interview | Podcast |
|---|------|----------|-----------|----------------|----------|-----------|---------|
| 91 | TEMPORAL-01 | Stress Cascade | FIRE (cap 0.65, EXPERIMENTAL) | FIRE | FIRE | FIRE | N/A |
| 92 | TEMPORAL-02 | Engagement Build | FIRE (cap 0.65) | FIRE | FIRE | FIRE — tracks candidate warming up over first 5 minutes | N/A |
| 93 | TEMPORAL-03 | Disengage Cascade | FIRE (cap 0.55) | FIRE | FIRE | FIRE — critical signal if candidate disengages mid-interview | N/A |
| 94 | TEMPORAL-04 | Objection Formation | FIRE (cap 0.60, only validated sequence) | RENAME: **"concern_formation"** | RENAME: **"disagreement_formation"** | RENAME: **"hesitation_formation"** | N/A |
| 95 | TEMPORAL-05 | Trust Repair | FIRE (cap 0.50-0.55, EXPERIMENTAL) | FIRE | FIRE | REINTERPRET: interviewer rebuilding rapport after tough question = good interviewer technique (positive) | N/A |
| 96 | TEMPORAL-06 | Decision Engagement (renamed) | FIRE (cap 0.45-0.50) | RENAME: **"commitment_sequence"** | GATE — no buying/decision progression in peer meetings | RENAME: **"decision_sequence"** (candidate deciding to accept/reject) | GATE |
| 97 | TEMPORAL-07 | Dominance Shift | FIRE (cap 0.55) | FIRE | FIRE | REINTERPRET: shift from interviewer to candidate = candidate gaining confidence (positive trajectory) | FIRE |
| — | TEMPORAL-08 | Behavioral Consistency (renamed) | FIRE (cap 0.55) | FIRE | FIRE | FIRE | FIRE |

---

## Summary: Content-Type Change Count

| Change Type | Sales Call | Client Meeting | Internal | Interview | Podcast |
|-------------|-----------|----------------|----------|-----------|---------|
| Rules with NO change | 97 (baseline) | 87 | 84 | 75 | 69 + N/A for video |
| ADAPT (detection threshold) | 1 (TALK-01) | 0 | 3 (FILLER-02, SENT-02, TALK-01) | 3 (PAUSE-01, TALK-01, BAL-01) | 2 (TALK-01, BAL-01) |
| RENAME (same detection, new label) | 0 | 5 (BUY, OBJ, PERS, FUSION-05, TEMPORAL-04/06) | 4 (OBJ, PERS, COMPOUND-04, TEMPORAL-04) | 6 (BUY, OBJ, PERS, FUSION-05, COMPOUND-04, TEMPORAL-04/06) | 0 |
| REINTERPRET (narrative only) | 0 | 0 | 2 (NEG-01, COMPOUND-04) | 7 (NEG-01, FUSION-04/10/12, COMPOUND-05/09, TEMPORAL-05/07) | 0 |
| GATE (suppress) | 0 | 1 (QUES-01 SPIN) | 3 (FUSION-13, BUY-01, TEMPORAL-06) | 3 (FUSION-13, QUES-01 SPIN, COMPOUND-12) | 8 (BUY, OBJ, PERS, NEG-01, CONF-01, FUSION-05/13, TEMPORAL-06) |
| DISABLED (technical) | 2 (MICRO-01, F-06) | 2 | 2 | 2 | 2 |
| Prompt switch | 0 | 1 (INTENT-01) | 1 (INTENT-01) | 1 (INTENT-01) | 1 (INTENT-01) |

**Key takeaway:** Sales calls need almost zero changes (1 threshold + 2 disabled). Interview needs the most interpretation changes (7 reinterpretations + 6 renames) but only 3 detection threshold changes. The per-speaker baseline does the heavy lifting.

---

## Architecture for Content-Type Adaptation

```
ContentTypeProfile class — loaded once per session

Methods:
  is_gated(rule_id) → bool          # Should this rule be suppressed?
  rename_signal(signal_type) → str   # Different label for this type?
  get_prompt_template(agent) → str   # Different LLM prompt?
  get_threshold(rule, param, default) → value  # Override detection threshold?
  get_narrative_note(rule_id) → str  # Interpretation guidance for narrative?

Only 5 rules use get_threshold() — the rest use is_gated(), rename_signal(), 
or get_narrative_note() which operate at the interpretation layer.
```

**Implementation: 1 new file (~250 lines), ~10 modified files (add `if profile.is_gated(...)` checks). No new LLM calls. No new rules. No detection logic changes except for the 5 threshold rules.**