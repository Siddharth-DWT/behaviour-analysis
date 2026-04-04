# NEXUS Content-Type Adaptation Plan — Complete 94-Rule Edition

## Summary

94 rules. 5 content types. 0 new rules. 0 extra LLM calls.
Every rule gets a content-type profile that adjusts thresholds, gates, renames, or reinterprets.

```
Phase 1 (NOW):     42 rules implemented → adapt all 42          ~6 days
Phase 2 (Video):   22 rules (Facial/Body/Gaze) → build + adapt  ~8 weeks  
Phase 2+ (Fusion): 12 remaining Fusion pairwise → build + adapt ~2 weeks
Phase 3 (Patterns):20 rules (Compound/Temporal) → build + adapt  ~4 weeks
```

---

## Architecture: ContentTypeProfile

Single class, loaded per session. Every rule calls it.

```python
profile = ContentTypeProfile("interview")
profile.is_gated("LANG-BUY-01")          # True for internal/podcast
profile.get_confidence_multiplier("VOICE-TONE-03")  # 0.6 for interview
profile.get_threshold("VOICE-FILLER-01", "spike_delta", 0.50)  # 1.0 for interview
profile.rename_signal("buying_signal")     # "candidate_interest_signal"
profile.get_prompt_template("language_intent")  # Interview-specific prompt
```

---

## PHASE 1: Current 42 Rules (Implement NOW — 6 days)

### Voice Agent — 16 rules

| # | Rule ID | Signal | sales_call | client_meeting | internal | interview | podcast | Adapt Type |
|---|---------|--------|-----------|----------------|----------|-----------|---------|------------|
| 1 | VOICE-CAL-01 | speaker_baseline | FIRE | FIRE | FIRE | FIRE | FIRE | None — universal foundation |
| 2 | VOICE-STRESS-01 | vocal_stress_score | FIRE | FIRE | FIRE | ADAPT: add +0.15 offset (interview stress is baseline) | FIRE | Threshold |
| 3 | VOICE-FILLER-01 | filler_detection | FIRE: spike >+50% | FIRE | ADAPT: spike >+75% | ADAPT: spike >+100% | ADAPT: spike >+30% | Threshold per type |
| 4 | VOICE-FILLER-02 | filler_credibility | FIRE: noticeable >1.3% | FIRE | ADAPT: noticeable >2.0% | GATE | FIRE | Threshold + Gate |
| 5 | VOICE-PITCH-01 | pitch_elevation_flag | FIRE: mild >+8% | FIRE | FIRE | ADAPT: mild >+12% | FIRE | Threshold |
| 6 | VOICE-PITCH-02 | monotone_flag | FIRE: var <-40% | FIRE | FIRE | ADAPT: var <-50% | FIRE | Threshold |
| 7 | VOICE-RATE-01 | speech_rate_anomaly | FIRE: >±25% | FIRE | FIRE | ADAPT: >±35% | FIRE | Threshold |
| 8 | VOICE-TONE-01 | warm | FIRE | FIRE | FIRE | FIRE | FIRE | None |
| 9 | VOICE-TONE-02 | cold | FIRE | FIRE | FIRE | REINTERPRET: "low_energy" | FIRE | Rename |
| 10 | VOICE-TONE-03 | nervous | FIRE | FIRE | FIRE | ADAPT: conf ×0.6 | FIRE | Confidence |
| 11 | VOICE-TONE-04 | confident | FIRE | FIRE | FIRE | ADAPT: conf ×1.2 | FIRE | Confidence |
| 12 | VOICE-TONE-05 | aggressive | FIRE | FIRE | FIRE | RENAME: "assertive" | RENAME: "assertive" | Rename |
| 13 | VOICE-TONE-06 | excited | FIRE | FIRE | FIRE | FIRE | REINTERPRET: positive | Reinterpret |
| 14 | VOICE-ENERGY-01 | energy_level | FIRE | FIRE | FIRE | FIRE | REINTERPRET: elevated=positive | Reinterpret |
| 15 | VOICE-PAUSE-01 | pause_classification | FIRE: hesitation >2000ms | FIRE | FIRE | ADAPT: hesitation >3000ms | FIRE | Threshold |
| 16 | VOICE-INT-01 | interruption_event | FIRE: overlap >200ms | FIRE | FIRE | REINTERPRET: interviewer=topic_control | ADAPT: overlap >400ms | Threshold + Rename |
| — | VOICE-TALK-01 | talk_time_ratio | FIRE: seller <65% optimal | FIRE | FIRE: peer >50% flagged | ADAPT: candidate 30-70% expected | ADAPT: guest 60-80% expected | Threshold |

### Language Agent — 12 rules (COMPLETE)

| # | Rule ID | Signal | sales_call | client_meeting | internal | interview | podcast | Adapt Type |
|---|---------|--------|-----------|----------------|----------|-----------|---------|------------|
| 17 | LANG-SENT-01 | sentiment_score | FIRE | FIRE | FIRE | FIRE | FIRE | None |
| 18 | LANG-SENT-02 | emotional_intensity | FIRE | FIRE | FIRE | FIRE | FIRE | None |
| 19 | LANG-BUY-01 | buying_signal | FIRE | RENAME: "client_engagement" | GATE | RENAME: "candidate_interest" | GATE | Gate + Rename |
| 20 | LANG-OBJ-01 | objection_signal | FIRE | RENAME: "client_concern" | RENAME: "concern_raised" | RENAME: "candidate_hesitation" | GATE | Gate + Rename |
| 21 | LANG-PWR-01 | power_language_score | FIRE | FIRE | FIRE | REINTERPRET: powerful=competent | FIRE | Reinterpret |
| 22 | LANG-PERS-01 | persuasion_technique | FIRE | ADAPT: conf ×0.7 | GATE | GATE | GATE | Gate + Confidence |
| 23 | LANG-QUES-01 | question_type | FIRE (SPIN) | FIRE (no SPIN) | FIRE (facilitation) | ADAPT: behavioral/situational/technical taxonomy | FIRE | Prompt switch |
| 24 | LANG-TOPIC-01 | topic_shift | FIRE | FIRE | FIRE | FIRE | FIRE | None |
| 25 | LANG-NEG-01 | gottman_horsemen | FIRE | FIRE | FIRE | ADAPT: conf ×0.7 | GATE | Gate + Confidence |
| 26 | LANG-EMP-01 | empathy_language | FIRE | FIRE | FIRE | REINTERPRET: soft_skill_signal | FIRE | Reinterpret |
| 27 | LANG-CLAR-01 | clarity_score | FIRE | FIRE | FIRE | ADAPT: conf ×1.2 | ADAPT: conf ×1.2 | Confidence |
| 28 | LANG-INTENT-01 | intent_classification | FIRE (sales prompt) | FIRE (meeting prompt) | FIRE (meeting prompt) | ADAPT: interview prompt | ADAPT: podcast prompt | Prompt switch |

### Conversation Agent — 8 rules

| # | Rule ID | Signal | sales_call | client_meeting | internal | interview | podcast | Adapt Type |
|---|---------|--------|-----------|----------------|----------|-----------|---------|------------|
| 29 | CONVO-TURN-01 | turn_taking_pattern | FIRE: mono <2/min | FIRE | FIRE | ADAPT: mono <1/min | ADAPT: mono <0.5/min | Threshold |
| 30 | CONVO-LAT-01 | response_latency | FIRE: delayed >1500ms | FIRE | FIRE | ADAPT: delayed >2500ms | ADAPT: delayed >3000ms | Threshold |
| 31 | CONVO-DOM-01 | dominance_score | FIRE: seller >65% flag | FIRE: one side >70% | FIRE: peer >50% | ADAPT: candidate 30-70% expected | ADAPT: host <50% expected | Threshold |
| 32 | CONVO-INT-01 | interruption_pattern | FIRE: >3/min | FIRE | FIRE | ADAPT: interviewer exempt, candidate >2/min | ADAPT: >5/min | Threshold |
| 33 | CONVO-RAP-01 | rapport_indicator | FIRE | FIRE | FIRE | FIRE | FIRE | None |
| 34 | CONVO-ENG-01 | conversation_engagement | FIRE | FIRE | FIRE | FIRE | FIRE | None |
| 35 | CONVO-BAL-01 | conversation_balance | FIRE: gini 0.15-0.30 normal for sales | FIRE | FIRE | ADAPT: gini 0.20-0.40 expected | ADAPT: gini 0.30-0.50 expected | Threshold |
| 36 | CONVO-CONF-01 | conflict_score | FIRE: min 2 indicators | FIRE | FIRE | ADAPT: min 3 indicators | GATE | Gate + Threshold |

### Fusion Agent — 6 rules (3 pairwise + 3 graph)

| # | Rule ID | Signal | sales_call | client_meeting | internal | interview | podcast | Adapt Type |
|---|---------|--------|-----------|----------------|----------|-----------|---------|------------|
| 37 | FUSION-02 | credibility_assessment | FIRE (cap 0.55) | FIRE (cap 0.55) | ADAPT: cap 0.45, stress gate 0.50 | ADAPT: cap 0.40 | GATE | Gate + Cap |
| 38 | FUSION-07 | verbal_incongruence | FIRE (cap 0.70) | FIRE (cap 0.60) | ADAPT: floor 0.35 | ADAPT: floor 0.35, cap 0.50 | GATE | Gate + Cap |
| 39 | FUSION-13 | urgency_authenticity | FIRE (cap 0.60) | ADAPT: only on pitch segments | GATE | GATE | GATE | Gate |
| 40 | FUSION-GRAPH-01 | tension_cluster | FIRE | FIRE | ADAPT: negative-only, min 3 | ADAPT: negative-only, min 4 | ADAPT: negative-only | Threshold |
| 41 | FUSION-GRAPH-02 | momentum_shift | FIRE | FIRE | FIRE | FIRE | FIRE | None |
| 42 | FUSION-GRAPH-03 | persistent_incongruence | FIRE | FIRE | ADAPT: threshold=3+dur/5+spk-2 | ADAPT: threshold +3 bonus | GATE | Gate + Threshold |

**Phase 1 Total: 42 rules, ~6 days implementation**

---

## PHASE 2: Facial Agent — 7 rules (Build in Weeks 7-8)

When building each rule, implement the content-type profile check from day 1.

| # | Rule ID | Signal | sales_call | client_meeting | internal | interview | podcast | Research Basis |
|---|---------|--------|-----------|----------------|----------|-----------|---------|---------------|
| 43 | FACE-CAL-01 | facial_baseline | FIRE | FIRE | FIRE | FIRE | N/A (audio podcast) | Per-speaker neutral face template |
| 44 | FACE-EMO-01 | primary_emotion (7-class) | FIRE | FIRE | FIRE | ADAPT: conf ×1.2 for candidate emotions (key diagnostic) | N/A | Ekman 1971, DeepFace, AffectNet. Base conf 0.55 |
| 45 | FACE-SMILE-01 | smile_type (Duchenne vs non) | FIRE | FIRE | FIRE | REINTERPRET: social smile from candidate = politeness (normal), from interviewer = encouragement (positive) | N/A | Ekman 1990 AU6+AU12. Base conf 0.60 |
| 46 | FACE-MICRO-01 | micro_expression | FIRE (cap 0.30) | FIRE (cap 0.30) | GATE (too subtle for meeting context) | ADAPT: cap 0.25 (interview stress confounds) | N/A | Frank 1997. EXPERIMENTAL, disable <10fps |
| 47 | FACE-ENG-01 | engagement_visual | FIRE | FIRE | FIRE | REINTERPRET: high candidate engagement = positive competency signal | N/A | Whitehill 2014. Base conf 0.50 |
| 48 | FACE-STRESS-01 | stress_visual (AU4+AU23+AU24) | FIRE | FIRE | FIRE | ADAPT: conf ×0.7 (interview facial stress is expected) | N/A | Ekman FACS. Base conf 0.45 |
| 49 | FACE-VA-01 | valence + arousal | FIRE | FIRE | FIRE | FIRE | N/A | Russell 1980 circumplex. Base conf 0.40-0.60 |

**Podcast note:** All facial rules return N/A for audio-only podcasts. If video podcast, treat as `interview` profile for host/guest dynamics.

### Content-Type Specifics for Facial:

**Interview adaptation is the biggest change:**
- Candidate's facial stress is EXPECTED and less diagnostic (same research finding as voice — Bogaard 2025)
- Interviewer's facial engagement/smile provides feedback quality signal
- Social (non-Duchenne) smiles from candidates = professional politeness, not masking
- Micro-expressions are less reliable because baseline arousal is elevated

**Internal meeting:**
- Micro-expressions gated — too subtle and too many false positives in casual group setting
- Stress visual is useful during specific agenda items (performance reviews, restructuring)
- Engagement visual is the PRIMARY facial signal for meetings

---

## PHASE 2: Body Agent — 8 rules (Build in Weeks 11-12)

| # | Rule ID | Signal | sales_call | client_meeting | internal | interview | podcast | Research Basis |
|---|---------|--------|-----------|----------------|----------|-----------|---------|---------------|
| 50 | BODY-CAL-01 | body_baseline | FIRE | FIRE | FIRE | FIRE | N/A | Navarro 2008 3 C's framework |
| 51 | BODY-POST-01 | posture_score + body_openness | FIRE | FIRE | FIRE | ADAPT: conf ×1.2 (posture is key interview signal, Carney 2010) | N/A | Mehrabian 1968. Base conf 0.40-0.55 |
| 52 | BODY-HEAD-01 | head_nod + head_shake | FIRE | FIRE | FIRE | FIRE — highest-value body signal across ALL types | N/A | McClave 2000. Base conf 0.55-0.75 |
| 53 | BODY-LEAN-01 | leaning_direction | FIRE | FIRE | FIRE | REINTERPRET: forward lean from candidate = interest (positive); backward = discomfort (note, not flag) | N/A | Mehrabian 1969. Base conf 0.45 |
| 54 | BODY-GEST-01 | gesture_type + hand_visible | FIRE | FIRE | FIRE | ADAPT: hand_concealment conf ×1.3 (hidden hands in interview = anxiety signal) | N/A | McNeill 1992. Base conf 0.45-0.80 |
| 55 | BODY-FIDG-01 | fidget_rate + movement_energy | FIRE | FIRE | FIRE | ADAPT: conf ×0.7 (interview fidgeting is expected, Navarro) | N/A | Mehrabian 1968. Base conf 0.50-0.55 |
| 56 | BODY-TOUCH-01 | self_touch + pacifying_type | FIRE | FIRE | FIRE | ADAPT: conf ×0.6 (self-soothing expected in interviews) | N/A | Ekman & Friesen 1969, Navarro. Base conf 0.35-0.50 |
| 57 | BODY-MIRROR-01 | body_mirroring_score | FIRE | FIRE | FIRE | REINTERPRET: candidate mirroring interviewer = rapport building (positive) | N/A | Chartrand & Bargh 1999. Base conf 0.40 |

**Interview is the critical type for Body Agent:**
- Posture and gesture are MORE diagnostic in interviews (Carney 2010 power posing)
- But fidgeting and self-touch are LESS diagnostic (everyone fidgets in interviews)
- Head nod/shake is universal — highest value signal regardless of type
- Mirroring is a positive rapport signal, not a deception signal

---

## PHASE 2: Gaze Agent — 7 rules (Build in Weeks 9-10)

| # | Rule ID | Signal | sales_call | client_meeting | internal | interview | podcast | Research Basis |
|---|---------|--------|-----------|----------------|----------|-----------|---------|---------------|
| 58 | GAZE-CAL-01 | gaze_baseline | FIRE | FIRE | FIRE | FIRE | N/A | Camera position calibration |
| 59 | GAZE-DIR-01 | gaze_direction | FIRE | FIRE | FIRE | FIRE | N/A | Argyle 1965. Base conf 0.45-0.65 |
| 60 | GAZE-CONTACT-01 | screen_engagement_pct | FIRE: speaking 40-60%, listening 60-75% | FIRE | FIRE | ADAPT: candidate speaking 30-50% OK (thinking/recalling reduces gaze) | N/A | Argyle 1972. Base conf 0.50 |
| 61 | GAZE-BLINK-01 | blink_rate_bpm | FIRE: stress >25 bpm | FIRE | FIRE | ADAPT: stress threshold >30 bpm (interview blink rate elevated, Bentivoglio 1997) | N/A | Base conf 0.45-0.80 |
| 62 | GAZE-ATT-01 | attention_score | FIRE | FIRE | FIRE | FIRE | N/A | Composite. Base conf 0.55 |
| 63 | GAZE-DIST-01 | distraction_count | FIRE: sustained break >3s | FIRE | ADAPT: break >5s (meeting multi-tasking is common) | ADAPT: break >4s | N/A | Argyle 1972. Base conf 0.45-0.65 |
| 64 | GAZE-SYNC-01 | gaze_synchrony | FIRE | FIRE | FIRE | FIRE | N/A | Wohltjen 2021 PNAS. Base conf 0.40 |

**Internal meeting adaptation:**
- Distraction threshold raised because meeting participants check laptops, take notes
- Screen engagement norms are looser (multi-person means less direct eye contact per person)

**Interview adaptation:**
- Candidate gaze during recall (looking away while thinking) is normal cognitive behavior, not avoidance
- Blink rate baseline is elevated — raise stress threshold from 25 to 30 bpm
- Gaze synchrony between interviewer-candidate is a strong rapport predictor

---

## PHASE 2+: Remaining 12 Fusion Pairwise Rules (Build in Weeks 13-14)

These require Facial, Body, or Gaze signals. Build with profiles from day 1.

| # | Rule ID | Cross-Modal Check | sales_call | client_meeting | internal | interview | podcast |
|---|---------|-------------------|-----------|----------------|----------|-----------|---------|
| 65 | FUSION-01 | Voice Tone × Facial Expression → Masking | FIRE (cap 0.75) | FIRE | FIRE | ADAPT: cap 0.55 (interview masking is normal social behavior) | N/A |
| 66 | FUSION-03 | Body Posture × Voice Energy → Manufactured Enthusiasm | FIRE (cap 0.65) | FIRE | GATE (enthusiasm irrelevant in meetings) | GATE (enthusiasm detection inappropriate for interviews) | N/A |
| 67 | FUSION-04 | Gaze Break × Filler Words → Uncertainty | FIRE (cap 0.70) | FIRE | FIRE | ADAPT: cap 0.50 (gaze+filler combo is common in interview recall) | N/A |
| 68 | FUSION-05 | Buying Signal × Body Language → Purchase Intent | FIRE (cap 0.70) | RENAME: "client_intent_verification" | GATE | RENAME: "candidate_intent_verification" | GATE |
| 69 | FUSION-06 | Micro-Expression × Language → Emotional Leakage | FIRE (cap 0.35) | FIRE (cap 0.35) | GATE (too speculative for meetings) | GATE (too speculative for interviews) | N/A |
| 70 | FUSION-07* | Head Nod × Speech Content → Unconscious Disagreement | FIRE (cap 0.65) | FIRE | FIRE | FIRE — universal, highest-value fusion signal | N/A |
| 71 | FUSION-08 | Eye Contact × Hedged Language → False Confidence | FIRE (cap 0.55) | FIRE | ADAPT: cap 0.40 | ADAPT: cap 0.40 (hedging + gaze break is normal interview nervousness) | N/A |
| 72 | FUSION-09 | Smile × Sentiment → Sarcasm/Masking | FIRE (cap 0.60) | FIRE | FIRE | ADAPT: cap 0.45 (social smiling during difficult answers is normal) | N/A |
| 73 | FUSION-10 | Response Latency × Facial Stress → Cognitive Load | FIRE (cap 0.60) | FIRE | FIRE | REINTERPRET: cognitive_load = thinking (neutral), not withholding (negative) | N/A |
| 74 | FUSION-11 | Talk Dominance × Gaze Submission → Anxious Dominance | FIRE (cap 0.65) | FIRE | FIRE | ADAPT: cap 0.45 (candidate can be dominant speaker but avoid eye contact from nerves) | N/A |
| 75 | FUSION-12 | Interruption × Body Position → Interruption Intent | FIRE (cap 0.55) | FIRE | FIRE | REINTERPRET: interviewer forward+interrupt = engagement (positive) | N/A |
| 76 | FUSION-14 | Empathy Language × Head Nod → Rapport Confirmation | FIRE (cap 0.70) | FIRE | FIRE | FIRE | N/A |
| 77 | FUSION-15 | Filler Words × Gaze Aversion → Compound Uncertainty | FIRE (cap 0.55) | FIRE | FIRE | ADAPT: cap 0.35 (filler+gaze combo expected in interviews) | N/A |

*Note: Current FUSION-07 in code is "verbal_incongruence" (power+sentiment). The DOCX spec FUSION-07 is "Head Nod × Speech Content". The head nod version requires Body Agent (Phase 2). The current verbal version stays as FUSION-07-AUDIO.*

**Key insight for Phase 2+ Fusion:**
- 5 rules GATED for interview (FUSION-03, 06, and podcast N/A for all video rules)
- 6 rules get confidence caps LOWERED for interview (masking, uncertainty, false confidence are all expected interview behaviors)
- FUSION-07 (head nod × speech) and FUSION-14 (empathy × head nod) are UNIVERSAL — they work equally well across all types
- All video Fusion rules return N/A for audio-only podcasts

---

## PHASE 3: Compound Patterns — 12 rules (Build in Weeks 15-16)

Multi-domain states requiring 3+ agents. Each gets content-type adaptation.

| # | Rule ID | Pattern Name | Domains | sales_call | client_meeting | internal | interview | podcast |
|---|---------|-------------|---------|-----------|----------------|----------|-----------|---------|
| 78 | COMPOUND-01 | Genuine Engagement | Voice+Face+Body+Gaze+Language+Convo | FIRE | FIRE | FIRE | FIRE — primary positive signal for candidate assessment | N/A (audio: use audio-only subset) |
| 79 | COMPOUND-02 | Active Disengagement | Voice+Face+Body+Gaze+Language+Convo | FIRE | FIRE | FIRE | REINTERPRET: candidate disengagement = lost interest in role (critical flag) | N/A |
| 80 | COMPOUND-03 | Emotional Suppression | Voice+Face+Body (mixed channels) | FIRE | FIRE | FIRE | ADAPT: conf ×0.7 (professional emotional control is expected in interviews) | N/A |
| 81 | COMPOUND-04 | Decision Readiness | 4 of 6 domains | FIRE — primary sales outcome signal (GREEN alert) | RENAME: "client_ready_to_commit" | RENAME: "consensus_reached" | RENAME: "candidate_decided" (positive = wants the job) | N/A |
| 82 | COMPOUND-05 | Cognitive Overload | Voice+Face+Gaze | FIRE | FIRE | FIRE | REINTERPRET: overload on technical questions = difficulty level signal, not candidate weakness | N/A |
| 83 | COMPOUND-06 | Conflict Escalation | 4 domains, both speakers | FIRE (RED alert) | FIRE | FIRE | ADAPT: conf ×0.8, only flag if ≥3 Gottman horsemen (interview tension ≠ conflict) | N/A |
| 84 | COMPOUND-07 | Silent Resistance | Voice+Face+Body (3 domains) | FIRE — "yes that means no" (ORANGE alert) | FIRE | ADAPT: rename "unspoken_disagreement" | ADAPT: rename "reserved_response" (candidate may be cautious, not resistant) | N/A |
| 85 | COMPOUND-08 | Rapport Peak | 4 domains, both speakers | FIRE (GREEN alert) | FIRE | FIRE | FIRE — indicates strong interviewer-candidate chemistry | N/A |
| 86 | COMPOUND-09 | Topic Avoidance | Voice+Face+Body | FIRE | FIRE | FIRE | REINTERPRET: candidate avoiding topic = possible weakness area (coaching note, not red flag) | N/A |
| 87 | COMPOUND-10 | Authentic Confidence | 4 domains | FIRE | FIRE | FIRE | FIRE — candidate assessment metric, high value | N/A |
| 88 | COMPOUND-11 | Anxiety Performance | Voice+Face+Body (controlled vs leaked) | FIRE | FIRE | GATE (anxiety masking in meetings is normal) | FIRE — the CORE interview diagnostic (anxious but performing well = coachable; anxious and failing = concern) | N/A |
| 89 | COMPOUND-12 | Deception Risk | 4 domains + 3 Fusion conflicts | FIRE (cap 0.55, ORANGE) | FIRE (cap 0.55) | ADAPT: cap 0.45 | GATE (deception detection in interviews is ethically problematic + high false positive rate) | N/A |

### Compound Patterns — Interview-Specific Interpretations:

The interview profile fundamentally shifts what these patterns MEAN:
- **Genuine Engagement** = candidate wants the role (strongest positive signal)
- **Active Disengagement** = candidate has lost interest (strongest negative signal)
- **Silent Resistance** becomes **Reserved Response** — cautious ≠ resistant
- **Anxiety Performance** is the SIGNATURE interview pattern — detecting someone who is nervous but performing well is the most valuable interview insight
- **Deception Risk** is GATED for interviews — the research (DePaulo 2003, Bogaard 2025) shows that stress/arousal in high-stakes interviews is indistinguishable from deception signals. Applying deception detection to interviews would produce unacceptable false positive rates

---

## PHASE 3: Temporal Sequences — 8 rules (Build in Weeks 15-16)

Multi-step patterns that unfold over time. Each gets content-type adaptation.

| # | Rule ID | Sequence Name | Time Scale | sales_call | client_meeting | internal | interview | podcast |
|---|---------|-------------|-----------|-----------|----------------|----------|-----------|---------|
| 90 | TEMPORAL-01 | Stress Cascade | 2-15s per step | FIRE | FIRE | FIRE | ADAPT: require 4 steps not 3 (interview stress cascades naturally across difficult questions) | N/A |
| 91 | TEMPORAL-02 | Engagement Build | 1-3 min | FIRE | FIRE | FIRE | FIRE — tracks candidate warming up over first 5 minutes | N/A |
| 92 | TEMPORAL-03 | Disengage Cascade | 30-120s | FIRE | FIRE | FIRE | FIRE — critical signal if candidate disengages mid-interview | N/A |
| 93 | TEMPORAL-04 | Objection Formation | 5-30s | FIRE — shows objection building before spoken | RENAME: "concern_formation" | RENAME: "disagreement_formation" | RENAME: "hesitation_formation" | N/A |
| 94 | TEMPORAL-05 | Trust Repair | 10-90s | FIRE | FIRE | FIRE | REINTERPRET: interviewer rebuilding rapport after tough question = good technique | N/A |
| 95 | TEMPORAL-06 | Buying Decision Sequence | 5-30 min | FIRE — tracks deal progression stages | RENAME: "commitment_sequence" | GATE (no buying in meetings) | RENAME: "decision_sequence" (candidate deciding to accept/reject) | GATE |
| 96 | TEMPORAL-07 | Dominance Shift | 30s-5 min | FIRE | FIRE | FIRE | REINTERPRET: shift from interviewer to candidate = candidate gaining confidence (positive trajectory) | FIRE |
| 97 | TEMPORAL-08 | Authenticity Erosion | 15-60 min | FIRE | FIRE | FIRE | ADAPT: require longer window (45-90 min for interview fatigue) | GATE |

### Temporal — Interview-Specific Value:

The most valuable temporal patterns for interviews are:
1. **Engagement Build** (TEMPORAL-02): Does the candidate warm up? Slow start → strong finish = good
2. **Disengage Cascade** (TEMPORAL-03): Candidate checking out = role mismatch
3. **Dominance Shift** (TEMPORAL-07): Candidate going from passive → active = confidence growing
4. **Stress Cascade** (TEMPORAL-01): Multiple simultaneous stress signals building = question difficulty
5. **Decision Sequence** (TEMPORAL-06 renamed): Candidate moving toward acceptance/rejection

---

## Complete Adaptation Statistics

### By Adaptation Type

| Adaptation Type | Phase 1 (42 rules) | Phase 2 Video (22) | Phase 2+ Fusion (12) | Phase 3 Patterns (20) | Total (97*) |
|----------------|--------------------|--------------------|----------------------|-----------------------|-------------|
| FIRE (no change) | 18 | 8 | 3 | 6 | 35 (36%) |
| Threshold adjust | 16 | 5 | 0 | 2 | 23 (24%) |
| Confidence multiplier | 6 | 4 | 0 | 3 | 13 (13%) |
| Gate (suppress) | 8 | 5 | 4 | 4 | 21 (22%) |
| Rename/Reinterpret | 10 | 5 | 5 | 6 | 26 (27%) |
| Prompt switch | 2 | 0 | 0 | 0 | 2 (2%) |

*97 = 94 spec + 3 graph-based fusion rules added beyond spec

Note: Rules can have multiple adaptation types (e.g., threshold + rename).

### By Content Type — What Gets Gated

| Content Type | Rules Gated (of 97) | Rules with Threshold Changes | Rules Renamed | Rules Unchanged |
|-------------|---------------------|-----------------------------|--------------|-----------------| 
| sales_call | 0 | 0 | 0 | 97 (100%) — baseline |
| client_meeting | 3 | 8 | 5 | 81 (83%) |
| internal | 8 | 12 | 4 | 73 (75%) |
| interview | 12 | 22 | 11 | 52 (54%) — most adapted |
| podcast | 16 + all video N/A | 8 | 4 | 69 (71%) |

Interview requires the most adaptation because it has the most different conversational dynamics from sales calls.

### By Content Type — Signal Interpretation Shifts

| Signal Pattern | sales_call | client_meeting | internal | interview | podcast |
|---------------|-----------|----------------|----------|-----------|---------|
| Stress + positive words | Credibility concern | Credibility concern | Normal multi-topic | Expected interview nerves | N/A |
| High dominance | Seller warning | Presenter expected | Peer dominance flag | Interviewer expected | Guest expected |
| Interruptions | Competitive | Competitive | Competitive | Topic control (interviewer) | Conversational overlap |
| Fast speech + energy | Urgency check | Enthusiasm | Engagement | Enthusiasm | Engaging host |
| Fidgeting + self-touch | Anxiety/deception | Discomfort | Normal | Expected nerves | N/A |
| Head shake + "yes" | Unconscious disagreement | Same | Same | Same (universal) | N/A |
| Low engagement | Prospect lost | Client bored | Participant checked out | Candidate not interested | Listener dropped off |
| Filler spike | Uncertainty | Uncertainty | Normal variation | Cognitive load (thinking) | Needs coaching |

---

## Implementation Timeline

```
Week 1 (Phase 1A):  ContentTypeProfile class + DB table + pipeline wiring
Week 1 (Phase 1B):  Voice Agent 16 rules → profile-aware
Week 1 (Phase 1C):  Language Agent 12 rules → profile-aware + prompt templates  
Week 2 (Phase 1D):  Conversation Agent 8 rules → profile-aware
Week 2 (Phase 1E):  Fusion Agent 6 rules → profile-aware (formalize existing if/else)
Week 2 (Phase 1F):  Dashboard content-type sections + signal renaming
Week 2 (Phase 1G):  Test all 5 types with real recordings, tune thresholds

Weeks 7-8:   Facial Agent (7 rules) — build with profiles from day 1
Weeks 9-10:  Gaze Agent (7 rules) — build with profiles from day 1
Weeks 11-12: Body Agent (8 rules) — build with profiles from day 1
Weeks 13-14: Remaining 12 Fusion pairwise — build with profiles
Weeks 15-16: 12 Compound + 8 Temporal — build with profiles

Total: 97 rules × 5 content types = 485 profile entries
       0 new rules created
       0 additional LLM calls
       5 prompt templates (swapped, not added)
       1 new code file (content_type_profile.py)
       ~10 modified files
```