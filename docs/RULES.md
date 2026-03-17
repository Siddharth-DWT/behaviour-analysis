# RULES.md — NEXUS Detection Rule Engine Reference

## Overview

94 detection rules across 7 agents. Each rule has a **Signal Definition Card** with:
- **Rule ID**: Unique identifier (e.g., VOICE-STRESS-01)
- **Signal Name**: Human-readable name
- **Research Basis**: Which study/paper supports this detection
- **Raw Features**: What data inputs the rule needs
- **Detection Logic**: IF-THEN with specific thresholds
- **Confidence Calculation**: How certainty is computed
- **Cross-Agent Validation**: Which other agents confirm/deny this signal

Full rule specifications are in the DOCX files under `docs/rule-engine-docs/`. This file is a quick reference and cross-reference guide.

---

## Signal Confidence Scale

All signals use a 0.0 to 1.0 confidence scale:

| Range | Meaning | Action |
|-------|---------|--------|
| 0.00 - 0.19 | Noise | Ignore — below detection threshold |
| 0.20 - 0.39 | Weak | Log internally — don't surface to user |
| 0.40 - 0.59 | Moderate | Show as subtle indicator in dashboard |
| 0.60 - 0.79 | Strong | Surface as notable moment / highlight |
| 0.80 - 0.85 | Very Strong | Flag as key insight (maximum for any signal) |

**Hard caps**: No single-domain signal exceeds 0.85. No deception-related signal exceeds 0.55. These caps are based on Bond & DePaulo (2006) and Levine (2014) showing that even trained professionals achieve only 54% accuracy at deception detection.

---

## Voice Agent — 18 Rules

### Core Rules (✅ Implemented in Code)

| Rule ID | Signal | Threshold | Confidence | Status |
|---------|--------|-----------|------------|--------|
| VOICE-CAL-01 | Speaker baseline calibration | First ~90s of speech, min 5 windows | Self-calibrating | ✅ Built |
| VOICE-STRESS-01 | Composite vocal stress | 7-weighted components (see below) | score × cal_conf | ✅ Built |
| VOICE-FILLER-01 | Filler word detection | Primary: um/uh/er; Secondary: like/you know | 0.90 × cal_conf | ✅ Built |
| VOICE-FILLER-02 | Filler credibility threshold | >1.3% noticeable, >2.5% significant, >4.0% severe | 0.85 | ✅ Built |
| VOICE-PITCH-01 | Pitch elevation flag | >+8% mild, >+15% significant, >+25% extreme | 0.50 × cal_conf | ✅ Built |
| VOICE-RATE-01 | Speech rate anomaly | >±25% from baseline | 0.40 × cal_conf | ✅ Built |
| VOICE-TONE-03 | Nervous/anxious tone | Multi-factor: F0↑ + variance↓ + rate↑ + jitter↑ | score × cal_conf | ✅ Built |
| VOICE-TONE-04 | Confident/authoritative | Multi-factor: F0↓ + variance↑ + energy↑ + jitter↓ | score × cal_conf | ✅ Built |

### Remaining Voice Rules (🔲 Documented, Not Yet Coded)

| Rule ID | Signal | Brief Description |
|---------|--------|-------------------|
| VOICE-TONE-01 | Warm/friendly tone | Low F0, wide range, moderate energy, smooth |
| VOICE-TONE-02 | Cold/distant tone | Flat F0, narrow range, low energy, monotone |
| VOICE-TONE-05 | Aggressive tone | High energy, fast rate, emphatic stress, low HNR |
| VOICE-TONE-06 | Excited/enthusiastic | High F0, wide range, fast rate, high energy |
| VOICE-VOL-01 | Volume shift | Significant energy change from baseline |
| VOICE-PAUSE-01 | Hesitation pause | >250ms within-utterance pause (Goldman-Eisler) |
| VOICE-PAUSE-02 | Strategic pause | >500ms pause BEFORE key content (emphasis tool) |
| VOICE-INT-01 | Interruption detection | Overlapping speech start during another's turn |
| VOICE-TALK-01 | Talk time imbalance | Per-speaker cumulative talk percentage |
| VOICE-PITCH-02 | Pitch drop (assertive) | F0 drops >10% below baseline at utterance end |

### VOICE-STRESS-01 Component Weights (Detailed)

```
Stress Score = Σ(weight × normalised_delta)

Component       Weight   Source                    Normalisation
─────────────────────────────────────────────────────────────────
F0 elevation    0.25     Streeter 1977             delta/0.30 (30% = max)
Jitter increase 0.20     Kappen 2022               delta/0.50
Rate change     0.15     Apple 1979                abs(delta)/0.40
Filler increase 0.15     Clark & Fox Tree 2002     delta/1.00
Pause increase  0.10     Goldman-Eisler 1968        delta/0.50
HNR decrease    0.10     Scherer 2003              inv(delta)/0.30
Shimmer increase 0.05    Kappen 2022               delta/0.50
```

---

## Language Agent — 12 Rules

| Rule ID | Signal | Method | Key Research |
|---------|--------|--------|-------------|
| LANG-SENT-01 | Per-sentence sentiment | DistilBERT + LIWC hybrid | Liu 2012, Pennebaker 2015 |
| LANG-EMO-01 | Emotional intensity | LIWC affect word density | Tausczik & Pennebaker 2010 |
| LANG-PERS-01 | Persuasion detection | Cialdini 7 principles via Claude API | Cialdini 2021 |
| LANG-BUY-01 | Buying signals | SPIN-derived patterns + Claude API | Rackham 1988 (35K calls) |
| LANG-OBJ-01 | Objection signals | Hedge counting + resistance patterns | Rackham 1988 |
| LANG-PWR-01 | Power language score | Powerless features counting | Lakoff 1975, O'Barr 1982 |
| LANG-QUES-01 | Question type classification | SPIN (Situation/Problem/Implication/Need) | Rackham 1988 |
| LANG-TOPIC-01 | Topic tracking | Semantic similarity between windows | Custom embedding |
| LANG-GOTT-01 | Gottman 4 Horsemen | Criticism/contempt/defensiveness/stonewalling | Gottman 1994 |
| LANG-EMP-01 | Empathy/rapport markers | Language Style Matching | Gonzales et al. 2010 |
| LANG-CLAR-01 | Clarity score | Sentence length + jargon ratio + passive voice | Flesch, Dale-Chall |
| LANG-INTENT-01 | Intent classification | Per-utterance via Claude API | Custom taxonomy |

### LANG-PWR-01 Powerless Features (Counted)

Lakoff (1975) and O'Barr & Atkins (1982) identified these markers of powerless speech:
- **Hedges**: "kind of", "sort of", "maybe", "perhaps", "I think", "I guess"
- **Tag questions**: "...right?", "...don't you think?", "...isn't it?"
- **Intensifiers**: "so", "very", "really" (over-emphasis signals uncertainty)
- **Hesitation forms**: "well", "you know", "I mean" (overlap with VOICE-FILLER-01)
- **Polite forms**: excessive "please", "if you don't mind", "I was wondering if"

Power score = 1.0 - (powerless_features / total_words × normalisation_factor)

### LANG-BUY-01 Buying Signal Patterns

From Rackham's SPIN Selling research (35,000 sales calls):
- **Future projection**: "When we implement...", "Once this is set up..."
- **Usage scenarios**: "How would this work with our...", "Could we use this for..."
- **Specification questions**: "What's the timeline?", "How many licenses?"
- **Positive reframing**: Repeating back benefits in own words
- **Social proof seeking**: "Who else uses this?", "Can I talk to a reference?"
- **Price/terms questions**: "What's the cost?", "Are there discounts?"
- **Implementation questions**: "How long to deploy?", "What's the onboarding?"

---

## Facial Agent — 7 Rules

| Rule ID | Signal | Method | Confidence Cap |
|---------|--------|--------|---------------|
| FACE-EMO-01 | 7-class emotion | DeepFace CNN + FACS AU validation | 0.75 |
| FACE-SMILE-01 | Duchenne vs non-Duchenne smile | AU6 (cheek) + AU12 (lip) co-occurrence | 0.70 |
| FACE-MICRO-01 | Micro-expression detection | Frame-diff + AU transient (EXPERIMENTAL) | 0.35 (disabled <10fps) |
| FACE-ENG-01 | Facial engagement composite | Head orientation + expression variability | 0.70 |
| FACE-STRESS-01 | Facial stress indicators | AU4 (brow) + AU23 (lip tightener) + AU24 (lip presser) | 0.65 |
| FACE-VA-01 | Valence-arousal continuous | AffectNet model → 2D coordinates | 0.60 |
| FACE-CAL-01 | Facial baseline calibration | First 60s: neutral face template + expression range | Self-calibrating |

### FACS Action Unit Reference (Key AUs Used)

| AU | Muscle | NEXUS Usage |
|----|--------|-------------|
| AU1 | Inner brow raise | Sadness, worry |
| AU2 | Outer brow raise | Surprise |
| AU4 | Brow lowerer | Anger, concentration, STRESS |
| AU5 | Upper lid raise | Fear, surprise |
| AU6 | Cheek raise | Duchenne smile (genuine) |
| AU7 | Lid tightener | Anger, determination |
| AU9 | Nose wrinkle | Disgust |
| AU12 | Lip corner pull | Smile (any type) |
| AU15 | Lip corner depress | Sadness |
| AU17 | Chin raise | Doubt, determination |
| AU20 | Lip stretch | Fear |
| AU23 | Lip tightener | STRESS, anger suppression |
| AU24 | Lip presser | STRESS, frustration |
| AU25 | Lips part | Surprise, speech |
| AU26 | Jaw drop | Surprise, shock |
| AU28 | Lip suck | Anxiety, self-soothing |

---

## Body Agent — 8 Rules

| Rule ID | Signal | Input | Key Research |
|---------|--------|-------|-------------|
| BODY-POST-01 | Posture / body openness | Shoulder width + arm position | Mehrabian 1969 |
| BODY-HEAD-01 | Head nod / shake detection | Pitch/yaw angular velocity | McClave 2000 |
| BODY-LEAN-01 | Forward / backward lean | Head size as distance proxy | Mehrabian 1968 |
| BODY-GEST-01 | Hand gesture classification | Hand visibility + movement type | McNeill 1992 |
| BODY-FIDG-01 | Fidget rate | High-freq low-amplitude movement | Harrigan et al. 1991 |
| BODY-TOUCH-01 | Self-touch / pacifying | Hand-face proximity detection | Navarro 2008 |
| BODY-MIRROR-01 | Body mirroring | Cross-speaker pose similarity | Chartrand & Bargh 1999 |
| BODY-CAL-01 | Body baseline calibration | First 60s: resting posture + movement range | Custom |

### BODY-HEAD-01 Incongruence Check

Head nod/shake is the **single most important body signal** for cross-modal validation because it operates unconsciously. The Fusion Agent uses this for FUSION-07 (Unconscious Disagreement):

```
IF head_nod.shake == TRUE
AND language_content.affirmative == TRUE
AND time_overlap < 2 seconds
THEN signal = "unconscious_disagreement"
     confidence = 0.70 × cal_conf
```

This is one of the highest-value fusion signals in the entire system.

### BODY-TOUCH-01 Stress Hierarchy (Navarro 2008)

| Self-Touch Location | Stress Level | Meaning |
|-------------------|-------------|---------|
| Neck touch / cover | Highest | Strongest pacifying (covers vulnerable carotid) |
| Face touch (cheek) | High | Self-soothing |
| Hair touch / play | Moderate-High | Anxiety displacement |
| Arm cross / self-hug | Moderate | Comfort-seeking barrier |
| Hand wringing | Moderate | Anxiety displacement |
| Object manipulation | Low-Moderate | Low-grade fidgeting |

---

## Gaze Agent — 7 Rules

| Rule ID | Signal | Method | Key Research |
|---------|--------|--------|-------------|
| GAZE-DIR-01 | Gaze direction | MediaPipe iris + head pose | Custom (redefined) |
| GAZE-CONTACT-01 | Screen engagement % | On-screen time / total time | Argyle 1972 |
| GAZE-BLINK-01 | Blink rate (BPM) | Eye Aspect Ratio threshold | Bentivoglio 1997 |
| GAZE-ATT-01 | Attention composite | Engagement + blink + stability | Custom weighted |
| GAZE-DIST-01 | Distraction events | Sustained gaze break >3s | Custom |
| GAZE-SYNC-01 | Mutual gaze synchrony | Cross-speaker gaze alignment | Wohltjen 2021 PNAS |
| GAZE-CAL-01 | Gaze baseline | Camera position estimation + resting gaze | Custom |

### Screen Engagement Norms (Argyle 1972, adapted for webcam)

| Context | Expected Screen Engagement |
|---------|--------------------------|
| While listening | 60-75% |
| While speaking | 40-60% |
| During rapport | 65-80% |
| During disagreement | 30-50% |
| While reading/notes | 10-30% |

Deviations from these norms (per speaking/listening state) trigger GAZE-CONTACT-01.

### Blink Rate Norms (Bentivoglio 1997)

| State | Normal BPM | Stress Indicator |
|-------|-----------|-----------------|
| Resting | 15-20 | >25 or <10 |
| Conversation | 20-26 | >35 or <12 |
| Reading | 3-5 | N/A |

---

## Conversation Agent — 7 Rules

| Rule ID | Signal | Method | Key Research |
|---------|--------|--------|-------------|
| CONVO-TURN-01 | Turn-taking dynamics | Sequential turn analysis | Sacks et al. 1974 |
| CONVO-LAT-01 | Response latency | Inter-speaker gap timing | Stivers 2009 |
| CONVO-RAP-01 | Multi-modal rapport | Tickle-Degnen 3 components | Tickle-Degnen 1990 |
| CONVO-CONF-01 | Multi-modal conflict | Gottman + all agent signals | Gottman 1994 |
| CONVO-DOM-01 | Dominance mapping | Multi-factor per speaker | Dunbar & Burgoon 2005 |
| CONVO-MUT-01 | Mutual engagement | MIN function across agents | Custom |
| CONVO-INT-01 | Interruption dynamics | Cooperative vs competitive | Tannen 1994 |

### CONVO-LAT-01 Response Latency Interpretation (Stivers 2009)

Stivers et al. (2009) found 200ms is the universal preferred response time across 10 languages:

| Latency | Interpretation |
|---------|---------------|
| 0-200ms | Preferred response (agreement, expected answer) |
| 200-500ms | Normal variation |
| 500-1000ms | Slight dispreference (thinking, mild disagreement) |
| 1000-2000ms | Clear dispreference (strong disagreement, bad news) |
| >2000ms | Significant dispreference (rejection, avoidance) |
| Negative (overlap) | Either high agreement or competitive interruption |

---

## Fusion Agent — 15 Pairwise Rules

| Rule ID | Pair | Signal Name | Max Confidence |
|---------|------|-------------|---------------|
| FUSION-01 | Voice × Face | Emotional masking | 0.75 |
| FUSION-02 | Content × Stress | Credibility assessment | 0.55 |
| FUSION-03 | Body × Voice energy | Manufactured enthusiasm | 0.65 |
| FUSION-04 | Gaze × Fillers | Uncertainty signal | 0.60 |
| FUSION-05 | Buying × Body | Purchase intent validation | 0.70 |
| FUSION-06 | Micro-expression × Stated | Emotional leakage | 0.35 |
| FUSION-07 | Head shake × Language | Unconscious disagreement | 0.70 |
| FUSION-08 | Gaze × Hedges | False confidence | 0.55 |
| FUSION-09 | Smile × Sentiment | Sarcasm / social masking | 0.60 |
| FUSION-10 | Latency × Facial | Cognitive load | 0.65 |
| FUSION-11 | Dominance × Gaze | Anxiety-driven dominance | 0.55 |
| FUSION-12 | Interruption × Body | Interruption intent | 0.60 |
| FUSION-13 | Persuasion × Pace | Urgency authenticity | 0.60 |
| FUSION-14 | Empathy × Nod | Rapport validation | 0.70 |
| FUSION-15 | Filler × Gaze | Sustained uncertainty | 0.65 |

---

## Compound Patterns — 12 Multi-Domain States

| ID | Pattern | Domains Required | Max Confidence |
|----|---------|-----------------|---------------|
| COMPOUND-01 | Genuine Engagement | Voice + Face + Body + Gaze | 0.80 |
| COMPOUND-02 | Active Disengagement | Voice + Face + Body + Gaze | 0.80 |
| COMPOUND-03 | Emotional Suppression | Voice + Face + Body | 0.70 |
| COMPOUND-04 | Decision Readiness | Language + Voice + Body + Gaze | 0.85 |
| COMPOUND-05 | Cognitive Overload | Voice + Face + Gaze + Convo | 0.75 |
| COMPOUND-06 | Conflict Escalation | Language + Voice + Face + Body + Convo | 0.75 |
| COMPOUND-07 | Silent Resistance | Language + Voice + Body + Face | 0.75 |
| COMPOUND-08 | Rapport Peak | Voice + Face + Body + Gaze + Convo | 0.80 |
| COMPOUND-09 | Topic Avoidance | Language + Voice + Gaze | 0.65 |
| COMPOUND-10 | Authentic Confidence | Voice + Face + Body + Language | 0.80 |
| COMPOUND-11 | Anxiety Performance | Voice + Face + Body | 0.70 |
| COMPOUND-12 | Deception Risk | All 6 domains | 0.55 (CAPPED) |

### COMPOUND-04 (Decision Readiness) — Highest Value for Sales

This is the most commercially valuable signal. It detects the moment a buyer shifts from evaluation to decision-making:

Required signals (must ALL be present within a 30-second window):
1. LANG-BUY-01 confidence > 0.40 (buying language active)
2. VOICE-TONE-04 or stress < 0.30 (calm/confident state)
3. BODY-LEAN-01 forward OR BODY-HEAD-01 nodding
4. GAZE-CONTACT-01 engagement > baseline

When all 4 fire together → Decision Readiness alert. This is the "close now" signal.

### COMPOUND-07 (Silent Resistance) — Detects "Yes" that Means "No"

Required signals (3+ within a 15-second window):
1. Language content is affirmative ("yes", "sure", "sounds good")
2. Voice shows stress elevation OR tone = nervous
3. Body shows one of: closed posture, backward lean, self-touch, head shake
4. Optional: Facial shows suppressed negative emotion

This pattern predicts deal collapse, project delays, and passive-aggressive follow-through.

---

## Temporal Sequences — 8 Cascade Patterns

| ID | Sequence | Duration | Stages | Early Warning |
|----|----------|----------|--------|--------------|
| TEMPORAL-01 | Stress Cascade | 2-15s | 4 stages | Stage 2 (voice change) |
| TEMPORAL-02 | Engagement Build | 1-3 min | 5 stages | N/A (positive) |
| TEMPORAL-03 | Disengagement Cascade | 30-120s | 4 stages | Stage 1 (gaze breaks) |
| TEMPORAL-04 | Objection Formation | 5-30s | 5 stages | Stage 2 (body tension) |
| TEMPORAL-05 | Trust Repair | 1-5 min | 4 stages | N/A (recovery) |
| TEMPORAL-06 | Buying Decision Sequence | 30s-5min | 5 stages | Stage 3 (narrowing) |
| TEMPORAL-07 | Dominance Shift | 1-5 min | 4 stages | Stage 1 (interruptions) |
| TEMPORAL-08 | Authenticity Erosion | 15-60 min | 4 stages | Stage 2 (micro-leaks) |

### TEMPORAL-04 (Objection Formation) — 5-Stage Early Warning

This sequence detects objections BEFORE they're spoken:

```
Stage 1 (0-5s):   Cognitive trigger — gaze break + brow furrow (processing)
Stage 2 (3-10s):  Body reaction — lean back OR self-touch OR posture close
Stage 3 (5-15s):  Voice change — F0 shift OR pause before speaking
Stage 4 (10-25s): Verbal hedge — "well...", "I mean...", "the thing is..."
Stage 5 (15-30s): Explicit objection — spoken resistance

Early warning fires at Stage 2 → salesperson gets 10-20 seconds of advance notice.
```

---

## Cross-Reference: Signal Dependencies

### Which agents need which other agents' data?

```
Voice Agent     → Independent (no cross-agent input needed)
Language Agent  → Needs Voice Agent transcript (word-level timestamps)
Facial Agent    → Independent (video frames only)
Body Agent      → Independent (video frames only)
Gaze Agent      → Shares face landmarks with Facial Agent
Conversation    → Needs Voice Agent diarization + Language Agent topics
Fusion Agent    → Consumes ALL other agents' outputs
```

### Which rules validate which other rules?

| Rule | Validated By | Invalidated By |
|------|-------------|----------------|
| VOICE-STRESS-01 (stress) | FACE-STRESS-01, BODY-FIDG-01 | FACE-EMO-01 = happy + confident |
| LANG-BUY-01 (buying) | BODY-LEAN-01 forward, GAZE-CONTACT-01 high | VOICE-TONE-03 nervous, BODY-POST-01 closed |
| FACE-SMILE-01 (Duchenne) | VOICE-TONE-01 warm, LANG-SENT-01 positive | VOICE-STRESS-01 high, BODY-TOUCH-01 active |
| LANG-OBJ-01 (objection) | BODY-LEAN-01 backward, CONVO-LAT-01 high | FACE-EMO-01 = happy, BODY-HEAD-01 nodding |
| CONVO-RAP-01 (rapport) | GAZE-SYNC-01, BODY-MIRROR-01 | VOICE-TONE-02 cold, LANG-GOTT-01 active |

---

## Alert Tiers

| Tier | Name | Trigger | Dashboard Behaviour |
|------|------|---------|-------------------|
| 1 | NOTICE | Single signal, confidence > 0.50 | Subtle timeline dot |
| 2 | ALERT | 2+ congruent signals OR compound pattern | Yellow highlight + notification |
| 3 | CRITICAL | High-confidence compound + temporal pattern | Red highlight + real-time push |
| 4 | INSIGHT | Post-session narrative-worthy moment | Star marker in report |

---

## Rule Configuration (Database)

All thresholds live in `rule_config` table. Schema:
```sql
rule_config (
  rule_id         VARCHAR PRIMARY KEY,   -- 'VOICE-STRESS-01'
  agent           VARCHAR,               -- 'voice'
  threshold_json  JSONB,                 -- {"high": 0.70, "moderate": 0.30, ...}
  weight          FLOAT DEFAULT 1.0,     -- Global weight multiplier
  enabled         BOOLEAN DEFAULT TRUE,  -- Kill switch per rule
  updated_at      TIMESTAMP
)
```

To tune a threshold without code changes:
```sql
UPDATE rule_config 
SET threshold_json = jsonb_set(threshold_json, '{high}', '0.65')
WHERE rule_id = 'VOICE-STRESS-01';
```
