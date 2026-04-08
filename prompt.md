# NEXUS Implementation Plan

**Research-Validated Threshold Changes + Phase 2 Build Specifications**
Based on Rule-by-Rule Academic Validation of All 94 Rules — April 2026

---

## Plan Summary

94 rules validated. 43 need changes. 3 should be disabled. 48 confirmed as-is.

- **Priority 1 (NOW):** 15 threshold fixes to 42 implemented rules — pure code changes, no new infrastructure
- **Priority 2 (Phase 2):** Build 22 video rules (Facial/Body/Gaze) with research-validated caps from day 1
- **Priority 3 (Phase 2+):** Build 12 Fusion pairwise + 20 Compound/Temporal with adjusted caps
- **Content-type adaptation:** Detect universally, interpret contextually. Only 5 rules need detection-logic changes per type. The rest need interpretation-layer changes only (labels, gating, prompts).

---

## Priority 1: Threshold Fixes to 42 Implemented Rules

These are pure code changes to existing rules.py files. No new agents, no new infrastructure. Each change is backed by the best available research paper. Estimated effort: 3-4 days.

### 1A. Voice Agent — 7 Threshold Changes

| Rule | Current | Change To | Research | Rationale |
|------|---------|-----------|----------|-----------|
| **STRESS-01** | F0 weight: 0.25, Filler weight: 0.15 | F0: **0.30**, Filler: **0.10** | Veiga 2025 meta-analysis | F0 is most reliable stress marker (SMD 0.55, p<0.001). Filler rate lacks direct stress validation in psychoacoustic literature. |
| **FILLER-02** | 1.3% / 2.5% / 4.0% | **2.5% / 4.0% / 6.0%** | Bortfeld 2001 (96 speakers) | 1.3% is floor of normal speech (1.3-4.4 per 100 words). Most normal speakers would be flagged at current threshold. |
| **PITCH-01** | 8% / 15% / 25% | **7% / 12% / 20%** | Veiga 2025 meta-analysis | Meta-analysis: avg stress F0 effect ~7.8% for males, ~10.7% for females. Current 15% "significant" misses many meaningful stress episodes. |
| **PITCH-02** | F0 range <30Hz absolute | **<40% of speaker's baseline F0 range** | Gender correction | 30Hz = 40% drop for males (baseline ~60Hz range) but 75% drop for females (baseline ~100Hz range). Percentage normalizes for gender. |
| **RATE-01** | ±25% | **±20%** or two-tier (±15% notable, ±25% significant) | Quené 2007 (JND study) | JND for speech tempo is ~5%. Normal between-speaker variation is 6-17%. Gap between 17% normal and 25% threshold is too large. |
| **PAUSE-01** | >250ms hesitation | **>200ms** hesitation | Heldner & Edlund 2010 | Modern consensus is 200ms (Laver 1994, Heldner 2010). Goldman-Eisler 1968's 250ms is outdated. |
| **INT-01** | >200ms overlap, ≤2 words backchannel | **>500ms** overlap, **≤3 words** backchannel | Heldner 2011 (JASA) | 200ms captures ~40% of normal cooperative overlaps as false positives. Perceptual detection threshold is ~120ms. 500ms filters out simultaneous starts and cooperative overlap. ≤2 words misses phrasal backchannels ("I see", "that's right"). |

### 1B. Language Agent — 4 Threshold + 6 Interpretation Changes

**Threshold Changes:**

| Rule | Current | Change To | Research | Rationale |
|------|---------|-----------|----------|-----------|
| **SENT-01** | Model 0.70 / LIWC 0.30 | Model **0.85-0.90** / LIWC **0.10-0.15** | Crossley 2017 (SEANCE) | Transformers achieve 85-95% accuracy. LIWC misses context, irony, sarcasm, negation. Retain LIWC for unique psychological dimensions (Authenticity, Clout) not direct sentiment. |
| **CLAR-01** | >25 words/sentence, >30% passive | **>18 words**, **>10% passive** | Biber 1999 (Longman Grammar) | Spoken language averages 12-20 words/sentence. Passives occur ~2-5% in conversation. >30% passive is essentially impossible in speech — a dead rule. |
| **TOPIC-01** | 15% word overlap | **Adaptive cosine similarity ~0.3** | Hearst 1997 (TextTiling) | TextTiling (1,494 citations) uses mean-1SD adaptive threshold, not fixed overlap. No published research validates 15% specifically. |
| **SENT-02** | >8% high, 4-8% moderate, <2% suppressed | **>6% high**, 3-6% moderate, **<1.5% suppressed** (professional speech) | Tausczik & Pennebaker 2010 | Average emotion word density is ~4%. Current threshold puts half of normal speech at moderate-to-high. |

**Interpretation/Gating Changes (same detection logic, different labels):**

| Rule | Current | Change | Research | Rationale |
|------|---------|--------|----------|-----------|
| **BUY-01** | Sales-gated only | **Universal** with content-type labels | Rackham 1988 | Same detection patterns (pricing/timeline/implementation questions) detect interest in interviews and client meetings. Labels: sales="buying_signal", client="engagement_signal", interview="interest_signal". |
| **OBJ-01** | Sales-gated only | **Universal** with content-type labels | Brill 2021 (Contrastive Pragmatics) | Hedge + contrastive marker patterns ("but", "however", "I'm not sure") are universal disagreement markers across all spoken contexts. Labels: sales="objection", client="concern", internal="disagreement", interview="hesitation". |
| **PERS-01** | Sales-gated only | **Universal** with content-type labels | Cialdini 2021 | 7 principles operate in politics, organizational dynamics, education, relationships — not just sales. Labels: client="influence_tactic", internal="influence_attempt", interview="impression_management". |
| **QUES-01** | SPIN classification for all types | **Split**: Open/closed/tag **universal**; SPIN **sales-only** | Rackham 1988 | SPIN (Situation/Problem/Implication/Need-payoff) was designed and validated exclusively on sales conversations. Open/closed/tag question distinction is universal across all interview, meeting, and conversational research. |
| **PWR-01** | Uniform powerless scoring | **Separate** hesitations from cognitive hedges | Jensen 2008 | Hedging ("I think", "sort of") increases perceived trustworthiness in scientific/expert communication. Only hesitation forms (um, uh) consistently signal low competence. Apply context-variable interpretation to deliberative hedges. |
| **NEG-01** | Gottman 4 horsemen uniform | Keep **criticism/contempt**; reframe **defensiveness** as "resistance", **stonewalling** as "disengagement" | Cortina workplace incivility | Gottman validated on married couples only (91% divorce prediction). No peer-reviewed workplace validation exists. Stonewalling at work may be adaptive boundary-setting, not emotional withdrawal. |

### 1C. Conversation Agent — 3 Changes

| Rule | Current | Change To | Research | Rationale |
|------|---------|-----------|----------|-----------|
| **TURN-01** | >8/min rapid | **>10/min** rapid | Meyer 2023 (J. of Cognition) | 8/min is upper edge of normal casual conversation (avg turn duration 2-4s = 6-8 combined turns/min). Truly competitive rapid turn-taking exceeds 10-12/min. |
| **DOM-01** | Interruption weight 0.20, Monologue weight 0.20 | Interruption **0.15**, Monologue **0.25** | Jayagopi 2009 (IEEE Trans.) | Interruption count "performed badly" as single dominance predictor. Speaking length alone achieved ~85.3% accuracy on AMI corpus. Increase monologue weight to reflect this. |
| **BAL-01** | Flag deviation from symmetry | Compare observed Gini to **expected Gini per content type** | Design decision | Interview (expected 20/80), podcast (30/70) have expected asymmetry. A Gini of 0.40 in peer meeting = problem; in interview = normal. Flag deviation from *expected* balance, not from symmetry. |

### 1D. Fusion Agent — 2 Changes

| Rule | Current | Change To | Research | Rationale |
|------|---------|-----------|----------|-----------|
| **FUSION-02** | Cap 0.55, name "credibility_assessment" | Cap **0.65**, rename to **"stress_sentiment_incongruence"** | Hartwig & Bond 2014 | Bond & DePaulo's 54% applies to untrained humans. Multi-cue systems achieve ~68% (Hartwig & Bond, 144 samples, 26,866 messages). "Credibility" implies truthfulness judgment — ethically indefensible at any confidence level. The system measures stress-sentiment mismatch, not truth. |
| **FUSION-13** | Gated to sales/pitch/presentation only | Expand to **any persuasive context** | Guyer 2024 (N=3,958) | Urgency-persuasion dynamics documented in negotiation (deadline pressure), customer escalation, political communication — not just sales. Add negotiation and customer escalation contexts. |

### 1E. Bug Fixes from Previous Audit

| Issue | Fix |
|-------|-----|
| **Tone priority chain** | Change from `warm > excited > aggressive > cold` to `aggressive > excited > warm > cold` (higher severity first). Currently aggressive can never fire if excited passes first. |
| **FILLER-01 zero-baseline** | Increase divisor from 1.0 to **3.0** for speakers with zero baseline filler rate. Currently a single "um" in a clean recording triggers filler_spike. |
| **NEG-01 stonewalling** | Require context: only flag "fine/whatever/okay" when preceded by a substantive question from other speaker. Currently flags normal acknowledgments. |
| **INTENT-01 prompts** | Add content-type-specific prompt templates. Interview intents (probe_competency, assess_fit) differ fundamentally from sales intents (buy, object, negotiate). Zero extra LLM calls — same call, different prompt content. |
| **TALK-01 sales sub-threshold** | Add >60% seller talk = significant imbalance flag for sales calls specifically. Gong data: top performers at 43%, underperformers at 64%. Current 70% threshold misses underperformer territory. |

---

## Priority 2: Build Video Agents with Research-Validated Caps

Build these agents with the corrected confidence caps from day 1. Do NOT use the DOCX spec caps where research contradicts them.

### 2A. Facial Agent — 7 Rules (Weeks 7-8)

| Rule | DOCX Cap | Build With | Key Research | Implementation Note |
|------|----------|-----------|-------------|---------------------|
| **EMO-01** | 0.55 | **0.55** ✓ | AffectNet 57-67% accuracy | Add per-emotion confidence weighting: Happy 0.70, Angry/Sad 0.55, Disgust/Contempt 0.35. Consider reducing to 4-5 class model. |
| **SMILE-01** | 0.60-0.65 | **0.45-0.55** ↓ | Girard 2020 (PMC7193529) | Duchenne hypothesis weakened — AU6 co-occurrence primarily predicted by AU12 intensity, not felt emotion. Add temporal onset features (93% accuracy per Cohn). Rename to "smile quality analysis". |
| **MICRO-01** | 0.15-0.30 | **DISABLE** ✗ | IEEE TPAMI 2021, Yan 2013 | ME lasts 40-200ms. At 30fps, frame interval is 33ms — a 40ms ME spans 1-2 frames, indistinguishable from noise. All major ME databases captured at 100-200fps. Even at 200fps, 3-class recognition only ~52%. Relabel as "rapid expression change" with cap 0.10-0.15 if retained. |
| **ENG-01** | 0.50 | **0.50** ✓ | Whitehill 2014 (IEEE TAC) | No change. Head pose + expression variability validated (2AFC=0.73, comparable to human accuracy 0.696). |
| **STRESS-01** | 0.45 | **0.35-0.40** ↓ | Giannakakis 2020 | AU23 F1=40-55%, AU24 F1=40-55% — among worst-detected AUs. Two of three constituent AUs are unreliable. Supplement with AU7, AU10, increased blink rate. |
| **VA-01** | 0.40-0.60 | **Valence 0.55, Arousal 0.40** | Toisoul 2021 (Nature MI) | Valence CCC ~0.76-0.82; arousal ~0.55-0.65. Split thresholds to reflect this asymmetry. |
| **CAL-01** | 60s | **90-120s** for formal contexts | Cohn 2004 | First 60s includes greeting behaviors (elevated smiling, social pleasantries). Implement greeting-phase detection to exclude initial pleasantry period. Add rolling recalibration every 10-15 minutes. |

### 2B. Gaze Agent — 7 Rules (Weeks 9-10)

| Rule | DOCX Cap | Build With | Key Research | Implementation Note |
|------|----------|-----------|-------------|---------------------|
| **DIR-01** | 0.45-0.65 | **0.45-0.65** ✓ | MPIIGaze (3-7° error) | Reframe as "screen-directed gaze". Binary on-screen/off-screen is the only reliable output. Cannot distinguish "looking at face on screen" from "reading chat". |
| **CONTACT-01** | 0.50 | **0.50** ✓ | Argyle 1972 | Reframe as "screen engagement". Argyle norms (speaker 40-60%, listener 60-75%) are NOT validated for video calls. Camera-screen offset (5-15° vertical) makes everyone appear to avoid eye contact. |
| **BLINK-01** | Stress >25 bpm | Stress **>30-35 bpm** ↑ | Bentivoglio 1997 (N=150) | **CRITICAL FIX:** >25 bpm is BELOW the normal conversational mean of 26 bpm per Bentivoglio. Would flag every normal conversation as stressed. Must raise to >30-35 bpm (~1+ SD above conversational mean). |
| **ATT-01** | 0.55 | **0.55** ✓ | Wohltjen 2024 | Composite approach validated. Exclude pupil dilation (webcam resolution insufficient). Build from screen gaze %, head pose stability, blink deviation, gaze break frequency. |
| **DIST-01** | >3s sustained break | **>8-10s** sustained break | Glenberg 1998, HAL 2024 | **CRITICAL FIX:** 3s is well within normal cognitive gaze aversion duration (avg ~6s per HAL 2024). Normal preferred gaze duration is 3.2-3.3s (Binetti 2016). >3s threshold would generate massive false positives during any cognitively demanding conversation. |
| **SYNC-01** | 0.40 | **0.40, flag EXPERIMENTAL** | Wohltjen 2021 (PNAS) | Rename to "Cross-Speaker Gaze Alignment". Remove Wohltjen citation — that paper measured pupillary synchrony with lab-grade eye trackers, not webcam gaze direction. In video calls, both speakers looking at screens simultaneously is default, not synchrony. |
| **CAL-01** | 60s | **60-120s** | Gudi 2020 (ECCV) | Camera position is dominant source of gaze variability (laptop vs external webcam = 5-15° offset). Use first 60-120s to estimate camera position from average head pose. Re-estimate every 5-10 minutes. |

### 2C. Body Agent — 8 Rules (Weeks 11-12)

| Rule | DOCX Cap | Build With | Key Research | Implementation Note |
|------|----------|-----------|-------------|---------------------|
| **POST-01** | 0.40-0.55 | **0.40-0.55** ✓ | Ding 2019 (99.5% binary) | Add confidence penalty (reduce to 0.35) when arms not detected. Arms visible only ~30-60% of webcam frames. Rely primarily on shoulder landmarks. |
| **HEAD-01** | 0.55-0.75 | **0.55-0.75** ✓ | Chen 2015 (ICCV) | HIGHEST-RELIABILITY body signal from webcam. Min angular velocity threshold ~15°/s for nods, ~20°/s for shakes. Weight heavily in composite scoring. |
| **LEAN-01** | 0.45 | **0.30-0.40** ↓ | PoseX (Chen 2021) | 3-5cm lean at 60cm distance produces only 5-8% head-size change — near noise floor of landmark detection. Require minimum 8-10% head-size change before triggering. Replace pure head-size with multi-landmark approach. |
| **GEST-01** | 0.45-0.80 | **0.45-0.80** ✓ | MediaPipe Hands (95.7% AP) | Focus on beat and deictic gestures as most reliably classifiable. Treat hand non-detection for >5s as separate signal (may indicate crossed arms). |
| **FIDG-01** | 0.50-0.55 | **0.35-0.45** ↓ | Zhang 2020 (Sensors, 85.4%) | Subtle finger fidgets fall below webcam detection threshold (~2-3 pixels min). Replace "anxiety" interpretation with "elevated movement" — fidgeting correlates with boredom, excitement, ADHD too. Require co-occurrence with other stress signals. |
| **TOUCH-01** | 0.35-0.50 | **0.35-0.50** ✓ | Beyan 2020 (ACM ICMI, 83.7% F1) | Remove Navarro's stress hierarchy (neck>face>hair>arm) — NOT empirically validated in peer-reviewed literature. Use bounding-box overlap rather than precise landmark proximity. Require sustained proximity >0.5s. |
| **MIRROR-01** | 0.40 | **0.20-0.30, flag EXPERIMENTAL** | No validated webcam system | Cross-feed mirroring detection across two independently framed video-call feeds is a fundamentally unsolved problem (different angles, distances, left-right flip ambiguity). No published system achieves reliable results. Restrict to gross posture matching with >5s temporal co-occurrence if retained. |
| **CAL-01** | 60s | **90-120s** | Bernieri & Rosenthal 1991 | Discard first 30s (settling-in behavior: camera adjustment, greeting). Posture baseline stabilizes slower than facial expression baseline. Implement per-signal baselines (separate for posture, head movement, gesture rate). |

---

## Priority 3: Fusion Pairwise + Compound + Temporal (Weeks 13-16)

### 3A. Remaining 12 Fusion Pairwise Rules

| Rule | Cross-Modal Check | DOCX Cap | Build Cap | Note |
|------|-------------------|----------|-----------|------|
| **F-01** | Tone × Face → Masking | 0.75 | **0.75** ✓ | Strong evidence (Watson 2013 fMRI). Note: incongruence ≠ deliberate masking. |
| **F-03** | Posture × Energy → Enthusiasm | 0.65 | **0.65** ✓ | Moderate evidence (Van den Stock 2011). |
| **F-04** | Gaze × Filler → Uncertainty | 0.70 | **0.70** ✓ | Strong for cognitive load, moderate for uncertainty. Consider relabel to "Cognitive Effort Confirmation". |
| **F-05** | Buy × Body → Purchase Intent | 0.70 | **0.55-0.60** ↓ | No peer-reviewed research validates multimodal purchase intent verification. Domain-specific inference. |
| **F-06** | Micro × Lang → Leakage | 0.35 | **DISABLE** or **0.20** | ME detection infeasible at ≤30fps. Burgoon 2018 systematically critiques all 6 propositions of ME theory. Jordan 2019: METT training did not improve lie detection. |
| **F-07*** | Head Nod × Speech → Disagree | 0.65 | **0.65** ✓ | Moderate-strong (Briñol & Petty 2003). Add cultural sensitivity flags — Bulgarian reversal, gender differences (women nod for listening >75%). |
| **F-08** | Eye × Hedge → False Confidence | 0.55 | **0.55** ✓ | Conservative cap appropriate. >80% of deception researchers agree gaze aversion is not diagnostic (Denault 2020). |
| **F-09** | Smile × Sentiment → Sarcasm | 0.60 | **0.60** ✓ | Moderate (IJCAI 2024 survey). Consider splitting "sarcasm" vs "social masking" as separate output labels. |
| **F-10** | Latency × Face → Cognitive Load | 0.60 | **0.60** ✓ | Moderate-strong (ADABase 2023). Specify AU4/AU7 preferred over AU23/AU24 for facial stress component. |
| **F-11** | Dominance × Gaze → Anxious | 0.65 | **0.65** ✓ | Moderate (Terburg 2012). Many alternative explanations (cultural norms, screen reading). |
| **F-12** | Interrupt × Body → Intent | 0.55 | **0.55** ✓ | Moderate. Cooperative interruptions = leaning forward + nodding; competitive = dominance signals. |
| **F-14** | Empathy × Nod → Rapport | 0.70 | **0.70** ✓ | **STRONGEST fusion rule.** Tickle-Degnen & Rosenthal 1990 (919+ citations). Could argue 0.75. |
| **F-15** | Filler × Gaze → Uncertainty | 0.55 | **MERGE with F-04** | Redundant with FUSION-04. Same signals (filler + gaze break), different labels and caps (0.70 vs 0.55). Creates logical inconsistency. Differentiate: F-04 = single event, merged rule = sustained pattern. |

### 3B. Compound Patterns — 12 Rules

| Rule | Pattern | DOCX Cap | Build Cap | Note |
|------|---------|----------|-----------|------|
| **C-01** | Genuine Engagement | 0.80 | **0.80** ✓ | Strong construct. Pellet-Rostaing 2023 (F1=0.76 multimodal). |
| **C-02** | Active Disengagement | 0.80 | **0.75** ↓ | Overlaps with fatigue, boredom, internal processing — not always intentional disengagement. |
| **C-03** | Emotional Suppression | 0.70 | **0.70** ✓ | Moderate-strong (Ekman leakage hierarchy). Flag that 15-30fps is insufficient for micro-expression component. |
| **C-04** | Decision Readiness | 0.85 | **0.65** ↓↓ | **LARGEST OVER-CONFIDENCE in the entire system.** No validated construct. Neuroscience "readiness potential" (Libet 1985) = motor cortex, not interpersonal decision. Ajzen's TPB = self-report, not behavioral. Rename to "Decision Engagement Signals". Label EXPERIMENTAL. |
| **C-05** | Cognitive Overload | 0.75 | **0.75** ✓ | Strong construct (Sweller's CLT). Emphasize speech-based indicators as primary webcam-accessible signals. |
| **C-06** | Conflict Escalation | 0.80 | **0.80** ✓ | Strong (Glasl 9-stage model + Gottman). Both-speaker requirement strengthens validity. |
| **C-07** | Silent Resistance | 0.70 | **0.65** ↓ | Massive cultural variation — collectivist cultures routinely practice verbal compliance with private disagreement as social norm, not resistance. Rename to "Verbal-Nonverbal Discordance". |
| **C-08** | Rapport Peak | 0.85 | **0.80** ↓ | Strong construct (Tickle-Degnen, 3,000+ citations) but detecting "peak" adds measurement uncertainty vs detecting presence. |
| **C-09** | Topic Avoidance | 0.70 | **0.70** ✓ | Moderate. Ensure system requires temporal contingency with topic content — not just avoidance-like behavior in isolation. |
| **C-10** | Authentic Confidence | 0.85 | **0.75** ↓ | Machine learning achieves 88% for confidence detection, but authentic/performed distinction relies on unvalidated leakage theory applied to confidence specifically. |
| **C-11** | Anxiety Performance | 0.65 | **0.65** ✓ | Core interview signal. Consider merging with C-03 as a subtype (both = controlled vs leaked channels). |
| **C-12** | Deception Risk | 0.55 | **0.55** ✓ | **BEST-CALIBRATED threshold in the entire system.** Precisely aligned with Bond & DePaulo 2006 (206 studies, 24,483 judges, 54% human accuracy) and Hartwig & Bond 2014 multi-cue ceiling (~70%). The 4-domain + 3-fusion-conflict requirement adds rigorous gating. Keep exactly as-is. |

### 3C. Temporal Sequences — 8 Rules

Only TEMPORAL-04 (Objection Formation) has a validated temporal ordering. All others are theoretical. Label as EXPERIMENTAL.

| Rule | Sequence | DOCX Cap | Build Cap | Note |
|------|----------|----------|-----------|------|
| **T-01** | Stress Cascade (voice→body→face, 2-15s) | 0.70 | **0.65** ↓ | Ordering reflects differential controllability (Ekman leakage hierarchy), not physiological timing — all systems activate simultaneously. Widen window to 2-30s. |
| **T-02** | Engagement Build (gaze→face→body→voice, 1-3min) | 0.65 | **0.65** ✓ | Consistent with attention→affect→approach models. Consider making voice parallel with body (co-occur rather than strictly sequential). |
| **T-03** | Disengage Cascade (face→gaze→body→voice, 30-120s) | 0.55 | **0.55** ✓ | Theoretical. Cap appropriately reflects this status. |
| **T-04** | Objection Formation (face→voice→language, 5-30s) | 0.60 | **0.60** ✓ | **ONLY VALIDATED temporal sequence.** Ekman dual-process: genuine emotional reactions appear on face before verbal formulation. Matsumoto & Hwang 2018 confirmed timing. Focus on "subtle expressions" (>200ms) rather than true micro-expressions for webcam feasibility. |
| **T-05** | Trust Repair (voice→face→body→gaze, 10-90s) | 0.75 | **0.50-0.55** ↓↓ | **MOST SPECULATIVE RULE.** Trust repair research (Sharma 2023) operates at relational level over weeks/months, not second-by-second multimodal signals. Proposed modality ordering has no direct empirical validation. Consider reframing as "Reconciliation Gesture Detection". |
| **T-06** | Buying Decision Seq (5-30min multi-stage) | 0.60 | **0.45-0.50** ↓ | Kotler's 5-stage buying process is a marketing model for understanding consumer psychology, not a real-time behavioral detection system. Observable behaviors do not map cleanly onto buying stages. Rename to "Decision Engagement Pattern". |
| **T-07** | Dominance Shift (voice→body→gaze, 30s-5min) | 0.55 | **0.55** ✓ | Theoretical. Dominance signals tend to manifest as simultaneous multimodal packages rather than sequential cascades. Add cultural sensitivity flags. |
| **T-08** | Authenticity Erosion (15-60min progressive) | 0.55 | **0.55** ✓ | Two converging literatures support progressive leakage: Ekman (sustained suppression → increasing leakage) and Hochschild 1983 (emotional labor depletes resources). Rename to "Behavioral Consistency Monitoring". Add ethical disclaimers. |

---

## Content-Type Adaptation: Minimal, Interpretation-Layer Only

The research overwhelmingly confirms: **detect universally, interpret contextually**. The per-speaker baseline handles most cross-context variation automatically. Only 5 rules need detection-logic changes per content type. The rest need interpretation-layer changes only.

### Detection-Layer Changes (Only 5 Rules)

| Rule | Sales | Client Mtg | Internal | Interview | Podcast |
|------|-------|-----------|----------|-----------|---------|
| **FILLER-02** | 2.5%/4%/6% | Same | noticeable +0.5% | Same | Same |
| **PAUSE-01 extended** | >2000ms | Same | Same | >3000ms | Same |
| **TALK-01** | >60% seller flag | Same | >50% peer flag | Candidate 30-70% expected | Guest 60-80% expected |
| **SENT-02** | >8%/>4%/<2% | Same | >6%/>3%/<1.5% (professional) | Same | Same |
| **CLAR-01** | >18 words | Same | Same | Same | Same |

### Interpretation-Layer Changes (Labels + Gating)

| Rule | Sales | Client Mtg | Internal | Interview | Podcast |
|------|-------|-----------|----------|-----------|---------|
| **BUY-01** | buying_signal | engagement_signal | GATE | interest_signal | GATE |
| **OBJ-01** | objection | concern | disagreement | hesitation | GATE |
| **PERS-01** | persuasion | influence_tactic | influence_attempt | impression_mgmt | GATE |
| **QUES-01 SPIN** | SPIN active | GATE SPIN | GATE SPIN | GATE SPIN | GATE SPIN |
| **NEG-01 stonewall** | stonewalling | stonewalling | disengagement | disengagement | GATE |
| **BAL-01** | Flag >0.35 Gini | Flag >0.35 | Flag >0.35 | Compare to expected | Compare to expected |
| **INTENT-01** | Sales prompt | Meeting prompt | Meeting prompt | Interview prompt | Podcast prompt |

**Implementation:** A `ContentTypeProfile` class loaded per session provides `is_gated()`, `rename_signal()`, and `get_prompt_template()`. One new file (~200 lines). No detection threshold changes needed beyond the 5 rules listed above.

---

## Implementation Timeline

| Week | Task | Rules Affected | Effort | Deliverable |
|------|------|---------------|--------|-------------|
| **1** | P1: Voice Agent threshold fixes (7 rules) | 16 implemented | 2 days | Updated voiceAgent/rules.py |
| **1** | P1: Language Agent threshold + interpretation (10) | 12 implemented | 2 days | Updated language-agent/rules.py + prompts |
| **2** | P1: Conversation + Fusion + Bug fixes | 14 implemented | 1 day | Updated rules.py x2 |
| **2** | P1: ContentTypeProfile class + interpretation layer | All 42 | 1 day | shared/utils/content_type_profile.py |
| **2** | P1: Test all 5 content types with real recordings | All 42 | 1 day | Validation report |
| **7-8** | P2: Facial Agent (7 rules, research caps) | 7 new | 2 weeks | services/facial-agent/ |
| **9-10** | P2: Gaze Agent (7 rules, research caps) | 7 new | 2 weeks | services/gaze-agent/ |
| **11-12** | P2: Body Agent (8 rules, research caps) | 8 new | 2 weeks | services/body-agent/ |
| **13-14** | P3: 12 Fusion pairwise (merge F-15 into F-04) | 12 new | 2 weeks | Updated fusion-agent/ |
| **15-16** | P3: 12 Compound + 8 Temporal (EXPERIMENTAL label) | 20 new | 2 weeks | services/pattern-engine/ |

---

## Final Statistics

| Category | Total Rules | No Change | Threshold Δ | Interpretation Δ | Disable |
|----------|------------|-----------|-------------|-------------------|---------|
| Voice Agent | 16 (implemented) | 9 | 7 | 0 | 0 |
| Language Agent | 12 (implemented) | 2 | 4 | 6 | 0 |
| Conversation Agent | 8 (implemented) | 5 | 2 | 1 | 0 |
| Fusion (audio) | 6 (implemented) | 4 | 1 | 1 | 0 |
| Facial Agent | 7 (build) | 2 | 3 | 1 | 1 (MICRO-01) |
| Gaze Agent | 7 (build) | 4 | 2 | 1 | 0 |
| Body Agent | 8 (build) | 4 | 2 | 1 | 1 (MIRROR exp) |
| Fusion (video) | 13 (build) | 10 | 1 | 1 | 1 (F-06) |
| Compound | 12 (build) | 6 | 5 | 1 | 0 |
| Temporal | 8 (build) | 4 | 3 | 1 | 0 |
| **TOTAL** | **97** | **50 (52%)** | **30 (31%)** | **14 (14%)** | **3 (3%)** |

**50 rules (52%) validated as-is — no changes needed.**
**30 rules (31%) need threshold adjustments backed by specific research papers.**
**14 rules (14%) need interpretation-layer changes (labels, gating, prompts) — not detection changes.**
**3 rules (3%) should be disabled or flagged experimental at webcam framerates.**

---

## Complete Content-Type Adaptation: All 97 Rules × 5 Types

### Legend

- **FIRE** = Rule runs with universal thresholds, no change
- **FIRE+** = Rule runs universally but with a small, research-justified threshold tweak for this type only
- **RENAME** = Same detection, different output label
- **GATE** = Rule suppressed (returns None) — irrelevant for this type
- **INTERP** = Same detection and label, but narrative/report interprets the signal differently
- **N/A** = Rule cannot run (audio-only podcast, no video feed)

### Voice Agent — 16 Rules × 5 Content Types

| # | Rule | Signal | Sales | Client Mtg | Internal | Interview | Podcast |
|---|------|--------|-------|-----------|----------|-----------|---------|
| 1 | VOICE-CAL-01 | speaker_baseline | FIRE | FIRE | FIRE | FIRE | FIRE |
| 2 | VOICE-STRESS-01 | vocal_stress_score | FIRE | FIRE | FIRE | FIRE | FIRE |
| 3 | VOICE-FILLER-01 | filler_detection | FIRE | FIRE | FIRE | FIRE | FIRE |
| 4 | VOICE-FILLER-02 | filler_credibility | FIRE | FIRE | FIRE+ noticeable +0.5% for informal | FIRE | FIRE |
| 5 | VOICE-PITCH-01 | pitch_elevation | FIRE | FIRE | FIRE | FIRE | FIRE |
| 6 | VOICE-PITCH-02 | monotone_flag | FIRE | FIRE | FIRE | FIRE | FIRE |
| 7 | VOICE-RATE-01 | speech_rate_anomaly | FIRE | FIRE | FIRE | FIRE | FIRE |
| 8 | VOICE-TONE-01 | warm | FIRE | FIRE | FIRE | FIRE | FIRE |
| 9 | VOICE-TONE-02 | cold | FIRE | FIRE | FIRE | FIRE | FIRE |
| 10 | VOICE-TONE-03 | nervous | FIRE | FIRE | FIRE | FIRE | FIRE |
| 11 | VOICE-TONE-04 | confident | FIRE | FIRE | FIRE | FIRE | FIRE |
| 12 | VOICE-TONE-05 | aggressive | FIRE | FIRE | FIRE | FIRE | FIRE |
| 13 | VOICE-TONE-06 | excited | FIRE | FIRE | FIRE | FIRE | FIRE |
| 14 | VOICE-ENERGY-01 | energy_level | FIRE | FIRE | FIRE | FIRE | FIRE |
| 15 | VOICE-PAUSE-01 | pause_classification | FIRE | FIRE | FIRE | FIRE+ extended_hesitation >3000ms (thinking time normal for complex answers) | FIRE |
| 16 | VOICE-INT-01 | interruption_event | FIRE | FIRE | FIRE | FIRE | FIRE |
| — | VOICE-TALK-01 | talk_time_ratio | FIRE+ >60% seller significant (Gong data) | FIRE | FIRE+ >50% peer meeting flag | FIRE+ candidate 30-70% expected range | FIRE+ guest 60-80% expected range |

**Voice summary:** 13 of 17 rules are fully universal — baseline-relative detection handles all context variation. Only FILLER-02 (informal tolerance), PAUSE-01 (interview thinking time), and TALK-01 (expected ratios per type) need per-type adjustments. All tone rules, stress, pitch, rate, energy, and interruption detection are universal because the per-speaker baseline already absorbs context-specific norms.

### Language Agent — 12 Rules × 5 Content Types

| # | Rule | Signal | Sales | Client Mtg | Internal | Interview | Podcast |
|---|------|--------|-------|-----------|----------|-----------|---------|
| 17 | LANG-SENT-01 | sentiment_score | FIRE | FIRE | FIRE | FIRE | FIRE |
| 18 | LANG-SENT-02 | emotional_intensity | FIRE | FIRE | FIRE+ >6% high, <1.5% suppressed (professional speech norms) | FIRE | FIRE |
| 19 | LANG-BUY-01 | buying/interest signal | FIRE | RENAME → "engagement_signal" | GATE | RENAME → "interest_signal" | GATE |
| 20 | LANG-OBJ-01 | objection/concern | FIRE | RENAME → "concern" | RENAME → "disagreement" | RENAME → "hesitation" | GATE |
| 21 | LANG-PWR-01 | power_language_score | FIRE | FIRE | FIRE | INTERP: powerful = competence signal, powerless = note (not flag) | FIRE |
| 22 | LANG-PERS-01 | persuasion_technique | FIRE | RENAME → "influence_tactic" | RENAME → "influence_attempt" | RENAME → "impression_management" | GATE |
| 23 | LANG-QUES-01 | question_type | FIRE (SPIN active) | FIRE (SPIN gated, open/closed only) | FIRE (SPIN gated) | FIRE (SPIN gated) | FIRE (SPIN gated) |
| 24 | LANG-TOPIC-01 | topic_shift | FIRE | FIRE | FIRE | FIRE | FIRE |
| 25 | LANG-NEG-01 | gottman_horsemen | FIRE | FIRE | FIRE, stonewalling → "disengagement" | FIRE, stonewalling → "disengagement" | GATE |
| 26 | LANG-EMP-01 | empathy_language | FIRE | FIRE | FIRE | INTERP: interviewer empathy = good technique; candidate empathy = soft skill | FIRE |
| 27 | LANG-CLAR-01 | clarity_score | FIRE | FIRE | FIRE | FIRE | FIRE |
| 28 | LANG-INTENT-01 | intent_classification | FIRE (sales prompt) | FIRE (meeting prompt) | FIRE (meeting prompt) | FIRE (interview prompt) | FIRE (podcast prompt) |

**Language summary:** Detection logic is 100% universal — no threshold changes per type (except SENT-02 for professional speech). All changes are labels (RENAME), gating (GATE for irrelevant rules like buying signals in internal meetings), and prompt templates (INTENT-01). The underlying hedge detection, sentiment scoring, empathy detection, and topic tracking run identically across all types.

### Conversation Agent — 8 Rules × 5 Content Types

| # | Rule | Signal | Sales | Client Mtg | Internal | Interview | Podcast |
|---|------|--------|-------|-----------|----------|-----------|---------|
| 29 | CONVO-TURN-01 | turn_taking_pattern | FIRE | FIRE | FIRE | FIRE | FIRE |
| 30 | CONVO-LAT-01 | response_latency | FIRE | FIRE | FIRE | INTERP: 600-1500ms = "deliberative" (neutral), not "delayed" | FIRE |
| 31 | CONVO-DOM-01 | dominance_score | FIRE | FIRE | FIRE | INTERP: interviewer structural dominance is expected | INTERP: guest quantitative dominance is expected |
| 32 | CONVO-INT-01 | interruption_pattern | FIRE | FIRE | FIRE | FIRE | FIRE |
| 33 | CONVO-RAP-01 | rapport_indicator | FIRE | FIRE | FIRE | FIRE | FIRE |
| 34 | CONVO-ENG-01 | conversation_engagement | FIRE | FIRE | FIRE | FIRE | FIRE |
| 35 | CONVO-BAL-01 | conversation_balance | FIRE | FIRE | FIRE | FIRE+ compare to expected Gini 0.20-0.40 | FIRE+ compare to expected Gini 0.30-0.50 |
| 36 | CONVO-CONF-01 | conflict_score | FIRE | FIRE | FIRE | FIRE | GATE (podcast disagreement = content, not conflict) |

**Conversation summary:** 6 of 8 rules are fully universal. BAL-01 needs expected-Gini comparison for asymmetric conversation types. LAT-01 and DOM-01 need interpretation changes for interviews (deliberative latency is normal; interviewer dominance is expected). Detection thresholds are identical across all types.

### Fusion Agent (Audio) — 6 Rules × 5 Content Types

| # | Rule | Signal | Sales | Client Mtg | Internal | Interview | Podcast |
|---|------|--------|-------|-----------|----------|-----------|---------|
| 37 | FUSION-02 | stress_sentiment_incongruence | FIRE (cap 0.65) | FIRE | FIRE | FIRE | FIRE |
| 38 | FUSION-07 | verbal_incongruence | FIRE (cap 0.70) | FIRE | FIRE | FIRE | FIRE |
| 39 | FUSION-13 | urgency_authenticity | FIRE (cap 0.60) | FIRE (presentations) | GATE (no persuasion context) | GATE | GATE |
| 40 | FUSION-GRAPH-01 | tension_cluster | FIRE | FIRE | FIRE | FIRE | FIRE |
| 41 | FUSION-GRAPH-02 | momentum_shift | FIRE | FIRE | FIRE | FIRE | FIRE |
| 42 | FUSION-GRAPH-03 | persistent_incongruence | FIRE | FIRE | FIRE | FIRE | FIRE |

**Fusion (audio) summary:** 5 of 6 rules are fully universal. Only FUSION-13 (urgency authenticity) needs gating — it measures persuasion+pace mismatch, which is irrelevant in internal meetings, interviews, and podcasts where no one is selling. FUSION-02 and FUSION-07 measure congruence/incongruence, which is universally meaningful regardless of context.

### Facial Agent — 7 Rules × 5 Content Types

| # | Rule | Signal | Sales | Client Mtg | Internal | Interview | Podcast |
|---|------|--------|-------|-----------|----------|-----------|---------|
| 43 | FACE-CAL-01 | facial_baseline | FIRE | FIRE | FIRE | FIRE+ extend to 90-120s (greeting phase longer in formal contexts) | N/A (audio) |
| 44 | FACE-EMO-01 | primary_emotion | FIRE | FIRE | FIRE | FIRE | N/A |
| 45 | FACE-SMILE-01 | smile_quality | FIRE | FIRE | FIRE | FIRE | N/A |
| 46 | FACE-MICRO-01 | DISABLED at ≤30fps | DISABLED | DISABLED | DISABLED | DISABLED | N/A |
| 47 | FACE-ENG-01 | engagement_visual | FIRE | FIRE | FIRE | FIRE | N/A |
| 48 | FACE-STRESS-01 | stress_visual | FIRE | FIRE | FIRE | FIRE | N/A |
| 49 | FACE-VA-01 | valence_arousal | FIRE | FIRE | FIRE | FIRE | N/A |

**Facial summary:** All rules are universal across video content types. MICRO-01 is disabled for ALL types at webcam framerates (research finding, not content-type decision). CAL-01 extends baseline window for formal contexts where greeting phase is longer. No facial rule needs content-type-specific thresholds — the per-speaker facial baseline handles all variation. Podcast is N/A for all facial rules unless video podcast (then treat as internal meeting profile).

### Body Agent — 8 Rules × 5 Content Types

| # | Rule | Signal | Sales | Client Mtg | Internal | Interview | Podcast |
|---|------|--------|-------|-----------|----------|-----------|---------|
| 50 | BODY-CAL-01 | body_baseline | FIRE | FIRE | FIRE | FIRE | N/A |
| 51 | BODY-POST-01 | posture_score | FIRE | FIRE | FIRE | FIRE | N/A |
| 52 | BODY-HEAD-01 | head_nod_shake | FIRE | FIRE | FIRE | FIRE | N/A |
| 53 | BODY-LEAN-01 | leaning_direction | FIRE | FIRE | FIRE | FIRE | N/A |
| 54 | BODY-GEST-01 | gesture_type | FIRE | FIRE | FIRE | FIRE | N/A |
| 55 | BODY-FIDG-01 | fidget_rate | FIRE | FIRE | FIRE | FIRE | N/A |
| 56 | BODY-TOUCH-01 | self_touch | FIRE | FIRE | FIRE | FIRE | N/A |
| 57 | BODY-MIRROR-01 | mirroring (EXPERIMENTAL) | FIRE (0.20-0.30) | FIRE | FIRE | FIRE | N/A |

**Body summary:** ALL 8 rules are fully universal across video content types. No content-type-specific thresholds needed. The per-speaker body baseline (CAL-01) handles individual posture differences. Head nod/shake (HEAD-01) is the highest-reliability body signal and is universally meaningful — a nod while saying "no" means the same thing regardless of whether it's a sales call or interview. Fidgeting interpretation (anxiety vs boredom vs excitement) depends on co-occurring signals, not content type. Podcast is N/A unless video.

### Gaze Agent — 7 Rules × 5 Content Types

| # | Rule | Signal | Sales | Client Mtg | Internal | Interview | Podcast |
|---|------|--------|-------|-----------|----------|-----------|---------|
| 58 | GAZE-CAL-01 | gaze_baseline | FIRE | FIRE | FIRE | FIRE | N/A |
| 59 | GAZE-DIR-01 | gaze_direction | FIRE | FIRE | FIRE | FIRE | N/A |
| 60 | GAZE-CONTACT-01 | screen_engagement | FIRE | FIRE | FIRE | FIRE | N/A |
| 61 | GAZE-BLINK-01 | blink_rate | FIRE | FIRE | FIRE | FIRE | N/A |
| 62 | GAZE-ATT-01 | attention_score | FIRE | FIRE | FIRE | FIRE | N/A |
| 63 | GAZE-DIST-01 | distraction_count | FIRE | FIRE | FIRE | FIRE | N/A |
| 64 | GAZE-SYNC-01 | gaze_alignment (EXPERIMENTAL) | FIRE | FIRE | FIRE | FIRE | N/A |

**Gaze summary:** ALL 7 rules are fully universal across video content types. The per-speaker gaze baseline (CAL-01) handles camera position differences and individual gaze patterns. The critical threshold fixes (BLINK-01 >30-35bpm, DIST-01 >8-10s) apply universally — they were wrong for ALL types, not just specific ones. Screen engagement norms (CONTACT-01) are already baseline-relative, so they auto-adapt. Podcast is N/A unless video.

### Fusion Pairwise (Video) — 13 Rules × 5 Content Types

| # | Rule | Cross-Modal | Sales | Client Mtg | Internal | Interview | Podcast |
|---|------|------------|-------|-----------|----------|-----------|---------|
| 65 | FUSION-01 | Tone × Face → Masking | FIRE (0.75) | FIRE | FIRE | FIRE | N/A |
| 66 | FUSION-03 | Posture × Energy → Enthusiasm | FIRE (0.65) | FIRE | FIRE | FIRE | N/A |
| 67 | FUSION-04 | Gaze × Filler → Cognitive Effort | FIRE (0.70) | FIRE | FIRE | FIRE | N/A |
| 68 | FUSION-05 | Buy × Body → Purchase Intent | FIRE (0.55-0.60) | RENAME → "engagement_verification" | GATE | RENAME → "interest_verification" | N/A |
| 69 | FUSION-06 | DISABLED (ME infeasible) | DISABLED | DISABLED | DISABLED | DISABLED | N/A |
| 70 | FUSION-07* | Head Nod × Speech → Disagree | FIRE (0.65) | FIRE | FIRE | FIRE | N/A |
| 71 | FUSION-08 | Eye × Hedge → False Confidence | FIRE (0.55) | FIRE | FIRE | FIRE | N/A |
| 72 | FUSION-09 | Smile × Sentiment → Sarcasm | FIRE (0.60) | FIRE | FIRE | FIRE | N/A |
| 73 | FUSION-10 | Latency × Face → Cognitive Load | FIRE (0.60) | FIRE | FIRE | INTERP: cognitive_load = thinking (neutral), not withholding | N/A |
| 74 | FUSION-11 | Dominance × Gaze → Anxious | FIRE (0.65) | FIRE | FIRE | FIRE | N/A |
| 75 | FUSION-12 | Interrupt × Body → Intent | FIRE (0.55) | FIRE | FIRE | FIRE | N/A |
| 76 | FUSION-14 | Empathy × Nod → Rapport | FIRE (0.70) | FIRE | FIRE | FIRE | N/A |
| 77 | FUSION-15 | MERGED into FUSION-04 | — | — | — | — | — |

**Fusion (video) summary:** 10 of 13 rules (including merged F-15) are fully universal. Cross-modal congruence/incongruence detection is content-agnostic — a smile while saying something negative means the same thing whether it's a sales call or interview. Only FUSION-05 (purchase intent) needs renaming for non-sales contexts, FUSION-06 is disabled (ME infeasible), and FUSION-10 gets interpretation softening for interviews (cognitive load during complex answers is expected, not suspicious).

### Compound Patterns — 12 Rules × 5 Content Types

| # | Rule | Pattern | Sales | Client Mtg | Internal | Interview | Podcast |
|---|------|---------|-------|-----------|----------|-----------|---------|
| 78 | COMPOUND-01 | Genuine Engagement (0.80) | FIRE | FIRE | FIRE | FIRE | N/A |
| 79 | COMPOUND-02 | Active Disengagement (0.75) | FIRE | FIRE | FIRE | FIRE | N/A |
| 80 | COMPOUND-03 | Emotional Suppression (0.70) | FIRE | FIRE | FIRE | FIRE | N/A |
| 81 | COMPOUND-04 | Decision Engagement (0.65, EXPERIMENTAL) | FIRE | RENAME → "commitment_signals" | RENAME → "consensus_signals" | RENAME → "decision_signals" | N/A |
| 82 | COMPOUND-05 | Cognitive Overload (0.75) | FIRE | FIRE | FIRE | INTERP: overload on technical Qs = difficulty level signal, not weakness | N/A |
| 83 | COMPOUND-06 | Conflict Escalation (0.80) | FIRE | FIRE | FIRE | FIRE | N/A |
| 84 | COMPOUND-07 | Verbal-Nonverbal Discordance (0.65) | FIRE | FIRE | FIRE | FIRE | N/A |
| 85 | COMPOUND-08 | Rapport Peak (0.80) | FIRE | FIRE | FIRE | FIRE | N/A |
| 86 | COMPOUND-09 | Topic Avoidance (0.70) | FIRE | FIRE | FIRE | INTERP: avoidance = possible weakness area (coaching note, not red flag) | N/A |
| 87 | COMPOUND-10 | Authentic Confidence (0.75) | FIRE | FIRE | FIRE | FIRE | N/A |
| 88 | COMPOUND-11 | Anxiety Performance (0.65) | FIRE | FIRE | FIRE | FIRE (core interview diagnostic) | N/A |
| 89 | COMPOUND-12 | Deception Risk (0.55 hard cap) | FIRE | FIRE | FIRE | FIRE | N/A |

**Compound summary:** 9 of 12 rules are fully universal. Multi-domain patterns are inherently content-agnostic — genuine engagement looks the same regardless of conversation type because it's detected from convergent signals across 3+ domains. Only COMPOUND-04 needs relabeling (the "readiness" construct applies differently), COMPOUND-05 needs interpretation softening for interviews, and COMPOUND-09 gets a coaching-oriented interpretation for interviews. No detection threshold changes per type.

**Note on COMPOUND-12 (Deception Risk) for interviews:** The research (DePaulo 2003, Bogaard 2025) shows interview stress is indistinguishable from deception signals. However, the 0.55 hard cap + 4-domain + 3-fusion-conflict requirement already provides sufficient false-positive protection. The rule fires so rarely (~0-1 times per session) that gating it for interviews would remove a legitimate (if rare) signal. Keep FIRE for all types but ensure the narrative report contextualizes: "This signal indicates behavioral inconsistency across multiple channels. In high-stakes settings like interviews, elevated arousal can produce similar patterns without deceptive intent."

### Temporal Sequences — 8 Rules × 5 Content Types

| # | Rule | Sequence | Sales | Client Mtg | Internal | Interview | Podcast |
|---|------|----------|-------|-----------|----------|-----------|---------|
| 90 | TEMPORAL-01 | Stress Cascade (0.65) | FIRE | FIRE | FIRE | FIRE | N/A |
| 91 | TEMPORAL-02 | Engagement Build (0.65) | FIRE | FIRE | FIRE | FIRE | N/A |
| 92 | TEMPORAL-03 | Disengage Cascade (0.55) | FIRE | FIRE | FIRE | FIRE | N/A |
| 93 | TEMPORAL-04 | Objection Formation (0.60) | FIRE | FIRE | FIRE | FIRE | N/A |
| 94 | TEMPORAL-05 | Trust Repair (0.50-0.55, EXPERIMENTAL) | FIRE | FIRE | FIRE | INTERP: interviewer rebuilding rapport after tough question = good technique | N/A |
| 95 | TEMPORAL-06 | Decision Engagement (0.45-0.50) | FIRE | RENAME → "commitment_progression" | GATE | RENAME → "decision_progression" | N/A |
| 96 | TEMPORAL-07 | Dominance Shift (0.55) | FIRE | FIRE | FIRE | INTERP: shift toward candidate = growing confidence (positive trajectory) | FIRE (audio) |
| 97 | TEMPORAL-08 | Behavioral Consistency (0.55) | FIRE | FIRE | FIRE | FIRE | FIRE (audio) |

**Temporal summary:** 6 of 8 rules are fully universal. Temporal sequences are content-agnostic by nature — a stress cascade (voice→body→face) unfolds the same way regardless of whether it's triggered by a tough sales objection or a difficult interview question. Only TEMPORAL-06 needs renaming/gating (buying decision sequence is sales-specific), and TEMPORAL-05/07 get interview-specific interpretation notes in the narrative report.

---

## Content-Type Adaptation Summary Across All 97 Rules

| Content Type | Rules FIRE (no change) | FIRE+ (small threshold tweak) | RENAME (label change) | GATE (suppressed) | INTERP (narrative change) | N/A (no video) |
|-------------|----------------------|------------------------------|----------------------|-------------------|--------------------------|----------------|
| **Sales Call** | 91 | 1 (TALK-01) | 0 | 0 | 0 | 0 |
| **Client Meeting** | 82 | 0 | 5 (BUY, OBJ, PERS, C-04, T-06) | 1 (FUSION-05 is renamed not gated) | 0 | 0 |
| **Internal Meeting** | 79 | 3 (FILLER-02, TALK-01, SENT-02) | 3 (OBJ, PERS, C-04) | 2 (BUY-01, FUSION-13) | 0 | 0 |
| **Interview** | 72 | 3 (PAUSE-01, TALK-01, FACE-CAL) | 4 (BUY, OBJ, PERS, T-06) | 0 | 7 (PWR, EMP, LAT, DOM, F-10, C-05, C-09) | 0 |
| **Podcast** | 52 | 1 (TALK-01) | 0 | 5 (BUY, OBJ, PERS, NEG, CONF) | 0 | 30 (all video rules N/A for audio podcast) |

**Key insight:** Sales calls need essentially zero adaptation (1 sub-threshold addition). Client meetings and internal meetings need 5-8 changes each, mostly label renames. Interviews need the most interpretation changes (7 INTERP notes) but still only 3 threshold tweaks. Podcasts need the most gating (5 rules irrelevant) but 30 rules are simply N/A because podcasts are audio-only.

**The per-speaker baseline is the primary adaptation mechanism.** An interview candidate's naturally elevated stress, faster speech rate, and higher pitch are absorbed into their baseline during the first 90s of speech. The rules then detect deviations from THAT baseline, not from some universal "normal." This is why 72 of 97 rules need zero content-type changes even for the most different conversation type (interviews).

---

## ContentTypeProfile Implementation

One class, loaded per session. ~200 lines of code. No new database tables needed (profiles are hardcoded initially, migrate to DB later).

```python
class ContentTypeProfile:
    def __init__(self, content_type: str):
        self.content_type = content_type
        self._profile = PROFILES.get(content_type, PROFILES["sales_call"])
    
    def is_gated(self, rule_id: str) -> bool:
        """Should this rule be suppressed for this content type?"""
        return rule_id in self._profile.get("gated_rules", set())
    
    def rename_signal(self, signal_type: str) -> str:
        """Get content-type-specific label for this signal."""
        return self._profile.get("renames", {}).get(signal_type, signal_type)
    
    def get_threshold_override(self, rule_id: str, param: str, default):
        """Get a per-type threshold override, or return the universal default."""
        return self._profile.get("thresholds", {}).get(rule_id, {}).get(param, default)
    
    def get_prompt_template(self, agent: str) -> str:
        """Get LLM prompt template for this content type."""
        return self._profile.get("prompts", {}).get(agent, "")
    
    def get_interpretation_note(self, rule_id: str) -> str:
        """Get narrative interpretation guidance for this content type."""
        return self._profile.get("interpretations", {}).get(rule_id, "")
    
    def get_expected_balance(self) -> dict:
        """Expected talk-time ratios that should NOT be flagged."""
        return self._profile.get("expected_balance", {})
```

**Profile definitions (5 types):**

```python
PROFILES = {
    "sales_call": {
        # Baseline — no gating, no renames, no interpretation changes
        "gated_rules": set(),
        "renames": {},
        "thresholds": {"VOICE-TALK-01": {"seller_significant": 0.60}},
        "expected_balance": {"seller": (0.35, 0.65)},
    },
    "client_meeting": {
        "gated_rules": set(),
        "renames": {
            "buying_signal": "engagement_signal",
            "objection_signal": "concern",
            "persuasion_technique": "influence_tactic",
        },
        "thresholds": {},
        "expected_balance": {"presenter": (0.50, 0.75)},
    },
    "internal": {
        "gated_rules": {"LANG-BUY-01", "FUSION-13"},
        "renames": {
            "objection_signal": "disagreement",
            "persuasion_technique": "influence_attempt",
        },
        "thresholds": {
            "VOICE-FILLER-02": {"noticeable_pct": 3.0},
            "LANG-SENT-02": {"high_pct": 0.06, "suppressed_pct": 0.015},
            "VOICE-TALK-01": {"peer_significant": 0.50},
        },
        "expected_balance": {},  # Peer meetings expect symmetry
    },
    "interview": {
        "gated_rules": set(),
        "renames": {
            "buying_signal": "interest_signal",
            "objection_signal": "hesitation",
            "persuasion_technique": "impression_management",
        },
        "thresholds": {
            "VOICE-PAUSE-01": {"extended_hesitation_ms": 3000},
            "VOICE-TALK-01": {"candidate_expected": (0.30, 0.70)},
        },
        "interpretations": {
            "LANG-PWR-01": "Powerful speech from candidate = competence signal. Powerless speech = note for coaching, not red flag.",
            "LANG-EMP-01": "Interviewer empathy = good technique. Candidate empathy = soft skill indicator.",
            "CONVO-LAT-01": "600-1500ms latency = deliberative (neutral). Complex answers need thinking time.",
            "CONVO-DOM-01": "Interviewer structural dominance is expected. Flag only if candidate <30% or >70%.",
            "FUSION-10": "Cognitive load = question difficulty signal, not withholding.",
            "COMPOUND-05": "Cognitive overload on technical questions = difficulty level signal.",
            "COMPOUND-09": "Topic avoidance = possible weakness area. Coaching note, not red flag.",
            "TEMPORAL-05": "Interviewer rebuilding rapport after tough question = good interview technique.",
            "TEMPORAL-07": "Dominance shift toward candidate = growing confidence (positive trajectory).",
        },
        "expected_balance": {"interviewer": (0.30, 0.55), "candidate": (0.40, 0.70)},
    },
    "podcast": {
        "gated_rules": {"LANG-BUY-01", "LANG-OBJ-01", "LANG-PERS-01", "LANG-NEG-01", "CONVO-CONF-01", "TEMPORAL-06"},
        "renames": {},
        "thresholds": {
            "VOICE-TALK-01": {"guest_expected": (0.60, 0.80)},
        },
        "expected_balance": {"host": (0.20, 0.50), "guest": (0.50, 0.80)},
    },
}
```

**Integration point:** Each agent's `evaluate()` method receives the profile object. Before running each rule, check `profile.is_gated(rule_id)`. After producing a signal, call `profile.rename_signal(signal_type)`. The Fusion narrative generator uses `profile.get_interpretation_note(rule_id)` to contextualize its report. Zero additional LLM calls — only prompt content changes for INTENT-01.