# NEXUS 42-rule academic validation: research findings and threshold recommendations

**Of the 42 implemented detection rules, 28 require some form of change — but only 11 need threshold modifications to their core detection logic.** The remaining 17 changes involve interpretation adjustments, gating expansions, or labeling refinements rather than fundamental threshold rewrites. The baseline-relative architecture is strongly validated: Łachut (2025) demonstrated that per-speaker calibration improved emotion recognition F1 from **0.619 to 0.753**, confirming that NEXUS's deviation-from-baseline approach auto-adapts across speaker types and partially across conversation types. The most significant finding across all rules is that content-type adaptation should primarily occur at the **interpretation layer**, not the detection layer — the system should detect universally but interpret contextually.

---

## Voice Agent: 16 rules, 7 changes needed

### VOICE-STRESS-01 — Composite stress score weights

**Best evidence:** Veiga et al. (2025), "The Fundamental Frequency of Voice as a Potential Stress Biomarker," *Stress and Health* — meta-analysis of 10 studies, 1,148 observations, real stress (not acted). Also Kappen et al. (2022/2024), *Scientific Reports* — within-participant designs with physiological validation.

The Veiga meta-analysis confirmed F0 elevation as the single most reliable stress biomarker (**SMD = 0.55, p < 0.001**). The PLOS ONE (2025) systematic review of 38 articles identified F0 and intensity as the two most consistently validated indicators. Kappen's network analysis found jitter as the most central node connecting speech to self-reported affect, but jitter's *direction* of change is inconsistent across studies — some find increase, others decrease under stress. Filler rate lacks direct empirical validation as a stress component in the core psychoacoustic literature.

**Recommendation: YES CHANGE weights.** Increase F0 elevation from 0.25 → **0.30** (strongest evidence). Decrease filler rate from 0.15 → **0.10** (weak empirical basis). Retain all other weights. Consider adding vocal intensity as a candidate feature at 0.10 weight, given PLOS ONE 2025 identifies it alongside F0 as the most reliable stress marker. Ensure jitter detection is bidirectional — jitter *decrease* may also indicate stress. The four-tier thresholds (0.3/0.5/0.7) are engineering decisions without direct composite validation but are reasonable as classification heuristics.

**Content-type adaptation:** The per-speaker baseline handles most variation. For interviews, extend baseline stabilization window to 5+ minutes, as Kappen (2024) showed social-evaluative stress produces different vocal patterns than cognitive stress.

### VOICE-FILLER-01 — Filler rate spike detection (+50% above baseline)

**Best evidence:** Bortfeld et al. (2001), *Language and Speech* — 96 speakers in real task-oriented conversations. Found an **82% increase** in filler rate simply from switching conversational roles (director vs. matcher). Topic difficulty alone causes 30–80% increases.

The +50% threshold sits well within the range of meaningful cognitive-load-induced changes while avoiding false positives from minor fluctuation. Laserna, Seih & Pennebaker (2014) analyzed 263 natural daily conversations and found that filled pauses (uh/um) and discourse markers (like, you know) have different demographic profiles and should ideally be tracked separately.

**Recommendation: UNIVERSAL — no change needed.** The +50% spike threshold is conservatively appropriate. The baseline mechanism handles cross-context variation. One enhancement worth considering: separate filled pauses from discourse markers per Laserna (2014), as they reflect different cognitive processes.

### VOICE-FILLER-02 — Absolute filler credibility thresholds

**Best evidence:** Bortfeld et al. (2001) and Christenfeld (1995), *Journal of Nonverbal Behavior*. Normal filled pause rate (uh/um only) ranges from **1.3 to 4.4 per 100 words** across corpora. Total disfluency rate including discourse markers reaches 4–6% of all words.

**The 1.3% "noticeable" threshold is at the floor of normal speech.** This means most normal speakers would be flagged. Christenfeld (1995) found listeners are profoundly bad at estimating filler frequency — they guessed 22.1 ums in a recording containing zero — but listeners do perceive heavy filler use negatively (uncomfortable, inarticulate, nervous). The current ±0.15 content-type adjustment is directionally correct but too small; Bortfeld found ~1 percentage point differences between easy and hard topics for fillers alone.

**Recommendation: YES CHANGE.** For filled pauses only (uh/um): raise to **>2.5% noticeable, >4.0% significant, >6.0% severe**. If counting all fillers (uh, um, like, you know): **>5.0%/7.0%/10.0%**. Increase content-type adjustments from ±0.15 to **±0.5 percentage points** or use multiplicative scaling (×0.8 for formal, ×1.3 for informal).

### VOICE-TONE-01 through 06 — Six emotion acoustic profiles

**Best evidence:** Juslin & Laukka (2003), *Psychological Bulletin* — meta-analysis of 104 studies, remains the single most comprehensive synthesis. Validated by Cespedes-Guevara & Eerola (2018) who analyzed 82 post-J&L studies and confirmed the profiles "in general coincide." Juslin, Laukka & Bänziger (2018) compared 1,877 voice clips across 23 datasets and found spontaneous expressions produce the **same directional patterns as acted but at lower magnitudes**.

The six NEXUS emotions map as follows to the research literature, with confidence levels:

- **Aggressive** (Anger): VERY HIGH confidence — strongest-validated profile across all literature. High F0, high variability, fast rate, high intensity, tense voice. Note: authentic irritation may show *slower* rate than acted anger per Laukka et al. (2011).
- **Nervous** (Fear/Anxiety): HIGH confidence — high F0, irregular variability, tense voice, high jitter, frequent hesitations. Add filled-pause frequency and voice-break rate as secondary discriminators.
- **Excited** (Happiness/Joy): HIGH confidence — high F0, wide variability, fast rate, high intensity. Kamiloğlu et al. (2020) clarified separation from low-arousal positive emotions. Add HNR and spectral tilt to distinguish from aggressive (excitement = more breathy/open).
- **Warm** (Tenderness): HIGH confidence — low F0, low variability, slow rate, soft intensity, breathy quality. Nussbaum et al. (2024) found timbre plays a more central role than previously appreciated for warmth recognition. Add HNR and spectral slope as secondary features.
- **Confident** (Dominance/Potency): MODERATE confidence — inferred from dimensional models, not directly validated as a discrete emotion. Low-moderate F0, steady variability, moderate-slow rate, high intensity, full resonant voice.
- **Cold** (Contempt): LOW-MODERATE confidence — primarily sourced from Banse & Scherer (1996); underrepresented in SER research. Flat F0, low variability, deliberate pace, controlled intensity, precise articulation.

**Recommendation: NO CHANGE to core profiles.** Juslin & Laukka 2003 remains the best available reference. The baseline-relative approach compensates for the main weakness (acted-speech calibration) by measuring deviations rather than absolutes. Minor enhancements: add voice quality features (HNR, spectral slope) for warm/excited differentiation; flag Cold as lowest-confidence profile. No content-type gating needed — baseline adaptation handles context differences.

### VOICE-PITCH-01 — Pitch elevation thresholds

**Best evidence:** Veiga et al. (2025) meta-analysis — **SMD = 0.55** overall, but with critical moderators. Women showed SMD = **0.71** (significant); men showed **nonsignificant** effects. Spontaneous speech showed SMD = **0.79**; standardized speech showed nonsignificant effects. Publication bias is a serious concern: trim-and-fill correction reduced the overall effect to a nonsignificant SMD of 0.17.

Converting to percentages: male baseline F0 ~120Hz with SD ~17Hz gives SMD 0.55 × 17 = ~9.4Hz = **~7.8%** elevation. Female baseline ~200Hz with SD ~30Hz gives SMD 0.71 × 30 = ~21Hz = **~10.7%**. The current 8% "mild" threshold captures approximately the average stress effect, which is appropriate for a detection system.

**Recommendation: YES CHANGE.** Lower thresholds to **>7% mild, >12% significant, >20% extreme** (from 8/15/25). The current 15% "significant" threshold may miss many meaningful stress episodes, and 25% "extreme" is overly conservative. Gender-stratified thresholds are strongly warranted: consider male mild **>5%**, female mild **>8%**, given the 2× effect size difference in the meta-analysis. For content types: effects are stronger in spontaneous speech — casual conversations should use more sensitive thresholds than scripted presentations.

### VOICE-PITCH-02 — Monotone detection

**Best evidence:** Juslin & Laukka (2003) confirmed low F0 variability as a consistent marker of low-arousal states. Laukka et al. (2011) validated this in authentic resigned speech from telephone services. Depression literature (BMC Psychiatry 2024) consistently characterizes depression by "monotony and flatness" with F0 range approaching zero.

The 40% variance drop criterion is reasonable. However, the **30Hz absolute range** threshold is problematic: male baseline F0 range is typically ~50–80Hz (so 30Hz = 40–60% reduction) while female range is ~80–120Hz (so 30Hz = 60–75% reduction). This creates a gender-asymmetric detection sensitivity.

**Recommendation: YES CHANGE (minor).** Keep the 40% variance drop. Replace the absolute <30Hz range with a **percentage-based threshold: <40% of speaker's baseline F0 range.** This normalizes for gender differences — for a male with 60Hz baseline range, threshold becomes <24Hz; for a female with 100Hz range, <40Hz. The 10-second window is appropriate.

### VOICE-RATE-01 — Speech rate anomaly

**Best evidence:** Quené (2007), *Journal of Phonetics* — the just-noticeable difference (JND) for speech tempo is approximately **5%**. Normal between-speaker variation ranges 6–17% across dialects (Jacewicz et al., 2009). Stress-related rate changes show heterogeneous directions: anxiety increases rate, disengagement decreases it.

The current ±25% threshold is 5× the JND and would miss many meaningful stress-related changes in the 10–20% range. Normal conversational variation (6–17%) would not be flagged, which is appropriate, but the gap between normal variation and the threshold is too large.

**Recommendation: YES CHANGE.** Lower to **±20%** or implement a two-tier system: **±15% notable, ±25% significant**. Track direction of change — faster signals arousal/anxiety, slower signals disengagement/depression.

**Content-type adaptation:** UNIVERSAL — no change needed. Per-speaker baselines handle context-specific rate norms.

### VOICE-ENERGY-01 — Energy level threshold

**Best evidence:** Bänziger & Scherer (2005) and PLOS ONE (2025) systematic review confirmed intensity alongside F0 as the two most reliable vocal stress/emotion markers. Psychoacoustically, 6dB represents a clear, unambiguous loudness shift (3–6× the JND of 1–2dB). Normal conversational dynamic range spans ~20–30dB, making 6dB approximately 20–30% of that range.

**Recommendation: UNIVERSAL — no change needed.** The ±6dB threshold is well-calibrated: clearly meaningful, far above the JND, and captures substantial emotional/stress shifts without excessive false positives. Consider adding a secondary ±3dB "notable" tier for finer granularity.

### VOICE-PAUSE-01 — Pause classification

**Best evidence:** Heldner & Edlund (2010), *Journal of Phonetics* — multiple conversational corpora. Modern research converges on **200ms** (Laver 1994; Heldner & Edlund use 180ms) rather than Goldman-Eisler's 1968 threshold of 250ms. Hieke et al. (1983) showed short pauses of 130–250ms are psychologically functional, not merely articulatory. Normal pause ratio in conversation is **40–50%** (Goldman-Eisler; Yang 2004). Clinical data shows high-symptom groups at 59%.

**Recommendation: YES CHANGE (hesitation threshold only).**

- Hesitation threshold: lower from >250ms to **>200ms** per modern consensus
- Pause ratio >55%: **NO CHANGE** — well-positioned above the 40–50% normal range
- Extended pause >2000ms: **MINOR CHANGE** — add context modifiers: **>3000ms for interviews** (complex thinking pauses are normal), keep 2000ms for other types

### VOICE-INT-01 — Interruption detection

**Best evidence:** Heldner & Edlund (2010) found ~40% of between-speaker intervals are overlaps — overlap is extremely common and mostly cooperative. Heldner (2011), *JASA*, established perceptual detection threshold for overlaps at **~120ms**. Zimmerman & West (1975) distinguished interruptions by *location* relative to transition-relevance places, not by duration. Call center research (2024) filtered overlaps <1 second as likely non-competitive.

The current 200ms threshold captures too many cooperative overlaps, simultaneous starts, and normal turn transitions. Research shows the modal overlap is <100ms, and overlaps up to ~500ms are common in collaborative speech. The ≤2 word backchannel filter misses phrasal backchannels like "I see," "that's right," "oh yeah" (2–3 words).

**Recommendation: YES CHANGE.** Raise overlap threshold from >200ms to **>500ms** for interruption classification. Raise backchannel filter from ≤2 to **≤3 words**. Consider 750–1000ms for higher-confidence interruption detection per call center research. Tannen (1990) documented "cooperative overlap" as culturally mediated — add cultural context awareness if feasible.

### VOICE-TALK-01 — Talk time ratio

**Best evidence:** Gong.io analysis of **25,000+ real sales calls** — optimal seller talk ratio is **43:57**. Low performers swing from 54% (wins) to 64% (losses). Demodesk (2025) analysis of 328 B2B meetings found discovery calls optimal at 40–60% seller talk, demos at 60–70%.

The 70%/80% thresholds are defensible as universal imbalance flags (70% is well above even worst-performing sellers at ~64%). However, for sales calls specifically, research clearly shows outcomes deteriorate above 60% seller talk time.

**Recommendation: YES CHANGE — add content-type sub-thresholds.** For sales calls: **>60% significant imbalance, >70% extreme** (seller side). For interviews: interviewer >60% OR interviewee >80% = significant. For presentations: raise to 80%/90% (presenter dominance expected). Keep universal 70%/80% for general meetings. Multi-person >2× expected share: **NO CHANGE**.

---

## Language Agent: 12 rules, 10 changes needed

### LANG-SENT-01 — Sentiment hybrid weighting

**Best evidence:** Tausczik & Pennebaker (2010) acknowledge LIWC "ignores context, irony, sarcasm, and idioms." SEANCE (Crossley et al., 2017) outperformed LIWC for sentiment by handling negation and POS tags — features LIWC completely lacks. Transformer models achieve **85–95%** sentiment accuracy; hybrid transformer-ML models reach 95.3%. No published research validates a 0.7/0.3 DistilBERT-LIWC weighting.

**Recommendation: YES CHANGE.** Increase transformer weight to **0.85–0.90**, reduce LIWC to **0.10–0.15**. Retain LIWC for its unique psychological dimensions (Authenticity, Clout, Emotional Tone) rather than direct sentiment contribution. Consider fine-tuning DistilBERT on conversational/spoken language data — models trained on text reviews don't capture speech disfluencies well.

### LANG-SENT-02 — Emotion word density thresholds

**Best evidence:** Tausczik & Pennebaker (2010) report **4.0% emotion words** as the average across 100M+ words (positive ~2.7%, negative ~1.3%). Sandler et al. (2024) found human dialogues at ~3.85% total emotion words. Professional speech likely runs lower (3–3.5%).

The current <2% "suppressed" threshold is defensible at half the average. However, calling 4–8% "moderate" when the average *is* 4% means half of normal speech gets classified as moderate-to-high.

**Recommendation: YES CHANGE — add content-type calibration.** For professional speech: **>6% high, 3–6% moderate, <1.5% suppressed**. For personal/emotional conversations: retain current >8%/4–8%/<2%.

### LANG-BUY-01 — Buying signals (SPIN patterns)

**Best evidence:** Rackham (1988), Huthwaite Corporation — 12-year study analyzing **35,000+ sales calls** across 20+ countries. Pricing, implementation, and timeline questions are consistently validated as strong buying signals across industry research.

The same detection patterns transfer beyond sales. In client meetings, cost/process/timing questions indicate deepening engagement. In interviews, candidate questions about start dates, team structure, and compensation signal strong interest.

**Recommendation: YES CHANGE — expand gating.** Convert from sales-gated to **UNIVERSAL WITH DIFFERENT LABELS**: sales = "buying_signal" (current), client meetings = "engagement_signal," interviews = "candidate_interest_signal." Keep SPIN classification itself sales-gated (the S→P→I→N progression is sales-specific), but the underlying pattern detection should run universally.

### LANG-OBJ-01 — Objection detection

**Best evidence:** Brill (2021), *Contrastive Pragmatics* — **53% of contrastive "but" uses** co-occurred with hedging in English spoken corpora. Brown & Levinson (1987) established that expressing disagreement is inherently face-threatening, making hedging a universal remedial strategy. These patterns appear across academic discourse, political interviews, business meetings, and informal conversations.

**Recommendation: YES CHANGE — expand to universal with relabeling.** Sales = "objection," client meetings = "concern/reservation," internal meetings = "disagreement/pushback," interviews = "hesitation/reservation." No threshold change needed — hedge + contrastive marker detection is well-validated universally.

### LANG-PWR-01 — Power language score

**Best evidence:** O'Barr & Atkins (1980) confirmed powerless features are status-related, not gender-related. However, modern research fundamentally challenges uniform interpretation: Jensen (2008), *Human Communication Research*, found hedging in scientific communication **increased** perceived trustworthiness. Piškorec showed hedges serve three functions — unintended powerlessness, strategic softening, and appropriate fuzziness. The Hutton Inquiry research found high-status witnesses using "powerless" features as deliberate power-negotiation tools.

**Recommendation: YES CHANGE — implement context-dependent scoring.** Separate hesitations (um, uh — consistently negative for competence perception) from cognitive hedges ("I think," "sort of" — mixed effects). In interviews, appropriate hedging should receive neutral-to-positive scoring. In sales, hedging may rightly be flagged. Weight hesitation features consistently but apply context-variable interpretation to deliberative hedges and tag questions.

### LANG-PERS-01 — Persuasion technique detection (Cialdini)

**Best evidence:** Cialdini himself states the principles operate in "politics, organizational dynamics, personal relationships, education, and public health." AACSB critique by Miller (2022) confirmed core mechanisms transfer to workplace contexts but labels/framing should adapt. Harvard Program on Negotiation documents active use in negotiations.

**Recommendation: YES CHANGE — expand gating.** From sales-only to **universal with context-appropriate labels**: sales = "persuasion_technique," client meetings = "influence_tactic," internal meetings = "influence_attempt" (lower-stakes framing), interviews = "impression_management." The 7 Cialdini principles are universal — only output labels change.

### LANG-QUES-01 — Question type classification

**Best evidence:** Open/closed question classification is a fundamental, universal distinction validated across medical interviews, UX research, coaching, and education. SPIN taxonomy (Rackham 1988) was specifically designed for and validated on sales conversations. No research validates SPIN as a classification system outside sales/consulting.

**Recommendation: YES CHANGE — split the rule.** Open/closed/tag → **UNIVERSAL** (all content types, same labels). SPIN classification → **GATED to sales_call** only. This is a clean architectural separation.

### LANG-NEG-01 — Gottman Four Horsemen in workplace contexts

**Best evidence:** Gottman (1993/1994) — predicts divorce with **~91% accuracy** in romantic relationships. However, **no peer-reviewed validation exists for professional contexts**. Workplace applications are entirely practitioner-driven (consulting blogs, training materials). Stonewalling in a marriage signals emotional withdrawal; in a workplace, it may indicate appropriate boundary-setting. Better workplace-validated frameworks exist: Thomas-Kilmann Conflict Mode Instrument, Cortina's workplace incivility research.

**Recommendation: YES CHANGE.** Retain **criticism and contempt** detection — these have strong face validity and overlap with workplace incivility research (Cortina et al.). **Reframe defensiveness** as "resistance/disagreement" in professional contexts (legitimate disagreement ≠ defensive reactivity). **Reframe stonewalling** as "disengagement" with context-dependent interpretation (may be adaptive in toxic meeting dynamics). Add workplace-validated constructs: passive-aggressive markers, professional disrespect indicators. Content-type adaptation is essential — hierarchical context (peer vs. manager-subordinate) changes interpretation significantly.

### LANG-EMP-01 — Empathy language detection

**Best evidence:** Sharma et al. (2020) developed the EPITOME framework identifying **three** empathy dimensions — Emotional Reaction, Interpretation, and Exploration — which is more nuanced than the current binary validation + reflection approach. Lee et al. (2023, EACL) found empathy identification systems "are not accurately accounting for context." EMNLP 2024 showed empathy is a collaborative practice requiring dyadic analysis.

**Recommendation: YES CHANGE.** Expand from two to **three dimensions** per EPITOME. Move beyond surface-level phrase matching to contextual analysis — the same phrase ("I understand") can be genuinely empathic or dismissive depending on conversational flow. Content-type adaptation needed: empathy expression differs in peer, manager-subordinate, and client-facing contexts.

### LANG-CLAR-01 — Clarity score thresholds

**Best evidence:** Biber et al. (1999), *Longman Grammar of Spoken and Written English* — the definitive corpus analysis. Average sentence length in spoken language is **12–20 words**, far shorter than written text. Chafe (1982) found passives occur approximately **5× more** in written than spoken corpora. Estimated passive voice rate in casual conversation: **~2–5%** of clauses.

The current >25 words/sentence threshold is too lenient for spoken language, and the >30% passive threshold is essentially a dead rule — virtually no conversational speaker would approach this rate.

**Recommendation: YES CHANGE.** For spoken language: lower sentence length to **>18 words** (roughly 1 SD above conversational mean). Lower passive voice to **>10%** (generous for speech — anything above 5% suggests overly formal or prepared language). Consider replacing raw sentence length with utterance complexity metrics (clause depth, subordination rate) which are more meaningful for speech clarity.

### LANG-INTENT-01 — LLM intent classification

**Best evidence:** Stolcke et al. (2000) Switchboard-DAMSL tagset confirmed that "content- and task-related distinctions will always play an important role in effective DA labeling." Voiceflow (2024) tested 500+ prompt variations and found prompt engineering produces significant accuracy improvements (p < 0.001), validating prompt switching. However, ArXiv (2024) found LLM performance is "influenced by the scope of intent labels" — taxonomy design matters.

**Recommendation: YES CHANGE.** Prompt switching is necessary but not sufficient. Add **different intent taxonomies per content type** — sales intents (buy, object, negotiate) differ fundamentally from meeting intents (propose, agree, assign_action) and interview intents (probe_competency, assess_fit). Architecture: universal base dialogue acts + content-type-specific intent overlays.

### LANG-TOPIC-01 — Topic shift detection

**Best evidence:** Hearst (1997), "TextTiling," *Computational Linguistics* (1,494+ citations) — the foundational method does NOT use a fixed percentage threshold. It computes cosine similarity and identifies boundaries where depth scores exceed **mean − 1 SD** — an adaptive, per-document threshold. No published research validates a 15% word overlap threshold specifically. Extended TextTiling with LLM embeddings tunes cosine similarity at ~0.19; BERT-based segmenters use ~0.3.

**Recommendation: YES CHANGE.** Replace fixed 15% word overlap with **adaptive cosine similarity** (using sentence embeddings) following Hearst's mean − 1 SD approach. If a fixed threshold is required, set cosine similarity at **~0.3 for embedding-based methods**. If retaining word overlap, increase to **~20–25%** — the current 15% is likely too aggressive, creating false topic boundaries.

---

## Conversation Agent: 8 rules, 5 changes needed

### CONVO-TURN-01 — Turn-taking rate

**Best evidence:** Meyer (2023), *Journal of Cognition*; Levinson & Torreira (2015), *Frontiers in Psychology*. Average turn duration in casual dyadic conversation is ~2–4 seconds, yielding roughly **6–8 combined speaker turns per minute**. The modal gap between turns is ~200ms per Stivers et al. (2009).

The >8/min "rapid" threshold sits at the upper edge of *normal* casual conversation. Truly competitive, rapid turn-taking exceeds 10–12/min.

**Recommendation: YES CHANGE.** Raise "rapid" threshold from >8/min to **>10/min**. Keep <2/min for monologue detection. Content-type adaptation needed: interviews and lectures naturally have lower turn rates (1–3/min combined).

### CONVO-LAT-01 — Response latency bands

**Best evidence:** Stivers et al. (2009), *PNAS* — 10 languages, cross-linguistic mean gap ~200ms, with modes from 0–200ms. Heldner & Edlund (2010) found ~40% of between-speaker intervals are overlaps, with median gap durations ~100ms. Confirmations are faster than disconfirmations; answers faster than non-answers.

The <200ms and 200–600ms bands are well-supported. However, **600–1500ms labeled as "normal" is misleading** — research shows most normal gaps fall within 0–500ms. A gap of 600–1000ms already signals processing difficulty, disagreement, or dispreference per Stivers' findings on disconfirmation latency.

**Recommendation: YES CHANGE.** Rename bands: <200ms = "overlapping" (keep), 200–600ms = "engaged" (keep), 600–1500ms → **"deliberative/processing"** (not "normal"), >1500ms = "delayed" (keep). Consider splitting deliberative: 600–1000ms = "deliberative," 1000–1500ms = "notably delayed." For interviews, the deliberative band should carry **neutral interpretation** — candidates need thinking time for complex questions.

### CONVO-DOM-01 — Dominance formula and thresholds

**Best evidence:** Jayagopi et al. (2009), *IEEE Transactions on Audio, Speech and Language Processing* — using the AMI meeting corpus, total speaking length correctly identified the most dominant person with **~85.3% accuracy**. Interruption count "performed badly" as a single cue. Itakura (2001) identified sequential dominance (topic control) as the strongest single indicator.

The 0.5 weight for talk_time is well-supported as the strongest predictor. The 0.2 weight for interruptions is **too high** given that Jayagopi found interruptions unreliable, and Hatch (1992) noted powerful speakers can control turns *without* interrupting.

**Recommendation: YES CHANGE.** Reduce interruption weight from 0.2 to **0.15**; increase monologue weight from 0.2 to **0.25**. Scale thresholds by participant count — in N-person conversations, expected dominance = 1/N, so thresholds should shift accordingly. Add topic-initiation as a signal if available, per Itakura's finding that sequential dominance is the strongest indicator.

### CONVO-RAP-01 — Rapport indicator weights

**Best evidence:** Tickle-Degnen & Rosenthal (1990), *Psychological Inquiry* (919+ citations) — the canonical rapport framework identifies three components: mutual attentiveness, positivity, and coordination. Gratch et al. (2007) demonstrated nonverbal feedback (backchannels, nods) significantly increases speaker engagement. Li (2010) found a **negative** correlation between backchannel frequency and conversation enjoyment in intercultural contexts — more is not always better.

The current weights map reasonably to the framework: backchannel (attentiveness), latency consistency (coordination), balance (mutual engagement), reciprocity (all three). However, **positivity/affect** — one of the three core rapport components — is currently unrepresented.

**Recommendation: UNIVERSAL — no change needed** to existing weights. The weights align with established theory. Minor enhancement: add a positivity/affect component if sentiment data is available, and normalize backchannel measurement to per-speaker baselines to handle cultural variation (Li 2010).

### CONVO-ENG-01 — Engagement score weights

**Best evidence:** Booth et al. (2023), *Proceedings of the IEEE* — engagement is a multi-dimensional construct comprising behavioral (48% of studies), cognitive (26%), and affective (26%) components. No gold-standard formula exists. The current components (response speed, backchannel, questions, participation, trend) align well with behavioral engagement research.

**Recommendation: UNIVERSAL — no change needed.** The components are well-chosen and the weights represent a reasonable heuristic consistent with the literature. Note that cognitive and affective engagement cannot be fully captured from behavioral proxies alone. Content-type awareness is helpful: in lectures, suppressed participation may reflect format constraints rather than disengagement.

### CONVO-BAL-01 — Gini coefficient thresholds

**Best evidence:** In a 2-person dyad, a 60/40 split yields Gini ≈ 0.10; 70/30 yields ≈ 0.20; 80/20 yields ≈ 0.30. The <0.15 "balanced" threshold captures approximately 55/45 to 60/40 splits. The >0.35 "imbalanced" threshold captures approximately 75/25 to 80/20 splits. These are reasonable for peer conversations.

The thresholds are fundamentally inappropriate for interviews (expected 20/80 interviewer-candidate) and podcasts (host 30%/guest 70%), where high Gini is expected and healthy.

**Recommendation: YES CHANGE — interpretation, not thresholds.** Keep the detection thresholds but add a **context_type parameter** that compares observed Gini to *expected* Gini for the conversation type. A Gini of 0.40 in a peer meeting is a problem; in an interview it is normal. Flag deviations from expected balance rather than deviations from symmetry.

### CONVO-CONF-01 — Conflict score minimum indicators

**Best evidence:** Gottman (1993) demonstrated multi-indicator approaches dramatically outperform single indicators. Single features like interruptions alone produce high false positive rates because competitive interruptions, raised voices, and dominance can appear in engaged, positive conversations. Hartwig & Bond (2014) confirmed constellation-based detection (multiple cues) achieves ~68% accuracy versus single-cue approaches.

**Recommendation: YES CHANGE (refinement).** Keep minimum 2 indicators but add **confidence tiers**: 2 indicators = "possible conflict," 3+ = "likely conflict." Add **repair attempt detection** — Gottman's research shows repair attempts are the most predictive moderating variable. Distinguish competitive interruptions from collaborative overlaps. Add temporal clustering — signals concentrated in a short window are more meaningful than the same signals spread across the full conversation.

---

## Fusion Agent: 6 rules, 2 changes needed

### FUSION-02 — Credibility assessment cap

**Best evidence:** Bond & DePaulo (2006), *Personality and Social Psychology Review* — meta-analysis of 206 documents, 24,483 judges, confirming **54% average human accuracy** for deception detection. However, Hartwig & Bond (2014), *Applied Cognitive Psychology*, found multi-cue constellation detection achieves **~68% accuracy** (R = .52, 144 samples, 26,866 messages). This is the correct reference for a multi-signal fusion system, not the single-judge figure. Constâncio et al. (2023) found ML deception detection ranges 51–100%, with realistic cross-validated accuracies of 60–75%.

The 0.55 cap is based on the wrong reference. Bond & DePaulo's 54% applies to untrained humans with no computational aids — explicitly not what an automated multi-signal fusion system performs. Hartwig & Bond's 67.9% multi-cue figure is the appropriate ceiling.

**Recommendation: YES CHANGE — raise cap to 0.65.** This acknowledges multi-cue paradigm performance while maintaining a buffer below the 68% theoretical ceiling. Do NOT exceed 0.70 because the system uses only two signal modalities (stress + sentiment), not the full constellation studied. **Critical labeling change: rename from "credibility" to "stress-sentiment incongruence."** "Credibility" implies truthfulness judgment, which is ethically indefensible at any confidence level. "Stress-sentiment incongruence" describes what the system actually measures.

### FUSION-07 — Verbal incongruence cap

**Best evidence:** DePaulo et al. (2003) meta-analysis of 158 cues — individual verbal cues are weak predictors (d < 0.25), but within-channel inconsistency is among the stronger signals. NLP models achieve 85–92% accuracy on sentiment classification. The mismatch detection (power-sentiment divergence) is a well-defined computational task.

**Recommendation: UNIVERSAL — no change needed.** The 0.70 cap is appropriate for within-text incongruence detection where the underlying NLP models are highly reliable. The uncertainty lies in interpretation, not measurement. Add an explicit caveat: power language + negative sentiment can reflect legitimate states (assertive complaint-making, high-confidence criticism) and should not be interpreted as deceptive.

### FUSION-13 — Urgency authenticity gating

**Best evidence:** Guyer et al. (2019/2024) found speech rate has a **curvilinear relationship** with persuasion — initially increases processing but eventually reverses (N = 3,958 across 6 studies). Van Zant & Berger (2020) confirmed paralinguistic persuasion attempts influence attitudes even when detected. Urgency-persuasion dynamics are well-documented in negotiations (deadline pressure), customer escalation, and political communication — not just sales contexts.

**Recommendation: YES CHANGE — expand gating only.** Keep the 0.60 cap (appropriate for this novel construct). Expand from "sales/pitch/presentation" to **any context where a speaker is attempting to motivate action** — add negotiation, customer escalation. Exclude or reinterpret for medical/emergency contexts where genuine urgency is expected and appropriate.

### FUSION-GRAPH-01 through 03 — Tension clusters, momentum, persistent incongruence

**Best evidence for 3+ signal threshold:** Hartwig & Bond (2014) demonstrated constellation-based detection achieves ~68% accuracy, with FBI research showing combined nonverbal + verbal clusters correctly classified 90% of participants. However, no research specifies an optimal minimum co-occurrence count — 3 is a reasonable statistical minimum where co-occurrence begins to exceed chance coincidence.

**Best evidence for adaptive thresholds:** Bayesian Change Point Detection achieved 91% accuracy in stress detection from multimodal signals (2024). Data-Adaptive Isolation algorithms avoid misspecification by using permutation-based thresholds. The field consensus is that adaptive beats fixed.

**Best evidence for persistent incongruence:** DePaulo et al. (2003) found deception cues are more pronounced when lies must be sustained over time. Interpersonal Deception Theory (Buller & Burgoon 1996) predicts leakage accumulation as cognitive resources deplete. Persistence increases *signal reliability* (reduces false positives) but does not increase *specificity* about cause.

**Recommendation: UNIVERSAL — no change needed for all three.** The 3+ signal threshold is a reasonable default (consider making it adaptive based on per-conversation signal density). Adaptive thresholds are correctly specified — recommend Bayesian Change Point Detection as the backbone methodology. Persistent incongruence detection is conceptually sound — label as "sustained signal mismatch" without implying specific cause.

---

## Cross-cutting findings that apply to all 42 rules

Three architectural conclusions emerge from this validation that transcend individual rules:

**The baseline-relative approach is the system's strongest asset.** Five independent research streams validate per-speaker normalization: Łachut 2025 (F1 improvement 0.619→0.753), Sethu 2007 (20% accuracy improvement), Gat 2022 (SOTA on IEMOCAP), Juslin, Laukka & Bänziger 2018 (spontaneous vs. acted differences are magnitude-only, not directional), and the Veiga 2025 finding that effects are stronger in spontaneous speech. Baseline-relative detection handles most speaker variation and much content-type variation automatically. However, it does **not** handle structural asymmetries — interviews, podcasts, and presentations have fundamentally different expected interaction patterns that require a context template overlaid on baseline detection.

**Content-type adaptation should occur at the interpretation layer, not the detection layer.** Of the 42 rules, only 5 need detection-logic changes per content type (VOICE-FILLER-02, VOICE-PAUSE-01 extended threshold, VOICE-TALK-01, LANG-SENT-02, LANG-CLAR-01). The remaining 37 should detect universally and interpret contextually. The system needs a minimum of four context templates — **peer/symmetric, interview/asymmetric, presentation/monologic, group meeting/multi-party** — that modulate interpretation of detected signals without changing detection thresholds.

**Publication bias and the acted-vs-spontaneous gap require conservative calibration.** Veiga 2025's trim-and-fill correction reduced the F0 stress effect from significant (SMD 0.55) to nonsignificant (SMD 0.17). Vogt & Wagner 2007 found minimal overlap in optimal feature sets between acted and spontaneous emotion. Laukka et al. 2011 confirmed authentic emotions show smaller effect sizes than acted portrayals. These findings argue for keeping detection thresholds slightly more sensitive than the meta-analytic averages would suggest — real conversational signals are subtler than laboratory findings, and the baseline-relative approach already accounts for this by measuring deviations rather than absolutes.

| Agent | Total Rules | Changes Needed | Threshold Changes | Interpretation/Gating Changes |
|-------|------------|----------------|-------------------|-------------------------------|
| Voice | 16 | 7 | 7 (STRESS, FILLER-02, PITCH-01, PITCH-02, RATE, PAUSE, INT) | 1 (TALK-01 sub-thresholds) |
| Language | 12 | 10 | 4 (SENT-01, SENT-02, CLAR-01, TOPIC-01) | 6 (BUY, OBJ, PWR, PERS, QUES, NEG, EMP, INTENT relabeling/gating) |
| Conversation | 8 | 5 | 3 (TURN, DOM, LAT rename) | 2 (BAL interpretation, CONF tiers) |
| Fusion | 6 | 2 | 1 (FUSION-02 cap raise) | 1 (FUSION-13 gating expansion) |
| **Total** | **42** | **24** | **15** | **10** |

The 14 rules requiring no change (VOICE-FILLER-01, VOICE-TONE-01–06, VOICE-ENERGY-01, CONVO-RAP-01, CONVO-ENG-01, FUSION-07, FUSION-GRAPH-01–03) have thresholds that are either directly validated by research or represent well-calibrated engineering decisions within established empirical ranges. The baseline-relative architecture provides a robust foundation that makes the system more resilient to threshold imprecision than a fixed-threshold system would be.

# Academic validation of NEXUS Phase 2+ detection rules

**Of 54 rules examined, 19 require threshold or confidence changes, 3 should be disabled or flagged experimental, and 32 can retain their current universal settings.** The system's baseline-relative architecture is its strongest methodological feature, consistently aligned with the research consensus that individual behavioral deviations outperform absolute thresholds. However, several rules cite in-person interaction research (Argyle, Mehrabian, Ekman) that does not directly transfer to video-call contexts, and webcam hardware constraints (15–30 fps, 720p–1080p) make certain detections—particularly micro-expressions and fine-grained gaze—technically infeasible.

The analysis below covers every rule across six categories: Facial Agent (7), Body Agent (8), Gaze Agent (7), Fusion Pairwise (13), Compound Patterns (12), and Temporal Sequences (8). For each rule, the best-fit research paper is identified, the threshold is evaluated, and a content-type recommendation is provided.

---

## Facial Agent: strong on engagement, weak on micro-expressions

### FACE-EMO-01 — 7-class emotion detection (base confidence 0.55)

**Best paper:** Mollahosseini et al. (2017), "AffectNet: A Database for Facial Expression, Valence, and Arousal Computing in the Wild." State-of-the-art 7-class accuracy on AffectNet ranges **57–67%**, with annotator agreement itself only ~60%, creating an inherent ceiling. Per-emotion accuracy is severely uneven: Happy reaches **94–95%**, but Disgust falls to **29%** and Contempt to **53%**. Khaireddin & Chen (2021) achieved 73.28% on FER2013, but that dataset uses posed, cropped faces—not webcam video.

**Threshold validation:** The 0.55 base confidence is **supported**. It correctly reflects near-ceiling performance for most emotion classes. However, a flat 0.55 across all seven classes misrepresents the massive accuracy differential.

**Recommendation:** UNIVERSAL — no content-type change needed. Add per-emotion confidence weighting: Happy 0.70, Neutral 0.65, Surprise 0.60, Sad/Angry 0.55, Fear/Disgust/Contempt 0.35–0.40. Consider reducing to a 4–5 class model for production, as three classes operate near chance.

### FACE-SMILE-01 — Duchenne vs. non-Duchenne smile (base confidence 0.60–0.65)

**Best paper:** Girard et al. (2020), "Reconsidering the Duchenne Smile" (PMC7193529). This study fundamentally challenges the Ekman/Davidson/Friesen (1990) Duchenne hypothesis, demonstrating that **AU6 co-occurrence is primarily predicted by AU12 intensity, not by felt positive emotion**. When controlling for AU12 intensity, AU6's predictive power for genuine emotion was "considerably reduced." Separately, OpenFace achieves AU12 detection AUC > 0.80 on conversational data, but AU6 is lower (~0.72).

**Threshold validation:** Needs adjustment. The Duchenne/non-Duchenne distinction rests on weakened theoretical ground. Temporal dynamics (onset speed, duration, symmetry) discriminate genuine from posed smiles at **93% accuracy** (Cohn et al.) and should supplement or replace the AU6 criterion.

**Recommendation:** UNIVERSAL — no content-type change. Lower the Duchenne-distinction confidence to **0.45–0.55**. Add temporal onset features. Rename internally to "smile quality analysis" rather than Duchenne detection.

### FACE-MICRO-01 — Micro-expression detection (base confidence 0.15–0.30)

**Best paper:** Yan et al. (2013) and the IEEE TPAMI survey (2021) confirm micro-expressions last **40–200 ms**, and all major ME databases (CASME II, SMIC) were captured at **100–200 fps**. At 30 fps, frame interval is 33 ms—a 40 ms micro-expression spans 1–2 frames, indistinguishable from noise. Even at 200 fps under lab conditions, 3-class ME recognition reaches only **~52%**. Downsampling from 200 fps to 25 fps dramatically degrades performance.

**Threshold validation:** Needs major adjustment. No validated ME detection system operates below 60 fps. The "disabled below 10 fps" threshold is **far too permissive**.

**Recommendation:** **DISABLE this rule entirely for ≤30 fps webcam input.** If functionality is desired, relabel as "rapid expression change detection" targeting expressions >200 ms, with confidence capped at 0.10–0.15. This is the single most important change in the entire rule set.

### FACE-ENG-01 — Facial engagement composite (base confidence 0.50)

**Best paper:** Whitehill et al. (2014), "The Faces of Engagement" (IEEE Trans. Affective Computing). Binary engagement classification achieved **2AFC = 0.73**, comparable to human accuracy (0.696). Most discriminating features: head pose roll (weight −0.57), AU10 (+0.51), AU1 (−0.44). Static facial features captured most engagement information (**r = 0.85** between frame-average and video labels), validating the head-orientation + expression-variability composite.

**Threshold validation:** Supported. The 0.50 base confidence is appropriate. Note that Whitehill's research was in educational tutoring contexts, not business calls—validation against conversational data is recommended but not required for threshold change.

**Recommendation:** UNIVERSAL — no change needed.

### FACE-STRESS-01 — Facial stress via AU4 + AU23 + AU24 (base confidence 0.45)

**Best paper:** Giannakakis et al. (2020) found AU10, AU04, and AU23 showed significant differences during stress (Stroop task). However, automated AU detection benchmarks reveal the critical problem: on BP4D, **AU23 F1 = 40–55%** and **AU24 F1 = 40–55%**—among the worst-detected AUs. On conversational data, AU23 had "relatively lower" AUC values across all systems (PMC8235167). AU4 is moderately detectable (F1 ~55–67%), but the combination depends on two unreliable components.

**Threshold validation:** Needs adjustment downward. The 0.45 base confidence is too high given that two of three constituent AUs have F1 scores below 55%.

**Recommendation:** UNIVERSAL — no content-type change. Lower confidence to **0.35–0.40**. Supplement AU23/24 with more reliably detected stress indicators: AU7 (lid tightener), AU10 (upper lip raiser), increased blink rate, and facial "freezing" (reduced movement). Replace Navarro (2008) citation with Giannakakis et al. (2020).

### FACE-VA-01 — Continuous valence-arousal (base confidence 0.40–0.60)

**Best paper:** Toisoul et al. (2021, Nature Machine Intelligence) achieved SOTA with CCC improvements of 17–20% over previous best on AffectNet. **Valence CCC ~0.76–0.82; arousal CCC ~0.55–0.65.** Human annotators are much better at judging valence than arousal from facial appearance—this is a fundamental perceptual limit, not just a technical one. Russell's (1980) circumplex model is broadly supported.

**Threshold validation:** Supported with refinement. The 0.40–0.60 range captures the valence-arousal asymmetry but could be sharpened.

**Recommendation:** UNIVERSAL — no change needed. Internally, split thresholds: valence 0.55–0.60, arousal 0.35–0.45. Supplement arousal with voice energy if available.

### FACE-CAL-01 — 60-second facial baseline

**Best paper:** Cohn et al. (2004) demonstrated individual facial expression differences are stable over **4–12 months** (Pearson r stability), strongly supporting the baseline-relative approach. OpenFace documentation confirms person-specific calibration significantly improves AU detection. However, the first 60 seconds of a conversation includes greeting behaviors (elevated smiling, social pleasantries) that do **not** represent true neutral baseline.

**Threshold validation:** Needs minor adjustment. Duration is sufficient but window selection is suboptimal.

**Recommendation:** For **interview** and **client_meeting**: extend to 90–120 s or implement greeting-phase detection to exclude the initial social pleasantry period. For **internal** and **podcast**: 60 s is acceptable. Implement adaptive baseline selection that detects the first sustained neutral/listening period rather than using the first 60 s blindly. Add rolling recalibration every 10–15 minutes.

---

## Body Agent: head nods excel, mirroring fails feasibility

### BODY-POST-01 — Posture and body openness (base confidence 0.40–0.55)

**Best paper:** Ding et al. (2019), "A real-time webcam-based method for assessing upper-body postures" (Machine Vision & Applications). Achieved **99.5% binary posture classification** and **88.2% for 19-level risk classification** from webcam. MediaPipe angular estimation achieves mean Pearson's r of **0.91 ± 0.08** for upper limb movements versus motion capture (Lafayette et al.). However, "body openness" requires arm visibility, which occurs in only ~30–60% of webcam-call frames due to tight framing.

**Threshold validation:** Supported with caveat. Confidence should drop when arms are absent from frame.

**Recommendation:** UNIVERSAL — no change needed. Add confidence penalty (reduce to 0.35) when arms are not detected. Rely primarily on shoulder landmarks and posture change from baseline rather than absolute openness scoring.

### BODY-HEAD-01 — Head nod/shake detection (base confidence 0.55–0.75)

**Best paper:** Chen et al. (2015, ICCV Workshop) achieved high F-scores for obvious nods using angular velocity at time interval m = 3–5 frames with SVM classification. Wei et al. (2013) reported **86% accuracy** with Kinect + HMM. MediaPipe Face Mesh provides robust 468-landmark head pose estimation, making angular velocity computation reliable. McClave (2000) appropriately documents communicative functions.

**Threshold validation:** **Supported.** This is the **highest-reliability body signal** from webcam. The 0.55–0.75 range correctly reserves the upper end for obvious, deliberate nods.

**Recommendation:** UNIVERSAL — no change needed. Use minimum angular velocity threshold of ~15°/s for nods, ~20°/s for shakes. Weight this signal heavily in composite scoring.

### BODY-LEAN-01 — Forward/backward lean via head-size proxy (base confidence 0.45)

**Best paper:** PoseX (Chen et al., 2021) uses proportional relations between face width, shoulder width, and ear-shoulder height to extract depth from RGB. BlazePose provides per-landmark z-coordinates, though accuracy is "experimental" per Google. A typical video-call lean of 3–5 cm at 60 cm distance produces only **5–8% head-size change**, near the noise floor of landmark detection.

**Threshold validation:** Needs adjustment downward. The signal-to-noise ratio for subtle leans is poor.

**Recommendation:** UNIVERSAL — no content-type change. Lower confidence to **0.30–0.40**. Replace pure head-size proxy with multi-landmark approach (nose z-value vs. shoulder z-values). Require minimum 8–10% head-size change before triggering.

### BODY-GEST-01 — Hand gesture classification (base confidence 0.45–0.80)

**Best paper:** MediaPipe Hands (Zhang et al., Google Research) achieves **95.7% average precision** in palm detection with 21 3D landmarks per hand. Moryossef et al. (2024) identifies critical flaws in hand ROI prediction for non-ideal orientations. In video calls, hands are frequently below frame or occluded—visibility may be only **30–60%** of frames.

**Threshold validation:** Supported. The wide 0.45–0.80 range correctly reflects high variability. Hand visibility as a binary signal is itself informative (absence may indicate crossed arms).

**Recommendation:** UNIVERSAL — no change needed. Focus on beat and deictic gestures as most reliably classifiable. Treat hand non-detection for >5 s as a separate signal.

### BODY-FIDG-01 — Fidget rate (base confidence 0.50–0.55)

**Best paper:** Zhang et al. (2020), "Video-Based Stress Detection through Deep Learning" (Sensors)—TSDNet combining facial expressions and action motions achieved **85.42% stress detection accuracy**. At 720p/30 fps, minimum reliably detectable movement is approximately **2–3 pixels** (~3–5 mm at typical distance). Subtle finger fidgets fall below this threshold. The fidgeting-anxiety association is context-dependent: fidgeting also correlates with boredom, excitement, and ADHD.

**Threshold validation:** Needs adjustment. Base confidence of 0.50–0.55 is too high for general fidget detection from webcam.

**Recommendation:** UNIVERSAL — no content-type change. Lower confidence to **0.35–0.45**. Only flag large, detectable movements (head position variance, major body shifts). Replace the "anxiety" interpretation with "elevated movement" as a context-neutral label. Require co-occurrence with other stress signals before interpreting as anxiety.

### BODY-TOUCH-01 — Self-touch / pacifying behavior (base confidence 0.35–0.50)

**Best paper:** Beyan et al. (2020, ACM ICMI) achieved **83.76% F1** for face-touch detection from video using CNN on 74K annotated frames. Hand-face proximity using MediaPipe is viable but degrades during occlusion (hand on face). **Navarro's stress hierarchy (neck > face > hair > arm cross) is NOT empirically validated** in peer-reviewed literature—it is practitioner guidance.

**Threshold validation:** Supported at lower end (0.35–0.50 range is appropriate).

**Recommendation:** UNIVERSAL — no change needed. Remove Navarro's body-region hierarchy. Use bounding-box overlap rather than precise landmark proximity for robustness against occlusion. Require sustained proximity (>0.5 s) to filter transient hand movement.

### BODY-MIRROR-01 — Cross-speaker body mirroring (base confidence 0.40)

**Best paper:** Chartrand & Bargh (1999) is well-replicated for in-person mirroring. Fujiwara & Daibo (2022) used OpenPose for automated posture matching in dyadic conversations. However, **automated mirroring detection across two separate, independently framed video-call feeds is a fundamentally different problem**: different camera angles, distances, framings, body proportions, and the left-right flip ambiguity. No published system achieves reliable mirroring detection in typical video-call conditions.

**Threshold validation:** Needs significant adjustment. 0.40 is too high for a technically dubious detection.

**Recommendation:** UNIVERSAL — **flag as EXPERIMENTAL across all content types**. Lower confidence to **0.20–0.30**, or remove from production scoring entirely. If retained, restrict to gross posture matching (both leaning forward/back) with >5 s temporal co-occurrence.

### BODY-CAL-01 — 60-second body baseline

**Best paper:** Bernieri & Rosenthal (1991) provide the foundational framework for behavioral coordination assessment. Behavioral coding research typically uses **2–5 minutes** for baseline establishment. The first 30–60 seconds of any interaction involve settling-in behavior (camera adjustment, greeting) that is not representative. Posture baseline stabilizes more slowly than facial expression baseline.

**Recommendation:** Extend to **90–120 seconds**. Discard first 30 s (settling period). Implement rolling baseline with exponential decay. Use per-signal baselines (separate for posture, head movement, gesture rate). Replace Navarro's 3 C's citation with Bernieri & Rosenthal (1991).

---

## Gaze Agent: blink detection excels, video-call norms are missing

### GAZE-DIR-01 — Gaze direction classification (base confidence 0.45–0.65)

**Best paper:** Zhang et al. (2015–2020) on MPIIGaze/ETH-XGaze: generic webcam gaze estimation achieves **3–7° angular error** without calibration and **1–2°** with person-specific calibration (Krafka et al.). MediaPipe Iris supports coarse directional classification but not fine-grained gaze-point estimation. Critically, "Don't look at the camera" (PMC12439499) found optimal perceived eye contact in video calls is ~2° below camera center—the camera-screen offset creates a systematic **5–15° vertical displacement** making all speakers appear to avoid eye contact.

**Threshold validation:** The lower bound (0.45) is appropriate for coarse classification; the upper bound should be reserved for binary screen-directed vs. away.

**Recommendation:** UNIVERSAL — no change needed. Reframe as "screen-directed gaze" rather than "eye contact." The binary on-screen/off-screen classification is the only reliable output.

### GAZE-CONTACT-01 — Screen engagement percentage (base confidence 0.50)

**Best paper:** Argyle & Cook (1976): listeners look at speakers ~75% of time; speakers look ~40–60%. These are in-person norms. In video calls, webcam **cannot distinguish** "looking at conversation partner's face on screen" from "looking at chat/notes/another window"—all register as screen-directed gaze. No validated equivalent norms exist for video-call screen engagement. Cao et al. (2021) found ~30% of video meetings involve email multitasking.

**Threshold validation:** The speaking 40–60% and listening 60–75% ranges derive from Argyle's in-person norms, which are **not validated for video calls**.

**Recommendation:** For **internal** meetings: widen acceptable range (speaking 35–70%, listening 55–85%) to account for multitasking tolerance. For **podcast**: adjust for note-reading norms. For all other types: UNIVERSAL is acceptable given baseline-relative detection. The key metric should be deviation from the speaker's own baseline, not absolute percentages.

### GAZE-BLINK-01 — Blink rate via EAR (base confidence 0.45–0.80)

**Best paper:** Soukupová & Čech (2016) established the EAR metric with **~96–98% blink detection accuracy** from webcam—the most technically sound detection in the entire rule set. Bentivoglio et al. (1997, N=150) is the definitive blink-rate study: **rest 17 bpm, conversation 26 bpm, reading 4.5 bpm**. Maffei & Angrilli (2019) found blink rate correlated with state anxiety (r = 0.418) during film viewing but **not at rest** (r = 0.144, p = 0.319).

**Threshold validation:** The resting norm "15–20 bpm" is validated. But the **stress threshold of >25 bpm is critically flawed**—it falls **below** the normal conversational mean of 26 bpm per Bentivoglio, meaning it would flag most normal conversations as stressed. Additionally, <10 bpm indicates concentrated attention/reading, not stress.

**Recommendation:** UNIVERSAL — no content-type change. **Fix the stress threshold to >30–35 bpm** (roughly 1+ SD above conversational mean). Must be baseline-relative, not absolute. Reframe <10 bpm as "concentrated attention." Lower stress-interpretation confidence to **0.35–0.45**; retain higher confidence (0.55–0.70) for attention/engagement inference.

### GAZE-ATT-01 — Attention composite (base confidence 0.55)

**Best paper:** Wohltjen & Wheatley (2024, Frontiers) showed blinking, gaze, and pupillary synchrony capture **dissociable aspects** of social attention, supporting composite approaches over single metrics. However, pupil dilation—a strong attention marker—**cannot be reliably measured from standard webcam** due to resolution and lighting variability.

**Threshold validation:** Supported. Composites that average noise from individual features are inherently more stable.

**Recommendation:** UNIVERSAL — no change needed. Build composite from screen-directed gaze %, head pose stability, blink rate deviation, and gaze break frequency. Exclude pupil dilation.

### GAZE-DIST-01 — Distraction events at >3 s (base confidence 0.45–0.65)

**Best paper:** Binetti et al. (2016, Royal Society Open Science) found preferred mutual gaze duration averages **3.2–3.3 s** (range 2–5 s). Glenberg et al. (1998, Memory & Cognition) showed gaze aversion increases with cognitive difficulty and **improves performance**—it is functional, not pathological. HAL (2024) found cognitive gaze aversion lasts **~6 s on average** with median onset at 1.09 s. Kendon (1967) documented that >70% of utterances begin with the speaker looking away.

**Threshold validation:** The **3 s threshold is not research-supported as a distraction indicator**. It is well within normal conversational gaze aversion duration and would generate massive false positives during any cognitively demanding conversation.

**Recommendation:** UNIVERSAL — no content-type change. **Increase threshold to 8–10 seconds**, especially during speaking turns. Differentiate by role: listener gaze breaks >5–6 s are more notable than speaker gaze breaks. Lower confidence to **0.35–0.50**. Require co-occurring signals (head orientation change + no speech) for distraction classification.

### GAZE-SYNC-01 — Mutual gaze synchrony (base confidence 0.40)

**Best paper:** Wohltjen & Wheatley (2021, PNAS) is the definitive study—but it measured **pupillary synchrony** using **dual lab-grade eye trackers** in face-to-face settings. The rule cites this paper but operationalizes a different construct (cross-speaker gaze alignment from separate webcam feeds). Pupillary synchrony cannot be measured from standard webcam. In video calls, both speakers looking at their screens simultaneously is the **default state**, not a synchrony indicator.

**Threshold validation:** The rule measures a different construct than the cited paper validates.

**Recommendation:** UNIVERSAL — **rename to "Cross-Speaker Gaze Alignment"** and remove the Wohltjen & Wheatley citation. The signal of interest is the temporal pattern of looking away and returning, not simultaneous screen gaze. Keep confidence at **0.40 or lower**. Flag as experimental.

### GAZE-CAL-01 — Per-speaker gaze baseline + camera position

**Best paper:** Gudi et al. (2020, ECCV Workshop) showed webcam gaze estimation with 4-point calibration dramatically improves accuracy. Camera position is the **dominant source of gaze variability** in video calls: laptop cameras sit 0–5° above screen center, external webcams 5–15° above or below. Without knowing camera position, all gaze analysis is confounded by geometry. Person-specific models achieve **<2° error** vs. 4–7° for generic models (PMC10147084).

**Recommendation:** UNIVERSAL — this is the **foundational infrastructure rule** for all gaze analysis. Use the first 60–120 s of each speaker to estimate camera position from average head pose. Re-estimate every 5–10 minutes. All other gaze rules must express thresholds as deviations from this baseline.

---

## Fusion pairwise rules: strongest for rapport and congruence, weakest for purchase intent

### FUSION-01 — Voice × Face → Masking/Congruence (cap 0.75)

**Best paper:** Watson et al. (2013, Frontiers in Human Neuroscience) demonstrated voice-face incongruence activates dedicated neural regions (right STG/STS) independent of task difficulty. Multiple fMRI and ERP studies confirm cross-modal emotional mismatch detection is automatic. **Scientific validity: Strong.**

**Recommendation:** UNIVERSAL — no change needed. Cap of 0.75 is appropriate. Note: detecting incongruence ≠ detecting deliberate masking—incongruence can arise from mixed emotions, display rules, or cognitive load.

### FUSION-03 — Body Posture × Voice Energy → Manufactured Enthusiasm (cap 0.65)

**Best paper:** Van den Stock et al. (2011, PLOS ONE) showed body-voice emotional congruence affects perception even below conscious awareness. The specific "manufactured enthusiasm" label is a reasonable inference but not directly validated. **Scientific validity: Moderate.**

**Recommendation:** UNIVERSAL — no change needed. Cap of 0.65 correctly reflects the inferential leap.

### FUSION-04 — Gaze Break × Filler Words → Uncertainty (cap 0.70)

**Best paper:** Doherty-Sneddon & Phelps (2005) showed gaze aversion primarily serves cognitive load management, not uncertainty specifically. Fillers are speech-planning markers. The combination signals cognitive effort, which **correlates with but is not identical to** uncertainty. **Scientific validity: Strong for cognitive load; Moderate for uncertainty.**

**Recommendation:** UNIVERSAL — no change needed. Consider relabeling as "Cognitive Effort Confirmation" rather than "Uncertainty."

### FUSION-05 — Buying Signal × Body Language → Purchase Intent (cap 0.70)

**Best paper:** No peer-reviewed research validates multimodal "purchase intent verification" combining verbal buying signals with body language in conversational settings. The concept relies on general incongruence detection applied to a commercial context. **Scientific validity: Weak/Speculative.**

**Recommendation:** UNIVERSAL — **lower cap to 0.55–0.60**. Flag as domain-specific inference. Most applicable to client_meeting contexts.

### FUSION-06 — Micro-Expression × Language + Voice → Emotional Leakage (cap 0.35)

**Best paper:** Burgoon (2018, PMC) systematically critiqued micro-expression theory, questioning all six necessary propositions. Jordan et al. (2019) found METT training did not improve lie detection accuracy. Even trained professionals achieve only ~47–50% ME identification. At 15–30 fps, reliable ME detection is technically infeasible (see FACE-MICRO-01). **Scientific validity: Weak.**

**Recommendation:** UNIVERSAL — **lower cap to 0.20–0.25** or DISABLE. If retained, relabel as "Subtle Expression × Language + Voice." Add documentation that this is not true micro-expression detection at standard framerates.

### FUSION-07 — Head Nod × Speech Content → Agreement/Disagreement (cap 0.65)

**Best paper:** Briñol & Petty (2003) showed head nodding influences attitude formation (self-validation effect). Andonova & Taylor (2012) documented cultural reversal (Bulgaria). Gender differences are significant: women nod for listening/understanding >75% of the time, not agreement (Acheson). **Scientific validity: Moderate-Strong.**

**Recommendation:** UNIVERSAL — no change needed. Add gender and cultural sensitivity flags in output documentation.

### FUSION-08 — Eye Contact × Hedged Language → False Confidence (cap 0.55)

**Best paper:** DePaulo et al. (2003) meta-analysis (1,300+ estimates, 158 cues): gaze aversion shows only weak/inconsistent association with deception. >80% of expert deception researchers agree gaze aversion is not diagnostic (Denault et al., 2020). Liars sometimes maintain **more** eye contact to appear credible. **Scientific validity: Weak-Moderate.**

**Recommendation:** UNIVERSAL — no change needed. The 0.55 deception-adjacent cap is appropriately conservative. Flag high false-positive risk in documentation.

### FUSION-09 — Smile × Sentiment → Sarcasm/Social Masking (cap 0.60)

**Best paper:** IJCAI 2024 Survey of Multimodal Sarcasm Detection confirms smile-sentiment mismatch is the core of sarcasm research. The MUStARD dataset provides a benchmark. Most research targets social media, not real-time conversation. **Scientific validity: Moderate.**

**Recommendation:** UNIVERSAL — no change needed. Consider splitting "sarcasm" and "social masking" as different output labels.

### FUSION-10 — Response Latency × Facial Stress → Cognitive Load (cap 0.60)

**Best paper:** ADABase (MDPI, 2023) validated multimodal cognitive load assessment using physiological signals + facial AUs. Response latency is a standard cognitive-load performance metric. SWELL-KW dataset confirms facial-expression correlates of cognitive load. **Scientific validity: Moderate-Strong.**

**Recommendation:** UNIVERSAL — no change needed. Specify which AUs constitute "facial stress" (AU4, AU7 preferred over AU23/24).

### FUSION-11 — Talk Dominance × Gaze Submission → Anxious Dominance (cap 0.65)

**Best paper:** Terburg et al. (2012) distinguished submissive gaze aversion from anxious gaze avoidance. A 2026 JMIR study replicated the gaze-avoidance/social-anxiety link in VR. The combined "anxious dominance" construct is theoretically grounded in social rank theory. **Scientific validity: Moderate.**

**Recommendation:** UNIVERSAL — no change needed. Note many alternative explanations exist (cultural norms, screen reading).

### FUSION-12 — Interruption × Body Position → Interruption Intent (cap 0.55)

**Best paper:** Frontiers in Psychology (2020) distinguished cooperative from competitive interruptions via nonverbal cues: cooperative interruptions involve leaning forward and nodding; competitive interruptions involve dominance signals. **Scientific validity: Moderate.**

**Recommendation:** UNIVERSAL — no change needed.

### FUSION-14 — Empathy Language × Head Nodding → Rapport (cap 0.70)

**Best paper:** Stivers (2008) demonstrated head-nodding during narratives signals stance alignment/empathy. Tickle-Degnen & Rosenthal (1990) defined rapport as mutual attentiveness + positivity + coordination, directly mapping to this combination. **Scientific validity: Strong.** This is the strongest fusion rule.

**Recommendation:** UNIVERSAL — no change needed. Cap could arguably be 0.75 given the robust evidence base.

### FUSION-15 — Filler Words × Gaze Aversion → Compound Uncertainty (cap 0.55)

**Best paper:** Same evidence base as FUSION-04. This rule **overlaps significantly** with FUSION-04 (Gaze Break × Filler Words → Uncertainty Confirmation), using essentially identical signals with different labels and different caps (0.70 vs. 0.55).

**Recommendation:** **MERGE with FUSION-04** or clearly differentiate scope (e.g., FUSION-04 = single event, FUSION-15 = sustained/repeated pattern). The 0.55 vs. 0.70 cap discrepancy for the same signals creates logical inconsistency.

---

## Compound patterns: engagement and conflict are solid, decision readiness is not

### COMPOUND-01 — Genuine Engagement (cap 0.80)

**Best paper:** Pellet-Rostaing et al. (2023, Frontiers in Computer Science) achieved **0.76 weighted F-score** with multimodal features (prosodic + gestural + syntactic). Engagement is a well-established construct with decades of research confirming it manifests across multiple behavioral channels simultaneously. **Construct validity: Strong.** Webcam detectability: High.

**Recommendation:** UNIVERSAL — no change needed. Cap of 0.80 is well-calibrated.

### COMPOUND-02 — Active Disengagement (cap 0.80)

**Best paper:** Leite et al. (2015, HRI) compared disengagement models. Gottman's "stonewalling" concept provides validated behavioral markers for extreme disengagement. However, active disengagement overlaps with fatigue, boredom, and internal processing. **Construct validity: Strong.**

**Recommendation:** UNIVERSAL — **lower cap to 0.75** to account for ambiguity between intentional disengagement and other withdrawal-like states.

### COMPOUND-03 — Emotional Suppression (cap 0.70)

**Best paper:** Ekman & Friesen (1969) established the leakage hierarchy (face most controlled, body leaks more). Porter & ten Brinke (2008, 2012) confirmed emotional leakage in deceptive facial expressions. At 15–30 fps, micro-expression detection is unreliable—the system must rely on macro-level channel discrepancies. **Construct validity: Moderate-Strong.** Webcam detectability: Low-Medium.

**Recommendation:** UNIVERSAL — no change needed. Cap of 0.70 is appropriate given the macro-level detection limitation. Flag that 15–30 fps is insufficient for the micro-expression component of leakage detection.

### COMPOUND-04 — Decision Readiness (cap 0.85)

**Best paper:** No directly validating paper exists. The neuroscience "readiness potential" (Libet 1985) refers to motor cortex preparation, not interpersonal decision communication. Ajzen's Theory of Planned Behavior (1991) defines behavioral intention via self-report, not behavioral observation. The specific convergence of 4/6 behavioral domains signaling "readiness" has **not been empirically validated**. **Construct validity: Weak/Novel.**

**Recommendation:** UNIVERSAL — **lower cap significantly to 0.65**. This is the most over-confident rule in the system. Rename to "Decision Engagement Signals." Label as experimental/novel. The 4-of-6 requirement provides some false-positive protection, but the construct itself lacks empirical grounding.

### COMPOUND-05 — Cognitive Overload (cap 0.75)

**Best paper:** Chen et al. (2012, ACM TIIS) validated speech, gesture, and pen-input changes under cognitive load. Sweller's Cognitive Load Theory is well-established. Webcam-accessible indicators include speech disfluency, reduced facial expressivity, increased blink rate, and increased response latency. Most high-accuracy systems use physiological sensors (EEG, EDA) not available from webcam. **Construct validity: Strong.**

**Recommendation:** UNIVERSAL — no change needed. Emphasize speech-based indicators as primary webcam-accessible signals.

### COMPOUND-06 — Conflict Escalation (cap 0.80)

**Best paper:** Vinciarelli et al. (2015), Social Signal Processing for Conflict Analysis. Glasl's 9-stage escalation model is well-validated. Gottman's research confirms conflict markers (vocal intensity, overlapping speech, contempt expressions, disrupted turn-taking) are coded reliably from video. Both-speaker requirement strengthens validity. **Construct validity: Strong.**

**Recommendation:** UNIVERSAL — no change needed.

### COMPOUND-07 — Silent Resistance (cap 0.70)

**Best paper:** No directly validating paper. Conceptually related to acquiescence bias (well-documented in survey methodology) and Gottman's stonewalling. The multimodal detection of verbal-nonverbal discrepancy is plausible but unvalidated as a detection target. **Critical cultural issue:** collectivist cultures routinely engage in verbal compliance with private disagreement as a social norm, not resistance. **Construct validity: Weak-Moderate.**

**Recommendation:** UNIVERSAL — **lower cap to 0.65**. Rename to "Verbal-Nonverbal Discordance" to be descriptively accurate. Must include cultural sensitivity documentation—this pattern has massive cultural variation.

### COMPOUND-08 — Rapport Peak (cap 0.85)

**Best paper:** Tickle-Degnen & Rosenthal (1990) defined rapport as mutual attentiveness + positivity + coordination (>3,000 citations). Ramseyer & Tschacher (2011) demonstrated behavioral synchrony predicts psychotherapy outcome. Meta-analysis (2024) found medium effect size (r = 0.32) for neural-behavioral synchrony. **Construct validity: Strong.**

**Recommendation:** UNIVERSAL — **lower cap to 0.80**. Detecting "peak" rapport requires precise synchrony measurement that introduces uncertainty. The construct is strong but the detection complexity warrants a small margin.

### COMPOUND-09 — Topic Avoidance (cap 0.70)

**Best paper:** Gottman's SPAFF coding system reliably codes avoidance behaviors from video. The key challenge is temporal contingency—behavior changes must correlate with specific topic introduction, requiring NLP integration. **Construct validity: Moderate.**

**Recommendation:** UNIVERSAL — no change needed. Ensure the system requires temporal contingency with topic content, not just avoidance-like behavior in isolation.

### COMPOUND-10 — Authentic Confidence (cap 0.85)

**Best paper:** Mori & Pell (2019, Frontiers in Communication) identified visual confidence markers: direct eye contact, serious face, upright posture. Machine learning achieved **88% accuracy** for confidence detection from gaze/head pose (Springer, 2025). However, distinguishing **authentic from performed** confidence relies on the same leakage theory as emotional suppression—and that distinction is not independently validated. **Construct validity: Moderate for confidence; Weak for the authentic/performed distinction.**

**Recommendation:** UNIVERSAL — **lower cap to 0.75**. Consider splitting into "Confidence Display" (higher cap) and "Confidence Authenticity Assessment" (lower cap). The 4-domain confirmation helps but doesn't resolve the authentic/performed challenge.

### COMPOUND-11 — Anxiety Performance (cap 0.65)

**Best paper:** No paper validates this as distinct from COMPOUND-03 (Emotional Suppression). Both involve controlled channels displaying one state while leaked channels reveal another. The only distinction is the specific emotion being masked. **Construct validity: Weak as a separate construct.**

**Recommendation:** UNIVERSAL — no change needed at 0.65. Consider merging with COMPOUND-03 as a subtype rather than maintaining as separate.

### COMPOUND-12 — Deception Risk (hard cap 0.55)

**Best paper:** Bond & DePaulo (2006) meta-analysis of 206 studies (24,483 judges): humans achieve **54% accuracy** (barely above chance). Hartwig & Bond (2014) found multi-cue analysis achieves **~70% accuracy** (ceiling across settings). DePaulo et al. (2003) meta-analysis: "few nonverbal cues reliably correlate with deception." Non-contact automated systems achieve **64–74%** on benchmarks with significant dataset specificity.

**Recommendation:** UNIVERSAL — **no change needed. The 0.55 hard cap is the best-calibrated threshold in the entire system**, precisely aligned with the scientific literature showing fundamental limits of behavioral deception detection. The 4-domain + 3-fusion-conflict requirement adds rigorous gating. Ensure documentation references Bond & DePaulo and emphasizes this is a "risk flag," not a deception verdict.

---

## Temporal sequences: objection formation validated, trust repair speculative

### TEMPORAL-01 — Stress Cascade, voice → body → face, 2–15 s (cap 0.70)

**Best paper:** Diamond & Kim (2002) describe phase-based stress response dynamics. Ekman & Friesen (1969) established the controllability hierarchy (voice least controlled, face most controlled). The ordering reflects **differential controllability** of channels (leakage visibility), not differential physiological activation timing—all systems activate near-simultaneously. **Temporal ordering: Partially Validated.**

**Recommendation:** UNIVERSAL — **lower cap to 0.65**. Widen window to 2–30 s per step. Reframe as "Stress Leakage Cascade" emphasizing progressive failure of suppression.

### TEMPORAL-02 — Engagement Build, gaze → face → body → voice, 1–3 min (cap 0.65)

**Best paper:** ACL Anthology (L18-1126) used temporal sequence mining to find engagement correlates with specific nonverbal signal sequences. The gaze → face → body ordering is consistent with attention → affect → approach models. Voice as the last modality is less clear—vocal backchannels often co-occur with gaze and facial engagement. **Temporal ordering: Partially Validated.**

**Recommendation:** UNIVERSAL — no change needed. Consider making voice co-occur with body (parallel steps 3–4) rather than strictly sequential. Broaden window to 30 s–5 min per step.

### TEMPORAL-03 — Disengage Cascade, face → gaze → body → voice, 30–120 s (cap 0.55)

**Best paper:** Essentially the reverse of TEMPORAL-02. Disengagement research suggests gaze aversion may precede or co-occur with facial expression changes rather than following them. **Temporal ordering: Theoretical.**

**Recommendation:** UNIVERSAL — no change needed. Cap of 0.55 appropriately reflects theoretical status. Consider allowing parallel face + gaze as combined first step.

### TEMPORAL-04 — Objection Formation, face → voice → language, 5–30 s (cap 0.60)

**Best paper:** Ekman's seminal work showed genuine emotional reactions appear on the face as micro/subtle expressions **before** verbal formulation—supported by dual-process models (fast emotional reaction → slower deliberative verbal response). Matsumoto & Hwang (2018, PMC6305322) confirmed micro-expressions differentiate truths from lies in terms of timing. **Temporal ordering: Validated.** This is the best-supported temporal sequence.

**Recommendation:** UNIVERSAL — no change needed. Focus on "subtle expressions" (>200 ms) rather than true micro-expressions for webcam feasibility. The 5–30 s window correctly captures the gap between facial flash and verbal articulation.

### TEMPORAL-05 — Trust Repair, voice → face → body → gaze, 10–90 s (cap 0.75)

**Best paper:** Sharma, Schoorman & Ballinger (2023) comprehensively reviewed trust repair research. Trust repair research operates at the **relational/behavioral level over weeks and months**, not at the second-by-second multimodal signal level. The proposed modality ordering has **no direct empirical validation**. **Temporal ordering: Speculative.** This is the most problematic rule.

**Recommendation:** UNIVERSAL — **lower cap to 0.50–0.55**. This is the highest-priority cap revision among temporal rules. Consider reframing as "Reconciliation Gesture Detection" and dropping the specific ordering requirement, or removing entirely.

### TEMPORAL-06 — Buying Decision Sequence, 5–30 min (cap 0.60)

**Best paper:** Kotler's (2012) 5-stage consumer buying process is the textbook standard but was developed as a **marketing model**, not a real-time behavioral detection system. The stages are cognitive/psychological, not defined by observable nonverbal signals. Observable behaviors do not map cleanly onto buying stages. **Temporal ordering: Speculative as behavioral sequence.**

**Recommendation:** UNIVERSAL — **lower cap to 0.45–0.50**. Reframe as "Decision Engagement Pattern." Flag as sales-context-specific. This rule should not be presented as detecting a validated behavioral phenomenon.

### TEMPORAL-07 — Dominance Shift, voice → body → gaze, 30 s–5 min (cap 0.55)

**Best paper:** Dominance research documents voice, body expansion, and gaze as simultaneous multimodal packages rather than sequential cascades. Cross-cultural research shows significant variation: dominant Chinese negotiators lean back and make more eye contact—opposite of Western patterns. **Temporal ordering: Theoretical.**

**Recommendation:** UNIVERSAL — no change needed. Add cultural sensitivity flags. Consider detecting dominance shift as co-occurring signals rather than strict sequence.

### TEMPORAL-08 — Authenticity Erosion, 15–60 min (cap 0.55)

**Best paper:** Two converging literatures support progressive leakage: Ekman's deception research (sustained suppression → increasing leakage) and Hochschild's (1983) emotional labor research (surface acting depletes resources via Conservation of Resources Theory). However, no studies measure progressive leakage at this specific temporal resolution within single conversations. **Temporal ordering: Partially Validated.**

**Recommendation:** UNIVERSAL — no change needed. Reframe as "Behavioral Consistency Monitoring" to reduce loaded terminology. Add ethical disclaimers about false positives (fatigue, boredom, comfort-level changes all cause progressive behavioral changes).

---

## The 19 rules requiring threshold changes

| Rule | Current cap | Recommended cap | Rationale |
|------|-----------|----------------|-----------|
| FACE-SMILE-01 | 0.60–0.65 | **0.45–0.55** (Duchenne distinction) | Duchenne hypothesis weakened by Girard et al. (2020) |
| FACE-MICRO-01 | 0.15–0.30 | **DISABLE** | 15–30 fps technically infeasible for micro-expression detection |
| FACE-STRESS-01 | 0.45 | **0.35–0.40** | AU23/24 F1 scores 40–55%, among worst-detected AUs |
| BODY-LEAN-01 | 0.45 | **0.30–0.40** | 5–8% head-size change near noise floor |
| BODY-FIDG-01 | 0.50–0.55 | **0.35–0.45** | Subtle fidgets below webcam detection threshold |
| BODY-MIRROR-01 | 0.40 | **0.20–0.30** | No validated cross-feed mirroring detection exists |
| GAZE-BLINK-01 | Stress >25 bpm | Stress **>30–35 bpm** | >25 is below normal conversational mean of 26 bpm |
| GAZE-DIST-01 | >3 s | **>8–10 s** | 3 s is within normal cognitive gaze aversion |
| FUSION-05 | 0.70 | **0.55–0.60** | No validated research on multimodal purchase intent |
| FUSION-06 | 0.35 | **0.20–0.25** or disable | ME detection infeasible at webcam framerates |
| COMPOUND-02 | 0.80 | **0.75** | Overlaps with fatigue and internal processing |
| COMPOUND-04 | 0.85 | **0.65** | Construct is novel/unvalidated; largest over-confidence |
| COMPOUND-07 | 0.70 | **0.65** | Massive cultural variation; weak construct validity |
| COMPOUND-08 | 0.85 | **0.80** | Peak detection adds uncertainty to strong construct |
| COMPOUND-10 | 0.85 | **0.75** | Authentic/performed distinction unvalidated |
| TEMPORAL-01 | 0.70 | **0.65** | Ordering is theoretical, not empirically measured |
| TEMPORAL-05 | 0.75 | **0.50–0.55** | Most speculative rule; trust repair studied at different timescale |
| TEMPORAL-06 | 0.60 | **0.45–0.50** | Marketing model, not behavioral detection system |
| FUSION-15 | 0.55 | **Merge with FUSION-04** | Redundant signals with inconsistent caps |

## Five cross-cutting findings that matter most

**The baseline-relative architecture is validated and essential.** Person-specific gaze models achieve <2° error vs. 4–7° for generic models. Individual differences in facial expression are stable over months (Cohn et al. 2004). Every piece of research examined confirms that within-person deviation dramatically outperforms absolute thresholds. This is the system's strongest design choice.

**Video calls are not in-person interactions.** Argyle, Mehrabian, Ekman, and Kleinke all studied face-to-face behavior. The camera-screen offset (5–15° systematic displacement), inability to distinguish "looking at face on screen" from "reading chat," and reduced social pressure of mediated communication create fundamentally different behavioral norms that have **not been systematically validated** in peer-reviewed research. Every rule citing pre-video-call research should be treated as theoretically motivated, not directly validated for its application context.

**The 0.55 deception-adjacent cap is the system's best-calibrated threshold.** Bond & DePaulo's meta-analysis (206 studies, 24,483 judges, 54% human accuracy) and Hartwig & Bond's multi-cue ceiling (~70%) provide robust empirical anchoring. The system's approach of flagging deception risk rather than asserting deception detection is scientifically and ethically appropriate.

**Micro-expression detection should be entirely disabled at ≤30 fps.** All major ME databases were captured at 100–200 fps. No validated detection system operates below 60 fps. Even at 200 fps under lab conditions, recognition accuracy reaches only ~52% for 3-class classification. Three rules depend on micro-expression input (FACE-MICRO-01, FUSION-06, and partially COMPOUND-03)—all should be adjusted or disabled for this component.

**Temporal sequence rules are the system's most speculative layer.** Only TEMPORAL-04 (Objection Formation) has a validated temporal ordering. The remaining seven sequences represent theoretically reasonable but empirically unmeasured cascading patterns. No published research maps complete multimodal cascading sequences with measured inter-step latencies in naturalistic conversation. These rules should be labeled as experimental and their caps reduced to reflect this theoretical status.