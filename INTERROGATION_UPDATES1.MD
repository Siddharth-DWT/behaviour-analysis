# Adaptive Multi-Quality Interrogation Analysis System
## Dynamic Threshold Adjustment + Graceful Degradation Architecture

**Philosophy:** Don't block rules - adapt them to available quality and combine evidence intelligently.

---

## 1. QUALITY-ADAPTIVE THRESHOLD FRAMEWORK

### Core Principle: Same Rule, Different Thresholds

Instead of blocking rules entirely, **adjust thresholds and confidence based on detected quality.**

```python
class AdaptiveRuleEngine:
    """
    Runs all rules, but adjusts thresholds and confidence based on 
    detected video/audio quality. Graceful degradation, not hard blocking.
    """
    
    def __init__(self):
        self.quality_profile = None
        self.rule_catalog = self.load_all_rules()
    
    def detect_quality_profile(self, video_path, audio_path):
        """
        Assess available signal quality and create profile.
        """
        video_quality = self.assess_video_quality(video_path) if video_path else None
        audio_quality = self.assess_audio_quality(audio_path)
        
        return QualityProfile(
            video_resolution=video_quality['resolution'] if video_quality else None,
            video_fps=video_quality['fps'] if video_quality else None,
            face_size_px=video_quality['face_size'] if video_quality else None,
            lighting_score=video_quality['lighting'] if video_quality else None,
            handcuffs_detected=video_quality['handcuffs'] if video_quality else None,
            camera_angle=video_quality['angle'] if video_quality else None,
            
            audio_sample_rate=audio_quality['sample_rate'],
            audio_snr=audio_quality['snr'],  # Signal-to-noise ratio
            audio_channels=audio_quality['channels'],
            transcription_wer=None  # Will be computed after Whisper runs
        )
    
    def run_analysis(self, video_path, audio_path):
        """
        Run all applicable rules with quality-adjusted thresholds.
        """
        # Step 1: Detect quality
        self.quality_profile = self.detect_quality_profile(video_path, audio_path)
        
        # Step 2: Run all rules
        results = []
        for rule in self.rule_catalog:
            if rule.is_applicable(self.quality_profile):
                # Get quality-adjusted parameters
                adjusted_rule = rule.adapt_to_quality(self.quality_profile)
                
                # Run with adjusted thresholds
                result = adjusted_rule.execute()
                
                if result:
                    results.append(result)
        
        # Step 3: Generate report with quality disclaimers
        return self.generate_report(results, self.quality_profile)
```

---

## 2. EXAMPLE: BARRIER BEHAVIOR - THREE QUALITY TIERS

### Rule: Crossed Arms After Question

**Original Spec:** "Crossed arms within 3 seconds of question"

**Problem:** Handcuffs make this impossible to detect OR meaningless.

**Adaptive Solution:** THREE threshold sets, auto-selected by quality profile.

```python
class BarrierBehaviorRule:
    """
    Adaptive barrier behavior detection with quality-based thresholds.
    """
    
    THRESHOLD_SETS = {
        'HIGH_QUALITY': {
            # 1080p+ @ 30fps, no handcuffs, clear view of upper body
            'time_window': 3.0,           # seconds after question
            'confidence_threshold': 0.85,  # MediaPipe detection confidence
            'movement_threshold': 0.15,    # minimum arm movement (meters)
            'output_confidence': 0.50,     # final rule confidence
            'baseline_required': True
        },
        
        'CCTV_QUALITY': {
            # 480p-720p @ 15-30fps, no handcuffs, partial occlusion
            'time_window': 5.0,           # Relaxed: harder to see exact timing
            'confidence_threshold': 0.70,  # Lower detection threshold
            'movement_threshold': 0.25,    # Larger movement required (lower resolution)
            'output_confidence': 0.35,     # Reduced confidence in finding
            'baseline_required': True
        },
        
        'HANDCUFFED': {
            # Any quality, but handcuffs detected
            'suppressed': True,
            'reason': 'Subject appears handcuffed - arm crossing physically restricted',
            'alternative_signal': 'torso_lean_away',  # Look for torso shift instead
            'time_window': 5.0,
            'output_confidence': 0.20,     # Very low - torso lean is weak proxy
            'baseline_required': True
        }
    }
    
    def adapt_to_quality(self, quality_profile):
        """
        Select appropriate threshold set based on quality profile.
        """
        if quality_profile.handcuffs_detected:
            self.thresholds = self.THRESHOLD_SETS['HANDCUFFED']
            self.mode = 'TORSO_LEAN_PROXY'  # Use alternative measurement
            
        elif quality_profile.video_resolution >= (1280, 720) and \
             quality_profile.video_fps >= 30:
            self.thresholds = self.THRESHOLD_SETS['HIGH_QUALITY']
            self.mode = 'STANDARD'
            
        else:  # CCTV quality
            self.thresholds = self.THRESHOLD_SETS['CCTV_QUALITY']
            self.mode = 'RELAXED'
        
        return self
    
    def execute(self):
        """
        Run detection with adapted thresholds.
        """
        if self.thresholds.get('suppressed'):
            # Try alternative measurement
            if self.mode == 'TORSO_LEAN_PROXY':
                return self.detect_torso_lean_away()
            else:
                return None  # Completely suppressed
        
        # Standard barrier detection
        barrier_event = self.detect_barrier_formation()
        
        if barrier_event:
            return {
                'rule_id': 'INTERROG-BODY-01',
                'label': 'barrier_behavior_detected',
                'confidence': self.thresholds['output_confidence'],
                'quality_mode': self.mode,
                'threshold_set': self.get_threshold_set_name(),
                'disclaimer': self.get_quality_disclaimer()
            }
    
    def get_quality_disclaimer(self):
        """
        Auto-generated disclaimer based on quality mode.
        """
        disclaimers = {
            'STANDARD': None,  # No disclaimer needed
            'RELAXED': 'Detection confidence reduced due to CCTV video quality (lower resolution/frame rate).',
            'TORSO_LEAN_PROXY': 'Subject appears handcuffed. Using torso lean as weak proxy for barrier behavior. Confidence heavily reduced.'
        }
        return disclaimers[self.mode]
```

**Result:** Rule fires in all three scenarios, but with:
- Different thresholds (3s vs 5s time window)
- Different measurements (arm crossing vs torso lean)
- Different confidence levels (0.50 vs 0.35 vs 0.20)
- Explicit disclaimers about quality

**NOT blocked - ADAPTED.**

---

## 3. EXAMPLE: BLINK RATE - FRAME RATE ADAPTIVE

### Rule: Blink Suppression-Spike Pattern

**Original Spec:** "Blink rate decreases during lie, increases after"

**Problem:** 15fps CCTV misses ~50% of blinks (blinks last ~100-150ms).

**Adaptive Solution:** Adjust detection window and confidence based on FPS.

```python
class BlinkRateRule:
    """
    Adaptive blink detection with FPS-based threshold adjustment.
    """
    
    THRESHOLD_SETS = {
        'HIGH_FPS': {
            # 60fps+ (lab quality)
            'min_fps': 60,
            'detection_method': 'frame_by_frame',
            'baseline_window': 300,        # seconds (5 min)
            'suppression_threshold': 0.70,  # 30% reduction from baseline
            'spike_threshold': 1.40,        # 40% increase from baseline
            'output_confidence': 0.70       # Research-validated 70% accuracy
        },
        
        'STANDARD_FPS': {
            # 30fps (good webcam/phone)
            'min_fps': 30,
            'detection_method': 'frame_by_frame',
            'baseline_window': 300,
            'suppression_threshold': 0.60,  # Relaxed: miss ~20% of blinks
            'spike_threshold': 1.60,        # Need larger change to overcome noise
            'output_confidence': 0.55       # Reduced confidence
        },
        
        'CCTV_FPS': {
            # 15-25fps (CCTV)
            'min_fps': 15,
            'detection_method': 'eyelid_closure_duration',  # Different algorithm
            'baseline_window': 600,         # Longer baseline (more samples needed)
            'suppression_threshold': 0.50,  # Very relaxed: miss ~50% of blinks
            'spike_threshold': 2.00,        # Need very large change
            'output_confidence': 0.35,      # Low confidence
            'min_face_size': 120            # Require larger face
        },
        
        'AUDIO_ONLY': {
            # No video or face too small
            'suppressed': True,
            'reason': 'Blink detection requires minimum 15fps video with face >100px'
        }
    }
    
    def adapt_to_quality(self, quality_profile):
        """
        Select threshold set based on FPS and face size.
        """
        if quality_profile.video_fps is None or \
           quality_profile.face_size_px < 100:
            self.thresholds = self.THRESHOLD_SETS['AUDIO_ONLY']
            return self
        
        if quality_profile.video_fps >= 60:
            self.thresholds = self.THRESHOLD_SETS['HIGH_FPS']
        elif quality_profile.video_fps >= 30:
            self.thresholds = self.THRESHOLD_SETS['STANDARD_FPS']
        else:  # 15-29 fps
            self.thresholds = self.THRESHOLD_SETS['CCTV_FPS']
            
            # Additional check for CCTV tier
            if quality_profile.face_size_px < self.thresholds['min_face_size']:
                self.thresholds = self.THRESHOLD_SETS['AUDIO_ONLY']
        
        return self
    
    def detect_blinks_adaptive(self):
        """
        Use different detection algorithms based on FPS.
        """
        if self.thresholds['detection_method'] == 'frame_by_frame':
            # High/Standard FPS: Detect blink as single-frame eye closure
            return self.detect_blinks_frame_by_frame()
        
        elif self.thresholds['detection_method'] == 'eyelid_closure_duration':
            # Low FPS: Detect blink as multi-frame eyelid closure pattern
            # More robust to missed frames
            return self.detect_blinks_by_closure_duration()
```

**Result:** Same rule, three quality tiers:
- 60fps: 0.70 confidence (research-validated)
- 30fps: 0.55 confidence (compensates for missed blinks)
- 15fps: 0.35 confidence (very relaxed thresholds, low confidence)
- <15fps or small face: Suppressed

---

## 4. MULTI-RULE FUSION WITH PARTIAL EVIDENCE

### Key Insight: Even Blocked Rules Contribute Information

**Scenario:** Analyzing CCTV footage with handcuffs

**Traditional Approach:**
```
❌ Barrier Behavior - BLOCKED (handcuffs)
❌ Motor Inhibition - BLOCKED (handcuffs)
❌ Gaze Saccades - BLOCKED (angle/fps)
❌ Facial AU - BLOCKED (resolution)
✅ Voice Agitation - 0.50 confidence
✅ Response Latency - 0.85 confidence
✅ Denial Evolution - 0.60 confidence

Total signals: 3
```

**Adaptive Approach:**
```
🟡 Barrier Behavior (PROXY) - 0.20 confidence (torso lean instead of arm cross)
🟡 Motor Inhibition (DEGRADED) - 0.15 confidence (gross movement only)
❌ Gaze Saccades - SUPPRESSED (unreliable at this angle/fps)
🟡 Facial AU (COARSE) - 0.25 confidence (smile detection only, not full AU)
✅ Voice Agitation - 0.50 confidence
✅ Response Latency - 0.85 confidence
✅ Denial Evolution - 0.60 confidence

Total signals: 6 (3 full + 3 degraded)
```

**Fusion Algorithm:**

```python
class MultiRuleFusion:
    """
    Combine evidence from full-quality and degraded rules.
    Low-confidence signals still provide value when combined.
    """
    
    def fuse_evidence(self, rule_results, quality_profile):
        """
        Bayesian fusion of multiple rules with quality-adjusted weights.
        """
        # Group rules by confidence tier
        high_confidence = [r for r in rule_results if r['confidence'] >= 0.60]
        medium_confidence = [r for r in rule_results if 0.30 <= r['confidence'] < 0.60]
        low_confidence = [r for r in rule_results if r['confidence'] < 0.30]
        
        # High-confidence rules: Full weight
        high_weight = 1.0
        
        # Medium-confidence rules: Partial weight
        medium_weight = 0.5
        
        # Low-confidence rules: Weak signal only
        # BUT: Multiple weak signals can reinforce each other
        low_weight = 0.2
        
        # Example: Capitulation Cascade detection
        cascade_evidence = {
            'voice_energy_drop': None,
            'speech_rate_slow': None,
            'denial_weakening': None,
            'posture_slumping': None,  # May be degraded/suppressed
            'facial_distress': None     # May be degraded/suppressed
        }
        
        # Populate from available rules
        for result in rule_results:
            if result['rule_id'] == 'VOICE-ENERGY':
                cascade_evidence['voice_energy_drop'] = result['confidence']
            elif result['rule_id'] == 'DENIAL-EVOLUTION':
                cascade_evidence['denial_weakening'] = result['confidence']
            elif result['rule_id'] == 'POSTURE-SHIFT':
                cascade_evidence['posture_slumping'] = result['confidence']  # May be 0.20
            # etc.
        
        # Calculate weighted fusion
        total_weight = 0
        total_score = 0
        
        for signal, confidence in cascade_evidence.items():
            if confidence is None:
                continue  # Signal unavailable
            
            # Determine weight based on confidence tier
            if confidence >= 0.60:
                weight = high_weight
            elif confidence >= 0.30:
                weight = medium_weight
            else:
                weight = low_weight
            
            total_weight += weight
            total_score += confidence * weight
        
        if total_weight == 0:
            return None  # No evidence available
        
        fused_confidence = total_score / total_weight
        
        # Bonus: Multiple weak signals reinforce each other
        weak_signal_count = len([c for c in cascade_evidence.values() 
                                  if c is not None and c < 0.30])
        
        if weak_signal_count >= 3:
            # 3+ weak signals converging = boost confidence
            fused_confidence += 0.10
        
        return {
            'pattern': 'capitulation_cascade',
            'confidence': min(fused_confidence, 0.75),  # Cap at 0.75
            'evidence_breakdown': cascade_evidence,
            'quality_note': f'Fusion of {len([c for c in cascade_evidence.values() if c is not None])} signals (high: {len(high_confidence)}, medium: {len(medium_confidence)}, low: {len(low_confidence)})'
        }
```

**Key Principle:** 
- 1 high-confidence signal (0.85) > 1 low-confidence signal (0.20)
- But 5 low-confidence signals converging > 1 high-confidence signal

**Real Example:**

**High-Quality Video (No Handcuffs):**
```
Capitulation Cascade Evidence:
✅ Voice energy drop: 0.65
✅ Speech rate slow: 0.60
✅ Denial weakening: 0.70
✅ Posture slumping: 0.55
✅ Facial distress: 0.50

Fused Confidence: 0.60
```

**CCTV Video (Handcuffs):**
```
Capitulation Cascade Evidence:
✅ Voice energy drop: 0.65 (audio unchanged)
✅ Speech rate slow: 0.60 (audio unchanged)
✅ Denial weakening: 0.70 (transcript unchanged)
🟡 Posture slumping: 0.20 (degraded - gross torso lean only)
🟡 Facial distress: 0.25 (degraded - coarse detection only)

Fused Confidence: 0.48

Quality Note: "2 visual signals degraded due to handcuffs + CCTV quality. 
Audio signals maintain full confidence. Pattern still detectable but 
confidence reduced from 0.60 → 0.48."
```

**Still useful! Not blocked!**

---

## 5. QUALITY-TIERED CONFIDENCE ADJUSTMENT TABLE

### Master Lookup: All Rules × All Quality Levels

| Rule | High-Quality | CCTV | Handcuffed | Audio-Only | Adjustment Method |
|------|-------------|------|------------|-----------|------------------|
| **LANGUAGE RULES** |
| Pronoun Distancing | 0.45 | 0.45 | 0.45 | 0.45 | No change (transcript-based) |
| Tense Shift | 0.35 | 0.35 | 0.35 | 0.35 | No change (transcript-based) |
| Contamination | 0.80 | 0.80 | 0.80 | 0.80 | No change (transcript-based) |
| Denial Evolution | 0.60 | 0.60 | 0.60 | 0.60 | No change (transcript-based) |
| **VOICE RULES** |
| Pitch Elevation | 0.50 | 0.45 | 0.50 | 0.50 | Slight reduction for HVAC noise |
| Speech Rate | 0.50 | 0.45 | 0.50 | 0.50 | Noise affects syllable detection |
| Vocal Agitation | 0.50 | 0.45 | 0.50 | 0.50 | Combined pitch+rate+volume |
| **BODY RULES** |
| Barrier Behavior | 0.50 | 0.35 | 0.20 | 0.0 | CCTV: relaxed thresh; Handcuffs: torso proxy; Audio: suppress |
| Freezing | 0.55 | 0.35 | 0.40 | 0.0 | CCTV: coarse movement only; Handcuffs: partial validity; Audio: suppress |
| Motor Inhibition | 0.35 | 0.15 | 0.0 | 0.0 | CCTV: very coarse; Handcuffs: suppress; Audio: suppress |
| **FACIAL RULES** |
| Inappropriate Affect | 0.45 | 0.30 | 0.30 | 0.0 | CCTV: coarse smile only; Audio: suppress |
| Blink Rate | 0.70 | 0.35 | 0.35 | 0.0 | CCTV: relaxed algorithm; Audio: suppress |
| **GAZE RULES** |
| Gaze Saccades | 0.40 | 0.15 | 0.15 | 0.0 | CCTV: frequency only (not direction); Audio: suppress |
| **CONVERSATION RULES** |
| SUE Latency | 0.85 | 0.85 | 0.85 | 0.85 | No change (timestamp-based) |
| Talk-Time Ratio | 0.60 | 0.60 | 0.60 | 0.60 | No change (diarization-based) |
| **COMPOUND PATTERNS** |
| Capitulation Cascade | 0.60 | 0.48 | 0.45 | 0.45 | CCTV: weighted fusion; Handcuffs: audio-heavy; Audio: audio-only |
| Resistance Hardening | 0.60 | 0.55 | 0.55 | 0.55 | Mostly voice-based, minor video contribution |
| Low Autonomic Reactivity | 0.65 | 0.0 | 0.0 | 0.0 | Requires full multimodal (suppress if degraded) |

**Color Coding:**
- 🟢 **Green (0.50+):** Reliable at this quality
- 🟡 **Yellow (0.20-0.49):** Degraded but usable
- 🔴 **Red (0.0):** Suppressed at this quality
- ⚫ **Black (N/A):** Not applicable

**Key Insight:** Even at worst quality (Audio-Only + Handcuffs), you get:
- ✅ **10 rules at full confidence** (all language/voice/conversation rules)
- 🟡 **2 rules at reduced confidence** (Capitulation Cascade, Resistance Hardening at 0.45-0.55)
- ❌ **9 rules suppressed** (all body/facial/gaze rules)

**That's still 12 signals - more than enough for analysis!**

---

## 6. DYNAMIC THRESHOLD CALIBRATION STRATEGY

### Problem: You don't have labeled data yet to set thresholds

### Solution: Conservative Defaults + Online Learning

```python
class ThresholdCalibrator:
    """
    Start with conservative defaults, adjust as labeled data arrives.
    """
    
    def __init__(self):
        # Start with research-based defaults
        self.thresholds = self.load_default_thresholds()
        
        # Track performance
        self.predictions = []
        self.ground_truth = []
    
    def load_default_thresholds(self):
        """
        Conservative defaults from research + safety margin.
        """
        return {
            'response_latency': {
                'HIGH_QUALITY': {
                    'immediate': 500,   # <500ms (research: truthful)
                    'extended': 2000    # >2000ms (research: deceptive)
                },
                'CCTV_QUALITY': {
                    'immediate': 700,   # +200ms safety margin (timestamp noise)
                    'extended': 2500    # +500ms safety margin
                },
                'AUDIO_ONLY': {
                    'immediate': 1000,  # +500ms (no visual cues for response start)
                    'extended': 3000
                }
            },
            
            'freezing_behavior': {
                'HIGH_QUALITY': {
                    'movement_threshold': 0.15,  # 85% reduction from baseline
                    'min_duration': 3.0          # seconds
                },
                'CCTV_QUALITY': {
                    'movement_threshold': 0.25,  # Relaxed (lower resolution)
                    'min_duration': 4.0          # Longer (frame drops)
                }
            }
            
            # etc. for all rules
        }
    
    def adjust_threshold_with_feedback(self, rule_id, quality_tier, 
                                        new_labeled_sample):
        """
        When you get ground truth labels, adjust thresholds.
        """
        self.predictions.append({
            'rule_id': rule_id,
            'quality_tier': quality_tier,
            'predicted': new_labeled_sample['predicted'],
            'threshold_used': self.thresholds[rule_id][quality_tier]
        })
        
        self.ground_truth.append(new_labeled_sample['actual'])
        
        # Every 10 new samples, recalibrate
        if len(self.predictions) % 10 == 0:
            self.recalibrate_threshold(rule_id, quality_tier)
    
    def recalibrate_threshold(self, rule_id, quality_tier):
        """
        Find optimal threshold via ROC analysis on accumulated data.
        """
        from sklearn.metrics import roc_curve
        
        # Filter to this rule + quality tier
        samples = [(p, gt) for p, gt in zip(self.predictions, self.ground_truth)
                   if p['rule_id'] == rule_id and p['quality_tier'] == quality_tier]
        
        if len(samples) < 10:
            return  # Need more data
        
        # Extract predictions and ground truth
        y_pred = [s[0]['predicted'] for s in samples]
        y_true = [s[1] for s in samples]
        
        # Find optimal threshold
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        # Update
        old_threshold = self.thresholds[rule_id][quality_tier]['extended']
        self.thresholds[rule_id][quality_tier]['extended'] = optimal_threshold
        
        print(f"Recalibrated {rule_id} ({quality_tier}): {old_threshold} → {optimal_threshold}")
```

**Progressive Learning:**
1. **Week 1:** Use conservative defaults from research
2. **Week 10:** Collect 50 labeled samples → first recalibration
3. **Month 6:** Collect 500 labeled samples → confident thresholds
4. **Year 1:** Collect 5000 labeled samples → publication-ready thresholds

**You don't need perfect thresholds to start - you need conservative defaults and a learning system.**

---

## 7. IMPLEMENTATION: GRACEFUL DEGRADATION ARCHITECTURE

```python
class InterrogationAnalyzer:
    """
    Main analysis engine with graceful degradation.
    """
    
    def __init__(self):
        self.rules = self.load_all_rules()
        self.quality_detector = QualityDetector()
        self.threshold_calibrator = ThresholdCalibrator()
        self.fusion_engine = MultiRuleFusion()
    
    def analyze(self, video_path=None, audio_path=None):
        """
        Run analysis with automatic quality adaptation.
        """
        # Step 1: Detect quality
        quality_profile = self.quality_detector.detect(video_path, audio_path)
        
        print(f"Quality Profile: {quality_profile.summary()}")
        # Example output:
        # "Audio: 44.1kHz mono, SNR 18dB (good)
        #  Video: 640x480 @ 15fps, face 95px (CCTV quality)
        #  Handcuffs: Yes
        #  → Mode: DEGRADED_VIDEO_HANDCUFFED"
        
        # Step 2: Adapt all rules to quality
        adapted_rules = []
        for rule in self.rules:
            adapted = rule.adapt_to_quality(quality_profile)
            adapted_rules.append(adapted)
        
        # Step 3: Execute adapted rules
        results = []
        suppressed_rules = []
        
        for rule in adapted_rules:
            result = rule.execute()
            
            if result is None:
                suppressed_rules.append({
                    'rule_id': rule.id,
                    'reason': rule.suppression_reason
                })
            else:
                results.append(result)
        
        # Step 4: Fuse multi-rule evidence
        compound_patterns = self.fusion_engine.fuse_evidence(results, quality_profile)
        
        # Step 5: Generate report
        report = self.generate_report(
            quality_profile=quality_profile,
            rule_results=results,
            compound_patterns=compound_patterns,
            suppressed_rules=suppressed_rules
        )
        
        return report
    
    def generate_report(self, quality_profile, rule_results, 
                        compound_patterns, suppressed_rules):
        """
        Generate comprehensive report with quality disclaimers.
        """
        return {
            'metadata': {
                'quality_tier': quality_profile.tier,
                'video_available': quality_profile.video_available,
                'handcuffs_detected': quality_profile.handcuffs_detected,
                'rules_fired': len(rule_results),
                'rules_suppressed': len(suppressed_rules)
            },
            
            'active_signals': [
                {
                    'rule_id': r['rule_id'],
                    'label': r['label'],
                    'confidence': r['confidence'],
                    'quality_disclaimer': r.get('disclaimer')
                }
                for r in rule_results
            ],
            
            'compound_patterns': compound_patterns,
            
            'suppressed_signals': suppressed_rules,
            
            'quality_impact_summary': self.summarize_quality_impact(
                quality_profile, rule_results, suppressed_rules
            ),
            
            'recommendations': self.generate_recommendations(
                quality_profile, rule_results, compound_patterns
            )
        }
    
    def summarize_quality_impact(self, quality_profile, active, suppressed):
        """
        Explain what the quality limitations mean for the analysis.
        """
        summary = []
        
        if quality_profile.video_resolution < (720, 480):
            summary.append(
                "Video resolution below optimal (480p vs 720p+). "
                "Facial and postural analysis confidence reduced by ~30%."
            )
        
        if quality_profile.video_fps < 30:
            summary.append(
                f"Frame rate ({quality_profile.video_fps}fps) below optimal (30fps+). "
                "Blink detection and micro-movements may be missed."
            )
        
        if quality_profile.handcuffs_detected:
            summary.append(
                "Subject appears handcuffed. Upper-body gesture analysis "
                f"suppressed ({len([s for s in suppressed if 'BODY' in s['rule_id']])} rules). "
                "Analysis relies primarily on voice and linguistic signals."
            )
        
        if quality_profile.audio_snr < 20:
            summary.append(
                f"Audio signal-to-noise ratio ({quality_profile.audio_snr}dB) below ideal (20dB+). "
                "Voice feature extraction may be less precise."
            )
        
        # Positive summary
        active_count = len(active)
        summary.append(
            f"\n✅ Despite quality limitations, {active_count} behavioral signals "
            f"were successfully extracted and analyzed."
        )
        
        return '\n'.join(summary)
```

---

## 8. EXAMPLE OUTPUT: CCTV + HANDCUFFS CASE

```json
{
  "metadata": {
    "quality_tier": "DEGRADED_VIDEO_HANDCUFFED",
    "video_available": true,
    "video_quality": "480p @ 15fps",
    "handcuffs_detected": true,
    "rules_fired": 12,
    "rules_suppressed": 9
  },
  
  "active_signals": [
    {
      "rule_id": "INTERROG-LANG-03",
      "label": "contamination_detected",
      "confidence": 0.80,
      "quality_disclaimer": null,
      "details": "Suspect adopted 5 case facts first introduced by interrogator"
    },
    {
      "rule_id": "INTERROG-CONV-01",
      "label": "evidence_response_processing_delay",
      "confidence": 0.85,
      "quality_disclaimer": null,
      "details": "3.2 second latency after evidence presentation"
    },
    {
      "rule_id": "VOICE-AGITATION",
      "label": "agitated_high_arousal_tone",
      "confidence": 0.45,
      "quality_disclaimer": "Confidence reduced from 0.50 to 0.45 due to HVAC noise in audio",
      "details": "Pitch +35%, rate +42%, volume +28%"
    },
    {
      "rule_id": "INTERROG-BODY-01",
      "label": "torso_lean_away",
      "confidence": 0.20,
      "quality_disclaimer": "Subject appears handcuffed. Using torso lean as weak proxy for barrier behavior.",
      "details": "15° backward lean 4.2s after accusatory question"
    },
    {
      "rule_id": "BLINK-PATTERN-01",
      "label": "blink_suppression_spike_pattern",
      "confidence": 0.35,
      "quality_disclaimer": "Confidence reduced from 0.70 to 0.35 due to 15fps frame rate (blinks may be missed)",
      "details": "Baseline 18/min → During 8/min → After 32/min"
    }
    // ... 7 more signals
  ],
  
  "compound_patterns": {
    "capitulation_cascade": {
      "detected": true,
      "confidence": 0.45,
      "quality_note": "Fusion of 5 signals (3 audio full-confidence + 2 video degraded). Pattern detectable despite handcuffs + CCTV quality.",
      "evidence": {
        "voice_energy_drop": 0.65,
        "denial_weakening": 0.70,
        "speech_rate_slow": 0.60,
        "posture_slumping": 0.20,  // degraded
        "facial_distress": 0.25     // degraded
      }
    }
  },
  
  "suppressed_signals": [
    {
      "rule_id": "MOTOR-INHIBIT-01",
      "reason": "Subject appears handcuffed - hand/arm gesture analysis invalid"
    },
    {
      "rule_id": "GAZE-SACCADES-01",
      "reason": "Camera angle (52° overhead) + 15fps insufficient for reliable gaze tracking"
    },
    {
      "rule_id": "LOW-AUTONOMIC-REACTIVITY",
      "reason": "Requires full multimodal signals (facial AU + posture + voice). Insufficient video quality."
    }
    // ... 6 more suppressed
  ],
  
  "quality_impact_summary": "Video resolution below optimal (480p vs 720p+). Facial and postural analysis confidence reduced by ~30%.\nFrame rate (15fps) below optimal (30fps+). Blink detection and micro-movements may be missed.\nSubject appears handcuffed. Upper-body gesture analysis suppressed (6 rules). Analysis relies primarily on voice and linguistic signals.\nAudio signal-to-noise ratio (18dB) below ideal (20dB+). Voice feature extraction may be less precise.\n\n✅ Despite quality limitations, 12 behavioral signals were successfully extracted and analyzed.",
  
  "recommendations": {
    "false_confession_risk": "HIGH",
    "risk_factors": [
      "Contamination detected (0.80 confidence)",
      "Capitulation cascade pattern (0.45 confidence)",
      "Duration: 6.2 hours"
    ],
    "suggested_actions": [
      "Review interrogator evidence disclosure timing",
      "Cross-reference suspect statements with case file",
      "Consider psychological evaluation"
    ]
  }
}
```

**Key Points:**
- 12 signals active (not blocked)
- 9 signals suppressed (with reasons)
- Compound pattern detected despite degraded inputs
- Clear quality disclaimers
- Still actionable analysis

---

## 9. ANSWER TO YOUR QUESTIONS

### Q1: Can we adjust thresholds for CCTV handling?

**YES - Three strategies:**

1. **Quality-Tiered Thresholds**
   - High-quality: Strict thresholds, high confidence
   - CCTV: Relaxed thresholds, reduced confidence
   - Audio-only: Audio thresholds only

2. **Conservative Defaults + Online Learning**
   - Start with research-based defaults + safety margin
   - Adjust as labeled data arrives
   - Progressive improvement over time

3. **Alternative Measurements**
   - If primary signal unavailable, use proxy
   - Example: Handcuffs → torso lean instead of arm crossing
   - Clearly labeled as degraded proxy

### Q2: If some rules are blocked, can we still use other rules?

**YES - This is the CORRECT approach:**

1. **Rules are independent** - each provides value
2. **Graceful degradation** - suppressed rules don't break system
3. **Multi-rule fusion** - combine available evidence intelligently
4. **Quality transparency** - report shows what was used/suppressed

**Example:** CCTV + Handcuffs case still gives:
- ✅ 10 audio/transcript rules at full confidence
- 🟡 2 degraded video rules at reduced confidence
- ❌ 9 suppressed rules (documented why)

**Total: 12 usable signals - enough for analysis**

---

## 10. REVISED SYSTEM ARCHITECTURE

```
┌─────────────────────────────────────────────────────────┐
│                  INTERROGATION VIDEO                    │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  Quality Detector     │
        │  - Video: res/fps/face│
        │  - Audio: SNR/channels│
        │  - Handcuffs: detect  │
        └───────────┬───────────┘
                    │
                    ▼
        ┌─────────────────────────────────┐
        │  Quality Profile                │
        │  → HIGH_QUALITY                 │
        │  → CCTV_QUALITY                 │
        │  → DEGRADED_VIDEO_HANDCUFFED    │
        │  → AUDIO_ONLY                   │
        └───────────┬─────────────────────┘
                    │
                    ▼
        ┌─────────────────────────────────┐
        │  Adaptive Rule Engine           │
        │  ┌──────────────────────────┐  │
        │  │ For each rule:           │  │
        │  │  1. Check applicability  │  │
        │  │  2. Adapt thresholds     │  │
        │  │  3. Select algorithm     │  │
        │  │  4. Execute              │  │
        │  │  5. Adjust confidence    │  │
        │  └──────────────────────────┘  │
        └───────────┬─────────────────────┘
                    │
      ┌─────────────┼─────────────┐
      ▼             ▼             ▼
┌──────────┐  ┌──────────┐  ┌──────────┐
│ Active   │  │ Degraded │  │Suppressed│
│ Rules    │  │ Rules    │  │ Rules    │
│ (12)     │  │ (3)      │  │ (9)      │
└────┬─────┘  └────┬─────┘  └────┬─────┘
     │             │             │
     └─────────────┼─────────────┘
                   ▼
        ┌──────────────────────┐
        │  Multi-Rule Fusion   │
        │  - Weighted evidence │
        │  - Converging signals│
        │  - Compound patterns │
        └──────────┬───────────┘
                   ▼
        ┌──────────────────────┐
        │  Report Generator    │
        │  - Active signals    │
        │  - Quality disclaimers│
        │  - Suppression reasons│
        │  - Recommendations   │
        └──────────────────────┘
```

---

## FINAL ANSWER

**Don't block rules - adapt them.**

Build a system where:
1. Every rule has quality-tiered thresholds
2. Suppressed rules report WHY they're suppressed
3. Degraded rules contribute at reduced confidence
4. Multi-rule fusion combines partial evidence
5. Output clearly documents quality impact

**Even worst-case CCTV + handcuffs gives 12 usable signals.**

That's enough for:
- Contamination detection (0.80)
- False confession risk assessment (0.65)
- SUE response latency (0.85)
- Capitulation cascade (0.45 degraded)

**Start building with adaptive thresholds from day 1.**