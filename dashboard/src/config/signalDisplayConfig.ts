export interface SignalDisplay {
  label: string;
  description: string;
  icon: string;
  color: string;
  category: 'face' | 'body' | 'gaze' | 'voice' | 'interaction' | 'pattern';
  priority: number; // 1=always show, 2=expanded, 3=show all
}

export function getSignalDisplay(
  signalType: string,
  valueText: string = '',
  _confidence: number = 0,
): SignalDisplay {
  const vtKey = `${signalType}:${valueText}`;
  if (SIGNAL_MAP[vtKey]) return SIGNAL_MAP[vtKey];
  if (SIGNAL_MAP[signalType]) return SIGNAL_MAP[signalType];
  return {
    label: signalType.replace(/_/g, ' '),
    description: '',
    icon: '●',
    color: '#9CA3AF',
    category: 'body',
    priority: 3,
  };
}

const SIGNAL_MAP: Record<string, SignalDisplay> = {

  // ── Face touch zones ────────────────────────────────────────────────────────
  'face_region_touch:chin_touch_evaluation': {
    label: 'Evaluating',
    description: "Touching chin while listening — weighing what's being said, decision-making in progress",
    icon: '🤔', color: '#F59E0B', category: 'body', priority: 1,
  },
  'face_region_touch:mouth_cover_suppression': {
    label: 'Holding Back',
    description: "Hand covering mouth — suppressing a reaction or opinion they're not sharing",
    icon: '🤐', color: '#F59E0B', category: 'body', priority: 1,
  },
  'face_region_touch:nose_touch_discomfort': {
    label: 'Uncomfortable',
    description: "Touching nose — self-soothing, something being discussed makes them uneasy",
    icon: '😟', color: '#EF4444', category: 'body', priority: 1,
  },
  'face_region_touch:cheek_touch_listening': {
    label: 'Attentive',
    description: "Light face touch while listening — engaged and processing what's being said",
    icon: '👂', color: '#10B981', category: 'body', priority: 2,
  },
  'face_region_touch:cheek_rest_fatigue': {
    label: 'Low Energy',
    description: 'Head resting on hand — losing focus, bored, or fatigued',
    icon: '💤', color: '#EF4444', category: 'body', priority: 1,
  },
  'face_region_touch:ear_touch_soothing': {
    label: 'Self-Soothing',
    description: 'Touching ear — anxiety or stress response, trying to calm themselves',
    icon: '😰', color: '#F59E0B', category: 'body', priority: 2,
  },
  'face_region_touch:neck_touch_vulnerability': {
    label: 'Feeling Exposed',
    description: "Touching neck — feels put on the spot or anxious about what's being discussed",
    icon: '😓', color: '#EF4444', category: 'body', priority: 1,
  },
  'face_region_touch:forehead_touch_frustration': {
    label: 'Frustrated',
    description: 'Touching forehead — cognitive strain, struggling with the topic',
    icon: '😤', color: '#EF4444', category: 'body', priority: 1,
  },

  // ── Body posture ────────────────────────────────────────────────────────────
  'self_touch': {
    label: 'Self-Touch',
    description: 'Hand touching face — self-soothing gesture, often indicates stress or deep thought',
    icon: '✋', color: '#F59E0B', category: 'body', priority: 2,
  },
  'arms_crossed': {
    label: 'Guarded',
    description: 'Arms crossed — closed posture, may feel defensive (only significant if changed from baseline)',
    icon: '🛡️', color: '#F59E0B', category: 'body', priority: 1,
  },
  'finger_steepling': {
    label: 'Confident',
    description: 'Fingertips pressed together — authority display, feels in control and ready to speak',
    icon: '👑', color: '#10B981', category: 'body', priority: 1,
  },
  'hands_clasped': {
    label: 'Restrained',
    description: 'Hands folded together — holding back energy, waiting patiently or exercising self-control',
    icon: '🤝', color: '#6B7280', category: 'body', priority: 2,
  },
  'head_supported:head_resting_disengagement': {
    label: 'Checked Out',
    description: 'Head resting on hand while listening — mentally disengaged from the conversation',
    icon: '💤', color: '#EF4444', category: 'body', priority: 1,
  },
  'head_supported:head_resting_contemplation': {
    label: 'Deep in Thought',
    description: 'Head resting on hand while speaking — pausing to think through a point',
    icon: '💭', color: '#F59E0B', category: 'body', priority: 2,
  },
  'body_fidgeting': {
    label: 'Restless',
    description: 'Elevated body movement — nervous energy, impatience, or discomfort',
    icon: '⚡', color: '#F59E0B', category: 'body', priority: 2,
  },
  'shoulder_tension': {
    label: 'Tense',
    description: 'Raised or tight shoulders — physical stress response, on guard',
    icon: '⚠️', color: '#F59E0B', category: 'body', priority: 2,
  },

  // ── Body lean + posture ──────────────────────────────────────────────────────
  'body_lean:forward_lean': {
    label: 'Interested',
    description: "Leaning forward — engaged, wants to hear more or feels connected to the topic",
    icon: '🔥', color: '#10B981', category: 'body', priority: 1,
  },
  'body_lean:backward_lean': {
    label: 'Pulling Back',
    description: 'Leaning away — creating distance, may disagree or have lost interest',
    icon: '↩️', color: '#F59E0B', category: 'body', priority: 1,
  },
  'posture:upright_power_posture': {
    label: 'Authoritative',
    description: 'Straight spine, open chest — projecting confidence and control',
    icon: '🏛️', color: '#10B981', category: 'body', priority: 2,
  },
  'posture:forward_slump': {
    label: 'Deflated',
    description: 'Slumped posture — energy drop, fatigue, or loss of confidence',
    icon: '📉', color: '#EF4444', category: 'body', priority: 2,
  },

  // ── Head gestures ────────────────────────────────────────────────────────────
  'head_nod': {
    label: 'Agreeing',
    description: 'Nodding while listening — agreement or encouragement to continue',
    icon: '✅', color: '#10B981', category: 'body', priority: 1,
  },
  'head_shake': {
    label: 'Disagreeing',
    description: "Shaking head — internal disagreement, even if words say otherwise",
    icon: '❌', color: '#EF4444', category: 'body', priority: 1,
  },

  // ── Posture transitions ──────────────────────────────────────────────────────
  'posture_transition:closing_up': {
    label: 'Shutting Down',
    description: 'Shifted from open to closed posture — something triggered a defensive reaction',
    icon: '🚪', color: '#EF4444', category: 'body', priority: 1,
  },
  'posture_transition:opening_up': {
    label: 'Warming Up',
    description: 'Shifted from closed to open posture — becoming more comfortable or receptive',
    icon: '🌤️', color: '#10B981', category: 'body', priority: 1,
  },
  'posture_transition:disengaging': {
    label: 'Losing Interest',
    description: 'Was engaged, now withdrawn — the topic or conversation lost them',
    icon: '📴', color: '#EF4444', category: 'body', priority: 1,
  },
  'posture_transition:re_engaging': {
    label: 'Back In',
    description: 'Was withdrawn, now re-engaged — something recaptured their attention',
    icon: '🔄', color: '#10B981', category: 'body', priority: 1,
  },
  'posture_transition:defensive_shift': {
    label: 'Getting Defensive',
    description: 'Abrupt shift to closed posture — feels challenged or threatened',
    icon: '⚠️', color: '#EF4444', category: 'body', priority: 1,
  },

  // ── Face / eyes ──────────────────────────────────────────────────────────────
  'lip_pursing': {
    label: 'Biting Their Tongue',
    description: "Lips pressed together while listening — has something to say but isn't voicing it",
    icon: '🤐', color: '#F59E0B', category: 'face', priority: 1,
  },
  'facial_stress': {
    label: 'Under Pressure',
    description: 'Brow tension and jaw clenching — stress visible on face, even if voice sounds calm',
    icon: '😣', color: '#EF4444', category: 'face', priority: 1,
  },
  'facial_emotion:happy': {
    label: 'Positive',
    description: 'Genuine positive expression detected',
    icon: '😊', color: '#10B981', category: 'face', priority: 2,
  },
  'facial_emotion:angry': {
    label: 'Frustrated',
    description: 'Anger visible in facial expression — furrowed brow, tightened jaw',
    icon: '😠', color: '#EF4444', category: 'face', priority: 1,
  },
  'facial_emotion:sad': {
    label: 'Discouraged',
    description: 'Sadness visible — downturned mouth, lowered brow',
    icon: '😔', color: '#EF4444', category: 'face', priority: 2,
  },
  'facial_emotion:surprised': {
    label: 'Surprised',
    description: 'Unexpected reaction — raised eyebrows, widened eyes',
    icon: '😲', color: '#F59E0B', category: 'face', priority: 2,
  },
  'facial_emotion:disgusted': {
    label: 'Put Off',
    description: 'Negative visceral reaction — nose wrinkle, upper lip raise',
    icon: '😒', color: '#EF4444', category: 'face', priority: 1,
  },
  'smile_type:duchenne': {
    label: 'Genuinely Happy',
    description: 'Real smile — eyes crinkle at the corners, authentic positive reaction',
    icon: '😄', color: '#10B981', category: 'face', priority: 1,
  },
  'smile_type:social': {
    label: 'Polite Smile',
    description: "Mouth-only smile — being socially appropriate but not genuinely happy",
    icon: '🙂', color: '#F59E0B', category: 'face', priority: 2,
  },
  'facial_engagement:high_engagement': {
    label: 'Expressive',
    description: 'High facial activity — actively reacting and engaged with the conversation',
    icon: '🎭', color: '#10B981', category: 'face', priority: 2,
  },
  'facial_engagement:low_engagement': {
    label: 'Flat Expression',
    description: 'Low facial activity — poker face, not visibly reacting',
    icon: '😐', color: '#F59E0B', category: 'face', priority: 2,
  },

  // ── Plain gaze types (no value_text suffix) ──────────────────────────────────
  'screen_contact': {
    label: 'Eye Contact',
    description: 'How much they look at the camera/screen during this period',
    icon: '👁️', color: '#3B82F6', category: 'gaze', priority: 2,
  },
  'attention_level': {
    label: 'Attention',
    description: 'Overall attentiveness based on gaze stability, blink rate, and head pose',
    icon: '🎯', color: '#3B82F6', category: 'gaze', priority: 2,
  },
  'blink_rate_anomaly': {
    label: 'Blink Rate',
    description: 'Blinking faster or slower than their baseline',
    icon: '👁️', color: '#F59E0B', category: 'gaze', priority: 3,
  },

  // ── Plain body types (no value_text suffix) ───────────────────────────────────
  'posture': {
    label: 'Posture',
    description: 'Body posture relative to their baseline — upright, slouched, or leaning',
    icon: '🧍', color: '#6B7280', category: 'body', priority: 2,
  },
  'body_lean': {
    label: 'Leaning',
    description: 'Body lean direction — forward (interested) or back (pulling away)',
    icon: '↕️', color: '#6B7280', category: 'body', priority: 2,
  },
  'head_body_incongruence': {
    label: 'Mixed Signals',
    description: 'Head gesture contradicts body posture — nodding while pulling away',
    icon: '🔀', color: '#F59E0B', category: 'pattern', priority: 1,
  },

  // ── Plain face types (no value_text suffix) ───────────────────────────────────
  'facial_engagement': {
    label: 'Expression Activity',
    description: 'How expressive their face is — high activity vs flat/poker face',
    icon: '🎭', color: '#3B82F6', category: 'face', priority: 2,
  },

  // ── Gaze ─────────────────────────────────────────────────────────────────────
  'sustained_distraction': {
    label: 'Not Paying Attention',
    description: 'Looking away from screen for extended time — mentally elsewhere',
    icon: '👀', color: '#EF4444', category: 'gaze', priority: 1,
  },
  'screen_contact:low_screen_contact': {
    label: 'Avoiding Eye Contact',
    description: 'Looking away from camera — discomfort, distraction, or multitasking',
    icon: '🙈', color: '#F59E0B', category: 'gaze', priority: 1,
  },
  'screen_contact:sustained_eye_contact': {
    label: 'Strong Eye Contact',
    description: 'Consistently looking at camera — focused, engaged, and present',
    icon: '🎯', color: '#10B981', category: 'gaze', priority: 2,
  },
  'attention_level:high_attention': {
    label: 'Fully Focused',
    description: 'Steady gaze, appropriate blink rate — giving full attention',
    icon: '🎯', color: '#10B981', category: 'gaze', priority: 2,
  },
  'attention_level:reduced_attention': {
    label: 'Distracted',
    description: 'Unstable gaze, frequent look-aways — attention is divided',
    icon: '📱', color: '#F59E0B', category: 'gaze', priority: 1,
  },
  'blink_rate_anomaly:elevated_blink_rate': {
    label: 'Stressed',
    description: 'Blinking faster than their baseline — cognitive load or anxiety',
    icon: '👁️', color: '#EF4444', category: 'gaze', priority: 2,
  },
  'blink_rate_anomaly:low_blink_rate': {
    label: 'Intense Focus',
    description: 'Blinking less than normal — deep concentration or surprise',
    icon: '🔍', color: '#F59E0B', category: 'gaze', priority: 3,
  },
  'gaze_direction_shift': {
    label: 'Looked Away',
    description: 'Shifted gaze direction — may be thinking, checking something, or losing interest',
    icon: '↗️', color: '#6B7280', category: 'gaze', priority: 3,
  },

  // ── Voice signals ─────────────────────────────────────────────────────────────
  'vocal_stress_score': {
    label: 'Voice Stress',
    description: 'Elevated vocal tension — pitch instability and tremor detected in speech',
    icon: '📈', color: '#EF4444', category: 'voice', priority: 1,
  },
  'filler_detection:elevated_fillers': {
    label: 'Lots of Fillers',
    description: '"Um", "uh", "like" — more filler words than baseline, may indicate uncertainty',
    icon: '💬', color: '#F59E0B', category: 'voice', priority: 1,
  },
  'filler_detection:filler_spike': {
    label: 'Filler Spike',
    description: 'Sudden jump in filler words — caught off guard or scrambling for words',
    icon: '⚡', color: '#EF4444', category: 'voice', priority: 1,
  },
  'tone_classification:confident': {
    label: 'Confident Tone',
    description: "Speaking with steady, authoritative tone — believes what they're saying",
    icon: '💪', color: '#10B981', category: 'voice', priority: 2,
  },
  'tone_classification:warm': {
    label: 'Warm Tone',
    description: 'Friendly, approachable speaking style — building rapport',
    icon: '☀️', color: '#10B981', category: 'voice', priority: 2,
  },
  'tone_classification:hesitant': {
    label: 'Hesitant',
    description: "Uncertain, wavering tone — not fully committed to what they're saying",
    icon: '❓', color: '#F59E0B', category: 'voice', priority: 1,
  },
  'tone_classification:confrontational': {
    label: 'Confrontational',
    description: 'Aggressive or challenging tone — conflict or pushback',
    icon: '⚔️', color: '#EF4444', category: 'voice', priority: 1,
  },
  'speech_rate_anomaly:speaking_fast': {
    label: 'Rushed',
    description: 'Speaking faster than baseline — excitement, nervousness, or trying to convince',
    icon: '🏃', color: '#F59E0B', category: 'voice', priority: 2,
  },
  'speech_rate_anomaly:speaking_slow': {
    label: 'Deliberate',
    description: 'Speaking slower than normal — choosing words carefully or losing energy',
    icon: '🐢', color: '#F59E0B', category: 'voice', priority: 2,
  },
  'pitch_elevation_flag': {
    label: 'Voice Pitch Spike',
    description: 'Pitch jumped above baseline — stress, excitement, or emphasis',
    icon: '📊', color: '#F59E0B', category: 'voice', priority: 2,
  },
  'monotone_flag': {
    label: 'Monotone',
    description: 'Flat, unchanging pitch — disengaged, reading a script, or fatigued',
    icon: '📏', color: '#F59E0B', category: 'voice', priority: 2,
  },
  'pause_classification:strategic_pause': {
    label: 'Strategic Pause',
    description: 'Deliberate silence for emphasis — skilled communication technique',
    icon: '⏸️', color: '#10B981', category: 'voice', priority: 2,
  },
  'pause_classification:hesitation_pause': {
    label: 'Hesitation',
    description: 'Unplanned pause — searching for words or uncertain about next point',
    icon: '⏳', color: '#F59E0B', category: 'voice', priority: 2,
  },
  'interruption_event': {
    label: 'Interrupted',
    description: 'One speaker cut off another — may indicate dominance, urgency, or disagreement',
    icon: '✂️', color: '#EF4444', category: 'voice', priority: 1,
  },

  // ── Language signals ─────────────────────────────────────────────────────────
  'buying_signal': {
    label: 'Buying Signal',
    description: "Language indicating interest in moving forward — asking about next steps, pricing, or timeline",
    icon: '💰', color: '#10B981', category: 'voice', priority: 1,
  },
  'objection_signal': {
    label: 'Objection',
    description: "Concern or pushback raised — needs to be addressed before they'll move forward",
    icon: '🚫', color: '#EF4444', category: 'voice', priority: 1,
  },
  'rapport_indicator': {
    label: 'Rapport Building',
    description: 'Language patterns indicate connection — shared references, agreement, warmth',
    icon: '🤝', color: '#10B981', category: 'voice', priority: 2,
  },
  'power_language_score': {
    label: 'Confident Language',
    description: 'Strong, decisive word choices — "will", "definitely" vs weak "might", "maybe"',
    icon: '💪', color: '#10B981', category: 'voice', priority: 2,
  },
  'conversation_engagement': {
    label: 'Engagement Level',
    description: "How actively they're participating in the conversation",
    icon: '📊', color: '#3B82F6', category: 'voice', priority: 2,
  },

  // ── Clusters ─────────────────────────────────────────────────────────────────
  'body_language_cluster:skepticism_cluster': {
    label: 'Skeptical',
    description: "Multiple cues aligned: chin touch + lean back + pursed lips — not buying what's being said",
    icon: '🔍', color: '#F59E0B', category: 'pattern', priority: 1,
  },
  'body_language_cluster:stress_anxiety_cluster': {
    label: 'Stressed',
    description: 'Multiple stress cues: neck touch + fidgeting + facial tension — uncomfortable with this topic',
    icon: '😰', color: '#EF4444', category: 'pattern', priority: 1,
  },
  'body_language_cluster:confidence_authority_cluster': {
    label: 'In Command',
    description: 'Multiple confidence cues: steepling + upright + steady gaze — feels fully in control',
    icon: '👑', color: '#10B981', category: 'pattern', priority: 1,
  },
  'body_language_cluster:disengagement_boredom_cluster': {
    label: 'Checked Out',
    description: 'Multiple disengagement cues: head resting + lean back + flat expression — mentally left the room',
    icon: '💤', color: '#EF4444', category: 'pattern', priority: 1,
  },

  // ── Cross-speaker interactions ────────────────────────────────────────────────
  'cross_speaker_interaction:agreement_reaction': {
    label: 'Agrees',
    description: "Nodding or smiling while the other person speaks — supporting what's being said",
    icon: '✅', color: '#10B981', category: 'interaction', priority: 1,
  },
  'cross_speaker_interaction:disagreement_reaction': {
    label: 'Disagrees',
    description: "Head shake or closed posture while listening — internal pushback on what's being said",
    icon: '❌', color: '#EF4444', category: 'interaction', priority: 1,
  },
  'cross_speaker_interaction:discomfort_reaction': {
    label: 'Uncomfortable',
    description: "Visible tension while the other person speaks — topic is sensitive for them",
    icon: '😬', color: '#F59E0B', category: 'interaction', priority: 1,
  },
  'cross_speaker_interaction:incongruent_reaction': {
    label: 'Mixed Signals',
    description: 'Face says one thing, body says another — smiling but tense, or nodding but arms crossed',
    icon: '🎭', color: '#EF4444', category: 'interaction', priority: 1,
  },
  'cross_speaker_interaction:disengagement_reaction': {
    label: 'Tuned Out',
    description: "Looking away or distracted while the other person speaks — not paying attention",
    icon: '📴', color: '#6B7280', category: 'interaction', priority: 2,
  },

  // ── Fusion compound patterns ──────────────────────────────────────────────────
  'genuine_engagement': {
    label: 'Genuinely Engaged',
    description: 'Voice, face, and body all signal authentic interest — high-quality interaction moment',
    icon: '⭐', color: '#10B981', category: 'pattern', priority: 1,
  },
  'active_disengagement': {
    label: 'Actively Disengaged',
    description: 'Multiple channels show withdrawal — voice flat, body closed, face disinterested',
    icon: '🔕', color: '#EF4444', category: 'pattern', priority: 1,
  },
  'emotional_suppression': {
    label: 'Suppressing Emotions',
    description: 'Holding back a visible reaction — stress present but being masked',
    icon: '🎭', color: '#F59E0B', category: 'pattern', priority: 1,
  },
  'decision_engagement': {
    label: 'Decision Point',
    description: 'Engagement spike around a buying signal — this is a critical conversion moment',
    icon: '🎯', color: '#10B981', category: 'pattern', priority: 1,
  },
  'cognitive_overload': {
    label: 'Overwhelmed',
    description: 'Too much information — high blink rate, stress, and processing difficulty',
    icon: '🧠', color: '#EF4444', category: 'pattern', priority: 1,
  },
  'peak_performance': {
    label: 'Peak Performance',
    description: 'Speaker is at their best — confident tone, strong language, engaged audience',
    icon: '🏆', color: '#10B981', category: 'pattern', priority: 1,
  },
  'conflict_escalation': {
    label: 'Tension Rising',
    description: 'Conflict indicators increasing — objections + stress + negative body language building',
    icon: '🌡️', color: '#EF4444', category: 'pattern', priority: 1,
  },
  'deception_cluster': {
    label: 'Inconsistency Detected',
    description: 'Multiple channels contradict — requires review (NOT a lie detector, just unusual pattern)',
    icon: '⚠️', color: '#EF4444', category: 'pattern', priority: 1,
  },
  'rapport_building': {
    label: 'Building Connection',
    description: 'Warm tone + mirroring + engagement — strong rapport being established',
    icon: '🤝', color: '#10B981', category: 'pattern', priority: 1,
  },
  'rapport_confirmation': {
    label: 'Rapport Confirmed',
    description: 'Voice tone, facial expression, and body language all confirm genuine rapport',
    icon: '🤝', color: '#10B981', category: 'pattern', priority: 1,
  },
  'verbal_nonverbal_discordance': {
    label: 'Words vs Body Mismatch',
    description: "What they're saying doesn't match how they look — voice and face tell different stories",
    icon: '🔀', color: '#F59E0B', category: 'pattern', priority: 1,
  },
  'credibility_assessment:credibility_concern': {
    label: 'Voice-Content Mismatch',
    description: "Positive words but stressed voice — may be uncomfortable with what they're claiming",
    icon: '⚠️', color: '#EF4444', category: 'pattern', priority: 1,
  },
  'verbal_incongruence:strong_verbal_incongruence': {
    label: 'Heavily Hedged Agreement',
    description: 'Saying yes but with heavy hedging language — agreeing without genuine conviction',
    icon: '🤷', color: '#F59E0B', category: 'pattern', priority: 1,
  },
  'urgency_authenticity:manufactured_urgency': {
    label: 'Artificial Urgency',
    description: 'Fast-paced persuasion with stress markers — urgency may be manufactured, not genuine',
    icon: '⏰', color: '#F59E0B', category: 'pattern', priority: 1,
  },
  'urgency_authenticity:authentic_urgency': {
    label: 'Genuine Excitement',
    description: 'Fast pace backed by confident vocal patterns — authentic enthusiasm detected',
    icon: '🚀', color: '#10B981', category: 'pattern', priority: 1,
  },
  'tone_face_masking': {
    label: 'Masking',
    description: "Voice tone doesn't match facial expression — putting on a performance",
    icon: '🎭', color: '#F59E0B', category: 'pattern', priority: 1,
  },
  'stress_suppression': {
    label: 'Hiding Stress',
    description: 'Voice sounds calm but face shows tension — actively suppressing visible stress',
    icon: '🫣', color: '#F59E0B', category: 'pattern', priority: 1,
  },

  // ── Temporal patterns ─────────────────────────────────────────────────────────
  'stress_trajectory:stress_trajectory_rising': {
    label: 'Stress Increasing',
    description: 'Vocal stress has been climbing throughout the session — growing discomfort',
    icon: '📈', color: '#EF4444', category: 'pattern', priority: 1,
  },
  'stress_trajectory:stress_trajectory_declining': {
    label: 'Relaxing',
    description: 'Vocal stress has decreased over the session — becoming more comfortable',
    icon: '📉', color: '#10B981', category: 'pattern', priority: 2,
  },
  'engagement_decay': {
    label: 'Engagement Fading',
    description: 'Engagement has declined over the session — losing the audience over time',
    icon: '📉', color: '#EF4444', category: 'pattern', priority: 1,
  },
  'adaptation_pattern': {
    label: 'Adapting',
    description: 'Speaker adjusted their style during the session — responded to feedback',
    icon: '🔄', color: '#10B981', category: 'pattern', priority: 2,
  },
  'escalation_ladder': {
    label: 'Conflict Escalating',
    description: 'Progressive increase in tension signals — conversation moving toward confrontation',
    icon: '🪜', color: '#EF4444', category: 'pattern', priority: 1,
  },
  'fatigue_detection': {
    label: 'Getting Tired',
    description: 'Behavioral signs of fatigue increasing — slower speech, less expression, more pauses',
    icon: '🔋', color: '#F59E0B', category: 'pattern', priority: 2,
  },
  'stress_recovery': {
    label: 'Good Recovery',
    description: 'Recovered quickly from stress spikes — resilient under pressure',
    icon: '💪', color: '#10B981', category: 'pattern', priority: 2,
  },

  // ── Laughter (FACE-LAUGH-01) ─────────────────────────────────────────────────
  'laughter:genuine_laughter': {
    label: 'Laughing',
    description: 'Genuine laughter — jaw oscillation + genuine eye crinkle (AU6+AU12). Real amusement.',
    icon: '😂', color: '#10B981', category: 'face', priority: 1,
  },
  'laughter:open_mouth_smile': {
    label: 'Big Smile',
    description: 'Wide open smile without the genuine eye crinkle — positive but not full laughter.',
    icon: '😄', color: '#F59E0B', category: 'face', priority: 2,
  },
  'laughter': {
    label: 'Laughing',
    description: 'Laughter or strong smile detected',
    icon: '😂', color: '#10B981', category: 'face', priority: 1,
  },

  // ── Hand gestures (BODY-GESTURE-02) ──────────────────────────────────────────
  'hand_gesture:approval': {
    label: 'Thumbs Up',
    description: 'Explicit approval gesture — agreement or endorsement',
    icon: '👍', color: '#10B981', category: 'body', priority: 1,
  },
  'hand_gesture:disapproval': {
    label: 'Thumbs Down',
    description: 'Explicit disapproval gesture — rejection or disagreement',
    icon: '👎', color: '#EF4444', category: 'body', priority: 1,
  },
  'hand_gesture:emphasis': {
    label: 'Emphasizing',
    description: 'Pointing gesture — adding weight to a point being made',
    icon: '☝️', color: '#F59E0B', category: 'body', priority: 2,
  },
  'hand_gesture:victory': {
    label: 'Victory Sign',
    description: 'V-sign — confidence or celebratory gesture',
    icon: '✌️', color: '#10B981', category: 'body', priority: 2,
  },
  'hand_gesture:tension': {
    label: 'Clenched Fist',
    description: 'Closed fist — suppressed emotion, determination, or controlled tension',
    icon: '✊', color: '#EF4444', category: 'body', priority: 1,
  },
  'hand_gesture': {
    label: 'Hand Gesture',
    description: 'Symbolic hand gesture detected',
    icon: '🖐️', color: '#F59E0B', category: 'body', priority: 2,
  },

  // ── Body cluster signals (evaluation, hidden-disagreement, frustration) ────────
  'evaluation_cluster': {
    label: 'Evaluating',
    description: 'Contemplation cluster: chin touch + head tilt + gaze shift — actively weighing a decision (Navarro 2008)',
    icon: '🤔', color: '#F59E0B', category: 'body', priority: 1,
  },
  'hidden_disagreement': {
    label: 'Suppressed Disagreement',
    description: 'Lip pursing + withdrawal cues — internal rejection not being voiced. Pease 2004: pursed lips is the anchor signal.',
    icon: '🙊', color: '#EF4444', category: 'body', priority: 1,
  },
  'frustration_cluster': {
    label: 'Frustration',
    description: 'Multiple frustration cues co-occurring: forehead touch + shoulder tension + restrictive posture',
    icon: '😤', color: '#EF4444', category: 'body', priority: 1,
  },

  // ── Gesture animation (BODY-GEST-01) ─────────────────────────────────────────
  'gesture_animation:very_animated_gestures': {
    label: 'Very Animated',
    description: 'High-velocity hand movements — enthusiastic, emphatic, high-energy communication style',
    icon: '🙌', color: '#10B981', category: 'body', priority: 2,
  },
  'gesture_animation:animated_gestures': {
    label: 'Gesturing',
    description: 'Elevated hand/arm movement — engaged and expressive communication',
    icon: '✋', color: '#10B981', category: 'body', priority: 2,
  },
  'gesture_animation': {
    label: 'Gesturing',
    description: 'Hand and arm movement level relative to baseline',
    icon: '✋', color: '#6B7280', category: 'body', priority: 2,
  },

  // ── Body mirroring (BODY-MIRROR-01) ──────────────────────────────────────────
  'body_mirroring:synchronized_lean': {
    label: 'Mirroring',
    description: 'Both participants leaning in the same direction simultaneously — strong rapport and alignment',
    icon: '🪞', color: '#10B981', category: 'interaction', priority: 1,
  },
  'body_mirroring': {
    label: 'Mirroring',
    description: 'Synchronized body movement between speakers — indicator of rapport',
    icon: '🪞', color: '#10B981', category: 'interaction', priority: 2,
  },

  // ── Gaze synchrony (GAZE-SYNC-01) ────────────────────────────────────────────
  'gaze_synchrony:synchronized_gaze_break': {
    label: 'Mutual Look-Away',
    description: 'Both participants looked away from the screen at the same time — shared distraction or discomfort',
    icon: '👀', color: '#F59E0B', category: 'gaze', priority: 2,
  },
  'gaze_synchrony': {
    label: 'Gaze Sync',
    description: 'Synchronized gaze patterns between speakers',
    icon: '👀', color: '#6B7280', category: 'gaze', priority: 3,
  },

  // ── Arm posture (BODY-ARM-01) ─────────────────────────────────────────────────
  'arm_posture:expansive': {
    label: 'Power Posture',
    description: 'Elbows wider than shoulders — expansive, dominant posture signals confidence (Carney 2010)',
    icon: '💪', color: '#10B981', category: 'body', priority: 2,
  },
  'arm_posture:contracted': {
    label: 'Closed Posture',
    description: 'Elbows pulled in close — contracted arm position, defensive or self-protective',
    icon: '🛡️', color: '#F59E0B', category: 'body', priority: 2,
  },
  'arm_posture': {
    label: 'Arm Position',
    description: 'Arm expansion relative to baseline — open vs closed posture',
    icon: '↔️', color: '#6B7280', category: 'body', priority: 2,
  },

  // ── Voice-face alignment (FUSION-15) ─────────────────────────────────────────
  'voice_face_alignment:congruent': {
    label: 'Authentic',
    description: 'Voice and face are saying the same thing — genuine, unmanaged expression',
    icon: '✅', color: '#10B981', category: 'pattern', priority: 2,
  },
  'voice_face_alignment:voice_positive_face_negative': {
    label: 'Forced Positivity',
    description: 'Voice sounds upbeat but face shows negative emotion — putting on a front',
    icon: '🎭', color: '#EF4444', category: 'pattern', priority: 1,
  },
  'voice_face_alignment:voice_negative_face_positive': {
    label: 'Polite Masking',
    description: 'Voice sounds tense or negative but face is smiling — suppressing a reaction',
    icon: '😬', color: '#F59E0B', category: 'pattern', priority: 1,
  },
  'voice_face_alignment:voice_stressed_face_calm': {
    label: 'Hidden Stress',
    description: 'Voice shows stress markers but face is composed — controlling visible reaction',
    icon: '🫣', color: '#F59E0B', category: 'pattern', priority: 1,
  },
  'voice_face_alignment:voice_calm_face_stressed': {
    label: 'Face Leaking',
    description: 'Voice sounds calm but face shows tension — stress leaking through facial expression',
    icon: '😰', color: '#EF4444', category: 'pattern', priority: 1,
  },
  'voice_face_alignment:energy_mismatch': {
    label: 'Energy Mismatch',
    description: 'Voice energy level does not match facial engagement — one channel is suppressed',
    icon: '⚡', color: '#F59E0B', category: 'pattern', priority: 2,
  },
  'voice_face_alignment': {
    label: 'Voice-Face Sync',
    description: 'Whether voice tone and facial expression are in sync',
    icon: '🎭', color: '#6B7280', category: 'pattern', priority: 2,
  },

  // ── Top-level voice signal types (no value_text subtype) ─────────────────────
  'energy_level': {
    label: 'Energy Level',
    description: 'Vocal energy relative to baseline — elevated indicates arousal, depressed indicates fatigue',
    icon: '⚡', color: '#F59E0B', category: 'voice', priority: 2,
  },
  'volume_shift': {
    label: 'Volume Shift',
    description: 'Speaking louder or quieter than baseline — dominance, emphasis, or withdrawal',
    icon: '🔊', color: '#8B5CF6', category: 'voice', priority: 2,
  },
  'pause_classification': {
    label: 'Pause Detected',
    description: 'Extended hesitation or thinking pause — cognitive load or uncertainty',
    icon: '⏸', color: '#6B7280', category: 'voice', priority: 2,
  },
  'strategic_pause': {
    label: 'Strategic Pause',
    description: 'Intentional emphasis pause before key content — deliberate communication',
    icon: '⏯', color: '#3B82F6', category: 'voice', priority: 1,
  },
  'talk_time_ratio': {
    label: 'Talk Time',
    description: 'Speaker dominance — proportion of total conversation time',
    icon: '⏱', color: '#10B981', category: 'voice', priority: 2,
  },
  'speech_rate_anomaly': {
    label: 'Speech Pace',
    description: 'Speaking significantly faster or slower than baseline — anxiety, rushing, or deliberation',
    icon: '🏃', color: '#F97316', category: 'voice', priority: 2,
  },
  'filler_detection': {
    label: 'Filler Words',
    description: 'Um, uh, like, you know — verbal fillers indicating uncertainty or cognitive load',
    icon: '💬', color: '#EF4444', category: 'voice', priority: 2,
  },
  'sentiment_score': {
    label: 'Sentiment',
    description: 'Positive or negative language sentiment from transcript analysis',
    icon: '💭', color: '#8B5CF6', category: 'voice', priority: 2,
  },
  'tone_classification': {
    label: 'Tone',
    description: 'Vocal tone — warm, cold, aggressive, excited, nervous, or confident',
    icon: '🎭', color: '#6366F1', category: 'voice', priority: 1,
  },

  // ── Interrogation: per-window behavioral signals ──────────────────────────────
  // Category matches the modality (face/body/gaze). Purple (#8B5CF6) distinguishes
  // interrogation-specific signals from standard behavioral signals.
  'blink_suppression_spike': {
    label: 'Blink Pattern',
    description: 'Blink suppression followed by rapid blinking — cognitive load response during questioning. Common in both truthful and deceptive subjects under pressure.',
    icon: '👁️', color: '#8B5CF6', category: 'face', priority: 1,
  },
  'motor_inhibition': {
    label: 'Movement Control',
    description: 'Unusually still body despite elevated stress — may indicate conscious movement suppression, freeze response, or simply focused attention.',
    icon: '🧊', color: '#8B5CF6', category: 'body', priority: 1,
  },
  'freezing_response': {
    label: 'Freeze Response',
    description: 'Sudden body stillness after accusatory statement — threat-detection freeze. Equally common in truthful and deceptive subjects under accusation.',
    icon: '❄️', color: '#8B5CF6', category: 'body', priority: 1,
  },
  'self_adaptor_increase': {
    label: 'Increasing Self-Touch',
    description: 'Self-touch rate increased across the session (Li et al. 2024, DePaulo 2003 d=0.10). Equally present in innocent suspects under prolonged pressure.',
    icon: '✋', color: '#8B5CF6', category: 'body', priority: 2,
  },
  'erratic_gaze_pattern': {
    label: 'Erratic Gaze',
    description: 'Elevated gaze randomness during questioning — may indicate cognitive load, visual search, avoidance, or simply looking around the room.',
    icon: '👀', color: '#8B5CF6', category: 'gaze', priority: 2,
  },
  // ── Interrogation: per-segment language signals ───────────────────────────────
  // Category 'pattern' so they appear in the sidebar and timeline.
  'detail_reduction': {
    label: 'Low Detail',
    description: 'Narrative lacks sensory details compared to earlier accounts. Strongest verbal deception cue (DePaulo 2003, d≈0.25-0.35). Also occurs in genuine memory gaps or fatigue.',
    icon: '📝', color: '#A78BFA', category: 'pattern', priority: 1,
  },
  'narrative_consistency_drift': {
    label: 'Story Drift',
    description: 'Same event described differently at different times (Granhag & Strömwall 1999). Also occurs from memory degradation or different questioning frames. No quantified effect size.',
    icon: '🔀', color: '#A78BFA', category: 'pattern', priority: 2,
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
  'evidence_response_processing_delay': {
    label: 'Delayed Response',
    description: 'Extended pause (>2s) after evidence presented — processing time. Equally present in innocent suspects confronted with unexpected information.',
    icon: '⏸️', color: '#EF4444', category: 'pattern', priority: 1,
  },
  'statement_contamination': {
    label: 'Information Adopted',
    description: 'Suspect using case-specific terms first introduced by interrogator — strongest false confession risk indicator (Garrett 2011: present in 97.5% of proven false confessions).',
    icon: '⚠️', color: '#EF4444', category: 'pattern', priority: 1,
  },

  // ── Interrogation: per-phase compound signals ─────────────────────────────────
  'capitulation_cascade': {
    label: 'Capitulation Pattern',
    description: 'Stress peak → freeze → weakening denial sequence — breakdown pattern. May indicate genuine breakdown, exhaustion, or pressure-induced compliance.',
    icon: '📉', color: '#EF4444', category: 'pattern', priority: 1,
  },
  'resistance_hardening': {
    label: 'Resistance Pattern',
    description: 'Increasing defensive behaviors over time — adopted resistance strategy. May indicate deception or genuine innocent resistance to false accusations.',
    icon: '📈', color: '#3B82F6', category: 'pattern', priority: 1,
  },

  // ── Interrogation: session-level signals ──────────────────────────────────────
  // Priority 3 = hidden from sidebar; consumed by InterrogationSummaryPanel only.
  'denial_weakening': {
    label: 'Denial Weakening',
    description: 'Denial strength decreased across session — from strong categorical denials to weak or acquiescent language.',
    icon: '📊', color: '#F59E0B', category: 'pattern', priority: 3,
  },
  'false_confession_risk': {
    label: 'False Confession Risk',
    description: 'Multi-factor risk assessment — evaluates duration, contamination, capitulation, denial evolution, processing delays, and technique coerciveness.',
    icon: '⚖️', color: '#EF4444', category: 'pattern', priority: 3,
  },
  'interrogator_technique': {
    label: 'Interrogation Technique',
    description: 'Classification of interrogator approach — PEACE (information-gathering), Reid (accusation-based), or Coercive (threats/promises).',
    icon: '🎭', color: '#6B7280', category: 'pattern', priority: 3,
  },
};
