"""
NEXUS Voice Agent (Agent 1)
FastAPI service for acoustic & prosodic analysis.

Implements 5 core rules from the Rule Engine:
  - VOICE-CAL-01: Per-speaker baseline calibration
  - VOICE-STRESS-01: Composite vocal stress score
  - VOICE-FILLER-01: Filler word detection & classification
  - VOICE-PITCH-01: Pitch elevation flag
  - VOICE-RATE-01: Speech rate anomaly detection
  - VOICE-TONE-03/04: Nervous + Confident tone classification

Endpoints:
  POST /analyse          → Process an audio file end-to-end
  POST /analyse/stream   → Process audio chunk (for future real-time)
  GET  /health           → Health check
"""
import os
import sys
import uuid
import json
import time
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

# isort: split
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# isort: split
from shared.models.requests import VoiceAnalysisRequest as AnalysisRequest, VoiceAnalysisResponse as AnalysisResponse

try:
    from shared.utils.audio_loader import load_audio as _load_audio
except ImportError:
    import librosa as _librosa
    def _load_audio(path, sr=16000):
        return _librosa.load(path, sr=sr, mono=True)

# Import from same directory (works in Docker /app context)
try:
    from feature_extractor import VoiceFeatureExtractor
    from calibration import CalibrationModule
    from rules import VoiceRuleEngine
    from transcriber import Transcriber
except ImportError:
    # Fallback for running from project root
    from services.voiceAgent.feature_extractor import VoiceFeatureExtractor
    from services.voiceAgent.calibration import CalibrationModule
    from services.voiceAgent.rules import VoiceRuleEngine
    from services.voiceAgent.transcriber import Transcriber

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nexus.voice")

app = FastAPI(
    title="NEXUS Voice Agent",
    description="Agent 1: Acoustic & prosodic analysis",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Globals (initialised on startup) ──
feature_extractor: Optional[VoiceFeatureExtractor] = None
transcriber: Optional[Transcriber] = None
rule_engine: Optional[VoiceRuleEngine] = None


@app.on_event("startup")
async def startup():
    global feature_extractor, transcriber, rule_engine
    logger.info("Starting NEXUS Voice Agent...")
    
    feature_extractor = VoiceFeatureExtractor()
    transcriber = Transcriber()
    rule_engine = VoiceRuleEngine()
    
    logger.info("Voice Agent ready.")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "agent": "voice",
        "version": "0.2.0",
        "transcriber_backend": transcriber.backend if transcriber else "not_loaded",
        "models_loaded": {
            "feature_extractor": feature_extractor is not None,
            "transcriber": transcriber is not None,
            "rule_engine": rule_engine is not None,
        }
    }



@app.post("/transcribe")
async def transcribe_only(request: AnalysisRequest):
    """
    Fast path: return transcript + diarisation without features/rules.
    Used by API Gateway to start Language Agent early while Voice features
    continue processing in parallel (Optimization 1).
    """
    file_path = Path(request.file_path)
    if not file_path.exists():
        raise HTTPException(404, f"Audio file not found: {file_path}")

    session_id = request.session_id or str(uuid.uuid4())
    start_time = time.time()

    logger.info(f"[{session_id}] Transcribe-only: {file_path.name}")

    transcript = transcriber.transcribe(str(file_path), num_speakers=request.num_speakers)

    elapsed = time.time() - start_time
    speakers = list(set(seg["speaker"] for seg in transcript["segments"]))
    logger.info(
        f"[{session_id}] Transcribe-only complete: "
        f"{transcript['duration_seconds']:.1f}s, {len(speakers)} speakers, "
        f"{len(transcript['segments'])} segments in {elapsed:.1f}s"
    )

    return {
        "session_id": session_id,
        "duration_seconds": transcript["duration_seconds"],
        "segments": transcript["segments"],
        "speakers": speakers,
    }


@app.post("/analyse", response_model=AnalysisResponse)
async def analyse_audio(request: AnalysisRequest):
    """
    Process a complete audio file through the Voice Agent pipeline.
    
    Pipeline:
    1. Transcribe audio (Whisper) + speaker diarisation
    2. Extract acoustic features per speaker per window
    3. Build per-speaker baselines (calibration)
    4. Run 5 core rules against features + baselines
    5. Return all signals
    """
    file_path = Path(request.file_path)
    if not file_path.exists():
        raise HTTPException(404, f"Audio file not found: {file_path}")
    
    session_id = request.session_id or str(uuid.uuid4())
    start_time = time.time()
    
    logger.info(f"[{session_id}] Analysing: {file_path.name}")

    # ── Load audio once (reused by diarisation + feature extraction) ──
    try:
        from shared.utils.audio_loader import load_audio
        audio_data = load_audio(str(file_path), sr=16000)
    except ImportError:
        import librosa as _lr
        audio_data = _lr.load(str(file_path), sr=16000, mono=True)

    # ── Step 1: Transcribe + diarise ──
    meeting_type = request.meeting_type or "sales_call"
    logger.info(f"[{session_id}] Step 1: Transcribing (num_speakers={request.num_speakers}, meeting_type={meeting_type})...")
    transcript = transcriber.transcribe(str(file_path), num_speakers=request.num_speakers, audio_data=audio_data, meeting_type=meeting_type)
    
    duration_sec = transcript["duration_seconds"]
    speakers = list(set(seg["speaker"] for seg in transcript["segments"]))
    logger.info(f"[{session_id}] Transcribed: {duration_sec:.1f}s, {len(speakers)} speakers, {len(transcript['segments'])} segments")
    
    # ── Step 2: Extract acoustic features ──
    logger.info(f"[{session_id}] Step 2: Extracting features...")
    try:
        features_by_speaker = feature_extractor.extract_all(
            str(file_path),
            transcript["segments"],
            audio_data=audio_data,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Audio file could not be decoded — it may be empty or corrupted. ({e})",
        )

    # ── Step 1: Transcribe + diarise (Whisper uses file path; diarisation reuses loaded audio) ──
    logger.info(f"[{session_id}] Step 1: Transcribing (num_speakers={request.num_speakers})...")
    transcript = transcriber.transcribe(
        str(file_path), num_speakers=request.num_speakers, preloaded_audio=(y, sr)
    )

    duration_sec = transcript["duration_seconds"]
    speakers = list(set(seg["speaker"] for seg in transcript["segments"]))
    logger.info(f"[{session_id}] Transcribed: {duration_sec:.1f}s, {len(speakers)} speakers, {len(transcript['segments'])} segments")

    # ── Step 2: Extract acoustic features (reuses loaded audio, parallel per speaker) ──
    logger.info(f"[{session_id}] Step 2: Extracting features...")
    try:
        features_by_speaker = feature_extractor.extract_all_from_array(
            y, sr, transcript["segments"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Feature extraction failed. ({e})",
        )
    
    # ── Step 3: Build baselines ──
    # Compute true speaking time per speaker from transcript (not feature windows)
    # so calibration confidence isn't penalised by skipped short windows.
    transcript_speech = {}
    for seg in transcript["segments"]:
        spk = seg["speaker"]
        transcript_speech[spk] = transcript_speech.get(spk, 0.0) + (seg["end_ms"] - seg["start_ms"]) / 1000.0

    logger.info(f"[{session_id}] Step 3: Calibrating baselines...")
    calibration = CalibrationModule()
    baselines = {}
    for speaker_id, features_list in features_by_speaker.items():
        baseline = calibration.build_baseline(
            speaker_id, session_id, features_list,
            transcript_speech_sec=transcript_speech.get(speaker_id, 0.0),
        )
        baselines[speaker_id] = baseline
        logger.info(
            f"[{session_id}] Baseline for {speaker_id}: "
            f"F0={baseline.f0_mean:.1f}Hz, rate={baseline.speech_rate_wpm:.0f}wpm, "
            f"confidence={baseline.calibration_confidence:.2f}"
        )
    
    # ── Step 4: Run rules ──
    logger.info(f"[{session_id}] Step 4: Running rule engine...")
    all_signals = []
    
    for speaker_id, features_list in features_by_speaker.items():
        baseline = baselines.get(speaker_id)
        if not baseline:
            continue
        
        for features in features_list:
            signals = rule_engine.evaluate(
                features=features,
                baseline=baseline,
                speaker_id=speaker_id,
                transcript_segments=[
                    s for s in transcript["segments"]
                    if s["speaker"] == speaker_id
                    and s["end_ms"] > features["window_start_ms"]
                    and s["start_ms"] < features["window_end_ms"]
                ]
            )
            all_signals.extend(signals)
    
    # ── Step 5: Build summary ──
    elapsed = time.time() - start_time
    logger.info(f"[{session_id}] Complete: {len(all_signals)} signals in {elapsed:.1f}s")
    
    summary = _build_summary(all_signals, baselines, transcript)
    
    speaker_data = [
        {
            "speaker_id": sid,
            "baseline": baselines[sid].to_dict() if sid in baselines else None,
            "signal_count": len([s for s in all_signals if s.get("speaker_id") == sid])
        }
        for sid in speakers
    ]
    
    return AnalysisResponse(
        session_id=session_id,
        duration_seconds=duration_sec,
        speakers=speaker_data,
        signals=[s if isinstance(s, dict) else s.to_dict() for s in all_signals],
        summary=summary,
        transcript_segments=transcript["segments"],
    )


@app.post("/analyse/upload")
async def analyse_upload(file: UploadFile = File(...)):
    """Upload an audio file and analyse it."""
    # Save to temp location
    upload_dir = Path("/app/data/recordings")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = upload_dir / f"{uuid.uuid4()}_{file.filename}"
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    return await analyse_audio(AnalysisRequest(file_path=str(file_path)))


def _build_summary(signals: list, baselines: dict, transcript: dict) -> dict:
    """Build a human-readable summary from all signals."""
    summary = {
        "total_signals": len(signals),
        "per_speaker": {},
        "key_moments": [],
        "stress_peaks": [],
        "filler_stats": {},
    }
    
    for speaker_id, baseline in baselines.items():
        speaker_signals = [s for s in signals if s.get("speaker_id") == speaker_id]
        
        stress_signals = [
            s for s in speaker_signals 
            if s.get("signal_type") == "vocal_stress_score"
        ]
        stress_values = [s["value"] for s in stress_signals if s.get("value") is not None]
        
        filler_signals = [
            s for s in speaker_signals 
            if s.get("signal_type") == "filler_detection"
        ]
        
        pitch_flags = [
            s for s in speaker_signals 
            if s.get("signal_type") == "pitch_elevation_flag"
        ]
        
        tone_signals = [
            s for s in speaker_signals 
            if s.get("signal_type") == "tone_classification"
        ]
        
        summary["per_speaker"][speaker_id] = {
            "baseline_f0_hz": round(baseline.f0_mean, 1),
            "baseline_rate_wpm": round(baseline.speech_rate_wpm, 0),
            "calibration_confidence": round(baseline.calibration_confidence, 2),
            "avg_stress": round(sum(stress_values) / len(stress_values), 3) if stress_values else 0,
            "max_stress": round(max(stress_values), 3) if stress_values else 0,
            "total_fillers": len(filler_signals),
            "pitch_elevation_events": len(pitch_flags),
            "tone_distribution": _count_tones(tone_signals),
        }
        
        # Find stress peaks (moments above 0.5)
        for s in stress_signals:
            if s.get("value", 0) > 0.5:
                summary["stress_peaks"].append({
                    "speaker": speaker_id,
                    "time_ms": s.get("window_start_ms"),
                    "stress_score": round(s["value"], 3),
                    "confidence": round(s.get("confidence", 0), 3),
                })
    
    # Sort stress peaks by score descending
    summary["stress_peaks"].sort(key=lambda x: x["stress_score"], reverse=True)
    summary["stress_peaks"] = summary["stress_peaks"][:10]  # Top 10
    
    return summary


def _count_tones(tone_signals: list) -> dict:
    counts = {}
    for s in tone_signals:
        tone = s.get("value_text", "unknown")
        counts[tone] = counts.get(tone, 0) + 1
    return counts
