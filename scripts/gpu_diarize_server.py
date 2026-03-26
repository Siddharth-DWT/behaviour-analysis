"""
GPU Diarization Server — Deploy on RTX 5090 alongside Whisper API.

Runs pyannote/speaker-diarization-3.1 on GPU for fast speaker diarization.
29-minute audio: ~45-90 seconds on GPU vs 15-20 minutes on CPU.

Usage:
    pip install pyannote.audio fastapi uvicorn python-multipart torch
    HF_TOKEN=hf_xxx python gpu_diarize_server.py

    # Or with uvicorn directly:
    HF_TOKEN=hf_xxx uvicorn gpu_diarize_server:app --host 0.0.0.0 --port 8010

Endpoints:
    POST /diarize     — Diarize an audio file, returns speaker timeline
    GET  /health      — Health check with GPU info
"""
import os
import time
import logging
import tempfile
from pathlib import Path

import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gpu-diarize")

app = FastAPI(title="GPU Diarization Server", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global pipeline (loaded once on first request) ──
_pipeline = None
HF_TOKEN = os.getenv("HF_TOKEN", "")


def get_pipeline():
    """Lazy-load pyannote pipeline on GPU."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN not set. Required for pyannote model access.")

    from pyannote.audio import Pipeline
    from huggingface_hub import login

    login(token=HF_TOKEN, add_to_git_credential=False)

    logger.info("Loading pyannote speaker-diarization-3.1 ...")
    _pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN,
    )

    if torch.cuda.is_available():
        _pipeline.to(torch.device("cuda"))
        logger.info(f"Pyannote loaded on GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("No GPU available — pyannote will run on CPU (slow)")

    return _pipeline


@app.get("/health")
async def health():
    gpu_available = torch.cuda.is_available()
    return {
        "status": "healthy",
        "service": "GPU Diarization Server",
        "gpu_available": gpu_available,
        "gpu_name": torch.cuda.get_device_name(0) if gpu_available else None,
        "pipeline_loaded": _pipeline is not None,
    }


@app.post("/diarize")
async def diarize(
    file: UploadFile = File(...),
    min_speakers: int = Form(default=2),
    max_speakers: int = Form(default=8),
    num_speakers: int = Form(default=0),  # 0 = auto-detect
):
    """
    Diarize an audio file and return speaker timeline.

    Args:
        file: Audio file (WAV, MP3, M4A, MP4, etc.)
        min_speakers: Minimum expected speakers (default 2)
        max_speakers: Maximum expected speakers (default 8)
        num_speakers: Exact speaker count (0 = auto-detect using min/max range)

    Returns:
        {
            "speakers": ["SPEAKER_00", "SPEAKER_01", ...],
            "timeline": [
                {"speaker": "SPEAKER_00", "start": 0.5, "end": 3.2},
                {"speaker": "SPEAKER_01", "start": 3.5, "end": 7.8},
                ...
            ],
            "num_speakers": 3,
            "duration": 1750.3,
            "processing_time": 42.5
        }
    """
    pipeline = get_pipeline()

    # Save uploaded file to temp
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        start_time = time.time()

        # Build diarization kwargs
        diarize_kwargs = {}
        if num_speakers > 0:
            diarize_kwargs["num_speakers"] = num_speakers
        else:
            if min_speakers > 0:
                diarize_kwargs["min_speakers"] = min_speakers
            if max_speakers > 0:
                diarize_kwargs["max_speakers"] = max_speakers

        logger.info(
            f"Diarizing {file.filename} ({len(content) / 1e6:.1f} MB) "
            f"with params: {diarize_kwargs}"
        )

        # Run pyannote diarization on GPU
        diarization = pipeline(tmp_path, **diarize_kwargs)

        # Extract timeline
        timeline = []
        speakers_seen = set()
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            timeline.append({
                "speaker": speaker,
                "start": round(turn.start, 3),
                "end": round(turn.end, 3),
            })
            speakers_seen.add(speaker)

        processing_time = time.time() - start_time

        # Get audio duration from the last turn end
        duration = max((t["end"] for t in timeline), default=0)

        logger.info(
            f"Diarization complete: {len(speakers_seen)} speakers, "
            f"{len(timeline)} turns, {processing_time:.1f}s"
        )

        return {
            "speakers": sorted(speakers_seen),
            "timeline": timeline,
            "num_speakers": len(speakers_seen),
            "duration": duration,
            "processing_time": round(processing_time, 2),
        }

    except Exception as e:
        logger.error(f"Diarization failed: {e}")
        raise HTTPException(500, f"Diarization failed: {str(e)}")

    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("DIARIZE_PORT", "8010"))
    uvicorn.run(app, host="0.0.0.0", port=port)
