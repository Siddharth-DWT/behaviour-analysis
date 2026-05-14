#!/usr/bin/env python3
"""
NEXUS End-to-End Pipeline Test
Uploads an audio/video file (or generates synthetic audio) to the monolith
backend at BACKEND_URL, then polls until the analysis completes and prints results.

Usage:
  python3 scripts/test_pipeline.py                                    # Synthetic sales call
  python3 scripts/test_pipeline.py --audio-file recording.mp3         # Any audio file
  python3 scripts/test_pipeline.py --audio-file lecture.mp4           # Video file
  python3 scripts/test_pipeline.py --url "https://youtube.com/watch?v=XXX"  # YouTube URL
  python3 scripts/test_pipeline.py --audio-file podcast.mp3 --type podcast --speakers 3
  python3 scripts/test_pipeline.py --skip-audio --type debate          # Reuse + override type
  python3 scripts/test_pipeline.py --no-intent                         # Skip LLM intent classification
  python3 scripts/test_pipeline.py --use-external-tts                  # GPU Coqui TTS for generation
  python3 scripts/test_pipeline.py --html                               # Generate HTML report
  python3 scripts/test_pipeline.py --audio-file call.wav --speakers 2 --html  # Full run + HTML
"""
import os
import sys
import json
import time
import argparse
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime

import httpx

# ── Config ──
BACKEND_URL  = os.getenv("BACKEND_URL", "http://localhost:8000")
NEXUS_TOKEN  = os.getenv("NEXUS_TOKEN", "")          # JWT from env (skip login)
NEXUS_EMAIL  = os.getenv("NEXUS_EMAIL", "admin@nexus.local")
NEXUS_PASS   = os.getenv("NEXUS_PASSWORD", "admin123")
POLL_INTERVAL = 10   # seconds between status polls
POLL_TIMEOUT  = 3600  # 1 hour max wait

# External APIs (GPU-accelerated, optional)
EXTERNAL_TTS_URL = os.getenv("EXTERNAL_TTS_URL", "http://110.227.200.12:8009")
EXTERNAL_API_KEY = os.getenv("EXTERNAL_API_KEY", "your-secret-api-key-is-this")

PROJECT_ROOT = Path(__file__).parent.parent
TEST_AUDIO_PATH = PROJECT_ROOT / "data" / "recordings" / "test_sales_call.wav"

# ── Colours for terminal output ──
C_RESET  = "\033[0m"
C_BOLD   = "\033[1m"
C_DIM    = "\033[2m"
C_RED    = "\033[91m"
C_GREEN  = "\033[92m"
C_YELLOW = "\033[93m"
C_BLUE   = "\033[94m"
C_PURPLE = "\033[95m"
C_CYAN   = "\033[96m"
C_ORANGE = "\033[38;5;208m"


# ═══════════════════════════════════════════════════════════════════
# STEP 0: Generate Synthetic Audio
# ═══════════════════════════════════════════════════════════════════

# Two-speaker fake sales conversation. Speaker A = seller, Speaker B = buyer.
# Includes hesitation words (um, well, uh) for filler detection,
# buying signals, objections, and varied emotional content.
CONVERSATION = [
    # (speaker_voice, text)
    ("Daniel", "Hello, thanks for taking the time to meet with me today. I'm excited to show you what our platform can do for your team."),
    ("Fred", "Sure, um, we've been looking at a few options. Can you walk me through the main features?"),
    ("Daniel", "Absolutely. Our AI analytics platform processes your customer data in real time. We've seen clients achieve a forty percent improvement in conversion rates within the first quarter."),
    ("Fred", "Well, that sounds impressive, but um, how does that compare to what we're already doing with our current tools?"),
    ("Daniel", "Great question. Unlike legacy tools, we use machine learning to identify patterns that traditional analytics simply miss. Let me share some specific examples."),
    ("Fred", "I'd like to see the pricing structure. Uh, what does the enterprise plan look like?"),
    ("Daniel", "The enterprise plan starts at two thousand per month. But here's the thing, the ROI typically pays for itself within sixty days. We guarantee measurable results."),
    ("Fred", "Hmm, well, that's a significant investment. We'd need to, um, discuss this internally. I'm not sure the budget allows for that kind of commitment right now."),
    ("Daniel", "I completely understand. Many of our best clients had the same concern initially. What if we started with a thirty day pilot at no cost? That way your team can see the results firsthand before making any commitment."),
    ("Fred", "A free trial could work. Um, what kind of support do you provide during the pilot period?"),
    ("Daniel", "Full dedicated support. You'll have a customer success manager assigned to your account, plus twenty four seven technical support. We want to make absolutely sure you succeed."),
    ("Fred", "That's, well, that's actually quite good. Um, how quickly can we get started? Our team has a quarterly review coming up and it would be great to have some initial data by then."),
    ("Daniel", "We can have you onboarded within forty eight hours. If you sign up this week, we can have preliminary results ready well before your quarterly review."),
    ("Fred", "Okay, I think, um, I think that could work. Let me talk to my director and, uh, get back to you by end of week. I'm cautiously optimistic about this."),
]


def generate_test_audio_external_tts(output_path: Path) -> Path:
    """
    Generate synthetic 2-speaker sales conversation using the external
    GPU-accelerated Coqui TTS API with voice cloning for distinct speakers.
    """
    print(f"\n{C_BOLD}{'=' * 60}{C_RESET}")
    print(f"{C_BOLD}  STEP 0: Generating Audio via External TTS (Coqui XTTS){C_RESET}")
    print(f"{C_BOLD}{'=' * 60}{C_RESET}")
    print(f"  {C_DIM}TTS API: {EXTERNAL_TTS_URL}{C_RESET}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    project_root = str(Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from shared.utils.external_apis import TTSClient

    tts = TTSClient(base_url=EXTERNAL_TTS_URL, api_key=EXTERNAL_API_KEY)

    if not tts.is_healthy():
        print(f"  {C_RED}External TTS API not healthy. Falling back to macOS say.{C_RESET}")
        return generate_test_audio(output_path)

    temp_dir = tempfile.mkdtemp(prefix="nexus_tts_")
    segment_files = []
    speaker_ref_dir = Path(temp_dir) / "speaker_refs"
    speaker_ref_dir.mkdir(exist_ok=True)

    print(f"\n  {C_CYAN}Generating speaker reference voices...{C_RESET}")

    speaker_a_ref = speaker_ref_dir / "speaker_a_ref.wav"
    try:
        ref_audio = tts.synthesize(
            "Hello, I'm excited to present our solution to you today. "
            "Our platform delivers exceptional results for enterprise clients.",
            language="en",
        )
        with open(speaker_a_ref, "wb") as f:
            f.write(ref_audio)
        print(f"  {C_GREEN}Speaker A (seller) reference: {len(ref_audio)} bytes{C_RESET}")
        use_clone_a = True
    except Exception as e:
        print(f"  {C_YELLOW}Speaker A reference failed: {e}. Using default voice.{C_RESET}")
        use_clone_a = False

    speaker_b_ref = speaker_ref_dir / "speaker_b_ref.wav"
    try:
        ref_audio_b = tts.synthesize(
            "Well, um, I have some concerns about this. "
            "Let me think about it and discuss with my team. "
            "We need to be careful about budget commitments.",
            language="en",
        )
        with open(speaker_b_ref, "wb") as f:
            f.write(ref_audio_b)
        print(f"  {C_GREEN}Speaker B (buyer) reference: {len(ref_audio_b)} bytes{C_RESET}")
        use_clone_b = True
    except Exception as e:
        print(f"  {C_YELLOW}Speaker B reference failed: {e}. Using default voice.{C_RESET}")
        use_clone_b = False

    print(f"\n  {C_CYAN}Generating conversation turns...{C_RESET}")

    for i, (voice, text) in enumerate(CONVERSATION):
        seg_path = os.path.join(temp_dir, f"seg_{i:03d}.wav")
        is_seller = (voice == "Daniel")
        speaker_label = "Seller" if is_seller else "Buyer"

        print(f"  {C_DIM}[{i+1}/{len(CONVERSATION)}] {speaker_label}: {text[:55]}...{C_RESET}")

        try:
            if is_seller and use_clone_a:
                audio = tts.synthesize_clone(text, speaker_wav_path=str(speaker_a_ref), language="en")
            elif not is_seller and use_clone_b:
                audio = tts.synthesize_clone(text, speaker_wav_path=str(speaker_b_ref), language="en")
            else:
                audio = tts.synthesize(text, language="en")

            with open(seg_path, "wb") as f:
                f.write(audio)
            segment_files.append(seg_path)

        except Exception as e:
            print(f"  {C_RED}TTS failed for segment {i}: {e}{C_RESET}")
            print(f"  {C_YELLOW}Falling back to macOS say for this segment.{C_RESET}")
            aiff_path = seg_path.replace(".wav", ".aiff")
            mac_voice = voice
            subprocess.run(["say", "-v", mac_voice, "-o", aiff_path, text], capture_output=True)
            subprocess.run(
                ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", aiff_path, seg_path],
                capture_output=True,
            )
            if os.path.exists(seg_path):
                segment_files.append(seg_path)
            try:
                os.unlink(aiff_path)
            except OSError:
                pass

    if not segment_files:
        print(f"  {C_RED}No audio segments generated. Aborting.{C_RESET}")
        sys.exit(1)

    print(f"\n  {C_CYAN}Concatenating {len(segment_files)} segments...{C_RESET}")

    concat_method = None
    for tool in ["ffmpeg", "sox"]:
        if subprocess.run(["which", tool], capture_output=True).returncode == 0:
            concat_method = tool
            break

    if concat_method == "ffmpeg":
        filter_parts = []
        input_args = []
        for j, seg in enumerate(segment_files):
            input_args.extend(["-i", seg])
            filter_parts.append(f"[{j}:a]")
        filter_str = "".join(filter_parts) + f"concat=n={len(segment_files)}:v=0:a=1[out]"
        cmd = (
            ["ffmpeg", "-y"] + input_args
            + ["-filter_complex", filter_str, "-map", "[out]",
               "-ar", "16000", "-ac", "1", str(output_path)]
        )
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  {C_RED}ffmpeg concat failed: {result.stderr[:200]}{C_RESET}")
            sys.exit(1)
    elif concat_method == "sox":
        cmd = ["sox"] + segment_files + ["-r", "16000", "-c", "1", str(output_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  {C_RED}sox concat failed: {result.stderr[:200]}{C_RESET}")
            sys.exit(1)
    else:
        _concat_wav_files(segment_files, str(output_path))

    for seg in segment_files:
        try:
            os.unlink(seg)
        except OSError:
            pass
    for ref in speaker_ref_dir.glob("*.wav"):
        try:
            ref.unlink()
        except OSError:
            pass
    try:
        speaker_ref_dir.rmdir()
        os.rmdir(temp_dir)
    except OSError:
        pass

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n  {C_GREEN}Audio generated via External TTS: {output_path}{C_RESET}")
    print(f"  {C_DIM}Size: {file_size_mb:.1f} MB{C_RESET}")
    print(f"  {C_DIM}Method: Coqui XTTS v2 with voice cloning{C_RESET}")

    return output_path


def generate_test_audio(output_path: Path) -> Path:
    """Generate a synthetic 2-speaker sales conversation using macOS say command."""
    print(f"\n{C_BOLD}{'═' * 60}{C_RESET}")
    print(f"{C_BOLD}  STEP 0: Generating Synthetic Test Audio{C_RESET}")
    print(f"{C_BOLD}{'═' * 60}{C_RESET}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_dir = tempfile.mkdtemp(prefix="nexus_test_")
    segment_files = []

    for i, (voice, text) in enumerate(CONVERSATION):
        seg_path = os.path.join(temp_dir, f"seg_{i:03d}.aiff")
        print(f"  {C_DIM}[{i+1}/{len(CONVERSATION)}] {voice}: {text[:60]}...{C_RESET}")

        cmd = ["say", "-v", voice, "-o", seg_path, text]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  {C_RED}say failed: {result.stderr}{C_RESET}")
            sys.exit(1)

        segment_files.append(seg_path)

    concat_method = None
    for tool in ["sox", "ffmpeg"]:
        if subprocess.run(["which", tool], capture_output=True).returncode == 0:
            concat_method = tool
            break

    if concat_method == "ffmpeg":
        filter_parts = []
        input_args = []
        for j, seg in enumerate(segment_files):
            input_args.extend(["-i", seg])
            filter_parts.append(f"[{j}:a]")
        filter_str = "".join(filter_parts) + f"concat=n={len(segment_files)}:v=0:a=1[out]"
        cmd = (
            ["ffmpeg", "-y"]
            + input_args
            + ["-filter_complex", filter_str, "-map", "[out]",
               "-ar", "16000", "-ac", "1", str(output_path)]
        )
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  {C_RED}ffmpeg concat failed: {result.stderr[:200]}{C_RESET}")
            sys.exit(1)
    elif concat_method == "sox":
        cmd = ["sox"] + segment_files + ["-r", "16000", "-c", "1", str(output_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  {C_RED}sox concat failed: {result.stderr[:200]}{C_RESET}")
            sys.exit(1)
    else:
        print(f"  {C_YELLOW}No sox/ffmpeg found. Using single-file fallback.{C_RESET}")
        wav_files = []
        for seg in segment_files:
            wav_seg = seg.replace(".aiff", ".wav")
            subprocess.run(
                ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", seg, wav_seg],
                capture_output=True,
            )
            wav_files.append(wav_seg)
        _concat_wav_files(wav_files, str(output_path))

    for seg in segment_files:
        try:
            os.unlink(seg)
            wav_ver = seg.replace(".aiff", ".wav")
            if os.path.exists(wav_ver):
                os.unlink(wav_ver)
        except OSError:
            pass
    try:
        os.rmdir(temp_dir)
    except OSError:
        pass

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n  {C_GREEN}Audio generated: {output_path}{C_RESET}")
    print(f"  {C_DIM}Size: {file_size_mb:.1f} MB{C_RESET}")

    return output_path


def _concat_wav_files(wav_files: list[str], output_path: str):
    """Concatenate multiple WAV files (same format) into one."""
    import wave

    if not wav_files:
        return

    with wave.open(wav_files[0], "rb") as wf:
        params = wf.getparams()

    with wave.open(output_path, "wb") as out:
        out.setparams(params)
        for wf_path in wav_files:
            with wave.open(wf_path, "rb") as wf:
                out.writeframes(wf.readframes(wf.getnframes()))


# ═══════════════════════════════════════════════════════════════════
# Authentication
# ═══════════════════════════════════════════════════════════════════

class _AuthClient:
    """Manages JWT token acquisition and renewal against the monolith."""

    def __init__(self, base_url: str):
        self._base = base_url.rstrip("/")
        self._token: str = NEXUS_TOKEN

    def _login(self) -> str:
        print(f"  {C_DIM}Authenticating as {NEXUS_EMAIL}...{C_RESET}")
        try:
            resp = httpx.post(
                f"{self._base}/auth/login",
                json={"email": NEXUS_EMAIL, "password": NEXUS_PASS},
                timeout=15,
            )
        except httpx.ConnectError:
            print(f"  {C_RED}Cannot connect to backend at {self._base}{C_RESET}")
            print(f"  {C_RED}Is it running? Start with: uvicorn backend.main:app --port 8000{C_RESET}")
            sys.exit(1)

        if resp.status_code != 200:
            print(f"  {C_RED}Login failed ({resp.status_code}): {resp.text[:200]}{C_RESET}")
            print(f"  {C_YELLOW}Set NEXUS_EMAIL / NEXUS_PASSWORD env vars or pass --token{C_RESET}")
            sys.exit(1)

        token = resp.json().get("access_token") or resp.json().get("token", "")
        if not token:
            print(f"  {C_RED}Login response missing token: {resp.text[:200]}{C_RESET}")
            sys.exit(1)

        print(f"  {C_GREEN}Authenticated.{C_RESET}")
        return token

    def headers(self) -> dict:
        if not self._token:
            self._token = self._login()
        return {"Authorization": f"Bearer {self._token}"}

    def get(self, path: str, **kwargs) -> httpx.Response:
        return httpx.get(f"{self._base}{path}", headers=self.headers(), **kwargs)

    def post(self, path: str, **kwargs) -> httpx.Response:
        return httpx.post(f"{self._base}{path}", headers=self.headers(), **kwargs)


# ═══════════════════════════════════════════════════════════════════
# STEP 1: Upload & Run Pipeline via Monolith REST API
# ═══════════════════════════════════════════════════════════════════

def run_pipeline_via_api(
    auth: _AuthClient,
    audio_path: Path,
    meeting_type: str = "sales_call",
    num_speakers: int = None,
) -> dict:
    """
    Upload audio_path to POST /sessions, poll until completed,
    then return a result dict mirroring the old multi-agent format:
      {
        "session_id": str,
        "duration_seconds": float,
        "speakers": list,
        "signals": list,
        "transcript_segments": list,
        "summary": dict,
        "report": dict | None,
        "language_signals": list,
        "fusion_signals": list,
        "alerts": list,
        "unified_states": list,
      }
    """
    print(f"\n{C_BOLD}{'═' * 60}{C_RESET}")
    print(f"{C_BOLD}  STEP 1: Uploading to {BACKEND_URL}/sessions{C_RESET}")
    print(f"{C_BOLD}{'═' * 60}{C_RESET}")
    print(f"  {C_DIM}File:         {audio_path.name}{C_RESET}")
    print(f"  {C_DIM}Meeting type: {meeting_type}{C_RESET}")
    if num_speakers:
        print(f"  {C_DIM}Speakers:     {num_speakers}{C_RESET}")

    config_dict: dict = {}
    if num_speakers:
        config_dict["num_speakers"] = num_speakers

    # Upload
    start = time.time()
    with open(audio_path, "rb") as fh:
        resp = auth.post(
            "/sessions",
            files={"file": (audio_path.name, fh, "application/octet-stream")},
            data={
                "title": audio_path.stem,
                "meeting_type": meeting_type,
                "config": json.dumps(config_dict),
            },
            timeout=120,
        )

    if resp.status_code not in (200, 201):
        print(f"  {C_RED}Upload failed ({resp.status_code}): {resp.text[:300]}{C_RESET}")
        sys.exit(1)

    upload_data = resp.json()
    session_id  = upload_data["session_id"]
    print(f"  {C_GREEN}Session created: {session_id}{C_RESET}")

    # Poll until complete
    print(f"\n{C_BOLD}  STEP 2: Waiting for pipeline to complete...{C_RESET}")
    print(f"  {C_DIM}Polling every {POLL_INTERVAL}s (timeout {POLL_TIMEOUT}s){C_RESET}")

    deadline = time.time() + POLL_TIMEOUT
    last_step = ""

    while time.time() < deadline:
        time.sleep(POLL_INTERVAL)

        try:
            progress_resp = auth.get(f"/sessions/{session_id}/progress", timeout=15)
        except httpx.RequestError:
            print(f"  {C_YELLOW}Poll request failed, retrying...{C_RESET}")
            continue

        if progress_resp.status_code != 200:
            # Fallback: use full session endpoint
            session_resp = auth.get(f"/sessions/{session_id}", timeout=15)
            if session_resp.status_code != 200:
                continue
            session_data = session_resp.json().get("session", {})
            status = session_data.get("status", "processing")
        else:
            progress = progress_resp.json()
            status   = progress.get("status", "processing")
            step     = progress.get("current_step", "")
            if step and step != last_step:
                elapsed = time.time() - start
                print(f"  {C_DIM}[{elapsed:5.0f}s] {step}{C_RESET}")
                last_step = step

        if status == "completed":
            break
        if status == "failed":
            print(f"  {C_RED}Pipeline failed for session {session_id}{C_RESET}")
            # Still retrieve what was stored
            break

    elapsed = time.time() - start
    print(f"\n  {C_GREEN}Pipeline finished in {elapsed:.0f}s (status={status}){C_RESET}")

    # Retrieve results
    print(f"\n{C_BOLD}  STEP 3: Retrieving results...{C_RESET}")

    session_detail = auth.get(f"/sessions/{session_id}", timeout=30).json()
    session_obj    = session_detail.get("session", {})

    signals_resp   = auth.get(f"/sessions/{session_id}/signals?limit=50000", timeout=30)
    all_signals    = signals_resp.json().get("signals", []) if signals_resp.status_code == 200 else []

    transcript_resp = auth.get(f"/sessions/{session_id}/transcript", timeout=30)
    transcript_segs = transcript_resp.json().get("segments", []) if transcript_resp.status_code == 200 else []

    report_resp    = auth.get(f"/sessions/{session_id}/report", timeout=30)
    report_data    = report_resp.json().get("report") if report_resp.status_code == 200 else None

    # Partition signals by agent for display purposes
    voice_signals    = [s for s in all_signals if s.get("agent") == "voice"]
    language_signals = [s for s in all_signals if s.get("agent") == "language"]
    fusion_signals   = [s for s in all_signals if s.get("agent") == "fusion"]
    alerts           = session_detail.get("alerts", [])

    # Build per-speaker summaries from stored signals
    speakers_seen = sorted(set(s.get("speaker_id", "unknown") for s in all_signals if s.get("speaker_id")))
    per_speaker_stats = _compute_per_speaker_stats(voice_signals, language_signals)

    print(f"  {C_GREEN}Signals retrieved: {len(all_signals)} total{C_RESET}")
    print(f"    Voice: {len(voice_signals)}, Language: {len(language_signals)}, Fusion: {len(fusion_signals)}")
    print(f"  Speakers: {speakers_seen}")

    # Synthesise a unified_states list from the latest fusion signals per speaker
    unified_states = _build_unified_states(fusion_signals, speakers_seen)

    return {
        "session_id":          session_id,
        "session":             session_obj,
        "duration_seconds":    session_obj.get("duration_seconds", 0) or 0,
        "speakers":            [{"speaker_id": s} for s in speakers_seen],
        "signals":             voice_signals,
        "language_signals":    language_signals,
        "fusion_signals":      fusion_signals,
        "unified_states":      unified_states,
        "alerts":              alerts,
        "transcript_segments": transcript_segs,
        "report":              report_data,
        "summary": {
            "per_speaker": per_speaker_stats,
            "sentiment_distribution": _sentiment_distribution(language_signals),
            "buying_signal_moments":  _buying_moments(language_signals),
            "objection_moments":      _objection_moments(language_signals),
            "stress_peaks":           _stress_peaks(voice_signals),
        },
    }


def _compute_per_speaker_stats(voice_signals: list, language_signals: list) -> dict:
    """Aggregate voice + language signals into per-speaker summary stats."""
    stats: dict[str, dict] = {}

    for s in voice_signals:
        sid   = s.get("speaker_id", "unknown")
        stype = s.get("signal_type", "")
        val   = _to_float(s.get("value"))
        entry = stats.setdefault(sid, {
            "stress_values": [], "filler_count": 0,
            "pitch_elevation_events": 0, "buying_signal_count": 0,
            "objection_count": 0, "sentiment_values": [],
        })
        if stype == "vocal_stress_score":
            entry["stress_values"].append(val)
        elif stype == "filler_detection":
            entry["filler_count"] += 1
        elif stype == "pitch_elevation_flag":
            entry["pitch_elevation_events"] += 1

    for s in language_signals:
        sid   = s.get("speaker_id", "unknown")
        stype = s.get("signal_type", "")
        val   = _to_float(s.get("value"))
        entry = stats.setdefault(sid, {
            "stress_values": [], "filler_count": 0,
            "pitch_elevation_events": 0, "buying_signal_count": 0,
            "objection_count": 0, "sentiment_values": [],
        })
        if stype == "buying_signal":
            entry["buying_signal_count"] += 1
        elif stype == "objection_signal":
            entry["objection_count"] += 1
        elif stype == "sentiment_score":
            entry["sentiment_values"].append(val)

    result = {}
    for sid, entry in stats.items():
        sv = entry["stress_values"]
        sent = entry["sentiment_values"]
        result[sid] = {
            "avg_stress":             round(sum(sv) / len(sv), 3) if sv else 0,
            "max_stress":             round(max(sv), 3) if sv else 0,
            "total_fillers":          entry["filler_count"],
            "pitch_elevation_events": entry["pitch_elevation_events"],
            "buying_signal_count":    entry["buying_signal_count"],
            "objection_count":        entry["objection_count"],
            "avg_sentiment":          round(sum(sent) / len(sent), 3) if sent else 0,
        }
    return result


def _sentiment_distribution(language_signals: list) -> dict:
    dist: dict[str, int] = {"positive": 0, "neutral": 0, "negative": 0}
    for s in language_signals:
        if s.get("signal_type") != "sentiment_score":
            continue
        val = _to_float(s.get("value"))
        if val > 0.1:
            dist["positive"] += 1
        elif val < -0.1:
            dist["negative"] += 1
        else:
            dist["neutral"] += 1
    return dist


def _buying_moments(language_signals: list) -> list:
    moments = []
    for s in language_signals:
        if s.get("signal_type") == "buying_signal":
            moments.append({
                "speaker":    s.get("speaker_id", "?"),
                "time_ms":    s.get("window_start_ms", 0),
                "strength":   _to_float(s.get("value")),
                "categories": s.get("metadata", {}).get("categories", []),
            })
    return sorted(moments, key=lambda x: x["time_ms"])


def _objection_moments(language_signals: list) -> list:
    moments = []
    for s in language_signals:
        if s.get("signal_type") == "objection_signal":
            moments.append({
                "speaker":    s.get("speaker_id", "?"),
                "time_ms":    s.get("window_start_ms", 0),
                "strength":   _to_float(s.get("value")),
                "categories": s.get("metadata", {}).get("categories", []),
            })
    return sorted(moments, key=lambda x: x["time_ms"])


def _stress_peaks(voice_signals: list) -> list:
    peaks = [
        {
            "speaker":      s.get("speaker_id", "?"),
            "time_ms":      s.get("window_start_ms", 0),
            "stress_score": _to_float(s.get("value")),
            "confidence":   _to_float(s.get("confidence")),
        }
        for s in voice_signals
        if s.get("signal_type") == "vocal_stress_score" and _to_float(s.get("value")) > 0.4
    ]
    return sorted(peaks, key=lambda x: x["stress_score"], reverse=True)


def _build_unified_states(fusion_signals: list, speakers: list) -> list:
    """Synthesise a minimal unified_states list from the latest fusion signal per speaker."""
    latest: dict[str, dict] = {}
    for s in fusion_signals:
        sid = s.get("speaker_id", "unknown")
        t   = s.get("window_start_ms", 0) or 0
        if sid not in latest or t > latest[sid].get("window_start_ms", 0):
            latest[sid] = s

    states = []
    for sid in speakers:
        sig = latest.get(sid, {})
        meta = sig.get("metadata", {}) if sig else {}
        states.append({
            "speaker_id":      sid,
            "stress_level":    _to_float(meta.get("stress_level")),
            "confidence_level": _to_float(meta.get("confidence_level", 0.5)),
            "engagement_level": _to_float(meta.get("engagement", 0.5)),
            "sentiment_score":  _to_float(meta.get("sentiment_score")),
            "authenticity_score": _to_float(meta.get("authenticity_score", 0.7)),
        })
    return states


def _to_float(v) -> float:
    if v is None or v == "":
        return 0.0
    try:
        return float(v)
    except (ValueError, TypeError):
        return 0.0


# ═══════════════════════════════════════════════════════════════════
# Transcript segment helpers (unchanged from original)
# ═══════════════════════════════════════════════════════════════════

def extract_transcript_segments_from_result(result: dict) -> list[dict]:
    """Extract transcript segments from the pipeline result dict."""
    transcript_segs = result.get("transcript_segments", [])
    if transcript_segs:
        segments = []
        for seg in transcript_segs:
            text = seg.get("text", "").strip()
            if text:
                segments.append({
                    "speaker":   seg.get("speaker", "Speaker_0"),
                    "start_ms":  seg.get("start_ms", 0),
                    "end_ms":    seg.get("end_ms", 0),
                    "text":      text,
                })
        if segments:
            return segments

    # Fallback: reconstruct from voice signal metadata
    segments = []
    for signal in result.get("signals", []):
        meta = signal.get("metadata", {})
        if meta and "transcript_text" in meta:
            segments.append({
                "speaker":  signal.get("speaker_id", "Speaker_0"),
                "start_ms": signal.get("window_start_ms", 0),
                "end_ms":   signal.get("window_end_ms", 0),
                "text":     meta["transcript_text"],
            })
    seen = set()
    unique = []
    for seg in segments:
        key = (seg["speaker"], seg["text"][:50])
        if key not in seen:
            seen.add(key)
            unique.append(seg)
    return unique


def extract_transcript_segments(result: dict, is_synthetic: bool) -> list[dict]:
    """Extract or synthesise transcript segments for display/analysis helpers."""
    segs = extract_transcript_segments_from_result(result)
    if segs and segs[0].get("text") != "[no transcript]":
        return segs

    if not is_synthetic:
        return segs or []

    # Synthetic fallback: map CONVERSATION to approximate timestamps
    dur_ms = int(result.get("duration_seconds", len(CONVERSATION) * 5) * 1000)
    seg_dur = dur_ms // len(CONVERSATION) if CONVERSATION else 5000
    return [
        {
            "speaker":  "Speaker_0" if voice == "Daniel" else "Speaker_1",
            "start_ms": i * seg_dur,
            "end_ms":   (i + 1) * seg_dur,
            "text":     text,
        }
        for i, (voice, text) in enumerate(CONVERSATION)
    ]


# ═══════════════════════════════════════════════════════════════════
# Display helpers
# ═══════════════════════════════════════════════════════════════════

def _print_pipeline_summary(result: dict):
    """Print voice + language + fusion summary after pipeline returns."""
    print(f"\n  {C_BLUE}── Per-Speaker Stress Summary ──{C_RESET}")
    for sid, stats in result["summary"]["per_speaker"].items():
        print(
            f"  {C_BOLD}{sid}{C_RESET}: "
            f"avg_stress={stats.get('avg_stress', 0):.3f}, "
            f"max_stress={stats.get('max_stress', 0):.3f}, "
            f"fillers={stats.get('total_fillers', 0)}, "
            f"pitch_events={stats.get('pitch_elevation_events', 0)}"
        )

    peaks = result["summary"].get("stress_peaks", [])
    if peaks:
        print(f"\n  {C_YELLOW}── Top Stress Peaks ──{C_RESET}")
        for p in peaks[:5]:
            t_sec = (p.get("time_ms", 0) or 0) / 1000
            print(
                f"  {p['speaker']} @ {t_sec:.1f}s: "
                f"stress={p['stress_score']:.3f} "
                f"(conf={p['confidence']:.3f})"
            )

    print(f"\n  {C_PURPLE}── Language Summary ──{C_RESET}")
    for sid, stats in result["summary"]["per_speaker"].items():
        print(
            f"  {C_BOLD}{sid}{C_RESET}: "
            f"avg_sentiment={stats.get('avg_sentiment', 0):+.3f}, "
            f"buying_signals={stats.get('buying_signal_count', 0)}, "
            f"objections={stats.get('objection_count', 0)}"
        )

    sent_dist = result["summary"].get("sentiment_distribution", {})
    if sent_dist:
        print(f"\n  {C_PURPLE}── Sentiment Distribution ──{C_RESET}")
        total = sum(sent_dist.values()) or 1
        for label, count in sent_dist.items():
            bar = "█" * int(30 * count / total)
            print(f"  {label:>10}: {bar} {count}")

    buy_moments = result["summary"].get("buying_signal_moments", [])
    if buy_moments:
        print(f"\n  {C_GREEN}── Buying Signal Moments ──{C_RESET}")
        for m in buy_moments[:5]:
            t_sec = (m.get("time_ms", 0) or 0) / 1000
            print(f"  {m['speaker']} @ {t_sec:.1f}s: strength={m['strength']:.3f}, categories={m.get('categories', [])}")

    obj_moments = result["summary"].get("objection_moments", [])
    if obj_moments:
        print(f"\n  {C_YELLOW}── Objection Moments ──{C_RESET}")
        for m in obj_moments[:5]:
            t_sec = (m.get("time_ms", 0) or 0) / 1000
            print(f"  {m['speaker']} @ {t_sec:.1f}s: strength={m['strength']:.3f}, categories={m.get('categories', [])}")

    print(f"\n  {C_ORANGE}── Unified Speaker States ──{C_RESET}")
    for state in result.get("unified_states", []):
        sid = state.get("speaker_id", "?")
        print(f"  {C_BOLD}{sid}{C_RESET}:")
        print(f"    Stress:       {state.get('stress_level', 0):.2f}")
        print(f"    Confidence:   {state.get('confidence_level', 0):.2f}")
        print(f"    Engagement:   {state.get('engagement_level', 0):.2f}")
        print(f"    Sentiment:    {state.get('sentiment_score', 0):+.2f}")
        print(f"    Authenticity: {state.get('authenticity_score', 0):.2f}")

    alerts = result.get("alerts", [])
    if alerts:
        print(f"\n  {C_RED}── Alerts ──{C_RESET}")
        for alert in alerts:
            sev = alert.get("severity", "?")
            sev_color = {
                "red": C_RED, "orange": C_ORANGE,
                "yellow": C_YELLOW, "green": C_GREEN,
            }.get(sev, C_DIM)
            print(
                f"  {sev_color}[{sev.upper()}]{C_RESET} "
                f"{alert.get('title', '?')} — {alert.get('speaker_id', '?')}"
            )
            print(f"    {C_DIM}{alert.get('description', '')}{C_RESET}")


def print_final_report(result: dict, segments: list[dict]):
    """Print the full pipeline report to stdout."""
    print(f"\n\n{'═' * 60}")
    print(f"{C_BOLD}{C_CYAN}  NEXUS PIPELINE REPORT{C_RESET}")
    print(f"{'═' * 60}")
    print(f"  Generated:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Session:    {result.get('session_id', '?')}")
    print(f"  Duration:   {result['duration_seconds']:.1f}s")
    print(f"  Speakers:   {len(result['speakers'])}")

    # ── Transcript with inline signals ──
    print(f"\n{C_BOLD}{'─' * 60}{C_RESET}")
    print(f"{C_BOLD}  TRANSCRIPT{C_RESET}")
    print(f"{'─' * 60}")

    all_signals = result.get("signals", []) + result.get("language_signals", [])

    for seg in segments:
        t_start = seg["start_ms"] / 1000
        speaker = seg["speaker"]

        matched = [
            s for s in all_signals
            if (s.get("window_start_ms", 0) or 0) < seg["end_ms"]
            and (s.get("window_end_ms", 0) or 0) > seg["start_ms"]
            and (s.get("speaker_id", "") == speaker or not s.get("speaker_id"))
        ]

        speaker_color = C_BLUE if speaker == "Speaker_0" else C_PURPLE
        print(f"\n  {C_DIM}[{t_start:6.1f}s]{C_RESET} {speaker_color}{C_BOLD}{speaker}{C_RESET}")
        print(f"  {seg['text']}")

        if matched:
            badges = []
            for s in matched:
                stype = s.get("signal_type", "?")
                val   = s.get("value")
                vtxt  = s.get("value_text", "")

                if stype == "vocal_stress_score" and val is not None:
                    fval = _to_float(val)
                    if fval > 0.5:
                        badges.append(f"{C_RED}STRESS:{fval:.2f}{C_RESET}")
                    elif fval > 0.3:
                        badges.append(f"{C_YELLOW}stress:{fval:.2f}{C_RESET}")
                elif stype == "filler_detection":
                    badges.append(f"{C_YELLOW}FILLER{C_RESET}")
                elif stype == "sentiment_score" and val is not None:
                    fval = _to_float(val)
                    if fval > 0.3:
                        badges.append(f"{C_GREEN}sent:+{fval:.2f}{C_RESET}")
                    elif fval < -0.3:
                        badges.append(f"{C_RED}sent:{fval:.2f}{C_RESET}")
                elif stype == "buying_signal":
                    badges.append(f"{C_GREEN}BUY:{vtxt}{C_RESET}")
                elif stype == "objection_signal":
                    badges.append(f"{C_ORANGE}OBJ:{vtxt}{C_RESET}")
                elif stype == "power_language_score" and val is not None:
                    if _to_float(val) < 0.35:
                        badges.append(f"{C_YELLOW}low-power{C_RESET}")
                elif stype == "intent_classification":
                    badges.append(f"{C_CYAN}intent:{vtxt}{C_RESET}")

            if badges:
                print(f"  {C_DIM}  signals:{C_RESET} {' | '.join(badges)}")

    # ── Signal Totals ──
    print(f"\n{C_BOLD}{'─' * 60}{C_RESET}")
    print(f"{C_BOLD}  SIGNAL TOTALS{C_RESET}")
    print(f"{'─' * 60}")
    print(f"  Voice signals:    {len(result.get('signals', []))}")
    print(f"  Language signals: {len(result.get('language_signals', []))}")
    print(f"  Fusion signals:   {len(result.get('fusion_signals', []))}")
    print(f"  Alerts:           {len(result.get('alerts', []))}")

    # ── Narrative Report ──
    report = result.get("report")
    if report:
        print(f"\n{C_BOLD}{'─' * 60}{C_RESET}")
        print(f"{C_BOLD}  NARRATIVE REPORT (LLM-Generated){C_RESET}")
        print(f"{'─' * 60}")

        exec_summary = ""
        if isinstance(report, dict):
            exec_summary = report.get("executive_summary", "")
        elif isinstance(report, str):
            exec_summary = report

        if exec_summary:
            print(f"\n  {C_CYAN}Executive Summary:{C_RESET}")
            for line in exec_summary.split("\n"):
                print(f"  {line}")

        if isinstance(report, dict):
            key_moments = report.get("key_moments", [])
            if key_moments:
                print(f"\n  {C_CYAN}Key Moments:{C_RESET}")
                for i, moment in enumerate(key_moments, 1):
                    if isinstance(moment, dict):
                        print(f"  {i}. {moment.get('description', moment)}")
                    else:
                        print(f"  {i}. {moment}")

            insights = report.get("cross_modal_insights", []) or report.get("insights", [])
            if insights:
                print(f"\n  {C_CYAN}Cross-Modal Insights:{C_RESET}")
                for insight in insights:
                    if isinstance(insight, dict):
                        print(f"  - {insight.get('description', insight)}")
                    else:
                        print(f"  - {insight}")

            recommendations = report.get("recommendations", [])
            if recommendations:
                print(f"\n  {C_CYAN}Recommendations:{C_RESET}")
                for rec in recommendations:
                    if isinstance(rec, dict):
                        print(f"  - {rec.get('text', rec)}")
                    else:
                        print(f"  - {rec}")

    print(f"\n{'═' * 60}")
    print(f"{C_GREEN}{C_BOLD}  PIPELINE COMPLETE{C_RESET}")
    print(f"{'═' * 60}\n")


# ═══════════════════════════════════════════════════════════════════
# Health check
# ═══════════════════════════════════════════════════════════════════

def check_health() -> bool:
    """Check the monolith /health endpoint."""
    try:
        resp = httpx.get(f"{BACKEND_URL}/health", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            agent_status = data.get("agents", {})
            agents_ok = all(v == "ok" for v in agent_status.values()) if agent_status else True
            color = C_GREEN if agents_ok else C_YELLOW
            print(f"  {color}[OK]{C_RESET} NEXUS Backend ({BACKEND_URL})")
            if agent_status:
                for name, status in agent_status.items():
                    sc = C_GREEN if status == "ok" else C_RED
                    print(f"       {sc}{name}: {status}{C_RESET}")
            return True
    except httpx.ConnectError:
        pass
    print(f"  {C_RED}[DOWN]{C_RESET} NEXUS Backend ({BACKEND_URL})")
    return False


# ═══════════════════════════════════════════════════════════════════
# Input preparation (unchanged from original)
# ═══════════════════════════════════════════════════════════════════

def prepare_input_audio(args) -> Path:
    """Prepare audio from any input source (URL, file, or synthetic)."""
    if args.url:
        print(f"\n{C_BOLD}{'=' * 60}{C_RESET}")
        print(f"{C_BOLD}  STEP 0: Downloading & Converting Media{C_RESET}")
        print(f"{C_BOLD}{'=' * 60}{C_RESET}")
        print(f"  {C_DIM}URL: {args.url}{C_RESET}")

        try:
            sys.path.insert(0, str(PROJECT_ROOT))
            from shared.utils.media_ingest import prepare_audio_sync, get_capabilities

            caps = get_capabilities()
            print(f"  {C_DIM}ffmpeg: {'yes' if caps['ffmpeg_available'] else 'no'}, "
                  f"yt-dlp: {'yes' if caps['ytdlp_available'] else 'no'}, "
                  f"afconvert: {'yes' if caps['afconvert_available'] else 'no'}{C_RESET}")

            audio_path = prepare_audio_sync(
                args.url,
                output_dir=str(PROJECT_ROOT / "data" / "recordings"),
                filename="url_download",
            )
            print(f"  {C_GREEN}Audio prepared: {audio_path}{C_RESET}")
            return Path(audio_path)

        except Exception as e:
            print(f"  {C_RED}Media ingestion failed: {e}{C_RESET}")
            sys.exit(1)

    if args.audio_file:
        input_path = Path(args.audio_file)
        if not input_path.exists():
            print(f"  {C_RED}File not found: {input_path}{C_RESET}")
            sys.exit(1)

        ext = input_path.suffix.lower()
        video_exts = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".flv"}
        audio_exts = {".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".opus"}

        if ext in video_exts or ext in audio_exts:
            print(f"\n{C_BOLD}{'=' * 60}{C_RESET}")
            print(f"{C_BOLD}  STEP 0: Converting Media to 16kHz WAV{C_RESET}")
            print(f"{C_BOLD}{'=' * 60}{C_RESET}")
            print(f"  {C_DIM}Input: {input_path} ({ext}){C_RESET}")

            try:
                sys.path.insert(0, str(PROJECT_ROOT))
                from shared.utils.media_ingest import prepare_audio_sync

                audio_path = prepare_audio_sync(
                    str(input_path),
                    output_dir=str(PROJECT_ROOT / "data" / "recordings"),
                    filename=input_path.stem,
                )
                print(f"  {C_GREEN}Converted: {audio_path}{C_RESET}")
                return Path(audio_path)

            except Exception as e:
                print(f"  {C_RED}Conversion failed: {e}{C_RESET}")
                sys.exit(1)
        else:
            print(f"\n  Using provided audio: {input_path}")
            return input_path

    if args.skip_audio and TEST_AUDIO_PATH.exists():
        print(f"\n  Reusing existing audio: {TEST_AUDIO_PATH}")
        return TEST_AUDIO_PATH

    if args.use_external_tts:
        return generate_test_audio_external_tts(TEST_AUDIO_PATH)
    else:
        return generate_test_audio(TEST_AUDIO_PATH)


# ═══════════════════════════════════════════════════════════════════
# Content classification + LLM enrichment (unchanged logic)
# ═══════════════════════════════════════════════════════════════════

def classify_content(segments: list[dict], args) -> dict:
    if args.type:
        print(f"\n  {C_CYAN}Content type (manual): {args.type}{C_RESET}")
        return {
            "content_type": args.type,
            "confidence":   1.0,
            "reasoning":    "User-specified override",
            "method":       "manual",
        }

    print(f"\n  {C_CYAN}Auto-detecting content type...{C_RESET}")
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from shared.utils.content_classifier import classify_content_type_sync

        result = classify_content_type_sync(segments)
        print(
            f"  {C_CYAN}Detected: {result['content_type']} "
            f"(confidence={result['confidence']:.2f}, method={result['method']}){C_RESET}"
        )
        if result.get("reasoning"):
            print(f"  {C_DIM}Reason: {result['reasoning']}{C_RESET}")
        return result

    except Exception as e:
        print(f"  {C_YELLOW}Classification failed: {e}. Defaulting to sales_call.{C_RESET}")
        return {
            "content_type": "sales_call",
            "confidence":   0.0,
            "reasoning":    f"Classification failed: {e}",
            "method":       "fallback",
        }


ROLE_MAP = {
    "sales_call":          ["Seller", "Prospect"],
    "podcast":             ["Host", "Guest"],
    "interview":           ["Interviewer", "Candidate"],
    "debate":              ["Speaker A", "Speaker B"],
    "meeting":             ["Facilitator", "Participant"],
    "lecture":             ["Lecturer", "Audience"],
    "presentation":        ["Presenter", "Audience"],
    "casual_conversation": ["Speaker A", "Speaker B"],
}


def assign_speaker_roles(segments: list[dict], content_type: str) -> dict:
    """Use keyword detection → LLM → heuristic to assign roles to speakers."""
    import re

    roles    = ROLE_MAP.get(content_type, ["Speaker A", "Speaker B"])
    speakers = sorted(set(s["speaker"] for s in segments))

    if content_type != "sales_call" or len(speakers) < 2:
        return {spk: roles[i % len(roles)] for i, spk in enumerate(speakers)}

    seller_patterns = [
        r"\bcalling\s+(you\s+)?from\b",
        r"\bmy\s+name\s+is\b",
        r"\bthis\s+is\s+\w+\s+(calling|from)\b",
        r"\bI('m|\s+am)\s+\w+\s+from\b",
        r"\bwe\s+(offer|provide|specialize|help)\b",
        r"\bour\s+(company|service|product|team|platform)\b",
        r"\bI\s+(wanted|was\s+wondering|was\s+calling)\b",
    ]
    prospect_patterns = [
        r"\bnot\s+(looking|interested|ready)\b",
        r"\bthank\s+you\s+for\s+(the\s+)?call",
        r"\bwe('re|\s+are)\s+(not|already)\b",
        r"\bwho\s+(is\s+this|are\s+you)\b",
        r"\bwhat\s+(company|is\s+this\s+about)\b",
    ]

    seller_score   = {spk: 0 for spk in speakers}
    prospect_score = {spk: 0 for spk in speakers}

    for seg in segments:
        spk  = seg["speaker"]
        text = seg["text"].lower()
        for pat in seller_patterns:
            if re.search(pat, text):
                seller_score[spk] += 1
        for pat in prospect_patterns:
            if re.search(pat, text):
                prospect_score[spk] += 1

    total_seller  = sum(seller_score.values())
    total_prospect = sum(prospect_score.values())
    if total_seller > 0 or total_prospect > 0:
        net    = {spk: seller_score[spk] - prospect_score[spk] for spk in speakers}
        ranked = sorted(speakers, key=lambda s: net[s], reverse=True)
        result = {ranked[0]: "Seller", ranked[1]: "Prospect"}
        print(f"  Speaker roles (keyword): {result}")
        return result

    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from shared.utils.llm_client import complete, is_configured
        if is_configured():
            excerpt = "\n".join(f'{s["speaker"]}: {s["text"]}' for s in segments[:16])
            system_prompt = (
                f"You are a conversation analyst. Given transcript excerpts from a "
                f"{content_type.replace('_', ' ')}, identify each speaker's role.\n\n"
                f"Likely roles: {', '.join(roles)}.\n"
                f"IMPORTANT: Do NOT assume Speaker_0 is the Seller. Analyze the actual words.\n"
                f"Return ONLY a JSON object mapping speaker IDs to roles."
            )
            raw = complete(system_prompt=system_prompt, user_prompt=excerpt, max_tokens=200, temperature=0.0)
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            parsed = json.loads(text)
            if isinstance(parsed, dict) and all(isinstance(v, str) for v in parsed.values()):
                print(f"  Speaker roles (LLM): {parsed}")
                return parsed
    except Exception as e:
        print(f"  {C_DIM}Role assignment LLM failed ({e}), using heuristic{C_RESET}")

    objection_counts: dict[str, int] = {}
    for seg in segments:
        low = seg["text"].lower()
        if any(kw in low for kw in ["not looking", "not interested", "concern", "expensive",
                                     "not sure", "issue", "worried", "no thank you", "thank you for the call"]):
            objection_counts[seg["speaker"]] = objection_counts.get(seg["speaker"], 0) + 1
    if objection_counts:
        prospect = max(objection_counts, key=objection_counts.get)
        result   = {spk: ("Prospect" if spk == prospect else "Seller") for spk in speakers}
        print(f"  Speaker roles (heuristic): {result}")
        return result

    return {spk: roles[i % len(roles)] for i, spk in enumerate(speakers)}


def analyze_call_outcome(
    segments: list[dict],
    result: dict,
    content_type: str,
) -> dict:
    """Analyze call outcome for sales_call content type. Returns outcome dict or None."""
    if content_type != "sales_call":
        return None

    per_speaker = result["summary"].get("per_speaker", {})
    alerts      = result.get("alerts", [])

    total_objections = sum(v.get("objection_count", 0) for v in per_speaker.values())
    total_buying     = sum(v.get("buying_signal_count", 0) for v in per_speaker.values())

    outcome = {
        "objection_detected":   total_objections > 0,
        "objection_count":      total_objections,
        "buying_signals_count": total_buying,
        "alerts_count":         len(alerts),
        "objection_resolved":   False,
    }

    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from shared.utils.llm_client import complete, is_configured
        if is_configured():
            transcript_text = "\n".join(f'{s["speaker"]}: {s["text"]}' for s in segments)
            system_prompt = (
                "You are an expert sales call analyst. Analyze the full conversation to "
                "determine the call outcome.\n\n"
                "Return ONLY a JSON object with these fields:\n"
                '"objection_handled": "yes" or "no" or "partially",\n'
                '"decision_readiness": "ready" or "not_ready" or "uncertain",\n'
                '"estimated_outcome": "positive" or "neutral" or "negative",\n'
                '"outcome_reasoning": "2-3 sentence explanation"'
            )
            signal_context = (
                f"Signal summary: {total_objections} objections, {total_buying} buying signals, "
                f"{len(alerts)} fusion alerts.\n"
            )
            raw = complete(
                system_prompt=system_prompt,
                user_prompt=f"Transcript:\n{transcript_text}\n\n{signal_context}",
                max_tokens=400,
                temperature=0.0,
            )
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            parsed = json.loads(text)
            outcome.update(parsed)
            return outcome
    except Exception as e:
        print(f"  {C_DIM}Call outcome LLM failed ({e}), using heuristic{C_RESET}")

    if total_buying > 0 and total_objections == 0:
        outcome.update({"objection_handled": "n/a", "estimated_outcome": "positive",
                        "decision_readiness": "ready",
                        "outcome_reasoning": "Buying signals detected with no objections"})
    elif total_buying > 0:
        outcome.update({"objection_handled": "partially", "estimated_outcome": "neutral",
                        "decision_readiness": "uncertain",
                        "outcome_reasoning": "Mixed signals: objections raised but buying interest present"})
    elif total_objections > 0:
        outcome.update({"objection_handled": "no", "estimated_outcome": "negative",
                        "decision_readiness": "not_ready",
                        "outcome_reasoning": "Objections raised without clear resolution or buying signals"})
    else:
        outcome.update({"objection_handled": "n/a", "estimated_outcome": "neutral",
                        "decision_readiness": "uncertain",
                        "outcome_reasoning": "No strong buying or objection signals detected"})

    return outcome


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="NEXUS End-to-End Pipeline Test",
        epilog="""
Examples:
  python scripts/test_pipeline.py                                    # Synthetic sales call
  python scripts/test_pipeline.py --audio-file recording.mp3         # Any audio file
  python scripts/test_pipeline.py --audio-file lecture.mp4           # Video file
  python scripts/test_pipeline.py --url "https://youtube.com/watch?v=XXX"  # YouTube URL
  python scripts/test_pipeline.py --audio-file podcast.mp3 --type podcast --speakers 3
  python scripts/test_pipeline.py --audio-file debate.wav --type debate --speakers 2
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--skip-audio",      action="store_true", help="Reuse existing test audio")
    parser.add_argument("--no-intent",       action="store_true", help="Skip LLM intent classification (unused, kept for compat)")
    parser.add_argument("--audio-file",      type=str,            help="Use a specific audio/video file")
    parser.add_argument("--url",             type=str,            help="Download audio from URL (YouTube, podcast, direct)")
    parser.add_argument("--type",            type=str,            default=None,
                        choices=["sales_call", "podcast", "interview", "lecture",
                                 "debate", "meeting", "presentation",
                                 "casual_conversation", "other"],
                        help="Override auto-detected content type")
    parser.add_argument("--speakers",        type=int,            default=None,
                        help="Expected number of speakers (1-10, default: auto)")
    parser.add_argument("--use-external-tts", action="store_true",
                        help="Use GPU-accelerated Coqui TTS API for audio generation")
    parser.add_argument("--html",            action="store_true",
                        help="Generate a self-contained HTML report (in addition to JSON)")
    parser.add_argument("--token",           type=str,            default=None,
                        help="JWT bearer token (overrides NEXUS_TOKEN env var)")
    parser.add_argument("--url-backend",     type=str,            default=None,
                        help="Backend URL (overrides BACKEND_URL env var, default: http://localhost:8000)")
    args = parser.parse_args()

    if args.speakers is not None and (args.speakers < 1 or args.speakers > 10):
        print(f"  {C_RED}--speakers must be between 1 and 10{C_RESET}")
        sys.exit(1)

    global BACKEND_URL, NEXUS_TOKEN
    if args.url_backend:
        BACKEND_URL = args.url_backend.rstrip("/")
    if args.token:
        NEXUS_TOKEN = args.token

    print(f"\n{C_BOLD}{C_CYAN}NEXUS End-to-End Pipeline Test{C_RESET}")
    print(f"{C_DIM}{'─' * 40}{C_RESET}")
    print(f"  {C_DIM}Backend: {BACKEND_URL}{C_RESET}")
    if args.url:
        print(f"  {C_DIM}Input: URL{C_RESET}")
    elif args.audio_file:
        print(f"  {C_DIM}Input: {args.audio_file}{C_RESET}")
    else:
        print(f"  {C_DIM}Input: Synthetic sales call{C_RESET}")
    if args.type:
        print(f"  {C_DIM}Content type: {args.type} (manual){C_RESET}")
    if args.speakers:
        print(f"  {C_DIM}Speakers: {args.speakers}{C_RESET}")

    # ── Health check ──
    print(f"\n  {C_BOLD}Checking backend...{C_RESET}")
    if not check_health():
        print(f"\n  {C_RED}Backend is required. Start with:{C_RESET}")
        print(f"  {C_DIM}  uvicorn backend.main:app --port 8000{C_RESET}")
        sys.exit(1)

    auth = _AuthClient(BACKEND_URL)

    # ── Step 0: Prepare audio ──
    audio_path = prepare_input_audio(args)
    is_synthetic = not bool(args.audio_file or args.url)

    # ── Step 1: Upload and run pipeline ──
    meeting_type = args.type or "sales_call"
    result = run_pipeline_via_api(auth, audio_path, meeting_type=meeting_type, num_speakers=args.speakers)

    # ── Print live summary ──
    _print_pipeline_summary(result)

    # ── Extract transcript segments ──
    segments = extract_transcript_segments(result, is_synthetic)
    print(f"\n  {C_DIM}Extracted {len(segments)} transcript segments{C_RESET}")

    # ── Auto-detect content type from transcript ──
    classification = classify_content(segments, args)
    content_type   = classification["content_type"]

    # ── Assign speaker roles ──
    print(f"\n  {C_CYAN}Assigning speaker roles...{C_RESET}")
    speaker_roles = assign_speaker_roles(segments, content_type)
    for sid, role in speaker_roles.items():
        print(f"    {C_BOLD}{sid}{C_RESET} → {C_CYAN}{role}{C_RESET}")

    # ── Analyze call outcome ──
    call_outcome = analyze_call_outcome(segments, result, content_type)
    if call_outcome:
        est        = call_outcome.get("estimated_outcome", "?")
        est_colour = {"positive": C_GREEN, "negative": C_RED}.get(est, C_YELLOW)
        print(f"\n  {C_CYAN}Call outcome: {est_colour}{est.upper()}{C_RESET}")
        print(f"    {C_DIM}{call_outcome.get('outcome_reasoning', '')}{C_RESET}")

    # ── Final report ──
    print_final_report(result, segments)

    # Save full results to JSON
    results_path = PROJECT_ROOT / "data" / "reports" / "test_pipeline_result.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    full_result = {
        "timestamp":      datetime.now().isoformat(),
        "audio_file":     str(audio_path),
        "session_id":     result.get("session_id"),
        "content_type":   content_type,
        "classification": classification,
        # Flatten into the shape the old scripts expected (for HTML report compat)
        "voice":    {
            "duration_seconds": result["duration_seconds"],
            "speakers":         result["speakers"],
            "signals":          result["signals"],
            "summary":          result["summary"],
        },
        "language": {
            "signals":       result["language_signals"],
            "summary":       result["summary"],
            "segment_count": len(segments),
            "speakers":      [s["speaker_id"] for s in result["speakers"]],
        },
        "fusion": {
            "fusion_signals": result["fusion_signals"],
            "unified_states": result["unified_states"],
            "alerts":         result["alerts"],
            "report":         result.get("report"),
            "speakers":       [s["speaker_id"] for s in result["speakers"]],
        },
        "speaker_roles": speaker_roles,
        "call_outcome":  call_outcome,
    }
    with open(results_path, "w") as f:
        json.dump(full_result, f, indent=2, default=str)
    print(f"  {C_DIM}Full results saved to: {results_path}{C_RESET}")

    if args.html:
        try:
            sys.path.insert(0, str(PROJECT_ROOT))
            from shared.utils.report_generator import generate_html_report
            html_path = generate_html_report(full_result, output_dir=str(results_path.parent))
            print(f"\n  {C_GREEN}{C_BOLD}HTML report generated:{C_RESET}")
            print(f"  {C_CYAN}{html_path}{C_RESET}")
            print(f"  {C_DIM}Open in browser: file://{html_path}{C_RESET}")
        except Exception as e:
            print(f"  {C_YELLOW}HTML report generation failed: {e}{C_RESET}")


if __name__ == "__main__":
    main()
