#!/usr/bin/env python3
"""
NEXUS End-to-End Pipeline Test
Accepts any audio/video file, URL, or generates synthetic audio, then runs:
  Voice Agent (8001) → Language Agent (8002) → Fusion Agent (8007)

Supports all content types: sales calls, podcasts, interviews, lectures,
debates, meetings, presentations, and casual conversations.

Usage:
  python3 scripts/test_pipeline.py                                    # Synthetic sales call
  python3 scripts/test_pipeline.py --audio-file recording.mp3          # Any audio file
  python3 scripts/test_pipeline.py --audio-file lecture.mp4            # Video file (extracts audio)
  python3 scripts/test_pipeline.py --url "https://youtube.com/watch?v=XXX"  # YouTube URL
  python3 scripts/test_pipeline.py --audio-file podcast.mp3 --type podcast --speakers 3
  python3 scripts/test_pipeline.py --skip-audio --type debate          # Reuse + override type
  python3 scripts/test_pipeline.py --voice-only                        # Stop after Voice Agent
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

# httpx is available; requests is not
import httpx

# ── Config ──
VOICE_URL = os.getenv("VOICE_AGENT_URL", "http://localhost:8001")
LANGUAGE_URL = os.getenv("LANGUAGE_AGENT_URL", "http://localhost:8002")
FUSION_URL = os.getenv("FUSION_AGENT_URL", "http://localhost:8007")
TIMEOUT = 300  # 5 minutes — Whisper can be slow on CPU

# External APIs (GPU-accelerated, optional)
EXTERNAL_TTS_URL = os.getenv("EXTERNAL_TTS_URL", "http://110.227.200.12:8009")
EXTERNAL_API_KEY = os.getenv("EXTERNAL_API_KEY", "your-secret-api-key-is-this")

PROJECT_ROOT = Path(__file__).parent.parent
TEST_AUDIO_PATH = PROJECT_ROOT / "data" / "recordings" / "test_sales_call.wav"

# ── Colours for terminal output ──
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_RED = "\033[91m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_BLUE = "\033[94m"
C_PURPLE = "\033[95m"
C_CYAN = "\033[96m"
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

    This produces much more natural-sounding audio than macOS `say`, and
    generates genuinely different voice characteristics for each speaker
    (which helps diarization).
    """
    print(f"\n{C_BOLD}{'=' * 60}{C_RESET}")
    print(f"{C_BOLD}  STEP 0: Generating Audio via External TTS (Coqui XTTS){C_RESET}")
    print(f"{C_BOLD}{'=' * 60}{C_RESET}")
    print(f"  {C_DIM}TTS API: {EXTERNAL_TTS_URL}{C_RESET}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Add shared module to path
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

    # Step 1: Generate speaker reference voices
    # We use the default TTS voice for Speaker A (seller) — clear, confident
    # For Speaker B (buyer), we generate a reference clip and use voice cloning
    # to get a distinctly different voice.
    print(f"\n  {C_CYAN}Generating speaker reference voices...{C_RESET}")

    # Speaker A reference — default TTS voice with seller-like speech
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

    # Speaker B reference — generate with a different text style to get tonal variation
    # Then clone from that for buyer segments
    speaker_b_ref = speaker_ref_dir / "speaker_b_ref.wav"
    try:
        # Use a different prompt style to get a naturally different voice quality
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

    # Step 2: Generate each conversation turn
    print(f"\n  {C_CYAN}Generating conversation turns...{C_RESET}")

    for i, (voice, text) in enumerate(CONVERSATION):
        seg_path = os.path.join(temp_dir, f"seg_{i:03d}.wav")
        is_seller = (voice == "Daniel")
        speaker_label = "Seller" if is_seller else "Buyer"

        print(f"  {C_DIM}[{i+1}/{len(CONVERSATION)}] {speaker_label}: {text[:55]}...{C_RESET}")

        try:
            if is_seller and use_clone_a:
                # Voice clone from Speaker A reference
                audio = tts.synthesize_clone(
                    text,
                    speaker_wav_path=str(speaker_a_ref),
                    language="en",
                )
            elif not is_seller and use_clone_b:
                # Voice clone from Speaker B reference
                audio = tts.synthesize_clone(
                    text,
                    speaker_wav_path=str(speaker_b_ref),
                    language="en",
                )
            else:
                # Fallback: default voice (no cloning)
                audio = tts.synthesize(text, language="en")

            with open(seg_path, "wb") as f:
                f.write(audio)
            segment_files.append(seg_path)

        except Exception as e:
            print(f"  {C_RED}TTS failed for segment {i}: {e}{C_RESET}")
            print(f"  {C_YELLOW}Falling back to macOS say for this segment.{C_RESET}")
            # Fallback: use macOS say for this segment
            aiff_path = seg_path.replace(".wav", ".aiff")
            mac_voice = voice  # Daniel or Fred
            subprocess.run(
                ["say", "-v", mac_voice, "-o", aiff_path, text],
                capture_output=True,
            )
            subprocess.run(
                ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
                 aiff_path, seg_path],
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

    # Step 3: Concatenate all segments into one WAV file
    print(f"\n  {C_CYAN}Concatenating {len(segment_files)} segments...{C_RESET}")

    # Try ffmpeg first, then sox, then manual WAV concat
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
        # Manual WAV concatenation
        _concat_wav_files(segment_files, str(output_path))

    # Cleanup
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

    # Generate individual WAV segments per speaker turn, then concatenate
    temp_dir = tempfile.mkdtemp(prefix="nexus_test_")
    segment_files = []

    for i, (voice, text) in enumerate(CONVERSATION):
        seg_path = os.path.join(temp_dir, f"seg_{i:03d}.aiff")
        print(f"  {C_DIM}[{i+1}/{len(CONVERSATION)}] {voice}: {text[:60]}...{C_RESET}")

        # macOS say outputs AIFF by default
        cmd = ["say", "-v", voice, "-o", seg_path, text]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  {C_RED}say failed: {result.stderr}{C_RESET}")
            sys.exit(1)

        segment_files.append(seg_path)

    # Add a short silence between segments using a tiny pause
    # Concatenate all AIFF files using sox if available, or ffmpeg
    concat_method = None
    for tool in ["sox", "ffmpeg"]:
        if subprocess.run(["which", tool], capture_output=True).returncode == 0:
            concat_method = tool
            break

    if concat_method == "ffmpeg":
        # Build ffmpeg concat filter
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
        # Fallback: just use the first segment converted to WAV via afconvert (macOS)
        print(f"  {C_YELLOW}No sox/ffmpeg found. Using single-file fallback.{C_RESET}")
        # Concat via afconvert — convert each to WAV then cat PCM data
        wav_files = []
        for seg in segment_files:
            wav_seg = seg.replace(".aiff", ".wav")
            subprocess.run(
                ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", seg, wav_seg],
                capture_output=True,
            )
            wav_files.append(wav_seg)

        # Manual WAV concatenation (all same format: 16kHz mono 16-bit)
        _concat_wav_files(wav_files, str(output_path))

    # Cleanup temp files
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
    import struct

    if not wav_files:
        return

    # Read params from first file
    with wave.open(wav_files[0], "rb") as wf:
        params = wf.getparams()

    with wave.open(output_path, "wb") as out:
        out.setparams(params)
        for wf_path in wav_files:
            with wave.open(wf_path, "rb") as wf:
                out.writeframes(wf.readframes(wf.getnframes()))


# ═══════════════════════════════════════════════════════════════════
# STEP 1: Voice Agent
# ═══════════════════════════════════════════════════════════════════

def run_voice_agent(audio_path: Path, num_speakers: int = None) -> dict:
    print(f"\n{C_BOLD}{'═' * 60}{C_RESET}")
    print(f"{C_BOLD}  STEP 1: Voice Agent → {VOICE_URL}/analyse{C_RESET}")
    print(f"{C_BOLD}{'═' * 60}{C_RESET}")

    payload = {
        "file_path": str(audio_path.resolve()),
        "meeting_type": "sales_call",
    }
    if num_speakers is not None:
        payload["num_speakers"] = num_speakers

    print(f"  {C_DIM}Sending: {audio_path.name}{C_RESET}")
    start = time.time()

    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            resp = client.post(f"{VOICE_URL}/analyse", json=payload)
    except httpx.ConnectError:
        print(f"  {C_RED}Cannot connect to Voice Agent at {VOICE_URL}{C_RESET}")
        print(f"  {C_RED}Is it running? Try: scripts/start_all.sh{C_RESET}")
        sys.exit(1)

    elapsed = time.time() - start

    if resp.status_code != 200:
        print(f"  {C_RED}Voice Agent error ({resp.status_code}): {resp.text[:300]}{C_RESET}")
        sys.exit(1)

    result = resp.json()

    print(f"\n  {C_GREEN}Voice Agent complete in {elapsed:.1f}s{C_RESET}")
    print(f"  Duration: {result['duration_seconds']:.1f}s")
    print(f"  Speakers: {len(result['speakers'])}")
    print(f"  Signals:  {len(result['signals'])}")

    # Print per-speaker baselines
    print(f"\n  {C_BLUE}── Per-Speaker Baselines ──{C_RESET}")
    for spk in result["speakers"]:
        bl = spk.get("baseline") or {}
        print(
            f"  {C_BOLD}{spk['speaker_id']}{C_RESET}: "
            f"F0={bl.get('f0_mean', '?')}Hz, "
            f"rate={bl.get('speech_rate_wpm', '?')}wpm, "
            f"cal_conf={bl.get('calibration_confidence', '?')}"
        )

    # Print voice summary
    summary = result.get("summary", {})
    print(f"\n  {C_BLUE}── Voice Summary ──{C_RESET}")
    for sid, stats in summary.get("per_speaker", {}).items():
        print(
            f"  {C_BOLD}{sid}{C_RESET}: "
            f"avg_stress={stats.get('avg_stress', 0):.3f}, "
            f"max_stress={stats.get('max_stress', 0):.3f}, "
            f"fillers={stats.get('total_fillers', 0)}, "
            f"pitch_events={stats.get('pitch_elevation_events', 0)}"
        )

    # Print stress peaks
    peaks = summary.get("stress_peaks", [])
    if peaks:
        print(f"\n  {C_YELLOW}── Top Stress Peaks ──{C_RESET}")
        for p in peaks[:5]:
            t_sec = (p.get("time_ms", 0) or 0) / 1000
            print(
                f"  {p['speaker']} @ {t_sec:.1f}s: "
                f"stress={p['stress_score']:.3f} "
                f"(conf={p['confidence']:.3f})"
            )

    return result


# ═══════════════════════════════════════════════════════════════════
# STEP 2: Language Agent
# ═══════════════════════════════════════════════════════════════════

def extract_transcript_segments_from_voice(voice_result: dict) -> list[dict]:
    """
    Extract transcript segments from voice agent response.
    Uses the transcript_segments field returned by the Voice Agent
    (diarised Whisper output with speaker labels, timestamps, and text).
    """
    # ── Primary: use transcript_segments from Voice Agent response ──
    transcript_segs = voice_result.get("transcript_segments", [])
    if transcript_segs:
        segments = []
        for seg in transcript_segs:
            text = seg.get("text", "").strip()
            if text:
                segments.append({
                    "speaker": seg.get("speaker", "Speaker_0"),
                    "start_ms": seg.get("start_ms", 0),
                    "end_ms": seg.get("end_ms", 0),
                    "text": text,
                })
        if segments:
            return segments

    # ── Fallback: look in signal metadata ──
    segments = []
    for signal in voice_result.get("signals", []):
        meta = signal.get("metadata", {})
        if meta and "transcript_text" in meta:
            segments.append({
                "speaker": signal.get("speaker_id", "Speaker_0"),
                "start_ms": signal.get("window_start_ms", 0),
                "end_ms": signal.get("window_end_ms", 0),
                "text": meta["transcript_text"],
            })

    # Deduplicate by text
    seen = set()
    unique_segments = []
    for seg in segments:
        key = (seg["speaker"], seg["text"][:50])
        if key not in seen:
            seen.add(key)
            unique_segments.append(seg)

    if unique_segments:
        return unique_segments

    # ── Last resort: group signals by speaker windows ──
    signal_groups = []
    current_group = None
    for signal in sorted(
        voice_result.get("signals", []),
        key=lambda s: s.get("window_start_ms", 0),
    ):
        speaker = signal.get("speaker_id", "Speaker_0")
        if current_group and current_group["speaker"] == speaker:
            current_group["end_ms"] = max(
                current_group["end_ms"],
                signal.get("window_end_ms", 0),
            )
        else:
            if current_group:
                signal_groups.append(current_group)
            current_group = {
                "speaker": speaker,
                "start_ms": signal.get("window_start_ms", 0),
                "end_ms": signal.get("window_end_ms", 0),
                "text": "[no transcript]",
            }
    if current_group:
        signal_groups.append(current_group)

    return signal_groups or segments


def extract_transcript_segments(voice_result: dict) -> list[dict]:
    """Extract transcript segments from voice agent result for language agent.
    For synthetic conversations, falls back to CONVERSATION variable mapping."""

    # First try to extract from voice agent metadata
    segments = extract_transcript_segments_from_voice(voice_result)
    if segments and segments[0].get("text") != "[no transcript]":
        return segments

    # Fallback: build synthetic segments from our known conversation
    # mapped to approximate time windows based on voice agent duration
    segments = []
    duration_ms = int(voice_result["duration_seconds"] * 1000)
    segment_duration = duration_ms // len(CONVERSATION) if CONVERSATION else 5000

    for i, (voice, text) in enumerate(CONVERSATION):
        speaker = "Speaker_0" if voice == "Daniel" else "Speaker_1"
        segments.append({
            "speaker": speaker,
            "start_ms": i * segment_duration,
            "end_ms": (i + 1) * segment_duration,
            "text": text,
        })

    return segments


def run_language_agent(
    segments: list[dict],
    run_intent: bool = True,
    content_type: str = "sales_call",
) -> dict:
    print(f"\n{C_BOLD}{'═' * 60}{C_RESET}")
    print(f"{C_BOLD}  STEP 2: Language Agent → {LANGUAGE_URL}/analyse{C_RESET}")
    print(f"{C_BOLD}{'═' * 60}{C_RESET}")
    print(f"  {C_DIM}Content type: {content_type}{C_RESET}")

    payload = {
        "segments": segments,
        "meeting_type": content_type,
        "content_type": content_type,
        "run_intent_classification": run_intent,
    }

    print(f"  {C_DIM}Sending {len(segments)} transcript segments{C_RESET}")
    start = time.time()

    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            resp = client.post(f"{LANGUAGE_URL}/analyse", json=payload)
    except httpx.ConnectError:
        print(f"  {C_RED}Cannot connect to Language Agent at {LANGUAGE_URL}{C_RESET}")
        sys.exit(1)

    elapsed = time.time() - start

    if resp.status_code != 200:
        print(f"  {C_RED}Language Agent error ({resp.status_code}): {resp.text[:300]}{C_RESET}")
        sys.exit(1)

    result = resp.json()

    print(f"\n  {C_GREEN}Language Agent complete in {elapsed:.1f}s{C_RESET}")
    print(f"  Segments: {result['segment_count']}")
    print(f"  Speakers: {result['speakers']}")
    print(f"  Signals:  {len(result['signals'])}")

    # Print language summary
    summary = result.get("summary", {})
    print(f"\n  {C_PURPLE}── Language Summary ──{C_RESET}")
    for sid, stats in summary.get("per_speaker", {}).items():
        print(
            f"  {C_BOLD}{sid}{C_RESET}: "
            f"avg_sentiment={stats.get('avg_sentiment', 0):+.3f}, "
            f"buying_signals={stats.get('buying_signal_count', 0)}, "
            f"objections={stats.get('objection_count', 0)}, "
            f"power_score={stats.get('avg_power_score', 0.5):.3f}"
        )
        intents = stats.get("intent_distribution", {})
        if intents:
            print(f"    Intents: {intents}")

    # Sentiment distribution
    sent_dist = summary.get("sentiment_distribution", {})
    if sent_dist:
        print(f"\n  {C_PURPLE}── Sentiment Distribution ──{C_RESET}")
        total = sum(sent_dist.values()) or 1
        for label, count in sent_dist.items():
            bar = "█" * int(30 * count / total)
            print(f"  {label:>10}: {bar} {count}")

    # Buying signal moments
    buy_moments = summary.get("buying_signal_moments", [])
    if buy_moments:
        print(f"\n  {C_GREEN}── Buying Signal Moments ──{C_RESET}")
        for m in buy_moments[:5]:
            t_sec = (m.get("time_ms", 0) or 0) / 1000
            print(f"  {m['speaker']} @ {t_sec:.1f}s: strength={m['strength']:.3f}, categories={m.get('categories', [])}")

    # Objection moments
    obj_moments = summary.get("objection_moments", [])
    if obj_moments:
        print(f"\n  {C_YELLOW}── Objection Moments ──{C_RESET}")
        for m in obj_moments[:5]:
            t_sec = (m.get("time_ms", 0) or 0) / 1000
            print(f"  {m['speaker']} @ {t_sec:.1f}s: strength={m['strength']:.3f}, categories={m.get('categories', [])}")

    return result


# ═══════════════════════════════════════════════════════════════════
# STEP 3: Fusion Agent
# ═══════════════════════════════════════════════════════════════════

def run_fusion_agent(
    voice_result: dict,
    language_result: dict,
    content_type: str = "sales_call",
) -> dict:
    print(f"\n{C_BOLD}{'═' * 60}{C_RESET}")
    print(f"{C_BOLD}  STEP 3: Fusion Agent → {FUSION_URL}/analyse{C_RESET}")
    print(f"{C_BOLD}{'═' * 60}{C_RESET}")
    print(f"  {C_DIM}Content type: {content_type}{C_RESET}")

    # Tag voice signals with agent field
    voice_signals = []
    for s in voice_result.get("signals", []):
        sig = {**s, "agent": "voice"}
        # Ensure speaker_id field exists
        if "speaker_id" not in sig and "speaker" in sig:
            sig["speaker_id"] = sig["speaker"]
        voice_signals.append(sig)

    # Tag language signals with agent field
    language_signals = []
    for s in language_result.get("signals", []):
        sig = {**s, "agent": "language"}
        if "speaker_id" not in sig and "speaker" in sig:
            sig["speaker_id"] = sig["speaker"]
        language_signals.append(sig)

    payload = {
        "voice_signals": voice_signals,
        "language_signals": language_signals,
        "meeting_type": content_type,
        "content_type": content_type,
        "generate_report": True,
        "voice_summary": voice_result.get("summary"),
        "language_summary": language_result.get("summary"),
    }

    print(f"  {C_DIM}Sending {len(voice_signals)} voice + {len(language_signals)} language signals{C_RESET}")
    start = time.time()

    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            resp = client.post(f"{FUSION_URL}/analyse", json=payload)
    except httpx.ConnectError:
        print(f"  {C_RED}Cannot connect to Fusion Agent at {FUSION_URL}{C_RESET}")
        sys.exit(1)

    elapsed = time.time() - start

    if resp.status_code != 200:
        print(f"  {C_RED}Fusion Agent error ({resp.status_code}): {resp.text[:300]}{C_RESET}")
        sys.exit(1)

    result = resp.json()

    print(f"\n  {C_GREEN}Fusion Agent complete in {elapsed:.1f}s{C_RESET}")
    print(f"  Speakers:       {result['speakers']}")
    print(f"  Fusion signals: {len(result['fusion_signals'])}")
    print(f"  Alerts:         {len(result['alerts'])}")

    # Print Unified Speaker States
    print(f"\n  {C_ORANGE}── Unified Speaker States ──{C_RESET}")
    for state in result.get("unified_states", []):
        sid = state.get("speaker_id", "?")
        print(f"  {C_BOLD}{sid}{C_RESET}:")
        print(f"    Stress:       {state.get('stress_level', 0):.2f}")
        print(f"    Confidence:   {state.get('confidence_level', 0):.2f}")
        print(f"    Engagement:   {state.get('engagement_level', 0):.2f}")
        print(f"    Sentiment:    {state.get('sentiment_score', 0):+.2f}")
        print(f"    Authenticity: {state.get('authenticity_score', 0):.2f}")

    # Print fusion signals
    fusion_signals = result.get("fusion_signals", [])
    if fusion_signals:
        print(f"\n  {C_ORANGE}── Fusion Signals ──{C_RESET}")
        for fs in fusion_signals:
            t_start = (fs.get("window_start_ms", 0) or 0) / 1000
            t_end = (fs.get("window_end_ms", 0) or 0) / 1000
            print(
                f"  [{t_start:.1f}s-{t_end:.1f}s] {fs.get('speaker_id', '?')}: "
                f"{fs.get('signal_type', '?')} = {fs.get('value_text', '?')} "
                f"(conf={fs.get('confidence', 0):.2f})"
            )

    # Print alerts
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

    return result


# ═══════════════════════════════════════════════════════════════════
# STEP 4: Print Final Report
# ═══════════════════════════════════════════════════════════════════

def print_final_report(
    voice_result: dict,
    language_result: dict,
    fusion_result: dict,
    segments: list[dict],
):
    print(f"\n\n{'═' * 60}")
    print(f"{C_BOLD}{C_CYAN}  NEXUS FULL PIPELINE REPORT{C_RESET}")
    print(f"{'═' * 60}")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Duration:  {voice_result['duration_seconds']:.1f}s")
    print(f"  Speakers:  {len(voice_result['speakers'])}")

    # ── Transcript with inline signals ──
    print(f"\n{C_BOLD}{'─' * 60}{C_RESET}")
    print(f"{C_BOLD}  TRANSCRIPT{C_RESET}")
    print(f"{'─' * 60}")

    all_signals = (
        voice_result.get("signals", [])
        + language_result.get("signals", [])
    )

    for seg in segments:
        t_start = seg["start_ms"] / 1000
        t_end = seg["end_ms"] / 1000
        speaker = seg["speaker"]

        # Find signals in this time window
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
                val = s.get("value")
                vtxt = s.get("value_text", "")
                conf = s.get("confidence", 0)

                if stype == "vocal_stress_score" and val is not None:
                    if val > 0.5:
                        badges.append(f"{C_RED}STRESS:{val:.2f}{C_RESET}")
                    elif val > 0.3:
                        badges.append(f"{C_YELLOW}stress:{val:.2f}{C_RESET}")
                elif stype == "filler_detection":
                    badges.append(f"{C_YELLOW}FILLER{C_RESET}")
                elif stype == "sentiment_score" and val is not None:
                    if val > 0.3:
                        badges.append(f"{C_GREEN}sent:+{val:.2f}{C_RESET}")
                    elif val < -0.3:
                        badges.append(f"{C_RED}sent:{val:.2f}{C_RESET}")
                elif stype == "buying_signal":
                    badges.append(f"{C_GREEN}BUY:{vtxt}{C_RESET}")
                elif stype == "objection_signal":
                    badges.append(f"{C_ORANGE}OBJ:{vtxt}{C_RESET}")
                elif stype == "power_language_score" and val is not None:
                    if val < 0.35:
                        badges.append(f"{C_YELLOW}low-power{C_RESET}")
                elif stype == "intent_classification":
                    badges.append(f"{C_CYAN}intent:{vtxt}{C_RESET}")

            if badges:
                print(f"  {C_DIM}  signals:{C_RESET} {' | '.join(badges)}")

    # ── Signal Totals ──
    print(f"\n{C_BOLD}{'─' * 60}{C_RESET}")
    print(f"{C_BOLD}  SIGNAL TOTALS{C_RESET}")
    print(f"{'─' * 60}")
    print(f"  Voice signals:    {len(voice_result.get('signals', []))}")
    print(f"  Language signals: {len(language_result.get('signals', []))}")
    print(f"  Fusion signals:   {len(fusion_result.get('fusion_signals', []))}")
    print(f"  Alerts:           {len(fusion_result.get('alerts', []))}")

    # ── Narrative Report ──
    report = fusion_result.get("report")
    if report:
        print(f"\n{C_BOLD}{'─' * 60}{C_RESET}")
        print(f"{C_BOLD}  NARRATIVE REPORT (LLM-Generated){C_RESET}")
        print(f"{'─' * 60}")

        exec_summary = report.get("executive_summary", "")
        if exec_summary:
            print(f"\n  {C_CYAN}Executive Summary:{C_RESET}")
            for line in exec_summary.split("\n"):
                print(f"  {line}")

        key_moments = report.get("key_moments", [])
        if key_moments:
            print(f"\n  {C_CYAN}Key Moments:{C_RESET}")
            for i, moment in enumerate(key_moments, 1):
                if isinstance(moment, dict):
                    print(f"  {i}. {moment.get('description', moment)}")
                else:
                    print(f"  {i}. {moment}")

        insights = report.get("cross_modal_insights", [])
        if not insights:
            insights = report.get("insights", [])
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

def check_health(name: str, url: str) -> bool:
    try:
        resp = httpx.get(f"{url}/health", timeout=5)
        if resp.status_code == 200:
            print(f"  {C_GREEN}[OK]{C_RESET} {name} ({url})")
            return True
    except httpx.ConnectError:
        pass
    print(f"  {C_RED}[DOWN]{C_RESET} {name} ({url})")
    return False


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def prepare_input_audio(args) -> Path:
    """
    Prepare audio from any input source (URL, file, or synthetic).
    Uses shared/utils/media_ingest.py for universal media handling.
    """
    # ── URL input ──
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

    # ── File input (audio or video) ──
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
            # Assume WAV or compatible audio — pass through
            print(f"\n  Using provided audio: {input_path}")
            return input_path

    # ── Reuse existing ──
    if args.skip_audio and TEST_AUDIO_PATH.exists():
        print(f"\n  Reusing existing audio: {TEST_AUDIO_PATH}")
        return TEST_AUDIO_PATH

    # ── Generate synthetic ──
    if args.use_external_tts:
        return generate_test_audio_external_tts(TEST_AUDIO_PATH)
    else:
        return generate_test_audio(TEST_AUDIO_PATH)


def classify_content(segments: list[dict], args) -> dict:
    """
    Auto-detect or use override for content type classification.
    """
    if args.type:
        print(f"\n  {C_CYAN}Content type (manual): {args.type}{C_RESET}")
        return {
            "content_type": args.type,
            "confidence": 1.0,
            "reasoning": "User-specified override",
            "method": "manual",
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
            "confidence": 0.0,
            "reasoning": f"Classification failed: {e}",
            "method": "fallback",
        }


# ═══════════════════════════════════════════════════════════════════
# LLM-Powered Enrichment: Speaker Roles + Call Outcome
# ═══════════════════════════════════════════════════════════════════

ROLE_MAP = {
    "sales_call": ["Seller", "Prospect"],
    "podcast": ["Host", "Guest"],
    "interview": ["Interviewer", "Candidate"],
    "debate": ["Speaker A", "Speaker B"],
    "meeting": ["Facilitator", "Participant"],
    "lecture": ["Lecturer", "Audience"],
    "presentation": ["Presenter", "Audience"],
    "casual_conversation": ["Speaker A", "Speaker B"],
}


def assign_speaker_roles(segments: list[dict], content_type: str) -> dict:
    """Use LLM to assign human-readable roles to speakers.
    Returns {"Speaker_0": "Seller", "Speaker_1": "Prospect"} etc."""
    roles = ROLE_MAP.get(content_type, ["Speaker A", "Speaker B"])
    speakers = sorted(set(s["speaker"] for s in segments))

    # Try LLM
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from shared.utils.llm_client import complete, is_configured
        if is_configured() and len(segments) >= 2:
            excerpt = "\n".join(
                f'{s["speaker"]}: {s["text"]}'
                for s in segments[:8]
            )
            system_prompt = (
                f"You are a conversation analyst. Given transcript excerpts from a "
                f"{content_type.replace('_', ' ')}, identify each speaker's role. "
                f"Likely roles: {', '.join(roles)}. "
                f"Return ONLY a JSON object mapping speaker IDs to roles, e.g. "
                f'{{"Speaker_0": "{roles[0]}", "Speaker_1": "{roles[1]}"}}'
            )
            raw = complete(system_prompt=system_prompt, user_prompt=excerpt, max_tokens=200, temperature=0.0)
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            parsed = json.loads(text)
            if isinstance(parsed, dict) and all(isinstance(v, str) for v in parsed.values()):
                return parsed
    except Exception as e:
        print(f"  {C_DIM}Role assignment LLM failed ({e}), using heuristic{C_RESET}")

    # Heuristic fallback
    if content_type == "sales_call" and len(speakers) >= 2:
        objection_counts = {}
        for seg in segments:
            low = seg["text"].lower()
            if any(kw in low for kw in ["not looking", "not interested", "concern", "expensive",
                                         "not sure", "issue", "worried", "no thank you", "thank you for the call"]):
                objection_counts[seg["speaker"]] = objection_counts.get(seg["speaker"], 0) + 1
        if objection_counts:
            prospect = max(objection_counts, key=objection_counts.get)
            return {spk: ("Prospect" if spk == prospect else "Seller") for spk in speakers}

    # Generic fallback
    return {spk: roles[i % len(roles)] for i, spk in enumerate(speakers)}


def analyze_call_outcome(
    segments: list[dict],
    language_result: dict,
    fusion_result: dict,
    content_type: str,
) -> dict:
    """Analyze call outcome for sales_call content type. Returns outcome dict or None."""
    if content_type != "sales_call":
        return None

    lang_summary = language_result.get("summary", {})
    per_speaker = lang_summary.get("per_speaker", {})
    alerts = fusion_result.get("alerts", [])
    objection_resolutions = lang_summary.get("objection_resolution", [])

    # Derive signal-based fields
    total_objections = sum(
        v.get("objection_count", v.get("total_objections", 0)) for v in per_speaker.values()
    )
    total_buying = sum(
        v.get("buying_signal_count", v.get("total_buying_signals", 0)) for v in per_speaker.values()
    )

    # Check if objections were resolved (buying signals followed objections)
    any_resolved = any(r["status"] == "handled_successfully" for r in objection_resolutions)

    outcome = {
        "objection_detected": total_objections > 0,
        "objection_count": total_objections,
        "buying_signals_count": total_buying,
        "alerts_count": len(alerts),
        "objection_resolved": any_resolved,
    }

    # Try LLM for subjective assessment
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from shared.utils.llm_client import complete, is_configured
        if is_configured():
            # Send FULL transcript for comprehensive analysis
            transcript_text = "\n".join(f'{s["speaker"]}: {s["text"]}' for s in segments)

            system_prompt = (
                "You are an expert sales call analyst. Analyze the full conversation to "
                "determine the call outcome. Pay close attention to:\n"
                "1. CONVERSATION ARC: How did the prospect's engagement change from start to end?\n"
                "2. OBJECTION HANDLING: Were initial objections overcome? Did the prospect warm up?\n"
                "3. NEXT STEPS: Were any follow-up actions agreed? (meetings, proposals, emails)\n"
                "4. INFORMATION SHARING: Did the prospect share contact info or internal details?\n"
                "5. BUYING SIGNALS: Look for specification questions, agreement, conditional interest.\n\n"
                "A call is POSITIVE if the prospect:\n"
                "- Agreed to a follow-up (call, email, meeting, proposal)\n"
                "- Shared contact information voluntarily\n"
                "- Asked detailed questions about the offering\n"
                "- Shifted from resistance to engagement\n\n"
                "A call is NEGATIVE only if the prospect:\n"
                "- Firmly declined with no opening for follow-up\n"
                "- Ended the call abruptly or with clear disinterest\n"
                "- Never engaged beyond polite responses\n\n"
                "Return ONLY a JSON object with these fields:\n"
                '"objection_handled": "yes" or "no" or "partially",\n'
                '"decision_readiness": "ready" or "not_ready" or "uncertain",\n'
                '"estimated_outcome": "positive" or "neutral" or "negative",\n'
                '"outcome_reasoning": "2-3 sentence explanation citing specific evidence from the transcript"'
            )

            # Build rich signal context
            signal_context = (
                f"Signal summary: {total_objections} objections, {total_buying} buying signals, "
                f"{len(alerts)} fusion alerts.\n"
            )
            if objection_resolutions:
                for r in objection_resolutions:
                    signal_context += (
                        f"Objection resolution ({r['speaker']}): {r['status']}"
                        f" — {r['detail']}\n"
                    )

            user_prompt = f"Transcript:\n{transcript_text}\n\n{signal_context}"

            raw = complete(system_prompt=system_prompt, user_prompt=user_prompt, max_tokens=400, temperature=0.0)
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            parsed = json.loads(text)
            outcome.update(parsed)
            return outcome
    except Exception as e:
        print(f"  {C_DIM}Call outcome LLM failed ({e}), using heuristic{C_RESET}")

    # Heuristic fallback (accounts for objection resolution)
    if any_resolved:
        outcome["objection_handled"] = "yes"
        outcome["estimated_outcome"] = "positive"
        outcome["decision_readiness"] = "uncertain"
        outcome["outcome_reasoning"] = (
            "Objections were raised but buying signals followed, "
            "indicating successful objection handling"
        )
    elif total_objections > 0 and total_buying > 0:
        outcome["objection_handled"] = "partially"
        outcome["estimated_outcome"] = "neutral"
        outcome["decision_readiness"] = "uncertain"
        outcome["outcome_reasoning"] = "Mixed signals: objections raised but buying interest present"
    elif total_buying > 0:
        outcome["objection_handled"] = "n/a"
        outcome["estimated_outcome"] = "positive"
        outcome["decision_readiness"] = "ready"
        outcome["outcome_reasoning"] = "Buying signals detected with no objections"
    elif total_objections > 0:
        outcome["objection_handled"] = "no"
        outcome["estimated_outcome"] = "negative"
        outcome["decision_readiness"] = "not_ready"
        outcome["outcome_reasoning"] = "Objections raised without clear resolution or buying signals"
    else:
        outcome["objection_handled"] = "n/a"
        outcome["estimated_outcome"] = "neutral"
        outcome["decision_readiness"] = "uncertain"
        outcome["outcome_reasoning"] = "No strong buying or objection signals detected"

    return outcome


def main():
    parser = argparse.ArgumentParser(
        description="NEXUS End-to-End Pipeline Test",
        epilog="""
Examples:
  python scripts/test_pipeline.py                                    # Synthetic sales call
  python scripts/test_pipeline.py --audio-file recording.mp3         # Any audio file
  python scripts/test_pipeline.py --audio-file lecture.mp4           # Video file (extracts audio)
  python scripts/test_pipeline.py --url "https://youtube.com/watch?v=XXX"  # YouTube URL
  python scripts/test_pipeline.py --audio-file podcast.mp3 --type podcast --speakers 3
  python scripts/test_pipeline.py --audio-file debate.wav --type debate --speakers 2
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--skip-audio", action="store_true", help="Reuse existing test audio")
    parser.add_argument("--voice-only", action="store_true", help="Stop after Voice Agent")
    parser.add_argument("--no-intent", action="store_true", help="Skip LLM intent classification")
    parser.add_argument("--audio-file", type=str, help="Use a specific audio/video file")
    parser.add_argument("--url", type=str, help="Download audio from URL (YouTube, podcast, direct)")
    parser.add_argument("--type", type=str, default=None,
                        choices=["sales_call", "podcast", "interview", "lecture",
                                 "debate", "meeting", "presentation",
                                 "casual_conversation", "other"],
                        help="Override auto-detected content type")
    parser.add_argument("--speakers", type=int, default=None,
                        help="Expected number of speakers (2-10, default: auto)")
    parser.add_argument("--use-external-tts", action="store_true",
                        help="Use GPU-accelerated Coqui TTS API for audio generation")
    parser.add_argument("--html", action="store_true",
                        help="Generate a self-contained HTML report (in addition to JSON)")
    args = parser.parse_args()

    # Validate speaker count
    if args.speakers is not None:
        if args.speakers < 1 or args.speakers > 10:
            print(f"  {C_RED}--speakers must be between 1 and 10{C_RESET}")
            sys.exit(1)

    print(f"\n{C_BOLD}{C_CYAN}NEXUS End-to-End Pipeline Test{C_RESET}")
    print(f"{C_DIM}{'─' * 40}{C_RESET}")
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

    # ── Health checks ──
    print(f"\n  {C_BOLD}Checking services...{C_RESET}")
    voice_ok = check_health("Voice Agent", VOICE_URL)
    lang_ok = check_health("Language Agent", LANGUAGE_URL)
    fusion_ok = check_health("Fusion Agent", FUSION_URL)

    if not voice_ok:
        print(f"\n  {C_RED}Voice Agent is required. Start it and try again.{C_RESET}")
        sys.exit(1)

    if not args.voice_only:
        if not lang_ok or not fusion_ok:
            print(f"\n  {C_YELLOW}Some agents are down. Running available stages only.{C_RESET}")

    # ── Step 0: Prepare audio ──
    audio_path = prepare_input_audio(args)

    # For external audio (non-synthetic), we won't have the CONVERSATION variable
    is_external_audio = bool(args.audio_file or args.url)

    # ── Step 1: Voice Agent ──
    voice_result = run_voice_agent(audio_path, num_speakers=args.speakers)

    if args.voice_only:
        print(f"\n  {C_GREEN}Voice-only mode complete.{C_RESET}")
        return

    # ── Extract transcript segments for Language Agent ──
    if is_external_audio:
        segments = extract_transcript_segments_from_voice(voice_result)
    else:
        segments = extract_transcript_segments(voice_result)
    print(f"\n  {C_DIM}Extracted {len(segments)} transcript segments for Language Agent{C_RESET}")

    # ── Auto-detect content type ──
    classification = classify_content(segments, args)
    content_type = classification["content_type"]

    # ── Step 2: Language Agent ──
    language_result = None
    if lang_ok:
        language_result = run_language_agent(
            segments,
            run_intent=not args.no_intent,
            content_type=content_type,
        )
    else:
        print(f"\n  {C_YELLOW}Skipping Language Agent (not running){C_RESET}")
        language_result = {"signals": [], "summary": {}, "segment_count": 0, "speakers": []}

    # ── Step 3: Fusion Agent ──
    fusion_result = None
    if fusion_ok and (voice_result["signals"] or language_result["signals"]):
        fusion_result = run_fusion_agent(
            voice_result, language_result, content_type=content_type,
        )
    else:
        print(f"\n  {C_YELLOW}Skipping Fusion Agent (not running or no signals){C_RESET}")
        fusion_result = {
            "fusion_signals": [], "unified_states": [], "alerts": [],
            "report": None, "speakers": [], "summary": {},
        }

    # ── Step 4a: Assign speaker roles ──
    print(f"\n  {C_CYAN}Assigning speaker roles...{C_RESET}")
    speaker_roles = assign_speaker_roles(segments, content_type)
    for sid, role in speaker_roles.items():
        print(f"    {C_BOLD}{sid}{C_RESET} → {C_CYAN}{role}{C_RESET}")

    # ── Step 4b: Analyze call outcome (sales_call only) ──
    call_outcome = analyze_call_outcome(segments, language_result, fusion_result, content_type)
    if call_outcome:
        est = call_outcome.get("estimated_outcome", "?")
        est_colour = {
            "positive": C_GREEN, "negative": C_RED,
        }.get(est, C_YELLOW)
        print(f"\n  {C_CYAN}Call outcome: {est_colour}{est.upper()}{C_RESET}")
        print(f"    {C_DIM}{call_outcome.get('outcome_reasoning', '')}{C_RESET}")

    # ── Step 5: Final Report ──
    print_final_report(voice_result, language_result, fusion_result, segments)

    # Save full results to JSON
    results_path = PROJECT_ROOT / "data" / "reports" / "test_pipeline_result.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    full_result = {
        "timestamp": datetime.now().isoformat(),
        "audio_file": str(audio_path),
        "content_type": content_type,
        "classification": classification,
        "voice": voice_result,
        "language": language_result,
        "fusion": fusion_result,
        "speaker_roles": speaker_roles,
        "call_outcome": call_outcome,
    }
    with open(results_path, "w") as f:
        json.dump(full_result, f, indent=2, default=str)
    print(f"  {C_DIM}Full results saved to: {results_path}{C_RESET}")

    # Generate HTML report if requested
    if args.html:
        try:
            sys.path.insert(0, str(PROJECT_ROOT))
            from shared.utils.report_generator import generate_html_report
            html_path = generate_html_report(
                full_result,
                output_dir=str(results_path.parent),
            )
            print(f"\n  {C_GREEN}{C_BOLD}HTML report generated:{C_RESET}")
            print(f"  {C_CYAN}{html_path}{C_RESET}")
            print(f"  {C_DIM}Open in browser: file://{html_path}{C_RESET}")
        except Exception as e:
            print(f"  {C_YELLOW}HTML report generation failed: {e}{C_RESET}")


if __name__ == "__main__":
    main()
