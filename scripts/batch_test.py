#!/usr/bin/env python3
"""
NEXUS Batch Pipeline Runner
Runs the full NEXUS pipeline on every audio/video file in a folder,
saves individual JSON + HTML reports, and generates a summary CSV.

Usage:
  python3 scripts/batch_test.py data/recordings/
  python3 scripts/batch_test.py data/recordings/ --type sales_call --speakers 2
  python3 scripts/batch_test.py data/recordings/ --output-dir data/reports/batch_2024
  python3 scripts/batch_test.py data/recordings/ --no-html          # JSON only
  python3 scripts/batch_test.py data/recordings/ --parallel 2       # 2 files at a time
"""
import os
import sys
import csv
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

import httpx

# ── Path setup ──
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.utils.report_generator import generate_html_report

# ── Config ──
VOICE_URL = os.getenv("VOICE_AGENT_URL", "http://localhost:8001")
LANGUAGE_URL = os.getenv("LANGUAGE_AGENT_URL", "http://localhost:8002")
FUSION_URL = os.getenv("FUSION_AGENT_URL", "http://localhost:8007")
TIMEOUT = 300  # 5 min per agent call

# Supported media extensions
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".opus"}
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".wmv", ".flv"}
MEDIA_EXTS = AUDIO_EXTS | VIDEO_EXTS

# ── Terminal colours ──
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_RED = "\033[91m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_BLUE = "\033[94m"
C_CYAN = "\033[96m"


# ══════════════════════════════════════════════════════════════
# SERVICE CALLS (self-contained, no dependency on test_pipeline)
# ══════════════════════════════════════════════════════════════

def check_health(name: str, url: str) -> bool:
    try:
        with httpx.Client(timeout=5) as c:
            r = c.get(f"{url}/health")
            ok = r.status_code == 200
            status = "UP" if ok else f"DOWN ({r.status_code})"
    except Exception:
        ok = False
        status = "DOWN (unreachable)"
    colour = C_GREEN if ok else C_RED
    print(f"    {colour}{name}: {status}{C_RESET}")
    return ok


def run_voice(audio_path: Path, num_speakers: Optional[int] = None) -> dict:
    payload = {"file_path": str(audio_path.resolve()), "meeting_type": "sales_call"}
    if num_speakers is not None:
        payload["num_speakers"] = num_speakers
    with httpx.Client(timeout=TIMEOUT) as c:
        resp = c.post(f"{VOICE_URL}/analyse", json=payload)
    if resp.status_code != 200:
        raise RuntimeError(f"Voice Agent {resp.status_code}: {resp.text[:200]}")
    return resp.json()


def extract_segments(voice_result: dict) -> list[dict]:
    """Extract transcript segments from voice result."""
    segs = voice_result.get("transcript_segments", [])
    if segs:
        out = []
        for s in segs:
            text = s.get("text", "").strip()
            if text:
                out.append({
                    "speaker": s.get("speaker", "Speaker_0"),
                    "start_ms": s.get("start_ms", 0),
                    "end_ms": s.get("end_ms", 0),
                    "text": text,
                })
        if out:
            return out

    # Fallback: signal metadata
    segments = []
    for sig in voice_result.get("signals", []):
        meta = sig.get("metadata", {})
        if meta and "transcript_text" in meta:
            segments.append({
                "speaker": sig.get("speaker_id", "Speaker_0"),
                "start_ms": sig.get("window_start_ms", 0),
                "end_ms": sig.get("window_end_ms", 0),
                "text": meta["transcript_text"],
            })
    seen = set()
    unique = []
    for s in segments:
        key = (s["speaker"], s["text"][:50])
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique or [{"speaker": "Speaker_0", "start_ms": 0, "end_ms": 0, "text": "[no transcript]"}]


def classify_content(segments: list[dict], content_type_override: Optional[str] = None) -> dict:
    if content_type_override:
        return {"content_type": content_type_override, "confidence": 1.0, "method": "manual"}
    try:
        from shared.utils.content_classifier import classify_content_type_sync
        return classify_content_type_sync(segments)
    except Exception:
        return {"content_type": "sales_call", "confidence": 0.0, "method": "fallback"}


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
    """Use LLM to assign human-readable roles to speakers."""
    roles = ROLE_MAP.get(content_type, ["Speaker A", "Speaker B"])
    speakers = sorted(set(s["speaker"] for s in segments))
    try:
        from shared.utils.llm_client import complete, is_configured
        if is_configured() and len(segments) >= 2:
            excerpt = "\n".join(f'{s["speaker"]}: {s["text"]}' for s in segments[:8])
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
    except Exception:
        pass
    # Heuristic fallback
    if content_type == "sales_call" and len(speakers) >= 2:
        objection_counts = {}
        for seg in segments:
            low = seg["text"].lower()
            if any(kw in low for kw in ["not looking", "not interested", "concern", "expensive",
                                         "not sure", "issue", "worried", "no thank you"]):
                objection_counts[seg["speaker"]] = objection_counts.get(seg["speaker"], 0) + 1
        if objection_counts:
            prospect = max(objection_counts, key=objection_counts.get)
            return {spk: ("Prospect" if spk == prospect else "Seller") for spk in speakers}
    return {spk: roles[i % len(roles)] for i, spk in enumerate(speakers)}


def analyze_call_outcome(segments: list[dict], language_result: dict, fusion_result: dict, content_type: str) -> dict:
    """Analyze call outcome for sales_call content type."""
    if content_type != "sales_call":
        return None
    lang_summary = language_result.get("summary", {})
    per_speaker = lang_summary.get("per_speaker", {})
    alerts = fusion_result.get("alerts", [])
    objection_resolutions = lang_summary.get("objection_resolution", [])
    total_objections = sum(v.get("objection_count", v.get("total_objections", 0)) for v in per_speaker.values())
    total_buying = sum(v.get("buying_signal_count", v.get("total_buying_signals", 0)) for v in per_speaker.values())
    any_resolved = any(r["status"] == "handled_successfully" for r in objection_resolutions)
    outcome = {
        "objection_detected": total_objections > 0,
        "objection_count": total_objections,
        "buying_signals_count": total_buying,
        "alerts_count": len(alerts),
        "objection_resolved": any_resolved,
    }
    try:
        from shared.utils.llm_client import complete, is_configured
        if is_configured():
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
    except Exception:
        pass
    # Heuristic fallback (accounts for objection resolution)
    if any_resolved:
        outcome.update({"objection_handled": "yes", "estimated_outcome": "positive",
                        "decision_readiness": "uncertain",
                        "outcome_reasoning": "Objections were raised but buying signals followed, indicating successful objection handling"})
    elif total_objections > 0 and total_buying > 0:
        outcome.update({"objection_handled": "partially", "estimated_outcome": "neutral",
                        "decision_readiness": "uncertain",
                        "outcome_reasoning": "Mixed signals: objections raised but buying interest present"})
    elif total_buying > 0:
        outcome.update({"objection_handled": "n/a", "estimated_outcome": "positive",
                        "decision_readiness": "ready",
                        "outcome_reasoning": "Buying signals detected with no objections"})
    elif total_objections > 0:
        outcome.update({"objection_handled": "no", "estimated_outcome": "negative",
                        "decision_readiness": "not_ready",
                        "outcome_reasoning": "Objections raised without clear resolution"})
    else:
        outcome.update({"objection_handled": "n/a", "estimated_outcome": "neutral",
                        "decision_readiness": "uncertain",
                        "outcome_reasoning": "No strong buying or objection signals detected"})
    return outcome


def run_language(segments: list[dict], content_type: str = "sales_call") -> dict:
    payload = {
        "segments": segments,
        "meeting_type": content_type,
        "content_type": content_type,
        "run_intent_classification": True,
    }
    with httpx.Client(timeout=TIMEOUT) as c:
        resp = c.post(f"{LANGUAGE_URL}/analyse", json=payload)
    if resp.status_code != 200:
        raise RuntimeError(f"Language Agent {resp.status_code}: {resp.text[:200]}")
    return resp.json()


def run_fusion(voice_result: dict, language_result: dict, content_type: str = "sales_call") -> dict:
    voice_signals = [{**s, "agent": "voice"} for s in voice_result.get("signals", [])]
    language_signals = [{**s, "agent": "language"} for s in language_result.get("signals", [])]
    payload = {
        "voice_signals": voice_signals,
        "language_signals": language_signals,
        "meeting_type": content_type,
        "content_type": content_type,
        "generate_report": True,
        "voice_summary": voice_result.get("summary"),
        "language_summary": language_result.get("summary"),
    }
    with httpx.Client(timeout=TIMEOUT) as c:
        resp = c.post(f"{FUSION_URL}/analyse", json=payload)
    if resp.status_code != 200:
        raise RuntimeError(f"Fusion Agent {resp.status_code}: {resp.text[:200]}")
    return resp.json()


# ══════════════════════════════════════════════════════════════
# SINGLE FILE PIPELINE
# ══════════════════════════════════════════════════════════════

def run_pipeline(
    audio_path: Path,
    num_speakers: Optional[int],
    content_type: Optional[str],
    lang_ok: bool,
    fusion_ok: bool,
) -> dict:
    """
    Run the full pipeline on a single file.
    Returns the combined result dict (or an error dict on failure).
    """
    result = {
        "audio_file": str(audio_path),
        "filename": audio_path.name,
        "timestamp": datetime.now().isoformat(),
        "status": "success",
        "error": None,
    }
    t0 = time.time()

    try:
        # Step 1: Voice
        voice_result = run_voice(audio_path, num_speakers=num_speakers)
        result["voice"] = voice_result

        # Extract segments
        segments = extract_segments(voice_result)

        # Classify content
        classification = classify_content(segments, content_type)
        ct = classification["content_type"]
        result["content_type"] = ct
        result["classification"] = classification

        # Step 2: Language
        if lang_ok:
            language_result = run_language(segments, content_type=ct)
        else:
            language_result = {"signals": [], "summary": {}, "segment_count": 0, "speakers": []}
        result["language"] = language_result

        # Step 3: Fusion
        if fusion_ok and (voice_result.get("signals") or language_result.get("signals")):
            fusion_result = run_fusion(voice_result, language_result, content_type=ct)
        else:
            fusion_result = {
                "fusion_signals": [], "unified_states": [], "alerts": [],
                "report": None, "speakers": [], "summary": {},
            }
        result["fusion"] = fusion_result

        # Step 4: Speaker roles + call outcome
        speaker_roles = assign_speaker_roles(segments, ct)
        result["speaker_roles"] = speaker_roles
        call_outcome = analyze_call_outcome(segments, language_result, fusion_result, ct)
        if call_outcome:
            result["call_outcome"] = call_outcome

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    result["elapsed_seconds"] = round(time.time() - t0, 1)
    return result


# ══════════════════════════════════════════════════════════════
# SUMMARY EXTRACTION
# ══════════════════════════════════════════════════════════════

def extract_csv_row(result: dict) -> dict:
    """Extract a summary row for the CSV from a pipeline result."""
    voice = result.get("voice", {})
    language = result.get("language", {})
    fusion = result.get("fusion", {})
    summary = voice.get("summary", {})
    per_speaker = summary.get("per_speaker", {})

    # Average stress across all speakers
    all_stress = [v.get("avg_stress", 0) for v in per_speaker.values()]
    avg_stress_str = " / ".join(
        f"{sid}:{v.get('avg_stress', 0):.3f}" for sid, v in per_speaker.items()
    ) if per_speaker else "n/a"

    # Key insight: pick first alert or top fusion signal
    alerts = fusion.get("alerts", [])
    fusion_signals = fusion.get("fusion_signals", [])
    if alerts:
        key_insight = alerts[0].get("title", "Alert detected")
    elif fusion_signals:
        fs = fusion_signals[0]
        key_insight = f"{fs.get('signal_type','?')}: {fs.get('value_text','')}"
    else:
        key_insight = "No cross-modal alerts"

    return {
        "filename": result.get("filename", ""),
        "status": result.get("status", "error"),
        "duration_s": round(voice.get("duration_seconds", 0), 1),
        "speakers": len(voice.get("speakers", [])),
        "content_type": result.get("content_type", "unknown"),
        "voice_signals": len(voice.get("signals", [])),
        "lang_signals": len(language.get("signals", [])),
        "fusion_signals": len(fusion_signals),
        "alerts_count": len(alerts),
        "avg_stress_per_speaker": avg_stress_str,
        "key_insight": key_insight,
        "elapsed_s": result.get("elapsed_seconds", 0),
        "error": result.get("error", ""),
    }


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="NEXUS Batch Pipeline Runner — process a folder of audio/video files",
        epilog="""
Examples:
  python scripts/batch_test.py data/recordings/
  python scripts/batch_test.py data/recordings/ --type sales_call --speakers 2
  python scripts/batch_test.py data/recordings/ --output-dir data/reports/batch_run
  python scripts/batch_test.py data/recordings/ --ext .wav .mp3
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input_dir", type=str, help="Directory containing audio/video files")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for reports (default: data/reports/batch_<timestamp>)")
    parser.add_argument("--type", type=str, default=None,
                        choices=["sales_call", "podcast", "interview", "lecture",
                                 "debate", "meeting", "presentation",
                                 "casual_conversation", "other"],
                        help="Override content type for all files")
    parser.add_argument("--speakers", type=int, default=None,
                        help="Number of speakers (applied to all files)")
    parser.add_argument("--ext", nargs="*", default=None,
                        help="Filter by file extensions (e.g. --ext .wav .mp3)")
    parser.add_argument("--no-html", action="store_true",
                        help="Skip HTML report generation (JSON only)")
    args = parser.parse_args()

    # Validate inputs
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"  {C_RED}Error: {input_dir} is not a directory{C_RESET}")
        sys.exit(1)

    if args.speakers is not None and (args.speakers < 1 or args.speakers > 10):
        print(f"  {C_RED}--speakers must be between 1 and 10{C_RESET}")
        sys.exit(1)

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        batch_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "data" / "reports" / f"batch_{batch_ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover media files
    allowed_exts = set(args.ext) if args.ext else MEDIA_EXTS
    files = sorted([
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in allowed_exts
    ])

    if not files:
        print(f"  {C_YELLOW}No media files found in {input_dir}{C_RESET}")
        print(f"  {C_DIM}Supported extensions: {', '.join(sorted(MEDIA_EXTS))}{C_RESET}")
        sys.exit(0)

    # Banner
    print(f"\n{C_BOLD}{C_CYAN}NEXUS Batch Pipeline Runner{C_RESET}")
    print(f"{C_DIM}{'─' * 50}{C_RESET}")
    print(f"  Input:   {input_dir} ({len(files)} files)")
    print(f"  Output:  {output_dir}")
    if args.type:
        print(f"  Type:    {args.type} (manual)")
    if args.speakers:
        print(f"  Speakers: {args.speakers}")
    print()

    # Health checks
    print(f"  {C_BOLD}Checking services...{C_RESET}")
    voice_ok = check_health("Voice Agent", VOICE_URL)
    lang_ok = check_health("Language Agent", LANGUAGE_URL)
    fusion_ok = check_health("Fusion Agent", FUSION_URL)

    if not voice_ok:
        print(f"\n  {C_RED}Voice Agent is required. Start it and try again.{C_RESET}")
        sys.exit(1)

    print()

    # Process each file
    csv_rows = []
    total_start = time.time()

    for i, audio_file in enumerate(files, 1):
        print(f"{C_BOLD}{'═' * 50}{C_RESET}")
        print(f"{C_BOLD}  [{i}/{len(files)}] {audio_file.name}{C_RESET}")
        print(f"{C_BOLD}{'═' * 50}{C_RESET}")

        result = run_pipeline(
            audio_path=audio_file,
            num_speakers=args.speakers,
            content_type=args.type,
            lang_ok=lang_ok,
            fusion_ok=fusion_ok,
        )

        # Save individual JSON
        json_name = f"{audio_file.stem}_result.json"
        json_path = output_dir / json_name
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2, default=str)

        # Save individual HTML
        html_path_str = None
        if not args.no_html and result["status"] == "success":
            try:
                html_path_str = generate_html_report(result, output_dir=str(output_dir))
            except Exception as e:
                print(f"  {C_YELLOW}HTML generation failed: {e}{C_RESET}")

        # Print summary for this file
        if result["status"] == "success":
            voice = result.get("voice", {})
            fusion = result.get("fusion", {})
            alerts = fusion.get("alerts", [])
            print(f"  {C_GREEN}OK{C_RESET} — "
                  f"{voice.get('duration_seconds', 0):.0f}s, "
                  f"{len(voice.get('speakers', []))} speakers, "
                  f"{len(voice.get('signals', []))} voice signals, "
                  f"{len(alerts)} alerts "
                  f"({result['elapsed_seconds']:.1f}s)")
        else:
            print(f"  {C_RED}FAILED{C_RESET}: {result.get('error', 'unknown')}")

        if html_path_str:
            print(f"  {C_DIM}JSON: {json_path}{C_RESET}")
            print(f"  {C_DIM}HTML: {html_path_str}{C_RESET}")
        else:
            print(f"  {C_DIM}JSON: {json_path}{C_RESET}")

        # Collect CSV row
        csv_rows.append(extract_csv_row(result))
        print()

    # Write summary CSV
    csv_path = output_dir / "batch_summary.csv"
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

    # Final summary
    total_elapsed = time.time() - total_start
    success_count = sum(1 for r in csv_rows if r["status"] == "success")
    error_count = sum(1 for r in csv_rows if r["status"] == "error")
    total_alerts = sum(r["alerts_count"] for r in csv_rows)

    print(f"\n{C_BOLD}{'═' * 50}{C_RESET}")
    print(f"{C_BOLD}{C_CYAN}  BATCH COMPLETE{C_RESET}")
    print(f"{C_BOLD}{'═' * 50}{C_RESET}")
    print(f"  Files processed: {len(files)}")
    print(f"  Succeeded:       {C_GREEN}{success_count}{C_RESET}")
    if error_count:
        print(f"  Failed:          {C_RED}{error_count}{C_RESET}")
    print(f"  Total alerts:    {total_alerts}")
    print(f"  Total time:      {total_elapsed:.1f}s")
    print(f"\n  {C_BOLD}Summary CSV:{C_RESET} {csv_path}")
    print(f"  {C_BOLD}Reports dir:{C_RESET} {output_dir}")
    print()


if __name__ == "__main__":
    main()
