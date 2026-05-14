#!/usr/bin/env python3
"""
NEXUS Batch Pipeline Runner
Runs the full NEXUS pipeline on every audio/video file in a folder by uploading
each file to the monolith backend, polling until analysis completes, then saving
individual JSON + HTML reports and a summary CSV.

Usage:
  python3 scripts/batch_test.py data/recordings/
  python3 scripts/batch_test.py data/recordings/ --type sales_call --speakers 2
  python3 scripts/batch_test.py data/recordings/ --output-dir data/reports/batch_2024
  python3 scripts/batch_test.py data/recordings/ --no-html          # JSON only
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
BACKEND_URL  = os.getenv("BACKEND_URL", "http://localhost:8000")
NEXUS_TOKEN  = os.getenv("NEXUS_TOKEN", "")
NEXUS_EMAIL  = os.getenv("NEXUS_EMAIL", "admin@nexus.local")
NEXUS_PASS   = os.getenv("NEXUS_PASSWORD", "admin123")
POLL_INTERVAL = 15   # seconds between status polls
POLL_TIMEOUT  = 3600  # 1 hour max wait per file

# Supported media extensions
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".opus"}
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".wmv", ".flv"}
MEDIA_EXTS = AUDIO_EXTS | VIDEO_EXTS

# ── Terminal colours ──
C_RESET  = "\033[0m"
C_BOLD   = "\033[1m"
C_DIM    = "\033[2m"
C_RED    = "\033[91m"
C_GREEN  = "\033[92m"
C_YELLOW = "\033[93m"
C_BLUE   = "\033[94m"
C_CYAN   = "\033[96m"


# ══════════════════════════════════════════════════════════════
# Authentication
# ══════════════════════════════════════════════════════════════

class _AuthClient:
    """Manages JWT token acquisition against the monolith."""

    def __init__(self, base_url: str):
        self._base  = base_url.rstrip("/")
        self._token = NEXUS_TOKEN

    def _login(self) -> str:
        try:
            resp = httpx.post(
                f"{self._base}/auth/login",
                json={"email": NEXUS_EMAIL, "password": NEXUS_PASS},
                timeout=15,
            )
        except httpx.ConnectError:
            print(f"  {C_RED}Cannot connect to backend at {self._base}{C_RESET}")
            sys.exit(1)

        if resp.status_code != 200:
            print(f"  {C_RED}Login failed ({resp.status_code}): {resp.text[:200]}{C_RESET}")
            sys.exit(1)

        token = resp.json().get("access_token") or resp.json().get("token", "")
        if not token:
            print(f"  {C_RED}Login response missing token: {resp.text[:200]}{C_RESET}")
            sys.exit(1)
        return token

    def headers(self) -> dict:
        if not self._token:
            self._token = self._login()
        return {"Authorization": f"Bearer {self._token}"}

    def get(self, path: str, **kwargs) -> httpx.Response:
        return httpx.get(f"{self._base}{path}", headers=self.headers(), **kwargs)

    def post(self, path: str, **kwargs) -> httpx.Response:
        return httpx.post(f"{self._base}{path}", headers=self.headers(), **kwargs)


# ══════════════════════════════════════════════════════════════
# Health check
# ══════════════════════════════════════════════════════════════

def check_health(base_url: str) -> bool:
    try:
        with httpx.Client(timeout=5) as c:
            r  = c.get(f"{base_url}/health")
            ok = r.status_code == 200
            if ok:
                data = r.json()
                agents_ok = all(v == "ok" for v in data.get("agents", {}).values())
                status    = "UP" if agents_ok else "UP (some agents degraded)"
            else:
                status = f"DOWN ({r.status_code})"
    except Exception:
        ok     = False
        status = "DOWN (unreachable)"
    colour = C_GREEN if ok else C_RED
    print(f"    {colour}NEXUS Backend: {status}{C_RESET}")
    return ok


# ══════════════════════════════════════════════════════════════
# Content classification + speaker role helpers
# ══════════════════════════════════════════════════════════════

def classify_content(segments: list[dict], content_type_override: Optional[str] = None) -> dict:
    if content_type_override:
        return {"content_type": content_type_override, "confidence": 1.0, "method": "manual"}
    try:
        from shared.utils.content_classifier import classify_content_type_sync
        return classify_content_type_sync(segments)
    except Exception:
        return {"content_type": "sales_call", "confidence": 0.0, "method": "fallback"}


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
    """Use LLM → heuristic to assign human-readable roles to speakers."""
    roles    = ROLE_MAP.get(content_type, ["Speaker A", "Speaker B"])
    speakers = sorted(set(s["speaker"] for s in segments))
    try:
        from shared.utils.llm_client import complete, is_configured
        if is_configured() and len(segments) >= 2:
            excerpt = "\n".join(f'{s["speaker"]}: {s["text"]}' for s in segments[:8])
            system_prompt = (
                f"You are a conversation analyst. Given transcript excerpts from a "
                f"{content_type.replace('_', ' ')}, identify each speaker's role. "
                f"Likely roles: {', '.join(roles)}. "
                f'Return ONLY a JSON object mapping speaker IDs to roles, e.g. '
                f'{{"Speaker_0": "{roles[0]}", "Speaker_1": "{roles[1]}"}}'
            )
            raw  = complete(system_prompt=system_prompt, user_prompt=excerpt, max_tokens=200, temperature=0.0)
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            parsed = json.loads(text)
            if isinstance(parsed, dict) and all(isinstance(v, str) for v in parsed.values()):
                return parsed
    except Exception:
        pass

    if content_type == "sales_call" and len(speakers) >= 2:
        objection_counts: dict[str, int] = {}
        for seg in segments:
            low = seg["text"].lower()
            if any(kw in low for kw in ["not looking", "not interested", "concern", "expensive",
                                         "not sure", "issue", "worried", "no thank you"]):
                objection_counts[seg["speaker"]] = objection_counts.get(seg["speaker"], 0) + 1
        if objection_counts:
            prospect = max(objection_counts, key=objection_counts.get)
            return {spk: ("Prospect" if spk == prospect else "Seller") for spk in speakers}

    return {spk: roles[i % len(roles)] for i, spk in enumerate(speakers)}


def analyze_call_outcome(
    segments: list[dict],
    per_speaker_stats: dict,
    alerts: list,
    content_type: str,
) -> dict:
    """Analyze call outcome for sales_call content type."""
    if content_type != "sales_call":
        return None

    total_objections = sum(v.get("objection_count", 0) for v in per_speaker_stats.values())
    total_buying     = sum(v.get("buying_signal_count", 0) for v in per_speaker_stats.values())

    outcome = {
        "objection_detected":   total_objections > 0,
        "objection_count":      total_objections,
        "buying_signals_count": total_buying,
        "alerts_count":         len(alerts),
    }

    try:
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
            raw  = complete(
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
    except Exception:
        pass

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
                        "outcome_reasoning": "Objections raised without clear resolution"})
    else:
        outcome.update({"objection_handled": "n/a", "estimated_outcome": "neutral",
                        "decision_readiness": "uncertain",
                        "outcome_reasoning": "No strong buying or objection signals detected"})
    return outcome


# ══════════════════════════════════════════════════════════════
# SINGLE FILE PIPELINE (via monolith REST API)
# ══════════════════════════════════════════════════════════════

def _to_float(v) -> float:
    if v is None or v == "":
        return 0.0
    try:
        return float(v)
    except (ValueError, TypeError):
        return 0.0


def _compute_per_speaker_stats(voice_signals: list, language_signals: list) -> dict:
    """Aggregate signals into per-speaker summary stats."""
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
        sv   = entry["stress_values"]
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


def run_pipeline(
    auth: _AuthClient,
    audio_path: Path,
    num_speakers: Optional[int],
    content_type: Optional[str],
) -> dict:
    """
    Upload one file, poll until pipeline completes, retrieve and return the
    full result dict. Returns an error dict if anything fails.
    """
    result = {
        "audio_file": str(audio_path),
        "filename":   audio_path.name,
        "timestamp":  datetime.now().isoformat(),
        "status":     "success",
        "error":      None,
    }
    t0 = time.time()

    try:
        # ── 1. Upload ──
        config_dict: dict = {}
        if num_speakers:
            config_dict["num_speakers"] = num_speakers

        meeting_type = content_type or "sales_call"

        with open(audio_path, "rb") as fh:
            resp = auth.post(
                "/sessions",
                files={"file": (audio_path.name, fh, "application/octet-stream")},
                data={
                    "title":        audio_path.stem,
                    "meeting_type": meeting_type,
                    "config":       json.dumps(config_dict),
                },
                timeout=120,
            )

        if resp.status_code not in (200, 201):
            raise RuntimeError(f"Upload failed ({resp.status_code}): {resp.text[:200]}")

        session_id = resp.json()["session_id"]
        print(f"  {C_DIM}Session: {session_id}{C_RESET}")

        # ── 2. Poll ──
        deadline  = time.time() + POLL_TIMEOUT
        last_step = ""
        status    = "processing"

        while time.time() < deadline:
            time.sleep(POLL_INTERVAL)

            try:
                pr = auth.get(f"/sessions/{session_id}/progress", timeout=15)
            except httpx.RequestError:
                continue

            if pr.status_code == 200:
                progress = pr.json()
                status   = progress.get("status", "processing")
                step     = progress.get("current_step", "")
                if step and step != last_step:
                    elapsed = time.time() - t0
                    print(f"  {C_DIM}[{elapsed:5.0f}s] {step}{C_RESET}")
                    last_step = step
            else:
                sr = auth.get(f"/sessions/{session_id}", timeout=15)
                if sr.status_code == 200:
                    status = sr.json().get("session", {}).get("status", "processing")

            if status in ("completed", "failed"):
                break

        # ── 3. Retrieve ──
        session_detail = auth.get(f"/sessions/{session_id}", timeout=30).json()
        session_obj    = session_detail.get("session", {})

        signals_resp    = auth.get(f"/sessions/{session_id}/signals?limit=50000", timeout=30)
        all_signals     = signals_resp.json().get("signals", []) if signals_resp.status_code == 200 else []

        transcript_resp = auth.get(f"/sessions/{session_id}/transcript", timeout=30)
        transcript_segs = transcript_resp.json().get("segments", []) if transcript_resp.status_code == 200 else []

        report_resp = auth.get(f"/sessions/{session_id}/report", timeout=30)
        report_data = report_resp.json().get("report") if report_resp.status_code == 200 else None

        voice_signals    = [s for s in all_signals if s.get("agent") == "voice"]
        language_signals = [s for s in all_signals if s.get("agent") == "language"]
        fusion_signals   = [s for s in all_signals if s.get("agent") == "fusion"]
        alerts           = session_detail.get("alerts", [])

        per_speaker_stats = _compute_per_speaker_stats(voice_signals, language_signals)
        speakers_seen     = sorted(set(s.get("speaker_id", "unknown") for s in all_signals if s.get("speaker_id")))

        # ── 4. Build transcript segments ──
        segments: list[dict] = []
        for seg in transcript_segs:
            text = seg.get("text", "").strip()
            if text:
                segments.append({
                    "speaker":  seg.get("speaker", "Speaker_0"),
                    "start_ms": seg.get("start_ms", 0),
                    "end_ms":   seg.get("end_ms", 0),
                    "text":     text,
                })
        if not segments:
            for s in voice_signals:
                meta = s.get("metadata", {})
                if meta and "transcript_text" in meta:
                    segments.append({
                        "speaker":  s.get("speaker_id", "Speaker_0"),
                        "start_ms": s.get("window_start_ms", 0),
                        "end_ms":   s.get("window_end_ms", 0),
                        "text":     meta["transcript_text"],
                    })

        # ── 5. Classification + enrichment ──
        classification  = classify_content(segments, content_type)
        ct              = classification["content_type"]
        speaker_roles   = assign_speaker_roles(segments, ct)
        call_outcome    = analyze_call_outcome(segments, per_speaker_stats, alerts, ct)

        # ── 6. Build result dict ──
        result.update({
            "session_id":     session_id,
            "content_type":   ct,
            "classification": classification,
            "speaker_roles":  speaker_roles,
            # Flatten into shape expected by extract_csv_row + generate_html_report
            "voice": {
                "duration_seconds": session_obj.get("duration_seconds", 0) or 0,
                "speakers":         [{"speaker_id": s} for s in speakers_seen],
                "signals":          voice_signals,
                "summary": {
                    "per_speaker": per_speaker_stats,
                },
            },
            "language": {
                "signals":       language_signals,
                "summary":       {"per_speaker": per_speaker_stats},
                "segment_count": len(segments),
                "speakers":      speakers_seen,
            },
            "fusion": {
                "fusion_signals": fusion_signals,
                "unified_states": [],
                "alerts":         alerts,
                "report":         report_data,
                "speakers":       speakers_seen,
            },
        })
        if call_outcome:
            result["call_outcome"] = call_outcome

    except Exception as e:
        result["status"] = "error"
        result["error"]  = str(e)

    result["elapsed_seconds"] = round(time.time() - t0, 1)
    return result


# ══════════════════════════════════════════════════════════════
# CSV summary extraction
# ══════════════════════════════════════════════════════════════

def extract_csv_row(result: dict) -> dict:
    """Extract a one-line summary row for the CSV from a pipeline result."""
    voice    = result.get("voice", {})
    language = result.get("language", {})
    fusion   = result.get("fusion", {})
    per_speaker = voice.get("summary", {}).get("per_speaker", {})

    avg_stress_str = (
        " / ".join(f"{sid}:{v.get('avg_stress', 0):.3f}" for sid, v in per_speaker.items())
        if per_speaker else "n/a"
    )

    alerts        = fusion.get("alerts", [])
    fusion_signals = fusion.get("fusion_signals", [])

    if alerts:
        key_insight = alerts[0].get("title", "Alert detected")
    elif fusion_signals:
        fs          = fusion_signals[0]
        key_insight = f"{fs.get('signal_type', '?')}: {fs.get('value_text', '')}"
    else:
        key_insight = "No cross-modal alerts"

    return {
        "filename":             result.get("filename", ""),
        "status":               result.get("status", "error"),
        "session_id":           result.get("session_id", ""),
        "duration_s":           round(voice.get("duration_seconds", 0), 1),
        "speakers":             len(voice.get("speakers", [])),
        "content_type":         result.get("content_type", "unknown"),
        "voice_signals":        len(voice.get("signals", [])),
        "lang_signals":         len(language.get("signals", [])),
        "fusion_signals":       len(fusion_signals),
        "alerts_count":         len(alerts),
        "avg_stress_per_speaker": avg_stress_str,
        "key_insight":          key_insight,
        "elapsed_s":            result.get("elapsed_seconds", 0),
        "error":                result.get("error", ""),
    }


# ══════════════════════════════════════════════════════════════
# Main
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
    parser.add_argument("input_dir",    type=str, help="Directory containing audio/video files")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for reports (default: data/reports/batch_<timestamp>)")
    parser.add_argument("--type",       type=str, default=None,
                        choices=["sales_call", "podcast", "interview", "lecture",
                                 "debate", "meeting", "presentation",
                                 "casual_conversation", "other"],
                        help="Override content type for all files")
    parser.add_argument("--speakers",   type=int, default=None,
                        help="Number of speakers (applied to all files)")
    parser.add_argument("--ext",        nargs="*", default=None,
                        help="Filter by file extensions (e.g. --ext .wav .mp3)")
    parser.add_argument("--no-html",    action="store_true",
                        help="Skip HTML report generation (JSON only)")
    parser.add_argument("--token",      type=str, default=None,
                        help="JWT bearer token (overrides NEXUS_TOKEN env var)")
    parser.add_argument("--url",        type=str, default=None,
                        help="Backend URL (overrides BACKEND_URL env var)")
    args = parser.parse_args()

    # Validate inputs
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"  {C_RED}Error: {input_dir} is not a directory{C_RESET}")
        sys.exit(1)

    if args.speakers is not None and (args.speakers < 1 or args.speakers > 10):
        print(f"  {C_RED}--speakers must be between 1 and 10{C_RESET}")
        sys.exit(1)

    global BACKEND_URL, NEXUS_TOKEN
    if args.url:
        BACKEND_URL = args.url.rstrip("/")
    if args.token:
        NEXUS_TOKEN = args.token

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        batch_ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
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
    print(f"  Backend: {BACKEND_URL}")
    print(f"  Input:   {input_dir} ({len(files)} files)")
    print(f"  Output:  {output_dir}")
    if args.type:
        print(f"  Type:    {args.type} (manual)")
    if args.speakers:
        print(f"  Speakers: {args.speakers}")
    print()

    # Health check
    print(f"  {C_BOLD}Checking backend...{C_RESET}")
    if not check_health(BACKEND_URL):
        print(f"\n  {C_RED}Backend is required. Start with:{C_RESET}")
        print(f"  {C_DIM}  uvicorn backend.main:app --port 8000{C_RESET}")
        sys.exit(1)
    print()

    # Authenticate once for the whole batch
    auth = _AuthClient(BACKEND_URL)
    # Force a login check before starting the batch
    auth.headers()
    print()

    # Process each file
    csv_rows    = []
    total_start = time.time()

    for i, audio_file in enumerate(files, 1):
        print(f"{C_BOLD}{'═' * 50}{C_RESET}")
        print(f"{C_BOLD}  [{i}/{len(files)}] {audio_file.name}{C_RESET}")
        print(f"{C_BOLD}{'═' * 50}{C_RESET}")

        result = run_pipeline(
            auth=auth,
            audio_path=audio_file,
            num_speakers=args.speakers,
            content_type=args.type,
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
            voice  = result.get("voice", {})
            fusion = result.get("fusion", {})
            alerts = fusion.get("alerts", [])
            print(
                f"  {C_GREEN}OK{C_RESET} — "
                f"{voice.get('duration_seconds', 0):.0f}s, "
                f"{len(voice.get('speakers', []))} speakers, "
                f"{len(voice.get('signals', []))} voice signals, "
                f"{len(alerts)} alerts "
                f"({result['elapsed_seconds']:.1f}s)"
            )
        else:
            print(f"  {C_RED}FAILED{C_RESET}: {result.get('error', 'unknown')}")

        if html_path_str:
            print(f"  {C_DIM}JSON: {json_path}{C_RESET}")
            print(f"  {C_DIM}HTML: {html_path_str}{C_RESET}")
        else:
            print(f"  {C_DIM}JSON: {json_path}{C_RESET}")

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
    error_count   = sum(1 for r in csv_rows if r["status"] == "error")
    total_alerts  = sum(r["alerts_count"] for r in csv_rows)

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
