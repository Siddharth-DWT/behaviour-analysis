"""
Re-run Fusion Agent on an existing session using signals already in PostgreSQL.
Used to test Fusion rule changes without re-processing audio.
"""
import asyncio
import sys
import os
import json
from collections import Counter
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx


API_URL = os.getenv("API_URL", "http://localhost:8000")
FUSION_URL = os.getenv("FUSION_URL", "http://localhost:8004")  # host port for fusion


async def rerun_fusion(session_id: str, token: str):
    async with httpx.AsyncClient(timeout=120) as client:
        headers = {"Authorization": f"Bearer {token}"}

        # 1. Get session info
        r = await client.get(f"{API_URL}/sessions/{session_id}", headers=headers)
        if r.status_code != 200:
            print(f"Failed to get session: {r.status_code} {r.text[:200]}")
            return
        session = r.json()
        meeting_type = session.get("meeting_type", "internal")
        print(f"Session: {session_id}")
        print(f"Title: {session.get('title')}, Type: {meeting_type}, Speakers: {session.get('speaker_count')}")

        # 2. Get existing signals
        r = await client.get(
            f"{API_URL}/sessions/{session_id}/signals",
            headers=headers,
            params={"limit": 5000},
        )
        body = r.json()
        all_signals = body if isinstance(body, list) else body.get("signals", [])

        voice_signals = [s for s in all_signals if s.get("agent") == "voice"]
        language_signals = [s for s in all_signals if s.get("agent") == "language"]
        conversation_signals = [s for s in all_signals if s.get("agent") == "conversation"]
        old_fusion = [s for s in all_signals if s.get("agent") == "fusion"]

        print(f"\nExisting signals: Voice={len(voice_signals)}, Language={len(language_signals)}, "
              f"Conversation={len(conversation_signals)}, Fusion={len(old_fusion)}")

        old_counts = Counter(s.get("signal_type") for s in old_fusion)
        print(f"\nBEFORE fusion signals:")
        for k, v in sorted(old_counts.items()):
            print(f"  {k:30s}: {v}")

        # 3. Get report for entities
        r = await client.get(f"{API_URL}/sessions/{session_id}/report", headers=headers)
        report = r.json() if r.status_code == 200 else {}
        report_content = report.get("report", {}).get("content", {})
        if isinstance(report_content, str):
            try:
                report_content = json.loads(report_content)
            except (json.JSONDecodeError, TypeError):
                report_content = {}
        entities = report_content.get("entities", {}) if isinstance(report_content, dict) else {}

        # 4. Get alerts count
        old_alerts = session.get("alerts", [])
        if isinstance(old_alerts, list):
            print(f"  {'alerts':30s}: {len(old_alerts)}")

        # 5. Build voice summary from existing data
        voice_summary = {}
        if conversation_signals:
            voice_summary["conversation"] = {"signals": conversation_signals}

        # 6. Convert signals to FusionSignalInput format (strip DB fields)
        def to_fusion_input(s):
            meta = s.get("metadata")
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except (json.JSONDecodeError, TypeError):
                    meta = {}
            return {
                "agent": s.get("agent", "voice"),
                "speaker_id": s.get("speaker_id") or s.get("speaker_label", "unknown"),
                "signal_type": s.get("signal_type", ""),
                "value": s.get("value"),
                "value_text": s.get("value_text", ""),
                "confidence": s.get("confidence", 0.5),
                "window_start_ms": s.get("window_start_ms", 0),
                "window_end_ms": s.get("window_end_ms", 0),
                "metadata": meta if isinstance(meta, dict) else {},
            }

        # 7. Call Fusion Agent directly
        fusion_request = {
            "voice_signals": [to_fusion_input(s) for s in voice_signals],
            "language_signals": [to_fusion_input(s) for s in language_signals],
            "session_id": session_id,
            "meeting_type": meeting_type,
            "generate_report": False,
            "voice_summary": voice_summary,
            "language_summary": {"entities": entities},
        }

        print(f"\nCalling Fusion Agent at {FUSION_URL}/analyse with meeting_type={meeting_type}...")
        r = await client.post(f"{FUSION_URL}/analyse", json=fusion_request)

        if r.status_code != 200:
            print(f"Fusion failed: {r.status_code}")
            print(r.text[:500])
            return

        result = r.json()
        new_fusion = result.get("fusion_signals", [])
        new_alerts = result.get("alerts", [])
        unified_states = result.get("unified_states", [])

        new_counts = Counter(s.get("signal_type") for s in new_fusion)
        print(f"\nAFTER fusion signals:")
        for k, v in sorted(new_counts.items()):
            print(f"  {k:30s}: {v}")
        print(f"  {'alerts':30s}: {len(new_alerts)}")

        if new_alerts:
            for a in new_alerts:
                print(f"    {a.get('severity', '?')}: {a.get('title', '?')}")

        # 8. Comparison
        print(f"\n{'='*60}")
        print("COMPARISON (BEFORE -> AFTER):")
        print(f"{'='*60}")
        all_types = sorted(set(list(old_counts.keys()) + list(new_counts.keys())))
        for t in all_types:
            old_v = old_counts.get(t, 0)
            new_v = new_counts.get(t, 0)
            delta = new_v - old_v
            if delta == 0:
                arrow = "  (unchanged)"
            elif delta < 0:
                arrow = f"  (v{abs(delta)})"
            else:
                arrow = f"  (^{delta})"
            print(f"  {t:30s}: {old_v:3d} -> {new_v:3d}{arrow}")

        old_alert_count = len(old_alerts) if isinstance(old_alerts, list) else 0
        print(f"  {'alerts':30s}: {old_alert_count:3d} -> {len(new_alerts):3d}")
        print(f"{'='*60}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/rerun_fusion.py <session_id> [jwt_token]")
        print("  session_id: full or partial UUID")
        print("  jwt_token: from /auth/login (or set via env TOKEN)")
        sys.exit(1)

    session_id = sys.argv[1]
    # Expand partial UUIDs
    if len(session_id) == 8:
        # Try to find full ID from partial
        pass

    token = (
        sys.argv[2] if len(sys.argv) > 2
        else os.getenv("TOKEN", "")
    )
    if not token:
        token = input("Enter JWT token: ").strip()

    asyncio.run(rerun_fusion(session_id, token))
