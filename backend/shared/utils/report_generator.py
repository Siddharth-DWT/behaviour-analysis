"""
NEXUS HTML Report Generator
Converts pipeline JSON output into a self-contained, dark-themed HTML report.

Colour system follows docs/UI.md exactly:
  Background:    #0F1117
  Surface:       #1A1D27
  Border:        #2D3348
  Text Primary:  #E8ECF4
  Text Secondary:#8B93A7
  Accent Blue:   #4F8BFF
  Accent Purple: #8B5CF6
  Stress High:   #EF4444
  Stress Med:    #F59E0B
  Stress Low:    #22C55E
  Alert Orange:  #F97316

Usage:
    from shared.utils.report_generator import generate_html_report
    html_path = generate_html_report(pipeline_json, output_dir="data/reports")
"""
import json
import html
import os
from pathlib import Path
from datetime import datetime
from typing import Optional


def generate_html_report(
    data: dict,
    output_dir: str = "data/reports",
    title: Optional[str] = None,
) -> str:
    """
    Generate a self-contained HTML report from pipeline JSON.

    Args:
        data: Full pipeline result dict (voice + language + fusion)
        output_dir: Directory to save the HTML file
        title: Optional title override

    Returns:
        Absolute path to the generated HTML file.
    """
    os.makedirs(output_dir, exist_ok=True)

    session_id = data.get("voice", {}).get("session_id", "unknown")[:8]
    ts = data.get("timestamp", datetime.now().isoformat())
    fname = f"{session_id}_report.html"
    out_path = Path(output_dir) / fname

    # ── Extract data ──
    voice = data.get("voice", {})
    language = data.get("language", {})
    fusion = data.get("fusion", {})
    duration = voice.get("duration_seconds", 0)
    speakers = voice.get("speakers", [])
    transcript_segments = voice.get("transcript_segments", [])
    voice_signals = voice.get("signals", [])
    lang_signals = language.get("signals", [])
    fusion_signals = fusion.get("fusion_signals", [])
    alerts = fusion.get("alerts", [])
    unified_states = fusion.get("unified_states", [])
    report = fusion.get("report", {})
    summary = voice.get("summary", {})
    lang_summary = language.get("summary", {})
    content_type = data.get("content_type", "unknown")
    audio_file = data.get("audio_file", "")

    speaker_roles = data.get("speaker_roles", {})
    auto_title = title or _infer_title(content_type, audio_file)

    # ── Build HTML ──
    parts = [
        _head(auto_title),
        _header(auto_title, ts, duration, len(speakers), content_type, len(alerts)),
        _executive_summary(report),
        _call_outcome_section(data),
        _speaker_cards(speakers, summary, lang_summary, unified_states, voice_signals, lang_signals, duration, speaker_roles=speaker_roles, alerts=alerts),
        _transcript_section(transcript_segments, voice_signals, lang_signals, speaker_roles=speaker_roles),
        _stress_timeline(voice_signals, duration, speaker_roles=speaker_roles),
        _alerts_section(alerts, fusion_signals, speaker_roles=speaker_roles),
        _key_moments_and_recommendations(report),
        _footer(ts, session_id),
    ]

    html_content = "\n".join(parts)
    out_path.write_text(html_content, encoding="utf-8")
    return str(out_path.resolve())


# ══════════════════════════════════════════════════════════════
# HTML SECTIONS
# ══════════════════════════════════════════════════════════════

def _head(title: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NEXUS — {_esc(title)}</title>
<style>
{_css()}
</style>
</head>
<body>
<div class="container">"""


def _css() -> str:
    return """
:root {
  --bg: #0F1117;
  --surface: #1A1D27;
  --surface-hover: #242836;
  --border: #2D3348;
  --text: #E8ECF4;
  --text-sec: #8B93A7;
  --text-muted: #565E73;
  --blue: #4F8BFF;
  --purple: #8B5CF6;
  --red: #EF4444;
  --amber: #F59E0B;
  --green: #22C55E;
  --orange: #F97316;
  --cyan: #06B6D4;
  --emerald: #10B981;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
  background: var(--bg); color: var(--text); line-height: 1.6;
}
.container { max-width: 960px; margin: 0 auto; padding: 24px 20px; }

/* Header */
.report-header {
  background: linear-gradient(135deg, #1A1D27 0%, #242836 100%);
  border: 1px solid var(--border); border-radius: 12px;
  padding: 32px; margin-bottom: 24px; position: relative; overflow: hidden;
}
.report-header::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
  background: linear-gradient(90deg, var(--blue), var(--purple), var(--orange));
}
.report-header h1 { font-size: 24px; font-weight: 700; margin-bottom: 4px; }
.report-header .nexus-badge {
  display: inline-block; background: var(--blue); color: #fff;
  font-size: 10px; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase;
  padding: 2px 8px; border-radius: 4px; margin-bottom: 12px;
}
.meta-row { display: flex; gap: 20px; flex-wrap: wrap; margin-top: 12px; }
.meta-item { font-size: 13px; color: var(--text-sec); }
.meta-item strong { color: var(--text); }

/* Cards */
.card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 24px; margin-bottom: 20px;
}
.card h2 {
  font-size: 15px; text-transform: uppercase; letter-spacing: 1px;
  color: var(--text-sec); margin-bottom: 16px; font-weight: 600;
}
.card h3 { font-size: 16px; font-weight: 600; margin-bottom: 12px; }

/* Speaker cards grid */
.speakers-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 16px; }
.speaker-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 20px;
}
.speaker-card .speaker-name {
  font-size: 16px; font-weight: 600; margin-bottom: 14px;
  display: flex; align-items: center; gap: 8px;
}
.speaker-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }

/* Gauge bars */
.gauge { margin-bottom: 10px; }
.gauge-label {
  display: flex; justify-content: space-between; font-size: 12px;
  color: var(--text-sec); margin-bottom: 4px;
}
.gauge-bar {
  height: 8px; border-radius: 4px; background: var(--surface-hover); overflow: hidden;
}
.gauge-fill { height: 100%; border-radius: 4px; transition: width 0.3s; }

/* Stats row */
.stats-row { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 12px; }
.stat-chip {
  background: var(--surface-hover); border-radius: 6px;
  padding: 4px 10px; font-size: 12px; color: var(--text-sec);
}
.stat-chip strong { color: var(--text); }

/* Transcript */
.transcript-seg {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 14px 16px; margin-bottom: 8px;
}
.transcript-seg .ts-header {
  display: flex; align-items: center; gap: 8px;
  font-size: 12px; margin-bottom: 6px;
}
.ts-time { color: var(--text-muted); font-family: monospace; font-size: 11px; }
.ts-speaker { font-weight: 600; font-size: 13px; }
.ts-text { font-size: 14px; line-height: 1.7; color: var(--text); margin-bottom: 8px; }
.badges { display: flex; gap: 6px; flex-wrap: wrap; }
.badge {
  display: inline-flex; align-items: center; font-size: 11px;
  padding: 2px 8px; border-radius: 10px; font-weight: 500;
}
.badge-stress    { background: rgba(239,68,68,0.15); color: var(--red); }
.badge-stress-lo { background: rgba(34,197,94,0.15); color: var(--green); }
.badge-filler    { background: rgba(245,158,11,0.15); color: var(--amber); }
.badge-sentiment-pos { background: rgba(34,197,94,0.15); color: var(--green); }
.badge-sentiment-neg { background: rgba(239,68,68,0.15); color: var(--red); }
.badge-objection { background: rgba(249,115,22,0.15); color: var(--orange); }
.badge-buying    { background: rgba(16,185,129,0.15); color: var(--emerald); }
.badge-intent    { background: rgba(79,139,255,0.15); color: var(--blue); }
.badge-power     { background: rgba(139,92,246,0.15); color: var(--purple); }

/* Stress timeline */
.timeline-chart { position: relative; margin-top: 12px; }
.timeline-bar-group { display: flex; align-items: flex-end; gap: 1px; height: 100px; }
.timeline-bar {
  flex: 1; min-width: 2px; border-radius: 2px 2px 0 0; transition: height 0.2s;
}
.timeline-labels {
  display: flex; justify-content: space-between;
  font-size: 10px; color: var(--text-muted); margin-top: 4px; font-family: monospace;
}
.timeline-legend { display: flex; gap: 16px; margin-top: 8px; font-size: 11px; color: var(--text-sec); }
.legend-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 4px; vertical-align: middle; }

/* Alerts */
.alert-item {
  background: var(--surface); border-left: 3px solid var(--orange);
  border-radius: 0 8px 8px 0; padding: 14px 16px; margin-bottom: 10px;
}
.alert-item.alert-critical { border-left-color: var(--red); }
.alert-item .alert-title { font-weight: 600; font-size: 14px; margin-bottom: 4px; }
.alert-item .alert-desc { font-size: 13px; color: var(--text-sec); }

/* Narrative */
.narrative { font-size: 15px; line-height: 1.8; color: var(--text); }
.narrative p { margin-bottom: 12px; }
.rec-list { list-style: none; }
.rec-list li {
  padding: 10px 14px; background: var(--surface-hover);
  border-radius: 8px; margin-bottom: 8px; font-size: 14px;
}
.rec-list li::before { content: '→ '; color: var(--blue); font-weight: 700; }

/* Footer */
.report-footer {
  text-align: center; font-size: 11px; color: var(--text-muted);
  padding: 20px 0; border-top: 1px solid var(--border); margin-top: 32px;
}

/* Section divider */
.section-divider { margin: 28px 0; border: none; border-top: 1px solid var(--border); }

@media (max-width: 600px) {
  .speakers-grid { grid-template-columns: 1fr; }
  .meta-row { flex-direction: column; gap: 6px; }
}
"""


def _header(title: str, ts: str, duration: float, num_speakers: int, content_type: str, num_alerts: int) -> str:
    dt = _parse_ts(ts)
    dur_str = _fmt_duration(duration)
    ct_label = content_type.replace("_", " ").title()
    return f"""
<div class="report-header">
  <span class="nexus-badge">NEXUS Analysis</span>
  <h1>{_esc(title)}</h1>
  <div class="meta-row">
    <span class="meta-item"><strong>{dt}</strong></span>
    <span class="meta-item">Duration: <strong>{dur_str}</strong></span>
    <span class="meta-item">Speakers: <strong>{num_speakers}</strong></span>
    <span class="meta-item">Type: <strong>{ct_label}</strong></span>
    <span class="meta-item">Alerts: <strong style="color: {'var(--orange)' if num_alerts else 'var(--green)'}">{num_alerts}</strong></span>
  </div>
</div>"""


def _executive_summary(report: dict) -> str:
    summary_text = report.get("executive_summary", "No executive summary available.")
    return f"""
<div class="card">
  <h2>Executive Summary</h2>
  <div class="narrative"><p>{_esc(summary_text)}</p></div>
</div>"""


def _call_outcome_section(data: dict) -> str:
    """Render Call Outcome card (sales_call only)."""
    outcome = data.get("call_outcome")
    if not outcome:
        return ""

    # Outcome badge colour
    est = outcome.get("estimated_outcome", "neutral")
    outcome_colour = {"positive": "var(--green)", "negative": "var(--red)"}.get(est, "var(--amber)")
    outcome_label = est.title()

    # Decision readiness
    readiness = outcome.get("decision_readiness", "uncertain")
    readiness_colour = {"ready": "var(--green)", "not_ready": "var(--red)"}.get(readiness, "var(--amber)")
    readiness_label = readiness.replace("_", " ").title()

    # Objection handling
    obj_handled = outcome.get("objection_handled", "n/a")
    obj_colour = {"yes": "var(--green)", "no": "var(--red)", "partially": "var(--amber)"}.get(obj_handled, "var(--text-sec)")

    reasoning = outcome.get("outcome_reasoning", "")
    obj_detected = outcome.get("objection_detected", False)
    obj_resolved = outcome.get("objection_resolved", False)
    buying_count = outcome.get("buying_signals_count", 0)
    alert_count = outcome.get("alerts_count", 0)

    return f"""
<div class="card" style="border-left:3px solid {outcome_colour}">
  <h2>Call Outcome</h2>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin-bottom:16px">
    <div style="text-align:center;padding:14px;background:var(--surface-hover);border-radius:8px">
      <div style="font-size:12px;color:var(--text-sec);margin-bottom:6px">Estimated Outcome</div>
      <div style="font-size:20px;font-weight:700;color:{outcome_colour}">{_esc(outcome_label)}</div>
    </div>
    <div style="text-align:center;padding:14px;background:var(--surface-hover);border-radius:8px">
      <div style="font-size:12px;color:var(--text-sec);margin-bottom:6px">Decision Readiness</div>
      <div style="font-size:20px;font-weight:700;color:{readiness_colour}">{_esc(readiness_label)}</div>
    </div>
    <div style="text-align:center;padding:14px;background:var(--surface-hover);border-radius:8px">
      <div style="font-size:12px;color:var(--text-sec);margin-bottom:6px">Objection Handled</div>
      <div style="font-size:20px;font-weight:700;color:{obj_colour}">{_esc(obj_handled.title())}</div>
    </div>
  </div>
  <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:12px">
    <span class="stat-chip" style="{'color:var(--orange)' if obj_detected else ''}">Objections: <strong>{'Yes' if obj_detected else 'No'}</strong></span>
    <span class="stat-chip" style="{'color:var(--emerald)' if obj_resolved else ''}">Objection Resolved: <strong>{'Yes' if obj_resolved else 'No'}</strong></span>
    <span class="stat-chip" style="{'color:var(--emerald)' if buying_count else ''}">Buying Signals: <strong>{buying_count}</strong></span>
    <span class="stat-chip">Fusion Alerts: <strong>{alert_count}</strong></span>
  </div>
  {f'<div style="font-size:14px;color:var(--text-sec);font-style:italic">{_esc(reasoning)}</div>' if reasoning else ''}
</div>"""


def _speaker_cards(
    speakers: list, summary: dict, lang_summary: dict,
    unified_states: list, voice_signals: list, lang_signals: list,
    duration: float, speaker_roles: dict = None, alerts: list = None,
) -> str:
    per_speaker = summary.get("per_speaker", {})
    lang_per_spk = lang_summary.get("per_speaker", {}) if isinstance(lang_summary, dict) else {}
    alerts_list = alerts or []

    # Build unified state lookup
    state_map = {}
    for us in unified_states:
        sid = us.get("speaker_id", "")
        state_map[sid] = us

    colours = ["var(--blue)", "var(--purple)", "var(--orange)", "var(--cyan)", "var(--emerald)"]

    # Compute interestingness score for sorting (most interesting first)
    def _interest_score(spk_item, idx):
        sid = spk_item.get("speaker_id", f"Speaker_{idx}")
        vs = per_speaker.get(sid, {})
        ls = lang_per_spk.get(sid, {})
        us = state_map.get(sid, {})
        stress_val = us.get("stress", vs.get("avg_stress", 0))
        stress_pct = min(100, int(stress_val * 100))
        sentiment_val = ls.get("avg_sentiment", us.get("sentiment", 0.5))
        sent_pct = min(100, max(0, int((sentiment_val + 1) / 2 * 100))) if sentiment_val < 1 else min(100, int(sentiment_val * 100))
        power_val = ls.get("avg_power_score", 0.5)
        power_pct = min(100, int(power_val * 100))
        buying = ls.get("total_buying_signals", 0)
        objections = ls.get("total_objections", 0)
        spk_alerts = sum(1 for a in alerts_list if a.get("speaker_id") == sid)
        return (
            stress_pct * 2
            + spk_alerts * 30
            + objections * 20
            + buying * 10
            + abs(50 - sent_pct)
            + (100 - power_pct)
        )

    indexed_speakers = [(i, spk) for i, spk in enumerate(speakers)]
    indexed_speakers.sort(key=lambda x: _interest_score(x[1], x[0]), reverse=True)

    cards_html = []
    for i, spk in indexed_speakers:
        sid = spk.get("speaker_id", f"Speaker_{i}")
        colour = colours[i % len(colours)]
        vs = per_speaker.get(sid, {})
        ls = lang_per_spk.get(sid, {})
        us = state_map.get(sid, {})

        stress_val = us.get("stress", vs.get("avg_stress", 0))
        sentiment_val = ls.get("avg_sentiment", us.get("sentiment", 0.5))
        power_val = ls.get("avg_power_score", 0.5)
        confidence_val = us.get("confidence", 0.5)
        fillers = vs.get("total_fillers", 0)
        filler_rate = _calc_filler_rate(sid, voice_signals, duration)
        buying = ls.get("total_buying_signals", 0)
        objections = ls.get("total_objections", 0)
        pitch = vs.get("baseline_f0_hz", spk.get("baseline", {}).get("f0_mean", 0))

        stress_pct = min(100, int(stress_val * 100))
        sent_pct = min(100, max(0, int((sentiment_val + 1) / 2 * 100))) if sentiment_val < 1 else min(100, int(sentiment_val * 100))
        power_pct = min(100, int(power_val * 100))
        conf_pct = min(100, int(confidence_val * 100))

        label = _speaker_label(sid, speaker_roles)

        cards_html.append(f"""
<div class="speaker-card">
  <div class="speaker-name"><span class="speaker-dot" style="background:{colour}"></span>{label}</div>
  {_gauge("Stress", stress_pct, _stress_colour(stress_pct))}
  {_gauge("Sentiment", sent_pct, "var(--green)" if sent_pct > 55 else ("var(--red)" if sent_pct < 45 else "var(--text-muted)"))}
  {_gauge("Power", power_pct, "var(--purple)")}
  {_gauge("Confidence", conf_pct, "var(--blue)")}
  <div class="stats-row">
    <span class="stat-chip">Fillers: <strong>{fillers}</strong> ({filler_rate:.1f}%)</span>
    <span class="stat-chip">Pitch: <strong>{pitch:.0f} Hz</strong></span>
    <span class="stat-chip" style="{'color:var(--emerald)' if buying else ''}">Buying: <strong>{buying}</strong></span>
    <span class="stat-chip" style="{'color:var(--orange)' if objections else ''}">Objections: <strong>{objections}</strong></span>
  </div>
</div>""")

    return f"""
<div class="card">
  <h2>Speaker Analysis</h2>
  <div class="speakers-grid">{"".join(cards_html)}</div>
</div>"""


def _gauge(label: str, pct: int, colour: str) -> str:
    return f"""<div class="gauge">
  <div class="gauge-label"><span>{label}</span><span>{pct}%</span></div>
  <div class="gauge-bar"><div class="gauge-fill" style="width:{pct}%;background:{colour}"></div></div>
</div>"""


def _transcript_section(segments: list, voice_signals: list, lang_signals: list, speaker_roles: dict = None) -> str:
    if not segments:
        return ""

    # Index lang signals by start_ms for quick lookup
    lang_by_start = {}
    for sig in lang_signals:
        key = sig.get("window_start_ms", sig.get("start_ms", 0))
        lang_by_start.setdefault(key, []).append(sig)

    colours = {"Speaker_0": "var(--blue)", "Speaker_1": "var(--purple)",
               "Speaker_2": "var(--orange)", "Speaker_3": "var(--cyan)"}

    rows = []
    for seg in segments:
        t_sec = seg.get("start_ms", 0) / 1000
        speaker = seg.get("speaker", "?")
        text = seg.get("text", "")
        colour = colours.get(speaker, "var(--text-sec)")

        # Gather badges for this segment
        badges_html = _segment_badges(seg, voice_signals, lang_signals)
        label = _speaker_label(speaker, speaker_roles) if speaker_roles else _esc(speaker)

        rows.append(f"""
<div class="transcript-seg">
  <div class="ts-header">
    <span class="ts-time">{_fmt_time(t_sec)}</span>
    <span class="ts-speaker" style="color:{colour}">{label}</span>
  </div>
  <div class="ts-text">{_esc(text)}</div>
  <div class="badges">{badges_html}</div>
</div>""")

    return f"""
<div class="card">
  <h2 style="display:flex;justify-content:space-between;align-items:center">
    Transcript
    <button id="badge-toggle" onclick="toggleBadges()" style="
      font-size:12px;padding:4px 12px;border-radius:6px;
      background:var(--surface-hover);color:var(--text-sec);
      border:1px solid var(--border);cursor:pointer;
    ">Show all signals</button>
  </h2>
  {"".join(rows)}
</div>"""


def _segment_badges(seg: dict, voice_signals: list, lang_signals: list) -> str:
    """Build inline signal badges for a transcript segment, sorted by priority.
    Deduplicates by signal type — keeps only the highest-value badge per type."""
    start = seg.get("start_ms", 0)
    end = seg.get("end_ms", 0)
    speaker = seg.get("speaker", "")

    # Collect badges as {base_type: (priority, value, html)} — dedup by base_type
    best: dict[str, tuple] = {}

    def _add(base_type: str, priority: int, val: float, html: str):
        if base_type not in best or val > best[base_type][1]:
            best[base_type] = (priority, val, html)

    # Voice signals in this segment's time range
    for vs in voice_signals:
        if vs.get("speaker_id") != speaker:
            continue
        ws = vs.get("window_start_ms", 0)
        we = vs.get("window_end_ms", 0)
        if ws > end or we < start:
            continue

        st = vs.get("signal_type", "")
        val = vs.get("value", 0)

        if st == "vocal_stress_score" and val > 0.3:
            cls = "badge-stress" if val > 0.5 else "badge-stress-lo"
            _add("stress", 1 if val > 0.5 else 3, val,
                 f'<span class="badge {cls}">stress {val:.0%}</span>')
        elif st == "filler_detection" and vs.get("value_text") == "elevated":
            cnt = vs.get("metadata", {}).get("filler_count", 0)
            if cnt > 0:
                _add("filler", 6, cnt,
                     f'<span class="badge badge-filler">fillers x{cnt}</span>')
        elif st == "pitch_analysis":
            vt = (vs.get("value_text") or "").lower()
            if "elevated" in vt or "high" in vt:
                _add("pitch", 5, val,
                     '<span class="badge badge-filler">pitch ↑</span>')
        elif st == "speech_rate_analysis":
            vt = (vs.get("value_text") or "").lower()
            if "fast" in vt or "rapid" in vt:
                _add("rate", 6, val,
                     '<span class="badge badge-intent">fast</span>')
            elif "slow" in vt:
                _add("rate", 6, val,
                     '<span class="badge badge-intent">slow</span>')

    # Language signals
    for ls in lang_signals:
        ws = ls.get("window_start_ms", ls.get("start_ms", 0))
        if ws < start or ws > end:
            continue
        if ls.get("speaker_id", "") != speaker:
            continue

        st = ls.get("signal_type", "")
        if st == "sentiment_score":
            val = ls.get("value", 0)
            if val > 0.3:
                _add("sentiment", 5, val,
                     f'<span class="badge badge-sentiment-pos">positive</span>')
            elif val < -0.3:
                _add("sentiment", 4, abs(val),
                     f'<span class="badge badge-sentiment-neg">negative</span>')
        elif st == "objection_signal" and ls.get("value", 0) > 0.3:
            cats = ls.get("metadata", {}).get("categories", [])
            label = cats[0] if cats else "objection"
            _add("objection", 2, ls.get("value", 0),
                 f'<span class="badge badge-objection">{_esc(label)}</span>')
        elif st == "buying_signal" and ls.get("value", 0) > 0.3:
            _add("buying", 2, ls.get("value", 0),
                 '<span class="badge badge-buying">buying signal</span>')
        elif st == "intent_classification":
            intent = ls.get("value_text", "")
            if intent and intent.upper() not in ("INFORM", "QUESTION"):
                _add("intent", 7, ls.get("value", 0),
                     f'<span class="badge badge-intent">{_esc(intent)}</span>')
        elif st == "power_language_score":
            val = ls.get("value", 0.5)
            if val < 0.3:
                _add("power", 7, 1 - val,
                     '<span class="badge badge-power">weak language</span>')
            elif val > 0.8:
                _add("power", 7, val,
                     '<span class="badge badge-buying">strong language</span>')

    # Sort by priority (most important first), limit to 4
    sorted_badges = sorted(best.values(), key=lambda x: x[0])
    final_badges = []
    for idx, (_, _, badge_html) in enumerate(sorted_badges[:4]):
        priority = "high" if idx < 2 else "low"
        tagged = badge_html.replace('class="badge ', f'data-priority="{priority}" class="badge ')
        final_badges.append(tagged)

    return "".join(final_badges)


def _stress_timeline(voice_signals: list, duration: float, speaker_roles: dict = None) -> str:
    """Build a CSS bar chart of stress over time, per speaker."""
    if duration <= 0:
        return ""

    # Collect stress signals per speaker
    speaker_stress = {}
    for vs in voice_signals:
        if vs.get("signal_type") != "vocal_stress_score":
            continue
        sid = vs.get("speaker_id", "?")
        t = vs.get("window_start_ms", 0) / 1000
        val = vs.get("value", 0)
        speaker_stress.setdefault(sid, []).append((t, val))

    if not speaker_stress:
        return ""

    # Build bars at 2.5s intervals
    n_bins = max(1, int(duration / 2.5))
    colours_map = {"Speaker_0": "var(--blue)", "Speaker_1": "var(--purple)",
                   "Speaker_2": "var(--orange)", "Speaker_3": "var(--cyan)"}

    tracks_html = []
    for sid in sorted(speaker_stress):
        points = speaker_stress[sid]
        bins = [0.0] * n_bins
        counts = [0] * n_bins
        for t, val in points:
            idx = min(int(t / 2.5), n_bins - 1)
            bins[idx] += val
            counts[idx] += 1
        for i in range(n_bins):
            if counts[i] > 0:
                bins[i] /= counts[i]

        colour = colours_map.get(sid, "var(--text-sec)")
        bars = []
        for val in bins:
            h = max(2, int(val * 100))
            c = _stress_colour_hex(val)
            bars.append(f'<div class="timeline-bar" style="height:{h}px;background:{c}"></div>')

        track_label = _speaker_label(sid, speaker_roles) if speaker_roles else _esc(sid)
        tracks_html.append(f"""
<div style="margin-bottom:12px">
  <div style="font-size:12px;color:{colour};margin-bottom:4px;font-weight:600">{track_label}</div>
  <div class="timeline-bar-group">{"".join(bars)}</div>
</div>""")

    # Time labels
    label_count = min(6, n_bins)
    step = duration / label_count
    labels = [_fmt_time(i * step) for i in range(label_count + 1)]
    labels_html = "".join(f"<span>{l}</span>" for l in labels)

    return f"""
<div class="card">
  <h2>Stress Timeline</h2>
  <div class="timeline-chart">
    {"".join(tracks_html)}
    <div class="timeline-labels">{labels_html}</div>
  </div>
  <div class="timeline-legend">
    <span><span class="legend-dot" style="background:var(--green)"></span>Low</span>
    <span><span class="legend-dot" style="background:var(--amber)"></span>Medium</span>
    <span><span class="legend-dot" style="background:var(--red)"></span>High</span>
  </div>
</div>"""


def _alerts_section(alerts: list, fusion_signals: list, speaker_roles: dict = None) -> str:
    if not alerts and not fusion_signals:
        return ""

    items = []
    for alert in alerts:
        sev = alert.get("severity", "warning")
        cls = "alert-critical" if sev in ("critical", "high") else ""
        title = alert.get("title", alert.get("pattern", "Alert"))
        speaker = alert.get("speaker_id", "")
        speaker_display = _speaker_label(speaker, speaker_roles) if speaker and speaker_roles else _esc(speaker)
        desc = alert.get("description", "")
        items.append(f"""
<div class="alert-item {cls}">
  <div class="alert-title">{'🔴' if sev in ('critical','high') else '⚠️'} {_esc(title)}{f' — {speaker_display}' if speaker else ''}</div>
  <div class="alert-desc">{_esc(desc)}</div>
</div>""")

    for fs in fusion_signals:
        sid = fs.get("speaker_id", "")
        sid_display = _speaker_label(sid, speaker_roles) if sid and speaker_roles else _esc(sid)
        sig_type = fs.get("signal_type", "")
        val_text = fs.get("value_text", "")
        conf = fs.get("confidence", 0)
        start = fs.get("window_start_ms", 0) / 1000
        end = fs.get("window_end_ms", 0) / 1000
        items.append(f"""
<div class="alert-item">
  <div class="alert-title">🟠 {_esc(sig_type.replace('_',' ').title())} — {sid_display}</div>
  <div class="alert-desc">{_esc(val_text)} (confidence: {conf:.2f}) @ {_fmt_time(start)}–{_fmt_time(end)}</div>
</div>""")

    return f"""
<div class="card">
  <h2>Alerts &amp; Fusion Insights ({len(alerts)} alerts, {len(fusion_signals)} fusion signals)</h2>
  {"".join(items)}
</div>"""


def _key_moments_and_recommendations(report: dict) -> str:
    parts = []

    # Key moments
    moments = report.get("key_moments", [])
    if moments:
        items = ""
        if isinstance(moments, list):
            for m in moments:
                if isinstance(m, str):
                    items += f"<li>{_esc(m)}</li>"
                elif isinstance(m, dict):
                    items += f"<li>{_esc(m.get('description', str(m)))}</li>"
        parts.append(f"""
<div class="card">
  <h2>Key Moments</h2>
  <ol style="padding-left:18px;color:var(--text);font-size:14px;line-height:2">{items}</ol>
</div>""")

    # Cross-modal insights
    insights = report.get("cross_modal_insights", [])
    if insights:
        items = ""
        if isinstance(insights, list):
            for ins in insights:
                if isinstance(ins, str):
                    items += f"<li>{_esc(ins)}</li>"
        elif isinstance(insights, str):
            items = f"<li>{_esc(insights)}</li>"
        if items:
            parts.append(f"""
<div class="card">
  <h2>Cross-Modal Insights</h2>
  <ul class="rec-list">{items}</ul>
</div>""")

    # Recommendations
    recs = report.get("recommendations", [])
    if recs:
        items = ""
        if isinstance(recs, list):
            for r in recs:
                if isinstance(r, str):
                    items += f"<li>{_esc(r)}</li>"
        elif isinstance(recs, str):
            items = f"<li>{_esc(recs)}</li>"
        if items:
            parts.append(f"""
<div class="card">
  <h2>Coaching Recommendations</h2>
  <ul class="rec-list">{items}</ul>
</div>""")

    return "\n".join(parts)


def _footer(ts: str, session_id: str) -> str:
    return f"""
<hr class="section-divider">
<div class="report-footer">
  Generated by <strong>NEXUS</strong> Multi-Agent Behavioural Analysis System<br>
  Session: {_esc(session_id)} · {_esc(ts)}<br>
  <span style="color:var(--text-muted)">Confidence ceiling: 0.85 · Deception cap: 0.55 · Probabilistic indicators only</span>
</div>
<script>
var nexusShowAll=false;
function toggleBadges(){{
  nexusShowAll=!nexusShowAll;
  var badges=document.querySelectorAll('.badge[data-priority="low"]');
  for(var i=0;i<badges.length;i++){{
    badges[i].style.display=nexusShowAll?'inline-flex':'none';
  }}
  document.getElementById('badge-toggle').textContent=
    nexusShowAll?'Show key signals only':'Show all signals';
}}
(function(){{
  var badges=document.querySelectorAll('.badge[data-priority="low"]');
  for(var i=0;i<badges.length;i++){{badges[i].style.display='none';}}
}})();
</script>
</div>
</body>
</html>"""


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def _speaker_label(sid: str, speaker_roles: dict) -> str:
    """Render a speaker name with role label if available."""
    role = speaker_roles.get(sid, "") if speaker_roles else ""
    if role:
        return f'{html.escape(role)} <span style="font-size:11px;color:var(--text-muted)">({html.escape(sid)})</span>'
    return html.escape(str(sid))


def _esc(text) -> str:
    if text is None:
        return ""
    return html.escape(str(text))


def _fmt_time(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


def _fmt_duration(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def _parse_ts(ts: str) -> str:
    try:
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%B %d, %Y at %H:%M")
    except Exception:
        return ts


def _infer_title(content_type: str, audio_file: str) -> str:
    if audio_file:
        name = Path(audio_file).stem
        name = name.replace("_", " ").replace("-", " ").title()
        return f"{name} — {content_type.replace('_',' ').title()}"
    return content_type.replace("_", " ").title() + " Analysis"


def _stress_colour(pct: int) -> str:
    if pct >= 60:
        return "var(--red)"
    if pct >= 35:
        return "var(--amber)"
    return "var(--green)"


def _stress_colour_hex(val: float) -> str:
    if val >= 0.6:
        return "#EF4444"
    if val >= 0.35:
        return "#F59E0B"
    return "#22C55E"


def _calc_filler_rate(speaker_id: str, voice_signals: list, duration: float) -> float:
    filler_count = 0
    for vs in voice_signals:
        if vs.get("speaker_id") != speaker_id:
            continue
        if vs.get("signal_type") == "filler_detection":
            filler_count += vs.get("metadata", {}).get("filler_count", 0)
    # Rough estimate: average speech is ~2.5 words per second
    word_estimate = max(1, duration * 1.5)  # conservative
    return (filler_count / word_estimate) * 100
