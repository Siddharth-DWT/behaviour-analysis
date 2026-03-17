# UI.md — NEXUS Dashboard Design Specification

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Framework | React 18 + TypeScript |
| Styling | Tailwind CSS |
| Charts | Recharts (signals timeline, area charts) |
| State | React Query (server state) + Zustand (client state) |
| Routing | React Router v6 |
| WebSocket | native WebSocket for live mode |
| Icons | Lucide React |
| Date/Time | date-fns |
| Build | Vite |

---

## Colour System

```
Background:     #0F1117 (near-black)
Surface:        #1A1D27 (card background)
Surface Hover:  #242836
Border:         #2D3348
Text Primary:   #E8ECF4
Text Secondary: #8B93A7
Text Muted:     #565E73

Accent Blue:    #4F8BFF (links, active states)
Accent Purple:  #8B5CF6 (fusion/cross-modal)

Signal Colours:
  Stress High:  #EF4444 (red-500)
  Stress Med:   #F59E0B (amber-500)
  Stress Low:   #22C55E (green-500)
  Neutral:      #6B7280 (gray-500)
  Confidence:   #3B82F6 (blue-500)
  Engagement:   #10B981 (emerald-500)
  Alert:        #F97316 (orange-500)

Agent Colours (for multi-agent timeline):
  Voice:        #4F8BFF (blue)
  Language:     #8B5CF6 (purple)
  Facial:       #F59E0B (amber)
  Body:         #10B981 (emerald)
  Gaze:         #EC4899 (pink)
  Conversation: #06B6D4 (cyan)
  Fusion:       #F97316 (orange)
```

---

## Layout Structure

```
┌─────────────────────────────────────────────────────────────────┐
│  TOPBAR  [NEXUS logo]  [Session name]  [Live/Recorded badge]   │
│          [Search]  [Settings gear]  [User avatar]              │
├──────┬──────────────────────────────────────────────────────────┤
│      │                                                         │
│  S   │            MAIN CONTENT AREA                            │
│  I   │                                                         │
│  D   │    Changes based on current view:                       │
│  E   │    - Session List                                       │
│  B   │    - Session Detail (Analysis)                          │
│  A   │    - Report View                                        │
│  R   │    - Settings                                           │
│      │                                                         │
│      │                                                         │
│      │                                                         │
├──────┴──────────────────────────────────────────────────────────┤
│  STATUS BAR  [Agent health dots]  [Processing status]          │
└─────────────────────────────────────────────────────────────────┘
```

### Sidebar Navigation (collapsed by default, expand on hover)

| Icon | Label | Route |
|------|-------|-------|
| 📊 | Dashboard | / |
| 📁 | Sessions | /sessions |
| 📈 | Analytics | /analytics |
| 👤 | Speakers | /speakers |
| ⚙️ | Settings | /settings |
| 📖 | Docs | /docs |

---

## View 1: Session List (`/sessions`)

The landing page showing all analysed sessions.

```
┌─────────────────────────────────────────────────────────────┐
│  Sessions                          [Upload New] [Filter ▼]  │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ 🟢 Q4 Sales Call — Acme Corp          Mar 14, 2:30pm │  │
│  │    Duration: 42 min  │  Speakers: 2  │  Alerts: 3    │  │
│  │    Key: Decision Readiness detected at 28:15          │  │
│  │    Tags: [sales] [high-value] [decision-ready]        │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ 🟡 Client Onboarding — Beta Inc       Mar 13, 10am   │  │
│  │    Duration: 55 min  │  Speakers: 3  │  Alerts: 7    │  │
│  │    Key: Silent Resistance detected (2 instances)      │  │
│  │    Tags: [client] [onboarding] [risk]                 │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ 🟢 Team Standup — Engineering         Mar 13, 9am    │  │
│  │    Duration: 18 min  │  Speakers: 5  │  Alerts: 0    │  │
│  │    Key: High engagement, balanced participation       │  │
│  │    Tags: [internal] [standup]                         │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  [Load more...]                                             │
└─────────────────────────────────────────────────────────────┘
```

### Session Card Data

| Field | Source |
|-------|--------|
| Status indicator | 🟢 positive / 🟡 mixed / 🔴 concerning |
| Title | User-provided or auto-generated from transcript |
| Date/time | Session creation timestamp |
| Duration | Audio/video file duration |
| Speaker count | From diarization |
| Alert count | Count of Tier 2+ alerts |
| Key insight | Highest-confidence compound pattern or most notable moment |
| Tags | Auto-generated from meeting_type + detected patterns |

### Filters

- Date range picker
- Meeting type (sales, client, internal, interview)
- Alert status (has alerts / no alerts)
- Speaker (search by name if identified)
- Minimum alert tier

---

## View 2: Session Detail (`/sessions/:id`)

The main analysis view. This is the most complex and important screen.

### Layout: 4-Panel Split

```
┌──────────────────────────────────┬──────────────────────────────┐
│                                  │                              │
│   TRANSCRIPT PANEL               │   SPEAKER CARDS PANEL        │
│   (Left, 55% width)             │   (Right, 45% width)         │
│                                  │                              │
├──────────────────────────────────┴──────────────────────────────┤
│                                                                 │
│   SIGNAL TIMELINE (Full width, bottom)                         │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│   ALERTS BAR (Full width, collapsible)                         │
└─────────────────────────────────────────────────────────────────┘
```

---

### Panel A: Transcript Panel (Left)

Scrollable transcript with inline signal annotations.

```
┌─────────────────────────────────────────────────┐
│  Transcript                    [Search] [Export] │
│                                                  │
│  ┌─ 00:42 ──────────────────────────────────┐   │
│  │ Speaker_0 (Sales Rep)                     │   │
│  │ "So let me walk you through our pricing   │   │
│  │  structure. We have three tiers..."       │   │
│  │                                           │   │
│  │  🔵 VOICE: confident (0.62)              │   │
│  │  🟣 LANG: persuasion — social proof (0.55)│   │
│  └───────────────────────────────────────────┘   │
│                                                  │
│  ┌─ 01:15 ──────────────────────────────────┐   │
│  │ Speaker_1 (Buyer)                         │   │
│  │ "Hmm, well, I mean... that's             │   │
│  │  interesting. What about, um, the         │   │
│  │  enterprise tier?"                        │   │
│  │                                           │   │
│  │  🔴 VOICE: stress elevated (0.58)        │   │
│  │  🔵 VOICE: filler spike (3.2%)           │   │
│  │  🟣 LANG: hedging + objection forming    │   │
│  │  🟠 FUSION: Objection Formation Stage 2  │   │
│  │  ⚠️ ALERT: Early objection warning       │   │
│  └───────────────────────────────────────────┘   │
│                                                  │
│  ┌─ 01:38 ──────────────────────────────────┐   │
│  │ Speaker_0 (Sales Rep)                     │   │
│  │ "Great question! The enterprise tier      │   │
│  │  includes everything in Professional      │   │
│  │  plus dedicated support..."               │   │
│  │                                           │   │
│  │  🔵 VOICE: rate elevated +32% (rushing)  │   │
│  │  🟣 LANG: enthusiasm language            │   │
│  │  🟠 FUSION: manufactured enthusiasm (0.52)│   │
│  └───────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

### Transcript Annotations

Each transcript segment can show inline signal badges. Badges are colour-coded by agent and show on hover:

| Badge | Colour | Example |
|-------|--------|---------|
| 🔵 VOICE | Blue | "stress: 0.67", "filler spike" |
| 🟣 LANG | Purple | "buying signal", "objection" |
| 🟡 FACE | Amber | "Duchenne smile", "stress face" |
| 🟢 BODY | Emerald | "forward lean", "head shake" |
| 🩷 GAZE | Pink | "distracted", "high engagement" |
| 🔷 CONVO | Cyan | "long latency", "interruption" |
| 🟠 FUSION | Orange | "Credibility issue", "Decision Readiness" |

**Interaction**: Clicking any badge scrolls the signal timeline to that moment and expands the signal details panel.

---

### Panel B: Speaker Cards (Right)

One card per speaker, showing real-time (or cumulative) state.

```
┌─────────────────────────────────────────┐
│  Speakers                               │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │ 👤 Speaker_0 — "Sales Rep"      │    │
│  │                                  │    │
│  │  Stress:     ████░░░░░░  38%    │    │
│  │  Engagement: ████████░░  82%    │    │
│  │  Confidence: ███████░░░  71%    │    │
│  │  Filler Rate: 0.8% ✅           │    │
│  │  Talk Time:  62%                 │    │
│  │  Tone: Confident                 │    │
│  │                                  │    │
│  │  ▸ Stress trend  [sparkline ↗]  │    │
│  │  ▸ Buying signals: 0           │    │
│  │  ▸ Alerts triggered: 1         │    │
│  └─────────────────────────────────┘    │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │ 👤 Speaker_1 — "Buyer"          │    │
│  │                                  │    │
│  │  Stress:     ██████░░░░  58%    │    │
│  │  Engagement: ██████░░░░  61%    │    │
│  │  Confidence: ████░░░░░░  42%    │    │
│  │  Filler Rate: 3.2% ⚠️           │    │
│  │  Talk Time:  38%                 │    │
│  │  Tone: Nervous                   │    │
│  │                                  │    │
│  │  ▸ Stress trend  [sparkline ↗↗] │    │
│  │  ▸ Buying signals: 2           │    │
│  │  ▸ Alerts triggered: 4         │    │
│  └─────────────────────────────────┘    │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │  RELATIONSHIP METRICS            │    │
│  │                                  │    │
│  │  Rapport Score: ████████░░ 78%  │    │
│  │  Dominance Balance:  S0 ◄══► S1 │    │
│  │  Engagement Sync: ███████░░ 72% │    │
│  │  Talk Time Balance: 62/38       │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

### Speaker Card Fields

| Field | Source | Update Frequency |
|-------|--------|-----------------|
| Stress gauge | VOICE-STRESS-01 (latest window) | Every 5s window |
| Engagement gauge | Composite: gaze + face + body + voice | Every 15s |
| Confidence gauge | VOICE-TONE-04 score | Every 5s window |
| Filler rate | VOICE-FILLER-01 (cumulative %) | Rolling |
| Talk time | Cumulative seconds / total seconds | Running total |
| Tone label | VOICE-TONE-03/04 (latest) | Every 5s window |
| Stress sparkline | Last 20 stress scores as tiny line chart | Rolling |
| Buying signals | Count of LANG-BUY-01 events | Cumulative |
| Alerts | Count of Tier 2+ alerts for this speaker | Cumulative |

### Relationship Card Fields

| Field | Source |
|-------|--------|
| Rapport | CONVO-RAP-01 multi-modal rapport score |
| Dominance balance | CONVO-DOM-01 per-speaker dominance → slider |
| Engagement sync | Cross-speaker engagement correlation |
| Talk time | Pie chart or bar showing per-speaker % |

---

### Panel C: Signal Timeline (Bottom)

A multi-track timeline showing all agent signals over the session duration.

```
┌─────────────────────────────────────────────────────────────────┐
│  Signal Timeline                           [Zoom: 1min ▼]      │
│                                                                 │
│  Time:  |0:00    |5:00    |10:00   |15:00   |20:00   |25:00   │
│         ├────────┼────────┼────────┼────────┼────────┼────────│
│  Voice  │▓▓░░▓▓▓▓▓░░░░▓▓▓████▓▓░░░░░░▓▓▓▓██████▓▓░░░░░░░░░│
│  Lang   │░░░░░░▓▓▓░░░░░░▓▓▓▓░░░░█▓▓░░░░▓▓▓▓░░▓██▓▓░░░░░░░░│
│  Face   │░░▓░░▓▓░▓▓░░░░░░▓▓▓░░░░░░░░▓▓▓░░░▓▓▓░░░░░░▓▓░░░░│
│  Body   │░░░░░░░▓░░░░░░░░▓▓░░░░░░░░░▓░░░░░▓▓░░░░░░░░▓░░░░│
│  Gaze   │▓▓▓▓▓▓░░▓▓▓▓▓▓▓░░░▓▓▓▓▓▓▓▓░░░▓▓▓▓▓▓▓░░░▓▓▓▓▓▓▓▓│
│  Convo  │▒░░░░░▒░░░░░▒░░░░░▒░░░░░▒░░░░░▒░░░░░▒░░░░░▒░░░░│
│         │                                                     │
│  Alerts │     ⚠          ⚠⚠           🔴              ⭐   │
│  Fusion │       ░░░▓▓▓░░░░░░▓▓██▓▓░░░░░░▓▓▓▓██▓▓░░░░░░░░░│
│         ├────────┼────────┼────────┼────────┼────────┼────────│
│                                                                 │
│  ◄ ████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ ►  │
│  ↑ Scroll handle (shows current view position in full session) │
└─────────────────────────────────────────────────────────────────┘
```

### Timeline Tracks

| Track | Content | Colour |
|-------|---------|--------|
| Voice | Stress score as heat strip (green→yellow→red) | Agent blue |
| Language | Sentiment polarity as heat strip (red→gray→green) | Agent purple |
| Facial | Valence as heat strip | Agent amber |
| Body | Movement energy as heat strip | Agent emerald |
| Gaze | Screen engagement as heat strip | Agent pink |
| Conversation | Turn events as discrete markers | Agent cyan |
| Alerts | Alert icons at triggered moments | By alert tier |
| Fusion | Compound pattern indicators | Agent orange |

### Timeline Interactions

- **Hover**: Shows tooltip with all active signals at that timestamp
- **Click**: Scrolls transcript panel to that moment, expands signal detail
- **Drag**: Select time range for focused analysis
- **Zoom**: 30s / 1min / 5min / full session zoom levels
- **Agent toggle**: Click track label to show/hide that agent's track

---

### Panel D: Alerts Bar (Collapsible)

```
┌─────────────────────────────────────────────────────────────────┐
│  Alerts (5)                                    [▼ Collapse]     │
│                                                                 │
│  🔴 28:15  CRITICAL — Decision Readiness detected (Speaker_1)  │
│     Buying language + calm voice + forward lean + high gaze     │
│     Confidence: 0.78  │  [Jump to moment →]                    │
│                                                                 │
│  ⚠️  12:45  ALERT — Objection Formation Stage 2 (Speaker_1)   │
│     Body tension + gaze break + hedge language beginning        │
│     Confidence: 0.61  │  [Jump to moment →]                    │
│                                                                 │
│  ⚠️  15:20  ALERT — Silent Resistance detected (Speaker_1)    │
│     Said "yes" + stress voice + closed posture + head shake     │
│     Confidence: 0.64  │  [Jump to moment →]                    │
│                                                                 │
│  📌  08:30  NOTICE — Manufactured Enthusiasm (Speaker_0)       │
│     High energy words + flat body language + forced smile        │
│     Confidence: 0.52  │  [Jump to moment →]                    │
│                                                                 │
│  ⭐  35:10  INSIGHT — Rapport Peak reached                     │
│     Mutual engagement high + mirroring + synchrony              │
│     Confidence: 0.71  │  [Jump to moment →]                    │
└─────────────────────────────────────────────────────────────────┘
```

### Alert Card Fields

| Field | Description |
|-------|-------------|
| Icon | 🔴 Critical / ⚠️ Alert / 📌 Notice / ⭐ Insight |
| Timestamp | When in the session this was detected |
| Pattern name | Compound or temporal pattern name |
| Speaker | Which speaker triggered it |
| Evidence | 1-line summary of contributing signals |
| Confidence | Overall confidence score |
| Action | "Jump to moment" scrolls transcript + timeline |

---

## View 3: Report View (`/sessions/:id/report`)

Post-session analysis report. Designed for sharing with team or client.

```
┌─────────────────────────────────────────────────────────────────┐
│  Session Report                     [Export PDF] [Export DOCX]  │
│  Q4 Sales Call — Acme Corp                                      │
│  March 14, 2025  •  42 minutes  •  2 participants              │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  EXECUTIVE SUMMARY                                        │  │
│  │                                                           │  │
│  │  [Claude-generated 3-4 sentence narrative summary of      │  │
│  │   the call, highlighting key behavioural patterns and     │  │
│  │   the most important cross-modal insight]                 │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  KEY MOMENTS                                              │  │
│  │                                                           │  │
│  │  1. 12:45 — Objection forming (early warning)            │  │
│  │     [2-line narrative: what happened behaviourally]       │  │
│  │                                                           │  │
│  │  2. 15:20 — Silent resistance detected                   │  │
│  │     [2-line narrative]                                    │  │
│  │                                                           │  │
│  │  3. 28:15 — Decision readiness (strongest signal)        │  │
│  │     [2-line narrative]                                    │  │
│  │                                                           │  │
│  │  4. 35:10 — Rapport peak                                 │  │
│  │     [2-line narrative]                                    │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────┐  ┌──────────────────────────────┐    │
│  │  SPEAKER ANALYSIS    │  │  COMMUNICATION DYNAMICS      │    │
│  │                      │  │                              │    │
│  │  [Per-speaker radar  │  │  [Rapport over time chart]   │    │
│  │   chart showing:     │  │  [Talk time pie chart]       │    │
│  │   - Confidence       │  │  [Dominance balance bar]     │    │
│  │   - Engagement       │  │  [Interruption count]        │    │
│  │   - Stress           │  │  [Response latency avg]      │    │
│  │   - Clarity          │  │                              │    │
│  │   - Rapport]         │  │                              │    │
│  └──────────────────────┘  └──────────────────────────────┘    │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  STRESS TIMELINE                                          │  │
│  │                                                           │  │
│  │  [Area chart: per-speaker stress scores over time]       │  │
│  │  [Annotated with key moment markers]                     │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  COACHING SUGGESTIONS                                     │  │
│  │                                                           │  │
│  │  For Sales Rep (Speaker_0):                              │  │
│  │  • Watch speech rate during pricing discussion (32%      │  │
│  │    above normal — may signal rushing)                    │  │
│  │  • Manufactured enthusiasm detected — try more           │  │
│  │    authentic engagement                                  │  │
│  │                                                           │  │
│  │  For Buyer (Speaker_1):                                  │  │
│  │  • High filler rate (3.2%) during technical questions    │  │
│  │  • Silent resistance pattern — consider addressing       │  │
│  │    concerns more directly                                │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  MEETING-TYPE SPECIFIC: SALES ANALYSIS                    │  │
│  │                                                           │  │
│  │  Buying Signals Detected: 4                              │  │
│  │  Objection Points: 2 (1 addressed, 1 unresolved)        │  │
│  │  Decision Readiness Moments: 1 (at 28:15)               │  │
│  │  Persuasion Techniques Used: 6 (social proof ×2,        │  │
│  │    scarcity ×1, authority ×2, reciprocity ×1)           │  │
│  │  Close Probability Estimate: Moderate-High               │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Report Sections by Meeting Type

| Meeting Type | Sections Shown |
|-------------|----------------|
| Sales Call | Executive Summary + Key Moments + Speaker Analysis + Stress Timeline + Buying Signals + Objection Map + Decision Readiness + Coaching |
| Client Meeting | Executive Summary + Key Moments + Satisfaction Indicators + Risk Flags + Rapport Trajectory + Action Items |
| Internal Meeting | Executive Summary + Participation Balance + Engagement Distribution + Decision Points + Action Items |
| Interview | Executive Summary + Candidate Analysis + Confidence Trajectory + Authenticity Score + Key Moments |

---

## View 4: Analytics (`/analytics`)

Cross-session analytics showing trends over time.

```
┌─────────────────────────────────────────────────────────────────┐
│  Analytics                              [Date Range ▼] [Team ▼]│
│                                                                 │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐  │
│  │ Sessions   │ │ Avg Rapport│ │ Close Rate │ │ Avg Stress │  │
│  │   142      │ │   72%      │ │   34%      │ │   0.38     │  │
│  │  ↑ +12%    │ │  ↑ +5%     │ │  ↑ +8%     │ │  ↓ -15%    │  │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘  │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  TEAM PERFORMANCE OVER TIME                               │  │
│  │  [Line chart: per-person rapport / stress / close rate]  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  MOST COMMON PATTERNS                                     │  │
│  │                                                           │  │
│  │  1. Silent Resistance (34 instances across 12 sessions)  │  │
│  │  2. Objection Formation (28 instances)                   │  │
│  │  3. Manufactured Enthusiasm (22 instances)               │  │
│  │  4. Decision Readiness (18 instances)                    │  │
│  │  5. Rapport Peak (45 instances)                          │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  SPEAKER PROFILES (Frequent participants)                 │  │
│  │                                                           │  │
│  │  [Cards for most frequent speakers with session-over-    │  │
│  │   session trends: avg stress, rapport, talk time, etc.]  │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## View 5: Speaker Profiles (`/speakers/:id`)

Cross-session view of a specific speaker.

```
┌─────────────────────────────────────────────────────────────────┐
│  Speaker Profile: John Smith                                    │
│  23 sessions  •  First seen: Jan 15  •  Last: Mar 14           │
│                                                                 │
│  ┌──────────────────────┐  ┌──────────────────────────────┐    │
│  │  BASELINE AVERAGES   │  │  TREND (last 30 days)        │    │
│  │                      │  │                              │    │
│  │  Avg F0: 142 Hz      │  │  [Sparklines for:           │    │
│  │  Avg Rate: 156 WPM   │  │   - Stress trend            │    │
│  │  Avg Fillers: 1.8%   │  │   - Confidence trend        │    │
│  │  Avg Stress: 0.35    │  │   - Engagement trend        │    │
│  │  Avg Engagement: 74% │  │   - Filler rate trend]      │    │
│  └──────────────────────┘  └──────────────────────────────┘    │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  STRESS TRIGGERS (learned from session history)           │  │
│  │                                                           │  │
│  │  Topics that historically elevate this speaker's stress:  │  │
│  │  • Pricing discussion (+42% stress vs baseline)          │  │
│  │  • Timeline questions (+28% stress)                      │  │
│  │  • Competitor comparisons (+35% stress)                  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  SESSION HISTORY                                          │  │
│  │  [List of all sessions this speaker appeared in]         │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## View 6: Settings (`/settings`)

```
┌─────────────────────────────────────────────────────────────────┐
│  Settings                                                       │
│                                                                 │
│  ┌─ General ─────────────────────────────────────────────────┐  │
│  │  Default meeting type: [Sales Call ▼]                     │  │
│  │  Auto-tag sessions: [✓ Enabled]                          │  │
│  │  Whisper model: [medium ▼]                               │  │
│  │  Language: [English ▼]                                    │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─ Alert Thresholds ────────────────────────────────────────┐  │
│  │  Minimum alert tier to show: [Alert (Tier 2) ▼]         │  │
│  │  Stress threshold for alert: [0.65 ────●──── ]          │  │
│  │  Decision Readiness min conf: [0.60 ────●──── ]         │  │
│  │  Enable deception risk alerts: [✗ Disabled]              │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─ Agent Configuration ─────────────────────────────────────┐  │
│  │  Voice Agent:  [✓ Enabled]  Port: 8001  [Test ►]        │  │
│  │  Language:     [✓ Enabled]  Port: 8002  [Test ►]        │  │
│  │  Facial:       [✗ Disabled] (not deployed)               │  │
│  │  Body:         [✗ Disabled] (not deployed)               │  │
│  │  Gaze:         [✗ Disabled] (not deployed)               │  │
│  │  Conversation: [✓ Enabled]  Port: 8006  [Test ►]        │  │
│  │  Fusion:       [✓ Enabled]  Port: 8007  [Test ►]        │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─ Domain Weights ──────────────────────────────────────────┐  │
│  │  Language:     [████████░░ 40%]                          │  │
│  │  Voice:        [█████░░░░░ 25%]                          │  │
│  │  Facial:       [████░░░░░░ 20%]                          │  │
│  │  Body:         [██░░░░░░░░ 10%]                          │  │
│  │  Gaze:         [█░░░░░░░░░  5%]                          │  │
│  │  [Reset to defaults]                                      │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─ API Keys ────────────────────────────────────────────────┐  │
│  │  Anthropic Claude: [●●●●●●●●●●●●sk-ant-...]  [Update]   │  │
│  │  HuggingFace (optional): [●●●●●●hf_...]      [Update]   │  │
│  │  Recall.ai (optional): [Not set]              [Set up]   │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─ Data Management ─────────────────────────────────────────┐  │
│  │  Storage used: 2.4 GB / 50 GB                            │  │
│  │  Recordings stored: 142                                   │  │
│  │  Auto-delete recordings after: [30 days ▼]               │  │
│  │  [Export all data]  [Delete all data]                     │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Library

### Reusable Components to Build

| Component | Description | Used In |
|-----------|-------------|---------|
| `<StressGauge />` | Horizontal bar 0-100% with colour gradient | Speaker cards |
| `<SignalBadge />` | Coloured pill with agent icon + signal name | Transcript |
| `<Sparkline />` | Tiny inline line chart (20 data points) | Speaker cards, profiles |
| `<AlertCard />` | Alert with tier icon, timestamp, evidence, link | Alert bar |
| `<SignalTimeline />` | Multi-track horizontal timeline with zoom | Session detail |
| `<RadarChart />` | 5-axis radar for speaker analysis | Reports |
| `<DominanceSlider />` | Two-ended slider showing balance | Speaker cards |
| `<TalkTimePie />` | Pie chart of talk time per speaker | Reports, speaker cards |
| `<HeatStrip />` | Thin horizontal colour bar (green→red) | Timeline tracks |
| `<TranscriptBlock />` | Styled transcript segment with annotations | Transcript panel |
| `<SessionCard />` | Summary card for session list | Session list |

---

## Responsive Breakpoints

| Breakpoint | Layout Change |
|-----------|---------------|
| Desktop (>1280px) | Full 4-panel layout |
| Laptop (1024-1280px) | Transcript + speaker cards stack vertically, timeline below |
| Tablet (768-1024px) | Single column: transcript → cards → timeline → alerts |
| Mobile (<768px) | Single column with collapsible sections, simplified cards |

---

## Live Mode Additions (Phase 4)

When connected to a live meeting via Recall.ai, the dashboard gains:

- **Live badge**: Pulsing green dot in topbar
- **Real-time speaker cards**: Values update every 5-15 seconds
- **Rolling transcript**: Auto-scrolls as new segments arrive
- **Live alert notifications**: Browser notification + sound for Tier 2+ alerts
- **Recording indicator**: Shows "Recording... 00:42:15"
- **Participant grid**: Thumbnail video of each participant (from Recall.ai)

```
┌─────────────────────────────────────────────────────────────────┐
│  🟢 LIVE  │  Q4 Sales Call  │  Recording... 00:42:15  │  [End] │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   [Same layout as Session Detail, but with live updating]       │
│                                                                  │
│   Transcript auto-scrolls ↓                                     │
│   Speaker cards pulse on update                                 │
│   Timeline grows rightward in real-time                         │
│   Alerts appear as browser notifications                        │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Implementation Priority

### Phase 1 Dashboard (Week 6)
Build ONLY these views / components:
1. Session List (basic cards)
2. Session Detail → Transcript panel (text only, no video)
3. Session Detail → Voice signals inline in transcript
4. Session Detail → Basic stress timeline (Recharts area chart)
5. Report View → Executive summary + key moments (Claude-generated)

### Phase 2 Dashboard (Weeks 13-14)
Add:
1. Speaker Cards with gauges
2. Full multi-track Signal Timeline
3. Alert Bar
4. Facial emotion badges in transcript
5. Radar charts in reports

### Phase 3 Dashboard (Weeks 19-22)
Add:
1. Analytics view with cross-session trends
2. Speaker Profiles view
3. Settings page with rule threshold sliders
4. Export to PDF/DOCX

### Phase 4 Dashboard (Weeks 25-28)
Add:
1. Live mode with WebSocket
2. Real-time updating UI
3. Browser notifications
4. Participant video grid
