# PLAN.md — NEXUS Development Roadmap

## Overview

Total estimated timeline: **40-48 weeks** from vertical slice to production.
Current status: **Phase 1 complete · Phase 3A in progress** (Neo4j single-session knowledge graph + hybrid RAG chat).
Phase 2 (visual agents) is deferred — audio-only pipeline is the active focus.

> **See `docs/STATUS.md` for the authoritative build state.** This file is the
> forward-looking roadmap; STATUS.md tracks what is actually shipped.

---

## Phase 0: Foundation ✅ COMPLETE
**Duration: 2 weeks | Status: DONE**

- [x] Research compendium (120+ studies across 6 domains)
- [x] Architecture plan (7-agent system design)
- [x] Feasibility analysis with gap assessment
- [x] Rule Engine documentation (94 rules, 2,158 paragraphs)
- [x] Compound patterns (12 multi-domain states)
- [x] Temporal sequences (8 cascade patterns)
- [x] Docker infrastructure (PostgreSQL+pgvector, Valkey)
- [x] Database schema (15 tables, seeded rule_config)
- [x] Shared libraries (signals model, message bus, config)
- [x] Voice Agent v0.1 (5 core rules implemented)

---

## Phase 1: Vertical Slice — Audio-Only MVP ✅ COMPLETE
**Duration: 4 weeks | Status: DONE (and significantly extended beyond original scope)**

Phase 1 actually delivered the original MVP plus content-type adaptation
(`ContentTypeProfile`), full RAG chat over pgvector, multi-backend
transcription cascade (Whisper+NeMo / Parakeet / AssemblyAI / Deepgram /
local), GPU diarization cascade, knowledge_store, and graph analytics in
the Fusion Agent. JWT auth was also pulled forward from Phase 4.

The original Phase 1 task list below is preserved for reference; nearly
all items are done.

### Week 3: Ground Truth + Voice Agent Validation
- [ ] Obtain 5-10 real sales call recordings (30-60 min each)
- [ ] Manual labeling: annotate stress moments, filler patterns, tone shifts, key moments
  - Use spreadsheet: timestamp | signal_type | human_judgement | notes
  - Minimum 3 recordings fully labeled
- [ ] Run Voice Agent on all recordings
- [ ] Compare output to labels → compute accuracy per rule
- [ ] Tune thresholds in `rule_config` table based on results
- [ ] Fix any feature extraction bugs discovered with real data
- [ ] Target: >60% agreement between agent and human labels

### Week 4: Language Agent
- [ ] Create `services/language-agent/` with same structure as voice-agent
- [ ] Implement LANG-SENT-01: Per-sentence sentiment (DistilBERT + LIWC fallback)
- [ ] Implement LANG-BUY-01: Buying signal detection (keyword patterns + Claude API)
- [ ] Implement LANG-OBJ-01: Objection signal detection (hedge counting + patterns)
- [ ] Implement LANG-PWR-01: Power language score (powerless feature counting)
- [ ] Implement LANG-INTENT-01: Intent classification (Claude API per utterance)
- [ ] Wire Language Agent to Redis Streams (publish signals)
- [ ] Validate against labeled data

### Week 5: Fusion Agent + Cross-Agent Communication
- [ ] Create `services/fusion-agent/`
- [ ] Implement Redis Streams subscription (consume from voice + language streams)
- [ ] Implement temporal alignment (5-10 second medium window)
- [ ] Implement FUSION-02: Speech Content × Voice Stress → Credibility check
- [ ] Implement FUSION-04: Gaze Break × Filler Words → Uncertainty (simplified: pause + filler correlation)
- [ ] Implement FUSION-13: Persuasion Language × Vocal Pace → Urgency authenticity
- [ ] Implement Unified Speaker State output
- [ ] Implement Claude API narrative generation (per-session report)
- [ ] **KEY VALIDATION**: Does fusion catch insights that individual agents miss?
  - Find 3-4 moments where cross-modal analysis reveals something single-modal doesn't
  - If yes → thesis validated, continue to Phase 2
  - If no → re-evaluate architecture before investing further

### Week 6: API Gateway + Basic Dashboard
- [ ] Create `services/api-gateway/` — FastAPI + WebSocket
- [ ] REST endpoints: POST /sessions (upload + analyse), GET /sessions/:id, GET /sessions/:id/signals, GET /sessions/:id/report
- [ ] WebSocket endpoint: /ws/sessions/:id (real-time signal push for future live mode)
- [ ] Create `dashboard/` — React + Tailwind + Recharts
- [ ] Dashboard v1: Session list → Session detail → Transcript + signal timeline + report
- [ ] See UI.md for full dashboard specification

**Phase 1 Deliverable**: A working demo that can process a recorded sales call and produce an analysis report with cross-modal insights. Shareable with beta users.

---

## Phase 2: Visual Layer
**Duration: 8 weeks | Target: Weeks 7-14**

### Weeks 7-8: Facial Agent
- [ ] Create `services/facial-agent/`
- [ ] Integrate MediaPipe Face Mesh (468 landmarks) for AU proxy detection
- [ ] Implement FACE-EMO-01: 7-class emotion detection (DeepFace)
- [ ] Implement FACE-SMILE-01: Duchenne vs non-Duchenne (AU6+AU12 co-occurrence)
- [ ] Implement FACE-STRESS-01: Facial stress composite (AU4+AU23+AU24)
- [ ] Implement FACE-ENG-01: Visual engagement score (composite)
- [ ] Implement FACE-VA-01: Valence-arousal continuous (AffectNet CNN)
- [ ] Implement FACE-CAL-01: Per-speaker facial baseline
- [ ] Skip FACE-MICRO-01 (micro-expressions) — unreliable at webcam framerate
- [ ] Validate on recorded video files (not live yet)

### Weeks 9-10: Gaze Agent
- [ ] Create `services/gaze-agent/`
- [ ] Integrate MediaPipe Iris tracking + head pose
- [ ] Implement GAZE-DIR-01: Gaze direction classification (with angular error dead zone)
- [ ] Implement GAZE-CONTACT-01: Screen engagement % (speaking vs listening split)
- [ ] Implement GAZE-BLINK-01: Blink rate (EAR method)
- [ ] Implement GAZE-ATT-01: Composite attention score
- [ ] Implement GAZE-DIST-01: Distraction event detection
- [ ] Implement GAZE-CAL-01: Per-speaker gaze baseline + camera position estimation
- [ ] Skip GAZE-SYNC-01 (mutual gaze synchrony) until multi-camera support

### Weeks 11-12: Body Agent
- [ ] Create `services/body-agent/`
- [ ] Integrate MediaPipe Holistic (33 body + 21 hand landmarks)
- [ ] Implement BODY-HEAD-01: Head nod/shake detection (most reliable webcam signal)
- [ ] Implement BODY-POST-01: Posture score + body openness (shoulders-up only mode)
- [ ] Implement BODY-LEAN-01: Forward/backward lean (head size proxy)
- [ ] Implement BODY-GEST-01: Hand visibility + gesture type classification
- [ ] Implement BODY-FIDG-01: Fidget rate + movement energy
- [ ] Implement BODY-TOUCH-01: Self-touch / pacifying detection
- [ ] Implement BODY-CAL-01: Per-speaker body baseline
- [ ] Skip BODY-MIRROR-01 (mirroring) until multi-camera support

### Weeks 13-14: Full Fusion + Testing
- [ ] Extend Fusion Agent with all 15 pairwise rules
- [ ] Implement remaining 12 FUSION rules (FUSION-01, 03, 05-12, 14, 15)
- [ ] Implement 4 highest-value compound patterns:
  - COMPOUND-01: Genuine Engagement
  - COMPOUND-02: Active Disengagement
  - COMPOUND-04: Decision Readiness
  - COMPOUND-07: Silent Resistance
- [ ] Implement 2 highest-value temporal sequences:
  - TEMPORAL-01: Stress Cascade
  - TEMPORAL-04: Objection Formation
- [ ] End-to-end testing with 20+ recorded video meetings
- [ ] Dashboard v2: Add facial emotion timeline, gaze heatmap, body language indicators
- [ ] Performance optimization: benchmark processing speed per video minute

**Phase 2 Deliverable**: Full 6-domain analysis on recorded video files. Complete cross-modal fusion with alerts and compound pattern detection.

---

## Phase 3: Intelligence Refinement
**Duration: 8 weeks | Target: Weeks 15-22**

> Phase 3 has been re-scoped around the **Neo4j knowledge graph** plan in
> `prompt.md`. The original "Remaining Compound Patterns" sub-phase is
> deferred until after the visual agents (Phase 2) come online, since most
> compound patterns require facial/body/gaze signals.

### Phase 3A: Single-Session Knowledge Graph 🚧 IN PROGRESS
- [x] Add Neo4j 5.26 to docker-compose (`neo4j:5.26-community` + APOC + GDS)
- [x] `services/api_gateway/neo4j_sync.py` — sync session data from PG to Neo4j after pipeline completes
- [x] Node types: Session, Speaker, Segment, Topic, Signal, FusionInsight, Entity (Person/Company/Product/Objection/Commitment), Alert
- [x] Structural + temporal + containment edges (PARTICIPATED_IN, PART_OF, NEXT, FOLLOWED_BY, PRECEDED, OCCURRED_DURING, DISCUSSES, MENTIONED_IN, RESOLVED_IN, etc.)
- [x] Causal edges via post-load Cypher MERGE (CONTRADICTS, REINFORCES, TRIGGERED)
- [x] Cross-speaker INFLUENCED edges within 30 s
- [x] Hybrid `/sessions/:id/chat` — pgvector text search + Neo4j Cypher search via LLM-generated queries
- [ ] **Validate** with real sessions: run all 8 prompt.md Cypher use-cases manually
- [ ] Add `COMPOSED_OF` / `COMBINES` provenance to fusion + composite signals (unlocks signal decomposition + fusion explanation queries)

### Phase 3B: Speaker Identity 🔲 NOT STARTED
- [ ] Add `voice_embedding VECTOR(256)` column to `speakers` table
- [ ] Extract per-speaker pyannote/embedding during diarization (model already loaded for community-1 fallback)
- [ ] Vector index in Neo4j Speaker nodes for cross-session matching
- [ ] Manual speaker labelling UI in dashboard

### Phase 3C: Cross-Session Intelligence 🔲 NOT STARTED
- [ ] Add `Session.outcome` (won/lost/unknown) column + dashboard tagger
- [ ] Entity resolver for cross-session companies/people/topics (`apoc.text.fuzzyMatch`)
- [ ] `INTERACTED_WITH`, `HAS_TREND`, `SHOWS_PATTERN` aggregations
- [ ] Winning vs losing pattern detection (Cypher)

### Phase 3D: Dashboard 🔲 NOT STARTED
- [ ] `/speakers/:id` Speaker Profile page (behavioural DNA radar, trajectory, interaction map)
- [ ] `/insights` Organization Insights page (team heatmap, topic sensitivity, won-vs-lost comparison)
- [ ] `/graph` Knowledge Graph Explorer (neovis.js)

### Deferred from original Phase 3 (require Phase 2 visual agents)
- [ ] Implement all 12 compound patterns (COMPOUND-01 through 12)
- [ ] Implement all 8 temporal sequences (TEMPORAL-01 through 08)

### Deferred to a separate reports milestone
- [ ] Per-meeting-type structured report templates (sales / client / internal / coaching)
- [ ] Export: PDF, DOCX, JSON, Slack message, CRM push

**Phase 3 Deliverable**: Single-session Neo4j knowledge graph, hybrid RAG chat, cross-session speaker identity + entity resolution, behavioural-DNA dashboard pages.

---

## Phase 4: Live Meeting Integration
**Duration: 8 weeks | Target: Weeks 23-30**

### Weeks 23-24: Recall.ai Integration
- [ ] Establish Recall.ai partnership/API access
- [ ] Implement bot creation endpoint (join Zoom/Meet/Teams via meeting URL)
- [ ] Implement real-time audio/video stream reception
- [ ] Route audio stream to Voice Agent in real-time chunks
- [ ] Route video frames to Facial/Body/Gaze agents

### Weeks 25-26: Real-Time Processing Optimization
- [ ] Implement tiered processing: Voice+Language at full speed, Facial at 3fps, Body/Gaze at 1fps
- [ ] Batch Fusion Agent cycles to every 15 seconds (not 10)
- [ ] Implement Claude API call batching (reduce per-session API cost)
- [ ] WebSocket real-time dashboard push (speaker cards updating live)
- [ ] Load test: target 5 concurrent sessions on single server

### Weeks 27-28: Authentication + Multi-Tenancy
- [x] JWT authentication (access tokens + refresh tokens) — **done early (Phase 1)**
- [x] Password hashing (bcrypt) — **done early**
- [x] User roles: admin, member, viewer — **done early**
- [x] Session access control (user ownership + admin override) — **done early**
- [x] Login/Signup pages in dashboard — **done early**
- [x] Protected routes + auto-refresh tokens — **done early**
- [ ] OAuth 2.0 / SSO integration (Google, Microsoft, SAML)
- [ ] Per-tenant data isolation (PostgreSQL row-level security)
- [ ] API key management for per-customer usage tracking

### Weeks 29-30: Production Hardening
- [ ] Error handling + circuit breakers for each agent
- [ ] Graceful degradation when agents fail (FUSION continues with available data)
- [ ] GPU deployment for video agents (NVIDIA A10G or T4)
- [ ] Kubernetes manifests for scalable deployment
- [ ] Monitoring: Prometheus metrics, Grafana dashboards
- [ ] GDPR/CCPA compliance: data retention policies, right-to-deletion
- [ ] Load test: target 10+ concurrent sessions

**Phase 4 Deliverable**: Live meeting analysis on Zoom/Meet/Teams with real-time dashboard.

---

## Phase 5: Scale & Launch
**Duration: 8+ weeks | Target: Weeks 31-40+**

- [ ] Beta launch with 3-5 enterprise customers
- [ ] Fine-tune models on collected real meeting data
- [ ] Add cultural calibration profiles (display rule adjustments)
- [ ] CRM integrations (Salesforce, HubSpot)
- [ ] Slack integration (automated post-meeting summaries)
- [ ] Session-over-session analytics (performance tracking per salesperson)
- [ ] Webhook API for custom integrations
- [ ] Mobile companion app (view reports on phone)
- [ ] Video annotated export (overlay analysis on original video)

---

## Cost Estimates

### Development Phase (Weeks 1-30)
| Item | Estimated Cost |
|------|---------------|
| Cloud VMs (dev) | $50-100/month |
| Claude API (development) | $50-200/month |
| OpenAI API (Whisper + embeddings) | $20-50/month |
| Domain + hosting | $20/month |
| **Total development** | **$150-400/month** |

### Production (per session)
| Component | Cost/Hour |
|-----------|----------|
| GPU compute (video agents) | $3.50-6.00 |
| Claude API (Language + Fusion) | $0.80-2.50 |
| Whisper (self-hosted or API) | $0.10-0.40 |
| Recall.ai bot | $0.50-1.00 |
| Infrastructure | $0.10-0.20 |
| **Total per session hour** | **$5.00-10.10** |

### Pricing Strategy
- Audio-only tier: $49/user/month (Voice + Language + Conversation)
- Full analysis tier: $149/user/month (all 6 domains + video)
- Enterprise: $299/user/month (cross-session profiles, CRM, custom alerts)
