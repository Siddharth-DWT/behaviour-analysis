-- ============================================================================
-- NEXUS Database Schema - Initialization Script
-- Runs automatically on first 'docker compose up'
-- ============================================================================

-- ========================
-- 1. EXTENSIONS
-- ========================

-- Vector similarity search (for research context & speaker embeddings)
CREATE EXTENSION IF NOT EXISTS vector;

-- UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Full text search improvements
CREATE EXTENSION IF NOT EXISTS pg_trgm;


-- ========================
-- 2. CORE TABLES
-- ========================

-- Organizations / Tenants (for future multi-tenancy)
CREATE TABLE organizations (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name            VARCHAR(255) NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Insert default org for development
INSERT INTO organizations (id, name) VALUES 
    ('00000000-0000-0000-0000-000000000001', 'Development');

-- Users
CREATE TABLE users (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    org_id          UUID REFERENCES organizations(id),
    email           VARCHAR(255) UNIQUE NOT NULL,
    name            VARCHAR(255) NOT NULL,
    role            VARCHAR(50) DEFAULT 'member',  -- admin, member, viewer
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

INSERT INTO users (org_id, email, name, role) VALUES
    ('00000000-0000-0000-0000-000000000001', 'dev@nexus.local', 'Developer', 'admin');


-- ========================
-- 3. SESSION TABLES
-- ========================

-- A session = one meeting / one recording being analysed
CREATE TABLE sessions (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    org_id          UUID REFERENCES organizations(id),
    title           VARCHAR(500),
    session_type    VARCHAR(50) NOT NULL DEFAULT 'recording',  
                    -- recording, live_call, uploaded_video, uploaded_audio
    meeting_type    VARCHAR(50),  
                    -- sales_call, client_meeting, internal, interview, other
    status          VARCHAR(50) NOT NULL DEFAULT 'created',    
                    -- created, processing, analysing, completed, failed
    media_url       TEXT,              -- path to audio/video file
    duration_ms     BIGINT,
    speaker_count   INT,
    created_by      UUID REFERENCES users(id),
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    started_at      TIMESTAMPTZ,       -- when analysis began
    completed_at    TIMESTAMPTZ        -- when analysis finished
);

CREATE INDEX idx_sessions_org ON sessions(org_id);
CREATE INDEX idx_sessions_status ON sessions(status);
CREATE INDEX idx_sessions_type ON sessions(meeting_type);


-- Speakers detected in a session
CREATE TABLE speakers (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id      UUID REFERENCES sessions(id) ON DELETE CASCADE,
    speaker_label   VARCHAR(50) NOT NULL,   -- "Speaker A", "Speaker B" from diarisation
    speaker_name    VARCHAR(255),           -- resolved name if known
    role            VARCHAR(50),            -- seller, buyer, facilitator, unknown
    
    -- Calibration baselines (populated during first 3-5 min)
    baseline_data   JSONB,                  -- all baseline values as JSON
    calibration_confidence FLOAT DEFAULT 0.0,
    
    -- Cumulative session stats
    total_talk_time_ms  BIGINT DEFAULT 0,
    talk_time_pct       FLOAT DEFAULT 0.0,
    total_words         INT DEFAULT 0,
    
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_speakers_session ON speakers(session_id);


-- ========================
-- 4. TRANSCRIPT TABLE
-- ========================

CREATE TABLE transcript_segments (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id      UUID REFERENCES sessions(id) ON DELETE CASCADE,
    speaker_id      UUID REFERENCES speakers(id),
    segment_index   INT NOT NULL,           -- sequential order
    start_ms        BIGINT NOT NULL,
    end_ms          BIGINT NOT NULL,
    text            TEXT NOT NULL,
    word_count      INT,
    
    -- Per-segment analysis (populated by LANGUAGE Agent)
    sentiment       VARCHAR(20),            -- positive, negative, neutral
    sentiment_score FLOAT,                  -- -1.0 to +1.0
    intent          TEXT,                    -- 1-sentence intent classification
    signals         JSONB,                  -- detected signals array
    
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_transcript_session ON transcript_segments(session_id);
CREATE INDEX idx_transcript_speaker ON transcript_segments(speaker_id);
CREATE INDEX idx_transcript_time ON transcript_segments(session_id, start_ms);

-- Full text search on transcript
CREATE INDEX idx_transcript_text_search ON transcript_segments 
    USING GIN (to_tsvector('english', text));


-- ========================
-- 5. SIGNAL TABLES (Time-Series)
-- ========================

-- Raw signals from all agents - the primary analysis data
CREATE TABLE signals (
    id              BIGSERIAL PRIMARY KEY,
    session_id      UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    speaker_id      UUID REFERENCES speakers(id),
    agent           VARCHAR(20) NOT NULL,    -- voice, language, facial, body, gaze, conversation, fusion
    signal_type     VARCHAR(80) NOT NULL,    -- e.g. "stress_score", "buying_signal", "engagement_visual"
    value           FLOAT,                   -- numeric value (0-1 for scores, -1 to +1 for sentiment)
    value_text      VARCHAR(255),            -- text value for enums (e.g. "warm", "nervous")
    confidence      FLOAT NOT NULL DEFAULT 0.5,
    window_start_ms BIGINT NOT NULL,
    window_end_ms   BIGINT NOT NULL,
    metadata        JSONB,                   -- any additional data (sub-signals, evidence, etc)
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Partitioning-ready indexes for time-series queries
CREATE INDEX idx_signals_session_time ON signals(session_id, window_start_ms);
CREATE INDEX idx_signals_session_agent ON signals(session_id, agent);
CREATE INDEX idx_signals_session_type ON signals(session_id, signal_type);
CREATE INDEX idx_signals_speaker ON signals(speaker_id, signal_type);


-- Alerts generated by FUSION Agent
CREATE TABLE alerts (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id      UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    speaker_id      UUID REFERENCES speakers(id),
    alert_type      VARCHAR(80) NOT NULL,    -- e.g. "masking_detected", "buying_signal_confirmed"
    severity        VARCHAR(20) NOT NULL,    -- green, yellow, orange, red
    title           VARCHAR(255) NOT NULL,
    description     TEXT,
    evidence        JSONB,                   -- contributing signals from each agent
    timestamp_ms    BIGINT NOT NULL,
    acknowledged    BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_alerts_session ON alerts(session_id, timestamp_ms);
CREATE INDEX idx_alerts_severity ON alerts(session_id, severity);


-- Compound patterns detected (COMPOUND-01 through COMPOUND-12)
CREATE TABLE compound_patterns (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id      UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    speaker_id      UUID REFERENCES speakers(id),
    pattern_id      VARCHAR(50) NOT NULL,    -- e.g. "COMPOUND-04" (decision_readiness)
    pattern_name    VARCHAR(100) NOT NULL,   -- e.g. "decision_readiness"
    domains_confirmed INT NOT NULL,          -- how many domains confirmed (3-6)
    confidence      FLOAT NOT NULL,
    start_ms        BIGINT NOT NULL,
    end_ms          BIGINT,
    evidence        JSONB,                   -- per-domain contributing signals
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_patterns_session ON compound_patterns(session_id, start_ms);


-- Temporal sequences detected (TEMPORAL-01 through TEMPORAL-08)
CREATE TABLE temporal_sequences (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id      UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    speaker_id      UUID REFERENCES speakers(id),
    sequence_id     VARCHAR(50) NOT NULL,    -- e.g. "TEMPORAL-04" (objection_formation)
    sequence_name   VARCHAR(100) NOT NULL,
    current_step    INT NOT NULL,
    total_steps     INT NOT NULL,
    confidence      FLOAT NOT NULL,
    trigger_ms      BIGINT NOT NULL,         -- when the sequence started
    current_ms      BIGINT NOT NULL,         -- current position in sequence
    status          VARCHAR(20) NOT NULL,    -- in_progress, completed, timed_out
    steps_data      JSONB,                   -- per-step signal data with timestamps
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_sequences_session ON temporal_sequences(session_id, trigger_ms);


-- ========================
-- 6. RULE ENGINE CONFIGURATION
-- ========================

-- Configurable thresholds and weights (adjustable without code changes)
CREATE TABLE rule_config (
    id              SERIAL PRIMARY KEY,
    rule_id         VARCHAR(50) NOT NULL,       -- e.g. "VOICE-STRESS-01"
    parameter       VARCHAR(100) NOT NULL,      -- e.g. "pitch_weight"
    value           FLOAT NOT NULL,             -- e.g. 0.25
    description     TEXT,
    updated_by      VARCHAR(100) DEFAULT 'system',
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(rule_id, parameter)
);

-- Domain weights for FUSION Agent
CREATE TABLE fusion_weights (
    id              SERIAL PRIMARY KEY,
    context         VARCHAR(50) NOT NULL DEFAULT 'default',  -- default, audio_only, video_only
    agent           VARCHAR(20) NOT NULL,       -- voice, language, facial, body, gaze
    weight          FLOAT NOT NULL,
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(context, agent)
);

-- Insert default domain weights
INSERT INTO fusion_weights (context, agent, weight) VALUES
    ('default', 'language', 0.40),
    ('default', 'voice', 0.25),
    ('default', 'facial', 0.20),
    ('default', 'body', 0.10),
    ('default', 'gaze', 0.05),
    ('audio_only', 'language', 0.50),
    ('audio_only', 'voice', 0.50),
    ('video_only', 'facial', 0.55),
    ('video_only', 'body', 0.30),
    ('video_only', 'gaze', 0.15);


-- ========================
-- 7. VECTOR STORAGE (pgvector)
-- ========================

-- Research knowledge base embeddings (for Claude API context)
CREATE TABLE research_embeddings (
    id              SERIAL PRIMARY KEY,
    rule_id         VARCHAR(50),                -- which rule this supports
    domain          VARCHAR(50),                -- voice, language, facial, body, gaze, conversation, fusion
    content         TEXT NOT NULL,              -- the text chunk
    source          VARCHAR(500),               -- citation
    embedding       vector(1536),               -- OpenAI text-embedding-3-small dimension
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Index for similarity search
CREATE INDEX idx_research_embedding ON research_embeddings 
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 20);


-- Speaker profile embeddings (for cross-session recognition & memory)
CREATE TABLE speaker_profiles (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    org_id          UUID REFERENCES organizations(id),
    name            VARCHAR(255) NOT NULL,
    email           VARCHAR(255),
    
    -- Aggregated baseline from multiple sessions
    avg_baseline    JSONB,                     -- averaged calibration data across sessions
    sessions_count  INT DEFAULT 0,
    
    -- Behavioural profile
    stress_triggers     JSONB,                 -- topics that historically trigger stress
    communication_style JSONB,                 -- power language score, empathy patterns, etc
    decision_pattern    JSONB,                 -- average time to decision, buying sequence stage history
    
    -- Voice embedding for speaker recognition
    voice_embedding     vector(256),           -- speaker embedding from voice model
    
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_speaker_profiles_org ON speaker_profiles(org_id);
CREATE INDEX idx_speaker_voice_embed ON speaker_profiles 
    USING ivfflat (voice_embedding vector_cosine_ops) WITH (lists = 10);


-- ========================
-- 8. REPORTS
-- ========================

CREATE TABLE session_reports (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id      UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    report_type     VARCHAR(50) NOT NULL DEFAULT 'post_session',  
                    -- post_session, coaching, executive_summary
    content         JSONB NOT NULL,            -- structured report data
    narrative       TEXT,                      -- Claude API generated narrative
    generated_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_reports_session ON session_reports(session_id);


-- ========================
-- 9. FEEDBACK LOOP
-- ========================

-- Human feedback on signal accuracy (for threshold calibration)
CREATE TABLE signal_feedback (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id      UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    signal_id       BIGINT REFERENCES signals(id),
    alert_id        UUID REFERENCES alerts(id),
    feedback_type   VARCHAR(20) NOT NULL,       -- accurate, inaccurate, uncertain
    notes           TEXT,
    given_by        UUID REFERENCES users(id),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_feedback_session ON signal_feedback(session_id);


-- ========================
-- 10. SEED: Initial Rule Config (VOICE Agent - 5 core rules for vertical slice)
-- ========================

-- VOICE-STRESS-01 weights
INSERT INTO rule_config (rule_id, parameter, value, description) VALUES
    ('VOICE-STRESS-01', 'pitch_weight', 0.25, 'Weight for F0 delta in stress composite'),
    ('VOICE-STRESS-01', 'jitter_weight', 0.20, 'Weight for jitter delta'),
    ('VOICE-STRESS-01', 'rate_weight', 0.15, 'Weight for speech rate change'),
    ('VOICE-STRESS-01', 'filler_weight', 0.15, 'Weight for filler rate increase'),
    ('VOICE-STRESS-01', 'pause_weight', 0.10, 'Weight for pause frequency change'),
    ('VOICE-STRESS-01', 'hnr_weight', 0.10, 'Weight for HNR decrease'),
    ('VOICE-STRESS-01', 'shimmer_weight', 0.05, 'Weight for shimmer change'),
    ('VOICE-STRESS-01', 'threshold_moderate', 0.30, 'Score above this = moderate stress'),
    ('VOICE-STRESS-01', 'threshold_elevated', 0.50, 'Score above this = elevated stress'),
    ('VOICE-STRESS-01', 'threshold_high', 0.70, 'Score above this = high stress');

-- VOICE-FILLER-01 thresholds
INSERT INTO rule_config (rule_id, parameter, value, description) VALUES
    ('VOICE-FILLER-01', 'spike_threshold_pct', 50.0, 'Filler rate increase % to flag spike'),
    ('VOICE-FILLER-02', 'credibility_noticeable_pct', 1.30, 'Filler % above which credibility impacted'),
    ('VOICE-FILLER-02', 'credibility_significant_pct', 2.50, 'Filler % for significant impact'),
    ('VOICE-FILLER-02', 'credibility_severe_pct', 4.00, 'Filler % for severe impact');

-- VOICE-PITCH-01 thresholds
INSERT INTO rule_config (rule_id, parameter, value, description) VALUES
    ('VOICE-PITCH-01', 'mild_elevation_pct', 8.0, 'F0 increase % for mild flag'),
    ('VOICE-PITCH-01', 'significant_elevation_pct', 15.0, 'F0 increase % for significant flag'),
    ('VOICE-PITCH-01', 'extreme_elevation_pct', 25.0, 'F0 increase % for extreme flag'),
    ('VOICE-PITCH-01', 'sustained_min_ms', 3000, 'Minimum duration in ms for significant flag');

-- VOICE-RATE-01 thresholds
INSERT INTO rule_config (rule_id, parameter, value, description) VALUES
    ('VOICE-RATE-01', 'elevated_threshold_pct', 25.0, 'Rate increase % to flag elevated'),
    ('VOICE-RATE-01', 'depressed_threshold_pct', -25.0, 'Rate decrease % to flag depressed'),
    ('VOICE-RATE-01', 'persist_min_ms', 15000, 'Minimum duration for confidence boost');

-- VOICE-CAL-01 calibration config
INSERT INTO rule_config (rule_id, parameter, value, description) VALUES
    ('VOICE-CAL-01', 'min_speech_sec_initial', 60, 'Seconds of speech for initial baseline'),
    ('VOICE-CAL-01', 'min_speech_sec_reliable', 180, 'Seconds for reliable baseline'),
    ('VOICE-CAL-01', 'ema_alpha', 0.10, 'Exponential moving average alpha for baseline update'),
    ('VOICE-CAL-01', 'confidence_60s', 0.30, 'Calibration confidence at 60s'),
    ('VOICE-CAL-01', 'confidence_120s', 0.60, 'Calibration confidence at 120s'),
    ('VOICE-CAL-01', 'confidence_180s', 0.80, 'Calibration confidence at 180s'),
    ('VOICE-CAL-01', 'confidence_300s', 0.95, 'Calibration confidence at 300s');

-- LANG-SENT-01 config
INSERT INTO rule_config (rule_id, parameter, value, description) VALUES
    ('LANG-SENT-01', 'model_weight', 0.70, 'Weight for ML model sentiment'),
    ('LANG-SENT-01', 'liwc_weight', 0.30, 'Weight for LIWC dictionary sentiment');

-- LANG-BUY-01 config
INSERT INTO rule_config (rule_id, parameter, value, description) VALUES
    ('LANG-BUY-01', 'single_signal_score', 0.40, 'Score for single buying signal type'),
    ('LANG-BUY-01', 'dual_signal_score', 0.70, 'Score for two buying signal types'),
    ('LANG-BUY-01', 'multi_signal_score', 0.90, 'Score for 3+ buying signal types'),
    ('LANG-BUY-01', 'hypothetical_penalty', -0.20, 'Confidence penalty for hypothetical phrasing');

-- FUSION-02 config
INSERT INTO rule_config (rule_id, parameter, value, description) VALUES
    ('FUSION-02', 'power_lang_threshold', 0.70, 'Power language score threshold for conflict check'),
    ('FUSION-02', 'stress_threshold', 0.50, 'Vocal stress threshold for conflict check'),
    ('FUSION-02', 'congruent_bonus', 0.20, 'Confidence bonus when both channels agree');


-- ========================
-- DONE
-- ========================
-- Schema version tracking
CREATE TABLE schema_version (
    version     INT PRIMARY KEY,
    applied_at  TIMESTAMPTZ DEFAULT NOW(),
    description TEXT
);

INSERT INTO schema_version (version, description) VALUES 
    (1, 'Initial schema - vertical slice: PostgreSQL + pgvector');
