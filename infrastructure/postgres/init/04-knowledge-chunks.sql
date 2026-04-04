-- ============================================================================
-- NEXUS Schema - Migration 04
-- Adds knowledge_chunks table for RAG-based session Q&A
-- Embedding dimension varies by provider (768 for Ollama/nomic, 1536 for OpenAI)
-- ============================================================================

CREATE TABLE IF NOT EXISTS knowledge_chunks (
    id VARCHAR(64) PRIMARY KEY,
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    chunk_type VARCHAR(50) NOT NULL,
    text TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    embedding vector,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_session ON knowledge_chunks(session_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_type ON knowledge_chunks(chunk_type);

INSERT INTO schema_version (version, description) VALUES
    (4, 'Knowledge chunks table for RAG chat')
ON CONFLICT (version) DO NOTHING;
