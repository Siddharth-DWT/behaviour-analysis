# services/api_gateway/neo4j_schema.py
"""
Neo4j index and constraint initialisation for NEXUS.
Called once at API Gateway startup — idempotent (IF NOT EXISTS guards).
"""
import logging

logger = logging.getLogger("nexus.gateway.neo4j_schema")

_SCHEMA_QUERIES = [
    # Uniqueness constraints
    "CREATE CONSTRAINT session_id IF NOT EXISTS FOR (s:Session) REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
    "CREATE CONSTRAINT signal_id IF NOT EXISTS FOR (s:Signal) REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT segment_id IF NOT EXISTS FOR (s:Segment) REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT speaker_id IF NOT EXISTS FOR (s:Speaker) REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT topic_id IF NOT EXISTS FOR (t:Topic) REQUIRE t.id IS UNIQUE",
    # Lookup indexes
    "CREATE INDEX signal_session IF NOT EXISTS FOR (s:Signal) ON (s.session_id)",
    "CREATE INDEX signal_type IF NOT EXISTS FOR (s:Signal) ON (s.signal_type)",
    "CREATE INDEX signal_speaker IF NOT EXISTS FOR (s:Signal) ON (s.speaker_label)",
    "CREATE INDEX signal_timestamp IF NOT EXISTS FOR (s:Signal) ON (s.timestamp_ms)",
    "CREATE INDEX segment_session IF NOT EXISTS FOR (s:Segment) ON (s.session_id)",
    "CREATE INDEX segment_speaker IF NOT EXISTS FOR (s:Segment) ON (s.speaker_label)",
    "CREATE INDEX topic_session IF NOT EXISTS FOR (t:Topic) ON (t.session_id)",
    "CREATE INDEX entity_session IF NOT EXISTS FOR (e:Entity) ON (e.session_id)",
    "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
    "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
    "CREATE INDEX speaker_session IF NOT EXISTS FOR (s:Speaker) ON (s.session_id)",
    "CREATE INDEX alert_session IF NOT EXISTS FOR (a:Alert) ON (a.session_id)",
    # ── Relationship property indexes for faster edge lookups (Neo4j 5.x) ──
    "CREATE INDEX rel_replied_gap IF NOT EXISTS FOR ()-[r:REPLIED_TO]-() ON (r.gap_ms)",
    "CREATE INDEX rel_influenced_lag IF NOT EXISTS FOR ()-[r:INFLUENCED]-() ON (r.lag_ms)",
]


async def init_neo4j_schema():
    """Create indexes and constraints. Safe to call on every startup."""
    try:
        from shared.utils.neo4j_client import get_neo4j_driver
    except ImportError:
        logger.warning("neo4j_client not available — skipping schema init")
        return

    driver = await get_neo4j_driver()
    if not driver:
        logger.warning("Neo4j unavailable — skipping schema init")
        return

    ok = 0
    async with driver.session() as session:
        for q in _SCHEMA_QUERIES:
            try:
                await session.run(q)
                ok += 1
            except Exception as e:
                logger.debug(f"Schema query skipped (may already exist): {e}")

    logger.info(f"Neo4j schema initialised ({ok}/{len(_SCHEMA_QUERIES)} queries ran)")
