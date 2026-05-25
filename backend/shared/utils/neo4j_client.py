"""
Neo4j async client for NEXUS knowledge graph.
Singleton pattern — call get_neo4j_driver() from anywhere.
"""
import os
import logging
from typing import Optional

logger = logging.getLogger("nexus.neo4j")

_driver = None


async def get_neo4j_driver():
    global _driver
    if _driver is None:
        try:
            from neo4j import AsyncGraphDatabase
        except ImportError:
            logger.warning("neo4j package not installed — graph features disabled")
            return None
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "nexus_graph_2026")
        try:
            _driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
            await _driver.verify_connectivity()
            logger.info(f"Neo4j connected: {uri}")
        except Exception as e:
            logger.warning(f"Neo4j not available: {e}. Graph features disabled.")
            _driver = None
    return _driver


async def close_neo4j():
    global _driver
    if _driver:
        await _driver.close()
        _driver = None


async def run_query(cypher: str, **params) -> list[dict]:
    """Execute a read-only Cypher query and return list of record dicts."""
    driver = await get_neo4j_driver()
    if not driver:
        return []
    try:
        async with driver.session() as session:
            result = await session.run(cypher, **params)
            return await result.data()
    except Exception as e:
        logger.error(f"Neo4j query failed: {e}\nCypher: {cypher[:300]}")
        return []
