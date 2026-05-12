"""Factory for optional graph store construction."""

from __future__ import annotations

from typing import Any

from rag_pipeline.neo4j_graph_store import Neo4jGraphStore


def create_graph_store(config: Any, driver: Any = None) -> Neo4jGraphStore | None:
    """Create a graph store when any graph feature is enabled."""
    enabled = any(
        bool(getattr(config, name, False))
        for name in ("graph_enabled", "graph_extraction_enabled", "graph_retrieval_enabled")
    )
    if not enabled:
        return None
    if getattr(config, "graph_store_provider", "neo4j") != "neo4j":
        raise ValueError("GRAPH_STORE_PROVIDER must be neo4j")
    missing = [
        name
        for name, value in {
            "NEO4J_URI": getattr(config, "neo4j_uri", None),
            "NEO4J_USER": getattr(config, "neo4j_user", None),
            "NEO4J_PASSWORD": getattr(config, "neo4j_password", None),
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError("Missing Neo4j environment variables: " + ", ".join(missing))
    store = Neo4jGraphStore(
        uri=config.neo4j_uri,
        user=config.neo4j_user,
        password=config.neo4j_password,
        database=getattr(config, "neo4j_database", "neo4j"),
        driver=driver,
    )
    store.ensure_constraints()
    return store
