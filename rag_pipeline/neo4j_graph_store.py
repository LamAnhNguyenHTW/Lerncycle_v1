"""Neo4j persistence and retrieval for user-scoped knowledge graphs."""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from typing import Any

from rag_pipeline.graph_schema import GraphExtraction
from rag_pipeline.graph_schema import normalize_node_name


def _transient_error_types() -> tuple[type[Exception], ...]:
    """Neo4j transient errors worth retrying; empty when neo4j is absent (tests)."""
    try:
        from neo4j.exceptions import ServiceUnavailable, SessionExpired
    except ImportError:
        return ()
    return (ServiceUnavailable, SessionExpired)


_TRANSIENT_ERRORS = _transient_error_types()
_MAX_RETRIES = 3
_RETRY_BACKOFF_SECONDS = 0.5
_CONCEPT_QUERY_STOPWORDS = {
    "a", "an", "and", "are", "as", "between", "by", "did", "do", "does",
    "for", "from", "how", "in", "into", "is", "of", "on", "or", "the",
    "these", "to", "together", "was", "were", "what", "when", "where",
    "which", "why", "with", "without", "das", "dem", "den", "der", "des",
    "die", "ein", "eine", "einem", "einen", "einer", "eines", "für", "hängen",
    "ist", "mit", "nach", "oder", "sind", "und", "von", "warum", "welche",
    "welcher", "welches", "werden", "wie", "wird", "wurden", "zwischen",
    "zusammen", "zum", "zur",
}


def _concept_search_terms(query: str) -> list[str]:
    """Return deterministic content terms for lexical Concept lookup."""
    terms: list[str] = []
    for token in re.findall(r"[\w-]+", query.casefold(), flags=re.UNICODE):
        if len(token) < 3 or token in _CONCEPT_QUERY_STOPWORDS or token in terms:
            continue
        terms.append(token)
    return terms[:20]


class Neo4jGraphStore:
    """Small Neo4j adapter with injectable driver for offline tests."""

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str | None = None,
        driver: Any = None,
    ) -> None:
        self.database = database
        if driver is None:
            from neo4j import GraphDatabase

            driver = GraphDatabase.driver(
                uri,
                auth=(user, password),
                max_connection_lifetime=300,
                keep_alive=True,
            )
        self.driver = driver

    def ensure_constraints(self) -> None:
        """Create Neo4j constraints and indexes used by GraphRAG."""
        statements = [
            "CREATE CONSTRAINT concept_user_name_type IF NOT EXISTS FOR (c:Concept) REQUIRE (c.user_id, c.normalized_name, c.node_type) IS UNIQUE",
            "CREATE CONSTRAINT chunk_user_id IF NOT EXISTS FOR (c:Chunk) REQUIRE (c.user_id, c.chunk_id) IS UNIQUE",
            "CREATE INDEX concept_user_id IF NOT EXISTS FOR (c:Concept) ON (c.user_id)",
            "CREATE INDEX concept_normalized_name IF NOT EXISTS FOR (c:Concept) ON (c.normalized_name)",
            "CREATE INDEX chunk_user_id IF NOT EXISTS FOR (c:Chunk) ON (c.user_id)",
            "CREATE INDEX chunk_source IF NOT EXISTS FOR (c:Chunk) ON (c.source_type, c.source_id)",
        ]
        for statement in statements:
            self._run(statement)

    def upsert_extraction(
        self,
        user_id: str,
        chunk: dict[str, Any],
        extraction: GraphExtraction,
    ) -> dict[str, int]:
        """Upsert a chunk, concepts, and grounded relationships."""
        chunk_payload = _chunk_payload(user_id, chunk)
        self._run(
            """
            MERGE (chunk:Chunk {user_id: $chunk.user_id, chunk_id: $chunk.chunk_id})
            SET chunk += $chunk
            """,
            {"chunk": chunk_payload},
        )
        node_count = 0
        for node in extraction.nodes:
            self._run(
                """
                MERGE (concept:Concept {
                  user_id: $user_id,
                  normalized_name: $normalized_name,
                  node_type: $node_type
                })
                SET concept.name = $name,
                    concept.description = $description,
                    concept.confidence = $confidence,
                    concept.updated_at = $updated_at
                WITH concept
                MATCH (chunk:Chunk {user_id: $user_id, chunk_id: $chunk_id})
                MERGE (concept)-[mentioned:MENTIONED_IN {
                  user_id: $user_id,
                  chunk_id: $chunk_id,
                  source_type: $source_type,
                  source_id: $source_id
                }]->(chunk)
                SET mentioned.page_index = $page_index
                """,
                {
                    "user_id": user_id,
                    "normalized_name": node.normalized_name,
                    "node_type": node.node_type,
                    "name": node.name,
                    "description": node.description,
                    "confidence": node.confidence,
                    "updated_at": _utc_now(),
                    "chunk_id": chunk_payload["chunk_id"],
                    "source_type": chunk_payload["source_type"],
                    "source_id": chunk_payload["source_id"],
                    "page_index": chunk_payload["page_index"],
                },
            )
            node_count += 1

        edge_count = 0
        for edge in extraction.edges:
            self._run(
                """
                MATCH (source:Concept {user_id: $user_id, normalized_name: $source_name})
                MATCH (target:Concept {user_id: $user_id, normalized_name: $target_name})
                MERGE (source)-[rel:RELATED {
                  user_id: $user_id,
                  source_name: $source_name,
                  target_name: $target_name,
                  relation_type: $relation_type,
                  chunk_id: $chunk_id
                }]->(target)
                SET rel.description = $description,
                    rel.confidence = $confidence,
                    rel.source_type = $source_type,
                    rel.source_id = $source_id,
                    rel.pdf_id = $pdf_id,
                    rel.page_index = $page_index,
                    rel.heading = $heading,
                    rel.snippet = $snippet,
                    rel.updated_at = $updated_at
                """,
                {
                    "user_id": user_id,
                    "source_name": normalize_node_name(edge.source),
                    "target_name": normalize_node_name(edge.target),
                    "relation_type": edge.relation_type,
                    "description": edge.description,
                    "confidence": edge.confidence,
                    "chunk_id": chunk_payload["chunk_id"],
                    "source_type": chunk_payload["source_type"],
                    "source_id": chunk_payload["source_id"],
                    "pdf_id": chunk_payload["pdf_id"],
                    "page_index": chunk_payload["page_index"],
                    "heading": chunk_payload["heading"],
                    "snippet": chunk_payload["snippet"],
                    "updated_at": _utc_now(),
                },
            )
            edge_count += 1
        return {"nodes_upserted": node_count, "relationships_upserted": edge_count}

    def delete_by_source(self, user_id: str, source_type: str, source_id: str) -> None:
        """Delete graph data for one source and clean orphan concepts."""
        params = {"user_id": user_id, "source_type": source_type, "source_id": source_id}
        self._run("MATCH ()-[r:RELATED {user_id: $user_id, source_type: $source_type, source_id: $source_id}]-() DELETE r", params)
        self._run("MATCH ()-[r:MENTIONED_IN {user_id: $user_id, source_type: $source_type, source_id: $source_id}]-() DELETE r", params)
        # DETACH: learning-structure SUPPORTED_BY relationships may still attach
        # to shared Chunk nodes; a plain DELETE would fail in that case.
        self._run("MATCH (chunk:Chunk {user_id: $user_id, source_type: $source_type, source_id: $source_id}) DETACH DELETE chunk", params)
        self._run("MATCH (concept:Concept {user_id: $user_id}) WHERE NOT (concept)--() DELETE concept", {"user_id": user_id})

    def delete_by_pdf_id(self, user_id: str, pdf_id: str) -> None:
        """Delete graph data for one PDF including its note/annotation chunks.

        Chunk nodes and RELATED relationships carry `pdf_id` for every
        source_type derived from the PDF, so this covers dependent note and
        annotation_comment graph data in one pass.
        """
        params = {"user_id": user_id, "pdf_id": pdf_id}
        self._run("MATCH ()-[r:RELATED {user_id: $user_id}]-() WHERE r.pdf_id = $pdf_id DELETE r", params)
        self._run("MATCH (chunk:Chunk {user_id: $user_id}) WHERE chunk.pdf_id = $pdf_id DETACH DELETE chunk", params)
        self._run("MATCH (concept:Concept {user_id: $user_id}) WHERE NOT (concept)--() DELETE concept", {"user_id": user_id})

    def search_concepts(
        self,
        user_id: str,
        query: str,
        source_types: list[str] | None = None,
        source_ids: list[str] | None = None,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Search concepts by substring, scoped by user and optional source filters."""
        terms = _concept_search_terms(query)
        if not terms:
            return []
        return self._records(
            """
            MATCH (concept:Concept {user_id: $user_id})
            WHERE any(term IN $terms WHERE toLower(concept.name) CONTAINS term)
            MATCH (concept)-[:MENTIONED_IN]->(chunk:Chunk {user_id: $user_id})
            WHERE ($source_types IS NULL OR chunk.source_type IN $source_types)
              AND ($source_ids IS NULL OR chunk.source_id IN $source_ids)
            WITH DISTINCT concept,
                 size([term IN $terms WHERE toLower(concept.name) CONTAINS term]) AS match_count
            WITH concept.normalized_name AS normalized_name,
                 head(collect(concept)) AS concept,
                 max(match_count) AS match_count
            ORDER BY match_count DESC, concept.name
            RETURN DISTINCT concept.name AS name,
                   concept.normalized_name AS normalized_name,
                   concept.node_type AS node_type,
                   null AS description
            LIMIT $top_k
            """,
            {
                "user_id": user_id,
                "terms": terms,
                "source_types": source_types,
                "source_ids": source_ids,
                "top_k": top_k,
            },
        )

    def get_neighborhood(
        self,
        user_id: str,
        concept_names: list[str],
        max_depth: int = 1,
        limit: int = 30,
        source_types: list[str] | None = None,
        source_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return relationship neighborhood around concept names."""
        records = self._records(
            """
            MATCH (source:Concept {user_id: $user_id})-[rel:RELATED]-(target:Concept {user_id: $user_id})
            WHERE source.normalized_name IN $concept_names
              AND ($source_types IS NULL OR rel.source_type IN $source_types)
              AND ($source_ids IS NULL OR rel.source_id IN $source_ids)
            RETURN source.name AS source,
                   target.name AS target,
                   rel.relation_type AS relation_type,
                   null AS description,
                   rel.chunk_id AS chunk_id,
                   rel.source_type AS source_type,
                   rel.source_id AS source_id,
                   rel.page_index AS page_index,
                   rel.snippet AS snippet
            LIMIT $limit
            """,
            {
                "user_id": user_id,
                "concept_names": [normalize_node_name(name) for name in concept_names],
                "max_depth": max_depth,
                "source_types": source_types,
                "source_ids": source_ids,
                "limit": limit,
            },
        )
        return {"relationships": records}

    def find_path_between_concepts(
        self,
        user_id: str,
        source_name: str,
        target_name: str,
        max_depth: int = 3,
        source_types: list[str] | None = None,
        source_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Find a shortest concept path scoped by user."""
        records = self._records(
            """
            MATCH path = shortestPath(
              (source:Concept {user_id: $user_id, normalized_name: $source_name})
              -[:RELATED*..3]-
              (target:Concept {user_id: $user_id, normalized_name: $target_name})
            )
            RETURN path
            LIMIT 1
            """,
            {
                "user_id": user_id,
                "source_name": normalize_node_name(source_name),
                "target_name": normalize_node_name(target_name),
                "max_depth": max_depth,
                "source_types": source_types,
                "source_ids": source_ids,
            },
        )
        return {"paths": records}

    def _records(self, statement: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        return self._execute(statement, parameters, materialize=True)

    def _run(
        self,
        statement: str,
        parameters: dict[str, Any] | None = None,
    ) -> Any:
        return self._execute(statement, parameters, materialize=False)

    def _execute(
        self,
        statement: str,
        parameters: dict[str, Any] | None,
        materialize: bool,
    ) -> Any:
        # All statements are idempotent (MERGE / CREATE IF NOT EXISTS / DELETE /
        # read-only MATCH), so retrying on a dropped connection is safe. A fresh
        # session per attempt pulls a live connection from the pool.
        for attempt in range(_MAX_RETRIES):
            try:
                return self._run_once(statement, parameters or {}, materialize)
            except _TRANSIENT_ERRORS:
                if attempt + 1 >= _MAX_RETRIES:
                    raise
                time.sleep(_RETRY_BACKOFF_SECONDS * (attempt + 1))

    def _run_once(
        self,
        statement: str,
        parameters: dict[str, Any],
        materialize: bool,
    ) -> Any:
        session = self.driver.session(database=self.database)
        if hasattr(session, "__enter__"):
            with session as active_session:
                result = active_session.run(statement, parameters)
                return _materialize_records(result) if materialize else result
        try:
            result = session.run(statement, parameters)
            return _materialize_records(result) if materialize else result
        finally:
            close = getattr(session, "close", None)
            if close:
                close()


def _materialize_records(result: Any) -> list[dict[str, Any]]:
    records = []
    for record in result or []:
        if hasattr(record, "data"):
            records.append(record.data())
        elif isinstance(record, dict):
            records.append(record)
    return records


def _chunk_payload(user_id: str, chunk: dict[str, Any]) -> dict[str, Any]:
    text = str(chunk.get("text") or chunk.get("content") or "")
    return {
        "user_id": user_id,
        "chunk_id": str(chunk.get("chunk_id") or chunk.get("id")),
        "source_type": chunk.get("source_type"),
        "source_id": chunk.get("source_id"),
        "pdf_id": chunk.get("pdf_id"),
        "page_index": chunk.get("page_index"),
        "heading": chunk.get("heading"),
        "snippet": " ".join(text.split())[:300],
        "updated_at": _utc_now(),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
