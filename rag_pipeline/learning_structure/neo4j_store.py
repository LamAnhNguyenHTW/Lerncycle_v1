"""Safe Neo4j writes for the namespaced Learning Graph layer."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from rag_pipeline.learning_structure.models import ChunkForExtraction, LearningTreeNode
from rag_pipeline.learning_structure.normalizer import normalize_title


GRAPH_SCOPE = "learning_structure"
_HARD_ARTIFACT_TITLES = {
    "agenda",
    "themen",
    "gliederung",
    "inhaltsverzeichnis",
    "outline",
    "recap",
    "vielen dank",
    "danke",
    "fragen",
    "q a",
    "kontakt",
    "ueber mich",
    "uber mich",
    "über mich",
    "about",
    "disclaimer",
}


class LearningGraphValidationError(ValueError):
    """Raised when invalid final learning-tree data reaches the Neo4j sink."""


def ensure_constraints(driver: Any, database: str | None = None) -> None:
    """Create id-based Learning Graph constraints without conflicting with GraphRAG."""
    statements = [
        "CREATE CONSTRAINT learning_topic_id IF NOT EXISTS FOR (t:LearningTopic) REQUIRE t.id IS UNIQUE",
        "CREATE CONSTRAINT learning_concept_id IF NOT EXISTS FOR (c:LearningConcept) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT learning_objective_id IF NOT EXISTS FOR (o:LearningObjective) REQUIRE o.id IS UNIQUE",
        "CREATE CONSTRAINT document_user_source IF NOT EXISTS FOR (d:Document) REQUIRE (d.user_id, d.source_type, d.source_id) IS UNIQUE",
        "CREATE CONSTRAINT chunk_user_id IF NOT EXISTS FOR (c:Chunk) REQUIRE (c.user_id, c.chunk_id) IS UNIQUE",
    ]
    for statement in statements:
        _run(driver, statement, database=database)


def write_learning_graph(
    user_id: str,
    document_meta: dict[str, Any],
    tree: LearningTreeNode,
    chunks: list[ChunkForExtraction],
    *,
    driver: Any,
    database: str | None = None,
) -> None:
    """Replace one user's Learning Graph for a source without touching GraphRAG data."""
    source_id = str(document_meta["source_id"])
    source_type = str(document_meta.get("source_type") or "pdf")
    base = {"user_id": user_id, "source_id": source_id, "source_type": source_type}
    _validate_tree_for_write(tree)
    _delete_existing_learning_layer(driver, base, database)
    _merge_document(driver, document_meta, base, database)
    _merge_chunks(driver, chunks, base, database)
    for child in tree.children:
        _write_node_recursive(driver, child, parent=None, base=base, database=database)


def _delete_existing_learning_layer(driver: Any, base: dict[str, str], database: str | None) -> None:
    _run(
        driver,
        """
        MATCH ()-[r {graph_scope: 'learning_structure'}]->()
        WHERE r.user_id = $user_id AND r.source_id = $source_id
        DELETE r
        """,
        base,
        database,
    )
    _run(
        driver,
        """
        MATCH (n)
        WHERE (n:LearningTopic OR n:LearningConcept OR n:LearningObjective)
          AND n.user_id = $user_id
          AND n.source_id = $source_id
          AND n.graph_scope = 'learning_structure'
        DETACH DELETE n
        """,
        base,
        database,
    )


def _merge_document(
    driver: Any,
    document_meta: dict[str, Any],
    base: dict[str, str],
    database: str | None,
) -> None:
    _run(
        driver,
        """
        MERGE (doc:Document {user_id: $user_id, source_type: $source_type, source_id: $source_id})
        SET doc.title = $title,
            doc.updated_at = $updated_at
        """,
        {
            **base,
            "title": document_meta.get("title"),
            "updated_at": _utc_now(),
        },
        database,
    )


def _merge_chunks(
    driver: Any,
    chunks: list[ChunkForExtraction],
    base: dict[str, str],
    database: str | None,
) -> None:
    for chunk in chunks:
        chunk_payload = {
            "chunk_id": chunk.chunk_id,
            "page_index": chunk.page_index,
            "heading_path": chunk.heading_path,
            "content_hash": chunk.content_hash,
        }
        _run(
            driver,
            """
            WITH $chunk AS chunk_payload
            MERGE (chunk:Chunk {user_id: $user_id, chunk_id: chunk_payload.chunk_id})
            SET chunk.source_id = $source_id,
                chunk.source_type = $source_type,
                chunk.page_index = chunk_payload.page_index,
                chunk.heading_path = chunk_payload.heading_path,
                chunk.content_hash = chunk_payload.content_hash,
                chunk.updated_at = $updated_at
            """,
            {**base, "chunk": chunk_payload, "updated_at": _utc_now()},
            database,
        )


def _write_node_recursive(
    driver: Any,
    node: LearningTreeNode,
    parent: LearningTreeNode | None,
    base: dict[str, str],
    database: str | None,
) -> None:
    if node.type in {"topic", "subtopic"}:
        _write_topic(driver, node, parent, base, database)
    elif node.type == "concept":
        _write_leaf(driver, node, parent, "LearningConcept", "HAS_LEARNING_CONCEPT", base, database)
    elif node.type == "objective":
        _write_leaf(driver, node, parent, "LearningObjective", "HAS_OBJECTIVE", base, database)
    for child in node.children:
        _write_node_recursive(driver, child, parent=node, base=base, database=database)


def _write_topic(
    driver: Any,
    node: LearningTreeNode,
    parent: LearningTreeNode | None,
    base: dict[str, str],
    database: str | None,
) -> None:
    payload = _node_payload(node, base)
    _run(
        driver,
        """
        MERGE (topic:LearningTopic {id: $node.id})
        SET topic += $node
        WITH topic
        MATCH (doc:Document {user_id: $user_id, source_type: $source_type, source_id: $source_id})
        FOREACH (_ IN CASE WHEN $parent_id IS NULL THEN [1] ELSE [] END |
          MERGE (doc)-[rel:HAS_LEARNING_TOPIC {graph_scope: 'learning_structure', user_id: $user_id, source_id: $source_id}]->(topic)
          SET rel.order_index = $node.order_index
        )
        FOREACH (_ IN CASE WHEN $parent_id IS NOT NULL THEN [1] ELSE [] END |
          MERGE (parent:LearningTopic {id: $parent_id})
          MERGE (parent)-[rel:HAS_SUBTOPIC {graph_scope: 'learning_structure', user_id: $user_id, source_id: $source_id}]->(topic)
          SET rel.order_index = $node.order_index
        )
        """,
        {**base, "node": payload, "parent_id": parent.id if parent else None},
        database,
    )
    _write_supported_by(driver, node, base, database)


def _write_leaf(
    driver: Any,
    node: LearningTreeNode,
    parent: LearningTreeNode | None,
    label: str,
    rel_type: str,
    base: dict[str, str],
    database: str | None,
) -> None:
    if parent is None:
        return
    payload = _node_payload(node, base)
    _run(
        driver,
        f"""
        MERGE (leaf:{label} {{id: $node.id}})
        SET leaf += $node
        WITH leaf
        MATCH (parent:LearningTopic {{id: $parent_id, user_id: $user_id}})
        MERGE (parent)-[rel:{rel_type} {{graph_scope: 'learning_structure', user_id: $user_id, source_id: $source_id}}]->(leaf)
        SET rel.order_index = $node.order_index
        """,
        {**base, "node": payload, "parent_id": parent.id},
        database,
    )
    _write_supported_by(driver, node, base, database)


def _write_supported_by(
    driver: Any,
    node: LearningTreeNode,
    base: dict[str, str],
    database: str | None,
) -> None:
    if not node.chunk_ids:
        raise ValueError(f"Learning node lacks chunk evidence: {node.id}")
    _run(
        driver,
        """
        MATCH (node {id: $node_id, user_id: $user_id})
        WITH node
        UNWIND $chunk_ids AS chunk_id
        MATCH (chunk:Chunk {user_id: $user_id, chunk_id: chunk_id})
        MERGE (node)-[rel:SUPPORTED_BY {graph_scope: 'learning_structure', user_id: $user_id, source_id: $source_id, chunk_id: chunk_id}]->(chunk)
        """,
        {**base, "node_id": node.id, "chunk_ids": node.chunk_ids},
        database,
    )


def _node_payload(node: LearningTreeNode, base: dict[str, str]) -> dict[str, Any]:
    return {
        "id": node.id,
        "title": node.label if node.type in {"topic", "subtopic"} else None,
        "normalized_title": normalize_title(node.label) if node.type in {"topic", "subtopic"} else None,
        "name": node.label if node.type == "concept" else None,
        "normalized_name": normalize_title(node.label) if node.type == "concept" else None,
        "objective": node.label if node.type == "objective" else None,
        "normalized_objective": normalize_title(node.label) if node.type == "objective" else None,
        "summary": node.summary,
        "level": _topic_level(node) if node.type in {"topic", "subtopic"} else None,
        "page_start": node.page_start,
        "page_end": node.page_end,
        "confidence": node.confidence,
        "order_index": node.order_index,
        "chunk_ids": node.chunk_ids,
        "user_id": base["user_id"],
        "source_id": base["source_id"],
        "source_type": base["source_type"],
        "graph_scope": GRAPH_SCOPE,
        "updated_at": _utc_now(),
    }


def _validate_tree_for_write(tree: LearningTreeNode) -> None:
    if tree.type != "document":
        raise LearningGraphValidationError("Learning graph root must be a document node")
    for child in tree.children:
        _validate_node_for_write(child, parent=None)


def _validate_node_for_write(node: LearningTreeNode, parent: LearningTreeNode | None) -> None:
    if node.type in {"topic", "subtopic"}:
        expected_type = "topic" if parent is None or parent.type == "document" else "subtopic"
        if node.type != expected_type:
            raise LearningGraphValidationError(f"Learning topic level mismatch for node {node.id}")
        if not node.chunk_ids:
            raise LearningGraphValidationError(f"Learning topic lacks chunk evidence: {node.id}")
        if not node.summary or len(node.summary.strip()) < 40:
            raise LearningGraphValidationError(f"Learning topic lacks non-trivial summary: {node.id}")
        if normalize_title(node.label) in _HARD_ARTIFACT_TITLES:
            raise LearningGraphValidationError(f"Learning topic title is denied: {node.label}")
    elif node.type in {"concept", "objective"}:
        if not node.chunk_ids:
            raise LearningGraphValidationError(f"Learning leaf lacks chunk evidence: {node.id}")
        if parent is None or parent.type not in {"topic", "subtopic"}:
            raise LearningGraphValidationError(f"Learning leaf lacks topic parent: {node.id}")
    elif node.type != "document":
        raise LearningGraphValidationError(f"Unsupported learning node type: {node.type}")

    for child in node.children:
        _validate_node_for_write(child, parent=node)


def _topic_level(node: LearningTreeNode) -> str:
    return "topic" if node.type == "topic" else "subtopic"


def _run(
    driver: Any,
    statement: str,
    parameters: dict[str, Any] | None = None,
    database: str | None = None,
) -> Any:
    session = driver.session(database=database)
    if hasattr(session, "__enter__"):
        with session as active_session:
            return active_session.run(statement, parameters or {})
    try:
        return session.run(statement, parameters or {})
    finally:
        close = getattr(session, "close", None)
        if close:
            close()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
