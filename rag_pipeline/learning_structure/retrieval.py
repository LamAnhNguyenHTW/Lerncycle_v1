"""Read Learning Graph trees from Neo4j."""

from __future__ import annotations

from typing import Any

from rag_pipeline.learning_structure.models import LearningTreeNode


def get_document_learning_tree(
    user_id: str,
    source_id: str,
    *,
    driver: Any,
    database: str | None = None,
) -> LearningTreeNode | None:
    """Return one document learning tree filtered by user, source, and scope."""
    records = _records(
        driver,
        """
        MATCH (doc:Document {user_id: $user_id, source_id: $source_id})
        MATCH (doc)-[root_rel:HAS_LEARNING_TOPIC {graph_scope: 'learning_structure', user_id: $user_id, source_id: $source_id}]->
          (root:LearningTopic {user_id: $user_id, source_id: $source_id, graph_scope: 'learning_structure'})
        WITH doc, collect(DISTINCT {id: root.id, order_index: root_rel.order_index}) AS roots
        OPTIONAL MATCH (n {user_id: $user_id, source_id: $source_id, graph_scope: 'learning_structure'})
        WHERE n:LearningTopic OR n:LearningConcept OR n:LearningObjective
        OPTIONAL MATCH (n)-[rel {graph_scope: 'learning_structure', user_id: $user_id, source_id: $source_id}]->(m)
        WHERE type(rel) IN ['HAS_SUBTOPIC', 'HAS_LEARNING_CONCEPT', 'HAS_OBJECTIVE']
          AND (m:LearningTopic OR m:LearningConcept OR m:LearningObjective)
        RETURN {
          id: 'document:' + doc.source_id,
          label: coalesce(doc.title, 'Document'),
          type: 'document',
          chunk_ids: [],
          children: []
        } AS document,
        roots,
        collect(DISTINCT n {.*, labels: labels(n)}) AS nodes,
        collect(DISTINCT {
          source_id: n.id,
          target_id: m.id,
          type: type(rel),
          order_index: rel.order_index
        }) AS relationships
        """,
        {"user_id": user_id, "source_id": source_id},
        database,
    )
    if not records:
        return None
    tree = records[0].get("tree")
    if isinstance(tree, LearningTreeNode):
        return tree
    if not isinstance(tree, dict):
        return _tree_from_record(records[0])
    return LearningTreeNode.model_validate(tree)


def _tree_from_record(record: dict[str, Any]) -> LearningTreeNode | None:
    document = record.get("document")
    if not isinstance(document, dict):
        return None

    node_map = {
        str(node.get("id")): _node_from_properties(node)
        for node in record.get("nodes", [])
        if isinstance(node, dict) and node.get("id")
    }
    if not node_map:
        return None

    child_ids_by_parent: dict[str, list[tuple[int | None, str]]] = {}
    subtopic_ids = set()
    for relationship in record.get("relationships", []):
        if not isinstance(relationship, dict):
            continue
        source_id = relationship.get("source_id")
        target_id = relationship.get("target_id")
        if not source_id or not target_id or target_id not in node_map:
            continue
        if relationship.get("type") == "HAS_SUBTOPIC":
            subtopic_ids.add(str(target_id))
        child_ids_by_parent.setdefault(str(source_id), []).append(
            (relationship.get("order_index"), str(target_id))
        )

    for node_id in subtopic_ids:
        node_map[node_id]["type"] = "subtopic"

    def attach_children(parent_id: str) -> list[dict[str, Any]]:
        children = []
        seen = set()
        for _order_index, child_id in sorted(
            child_ids_by_parent.get(parent_id, []),
            key=lambda item: ((item[0] is None), item[0] or 0, node_map[item[1]]["label"]),
        ):
            if child_id in seen:
                continue
            seen.add(child_id)
            child = dict(node_map[child_id])
            child["children"] = attach_children(child_id)
            children.append(child)
        return children

    root_ids = [
        (root.get("order_index"), str(root.get("id")))
        for root in record.get("roots", [])
        if isinstance(root, dict) and str(root.get("id")) in node_map
    ]
    if not root_ids:
        return None

    root_children = []
    for _order_index, root_id in sorted(
        root_ids,
        key=lambda item: ((item[0] is None), item[0] or 0, node_map[item[1]]["label"]),
    ):
        root = dict(node_map[root_id])
        root["type"] = "topic"
        root["children"] = attach_children(root_id)
        root_children.append(root)

    document["children"] = root_children
    return LearningTreeNode.model_validate(document)


def _node_from_properties(node: dict[str, Any]) -> dict[str, Any]:
    labels = set(node.get("labels") or [])
    if "LearningConcept" in labels:
        node_type = "concept"
        label = node.get("name")
    elif "LearningObjective" in labels:
        node_type = "objective"
        label = node.get("objective")
    else:
        node_type = "topic"
        label = node.get("title")

    return {
        "id": str(node["id"]),
        "label": str(label or node["id"]),
        "type": node_type,
        "summary": node.get("summary"),
        "page_start": node.get("page_start"),
        "page_end": node.get("page_end"),
        "confidence": node.get("confidence"),
        "order_index": node.get("order_index"),
        "chunk_ids": node.get("chunk_ids") or [],
        "children": [],
    }


def _records(
    driver: Any,
    statement: str,
    parameters: dict[str, Any],
    database: str | None,
) -> list[dict[str, Any]]:
    session = driver.session(database=database)
    if hasattr(session, "__enter__"):
        with session as active_session:
            result = active_session.run(statement, parameters)
            return _materialize_records(result)
    try:
        result = session.run(statement, parameters)
        return _materialize_records(result)
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
            records.append(dict(record))
    return records
