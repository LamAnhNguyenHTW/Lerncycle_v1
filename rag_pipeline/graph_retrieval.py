"""Symbolic graph retrieval helpers for GraphRAG."""

from __future__ import annotations

import re
from typing import Any


GRAPH_INTENT_TRIGGERS = (
    "wie hängt",
    "wie haengt",
    "zusammenhang",
    "beziehung",
    "relation",
    "mindmap",
    "konzeptkarte",
    "welche konzepte",
    "wichtigste begriffe",
    "themen bauen aufeinander auf",
    "voraussetzungen",
    "abhängig von",
    "abhaengig von",
    "related to",
    "relationship between",
    "concept map",
    "key concepts",
    "dependencies",
    "prerequisites",
    "connected concepts",
    "how do these concepts connect",
)


def detect_graph_intent(query: str) -> bool:
    """Return True for relationship or concept-map questions."""
    normalized = query.strip().lower()
    if normalized.startswith(("was ist ", "define ", "explain ", "erkläre ", "erklaere ")):
        return False
    if normalized.startswith("how is ") and " related" not in normalized:
        return False
    return any(trigger in normalized for trigger in GRAPH_INTENT_TRIGGERS)


def retrieve_graph_context(
    query: str,
    user_id: str,
    source_types: list[str] | None = None,
    source_ids: list[str] | None = None,
    top_k: int = 8,
    max_chars: int = 6000,
    graph_store: Any = None,
) -> dict[str, Any]:
    """Retrieve graph neighborhood context and safe source summaries."""
    if graph_store is None:
        return _empty()
    concepts = graph_store.search_concepts(
        user_id=user_id,
        query=query,
        source_types=source_types,
        source_ids=source_ids,
        top_k=top_k,
    )
    concept_names = [
        str(concept.get("normalized_name") or concept.get("name") or "").strip().lower()
        for concept in concepts
        if concept.get("normalized_name") or concept.get("name")
    ]
    if not concept_names:
        return _empty()
    neighborhood = graph_store.get_neighborhood(
        user_id=user_id,
        concept_names=concept_names,
        max_depth=1,
        limit=max(top_k * 4, 10),
        source_types=source_types,
        source_ids=source_ids,
    )
    relationships = list(neighborhood.get("relationships") or [])
    context_lines: list[str] = []
    backing_chunk_ids: list[str] = []
    for rel in relationships:
        line = _relationship_line(rel)
        if not line:
            continue
        context_lines.append(line)
        chunk_id = rel.get("chunk_id")
        if chunk_id and chunk_id not in backing_chunk_ids:
            backing_chunk_ids.append(str(chunk_id))
    context_text = "\n".join(context_lines)
    if len(context_text) > max_chars:
        context_text = context_text[:max_chars].rstrip()
    sources = []
    if context_text:
        sources.append(
            {
                "chunk_id": "knowledge-graph:" + ",".join(backing_chunk_ids[:5]),
                "source_type": "knowledge_graph",
                "source_id": source_ids[0] if source_ids else None,
                "title": "Knowledge Graph",
                "heading": "Concept relationships",
                "page": None,
                "score": None,
                "snippet": _snippet(context_text, 200),
                "metadata": {
                    "backing_chunk_ids": backing_chunk_ids[:20],
                    "node_names": [concept.get("name") for concept in concepts[:top_k]],
                    "relationship_count": len(relationships),
                },
            }
        )
    return {
        "context_text": context_text,
        "sources": sources,
        "nodes": concepts,
        "relationships": relationships,
    }


def _relationship_line(rel: dict[str, Any]) -> str:
    source = rel.get("source")
    target = rel.get("target")
    relation = rel.get("relation_type")
    if not source or not target or not relation:
        return ""
    description = rel.get("description")
    suffix = f" ({description})" if description else ""
    page = rel.get("page_index")
    page_text = f", page {page + 1}" if isinstance(page, int) else ""
    chunk = rel.get("chunk_id")
    backing = f" [chunk: {chunk}{page_text}]" if chunk else ""
    return f"{source} --{relation}--> {target}{suffix}{backing}"


def _snippet(text: str, max_chars: int) -> str:
    collapsed = re.sub(r"\s+", " ", text).strip()
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[: max_chars - 3].rstrip() + "..."


def _empty() -> dict[str, Any]:
    return {"context_text": "", "sources": [], "nodes": [], "relationships": []}
