"""Validated graph extraction schema for Neo4j GraphRAG."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


ALLOWED_NODE_TYPES = {
    "concept",
    "process",
    "method",
    "tool",
    "metric",
    "artifact",
    "person_role",
    "organization",
    "system",
    "data_object",
    "other",
}

_NODE_TYPE_ALIASES: dict[str, str] = {
    "person": "person_role",
    "role": "person_role",
    "actor": "person_role",
    "org": "organization",
    "algorithm": "method",
    "formula": "method",
    "framework": "method",
    "model": "concept",
    "theory": "concept",
    "principle": "concept",
    "term": "concept",
    "definition": "concept",
    "event": "other",
    "source": "other",
    "course module": "other",
    "module": "other",
    "topic": "concept",
}


def normalize_node_type(node_type: str) -> str:
    """Normalize GPT-returned node types to an allowed value."""
    raw = str(node_type).strip().lower()
    if raw in ALLOWED_NODE_TYPES:
        return raw
    return _NODE_TYPE_ALIASES.get(raw, "other")


def normalize_node_name(name: str) -> str:
    """Normalize concept names for stable user-scoped dedupe."""
    return re.sub(r"\s+", " ", str(name).strip().lower())


def normalize_relation_type(relation_type: str) -> str:
    """Normalize relation types to lowercase snake_case."""
    value = str(relation_type).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_")


def _validate_confidence(value: float | int | None) -> float | None:
    if value is None:
        return None
    confidence = float(value)
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1")
    return confidence


@dataclass(frozen=True)
class GraphNode:
    """One extracted graph node."""

    name: str
    node_type: str = "concept"
    description: str | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        if not normalize_node_name(self.name):
            raise ValueError("Graph node name must not be empty")
        if self.node_type not in ALLOWED_NODE_TYPES:
            raise ValueError(f"Unsupported graph node type: {self.node_type}")
        object.__setattr__(self, "confidence", _validate_confidence(self.confidence))

    @property
    def normalized_name(self) -> str:
        return normalize_node_name(self.name)


@dataclass(frozen=True)
class GraphEdge:
    """One extracted relationship between two graph nodes."""

    source: str
    target: str
    relation_type: str
    description: str | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        relation = normalize_relation_type(self.relation_type)
        if not relation:
            raise ValueError("Graph edge relation_type must not be empty")
        if normalize_node_name(self.source) == normalize_node_name(self.target):
            raise ValueError("Graph self-edges are not allowed")
        object.__setattr__(self, "relation_type", relation)
        object.__setattr__(self, "confidence", _validate_confidence(self.confidence))


@dataclass(frozen=True)
class GraphExtraction:
    """A validated graph extraction payload."""

    nodes: list[GraphNode]
    edges: list[GraphEdge]

    def __post_init__(self) -> None:
        nodes = _dedupe_nodes(self.nodes)
        known_names = {node.normalized_name for node in nodes}
        edges = _dedupe_edges(self.edges, known_names)
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "edges", edges)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "GraphExtraction":
        """Build and validate an extraction from decoded JSON."""
        if not isinstance(payload, dict):
            raise ValueError("Graph extraction payload must be an object")
        nodes = [
            GraphNode(
                name=str(item.get("name", "")),
                node_type=normalize_node_type(item.get("node_type") or "concept"),
                description=item.get("description"),
                confidence=item.get("confidence"),
            )
            for item in payload.get("nodes", [])
            if isinstance(item, dict)
        ]
        edges = [
            GraphEdge(
                source=str(item.get("source", "")),
                target=str(item.get("target", "")),
                relation_type=str(item.get("relation_type", "")),
                description=item.get("description"),
                confidence=item.get("confidence"),
            )
            for item in payload.get("edges", [])
            if isinstance(item, dict)
        ]
        return cls(nodes=nodes, edges=edges)


def _dedupe_nodes(nodes: list[GraphNode]) -> list[GraphNode]:
    seen: set[tuple[str, str]] = set()
    deduped: list[GraphNode] = []
    for node in nodes:
        key = (node.normalized_name, node.node_type)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(node)
    return deduped


def _dedupe_edges(
    edges: list[GraphEdge],
    known_names: set[str],
) -> list[GraphEdge]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[GraphEdge] = []
    for edge in edges:
        source = normalize_node_name(edge.source)
        target = normalize_node_name(edge.target)
        if source not in known_names or target not in known_names:
            raise ValueError("Graph edge references an unknown node")
        key = (source, target, edge.relation_type)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(edge)
    return deduped
