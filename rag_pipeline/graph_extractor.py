"""LLM-backed extraction of compact knowledge graph payloads from chunks."""

from __future__ import annotations

import json
import re
from typing import Any

from rag_pipeline.graph_schema import GraphExtraction
from rag_pipeline.graph_schema import normalize_node_name
from rag_pipeline.llm_client import OpenAILlmClient


GRAPH_EXTRACTION_SYSTEM_PROMPT = (
    "Extract a compact knowledge graph from the learning material. Return strict JSON "
    "with keys nodes and edges only. Nodes require name and node_type. Edges require "
    "source, target, and relation_type. Use only facts in the chunk. Do not invent "
    "concepts, do not add filler, and do not include markdown."
)


class GraphExtractionError(Exception):
    """Raised when a chunk graph extraction cannot be validated."""


class GraphExtractor:
    """Extract graph nodes and edges from already-created RAG chunks."""

    def __init__(
        self,
        llm_client: Any = None,
        max_nodes: int = 12,
        max_edges: int = 20,
    ) -> None:
        self.llm_client = llm_client
        self.max_nodes = max_nodes
        self.max_edges = max_edges

    def extract_from_chunk(self, chunk: dict[str, Any]) -> GraphExtraction:
        """Extract and validate graph data from one RAG chunk."""
        text = str(chunk.get("text") or chunk.get("content") or "").strip()
        if not text:
            return GraphExtraction(nodes=[], edges=[])

        client = self.llm_client or OpenAILlmClient()
        prompt = self._build_prompt(chunk, text)
        try:
            raw = client.complete(
                system_prompt=GRAPH_EXTRACTION_SYSTEM_PROMPT,
                user_prompt=prompt,
            )
            payload = json.loads(_strip_json_fence(raw))
            extraction = GraphExtraction.from_payload(payload)
            return _limit_extraction(extraction, self.max_nodes, self.max_edges)
        except Exception as exc:
            raise GraphExtractionError(str(exc)) from exc

    def _build_prompt(self, chunk: dict[str, Any], text: str) -> str:
        heading = chunk.get("heading") or ""
        source_type = chunk.get("source_type") or ""
        page = chunk.get("page_index")
        return "\n".join(
            [
                f"Heading: {heading}",
                f"Source type: {source_type}",
                f"Page: {page if page is not None else ''}",
                "Text:",
                text[:4000],
                "",
                "Return JSON: {\"nodes\": [...], \"edges\": [...]}",
            ]
        )


def _strip_json_fence(raw: str) -> str:
    text = raw.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text


def _limit_extraction(
    extraction: GraphExtraction,
    max_nodes: int,
    max_edges: int,
) -> GraphExtraction:
    nodes = extraction.nodes[:max_nodes]
    known = {node.normalized_name for node in nodes}
    edges = [
        edge
        for edge in extraction.edges
        if normalize_node_name(edge.source) in known and normalize_node_name(edge.target) in known
    ][:max_edges]
    return GraphExtraction(nodes=nodes, edges=edges)
