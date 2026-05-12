from __future__ import annotations

import pytest

from rag_pipeline.graph_extractor import GraphExtractionError, GraphExtractor


class FakeLlm:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = []

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        return self.response


def _chunk() -> dict:
    return {
        "chunk_id": "chunk-1",
        "source_type": "pdf",
        "page_index": 1,
        "heading": "Intro",
        "text": "Process Mining uses Event Logs.",
        "metadata": {"doc_items": [{"bbox": [1]}], "highlight_areas": [{"x": 1}]},
    }


def _payload() -> str:
    return (
        '{"nodes":[{"name":"Process Mining","node_type":"concept"},'
        '{"name":"Event Logs","node_type":"data_object"}],'
        '"edges":[{"source":"Process Mining","target":"Event Logs","relation_type":"uses"}]}'
    )


def test_graph_extractor_extracts_nodes_and_edges() -> None:
    extractor = GraphExtractor(llm_client=FakeLlm(_payload()))

    extraction = extractor.extract_from_chunk(_chunk())

    assert [node.name for node in extraction.nodes] == ["Process Mining", "Event Logs"]
    assert extraction.edges[0].relation_type == "uses"


def test_graph_extractor_strips_json_code_fence() -> None:
    extractor = GraphExtractor(llm_client=FakeLlm(f"```json\n{_payload()}\n```"))

    extraction = extractor.extract_from_chunk(_chunk())

    assert len(extraction.nodes) == 2


def test_graph_extractor_limits_nodes_and_edges() -> None:
    response = (
        '{"nodes":[{"name":"A"},{"name":"B"},{"name":"C"}],'
        '"edges":[{"source":"A","target":"B","relation_type":"x"},'
        '{"source":"B","target":"C","relation_type":"x"}]}'
    )
    extractor = GraphExtractor(llm_client=FakeLlm(response), max_nodes=2, max_edges=1)

    extraction = extractor.extract_from_chunk(_chunk())

    assert len(extraction.nodes) == 2
    assert len(extraction.edges) == 1


def test_graph_extractor_rejects_malformed_json() -> None:
    extractor = GraphExtractor(llm_client=FakeLlm("not-json"))

    with pytest.raises(GraphExtractionError):
        extractor.extract_from_chunk(_chunk())


def test_graph_extractor_rejects_unknown_edge_node() -> None:
    extractor = GraphExtractor(
        llm_client=FakeLlm(
            '{"nodes":[{"name":"A"}],"edges":[{"source":"A","target":"B","relation_type":"x"}]}'
        )
    )

    with pytest.raises(GraphExtractionError):
        extractor.extract_from_chunk(_chunk())


def test_graph_extractor_prompt_excludes_raw_metadata() -> None:
    llm = FakeLlm(_payload())
    extractor = GraphExtractor(llm_client=llm)

    extractor.extract_from_chunk(_chunk())

    prompt = llm.calls[0]["user_prompt"]
    assert "metadata" not in prompt
    assert "doc_items" not in prompt
    assert "bbox" not in prompt
    assert "highlight_areas" not in prompt


def test_graph_extractor_uses_injected_llm_client() -> None:
    llm = FakeLlm(_payload())
    extractor = GraphExtractor(llm_client=llm)

    extractor.extract_from_chunk(_chunk())

    assert len(llm.calls) == 1


def test_graph_extractor_handles_empty_text() -> None:
    llm = FakeLlm(_payload())
    extractor = GraphExtractor(llm_client=llm)

    extraction = extractor.extract_from_chunk({"text": ""})

    assert extraction.nodes == []
    assert extraction.edges == []
    assert llm.calls == []
