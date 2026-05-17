from __future__ import annotations

from rag_pipeline.rag_answer import answer_with_rag


class FakeLlmClient:
    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        return "Antwort"


def _result() -> dict:
    return {
        "chunk_id": "chunk-1",
        "text": "Process Mining verbindet Data Science und Process Science.",
        "score": 0.9,
        "source_type": "pdf",
        "source_id": "pdf-1",
        "pdf_id": "pdf-1",
        "page_index": 0,
        "title": "GPAA.pdf",
        "heading": "Definition",
        "metadata": {"filename": "GPAA.pdf"},
    }


def test_answer_with_rag_returns_timing_report_when_enabled() -> None:
    response = answer_with_rag(
        "Was ist Process Mining?",
        "user-1",
        llm_client=FakeLlmClient(),
        retrieval_fn=lambda **_: [_result()],
        enable_timing=True,
    )

    timing = response["timing"]
    stages = {stage["stage"]: stage for stage in timing["stages"]}

    assert "query_understanding" in stages
    assert "qdrant_retrieve" in stages
    assert "rerank" in stages
    assert "graph_retrieve" in stages
    assert "memory_retrieve" in stages
    assert "web_retrieve" in stages
    assert "context_build" in stages
    assert "llm_total" in stages
    assert "llm_first_token" not in stages
    assert stages["graph_retrieve"]["status"] == "skipped"
    assert stages["web_retrieve"]["status"] == "skipped"
    assert stages["memory_retrieve"]["status"] == "skipped"
    assert timing["total_ms"] >= 0


def test_answer_with_rag_omits_timing_when_disabled() -> None:
    response = answer_with_rag(
        "Was ist Process Mining?",
        "user-1",
        llm_client=FakeLlmClient(),
        retrieval_fn=lambda **_: [_result()],
    )

    assert "timing" not in response
