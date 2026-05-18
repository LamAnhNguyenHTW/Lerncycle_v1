from __future__ import annotations

import time

from rag_pipeline.rag_answer import answer_with_rag


class FakeLlm:
    def __init__(self) -> None:
        self.user_prompt = ""

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.user_prompt = user_prompt
        return "answer"


class SlowReranker:
    def rerank(self, query, results, top_k):
        time.sleep(5)
        return [{**results[0], "text": "slow reranked"}]


class FailingReranker:
    def rerank(self, query, results, top_k):
        raise RuntimeError("reranker failed")


def _result(text: str = "retrieval order") -> dict:
    return {
        "chunk_id": "chunk-1",
        "text": text,
        "score": 0.9,
        "source_type": "pdf",
        "source_id": "pdf-1",
        "page_index": 0,
        "metadata": {},
    }


def test_reranker_timeout_uses_retrieval_order_candidates() -> None:
    llm = FakeLlm()
    answer_with_rag(
        "query",
        "user-1",
        llm_client=llm,
        retrieval_fn=lambda **_: [_result()],
        reranker=SlowReranker(),
        reranking_enabled=True,
        reranking_top_k=1,
        rerank_timeout_s=0.1,
    )

    assert "retrieval order" in llm.user_prompt
    assert "slow reranked" not in llm.user_prompt


def test_reranker_exception_uses_retrieval_order_candidates() -> None:
    llm = FakeLlm()
    answer_with_rag(
        "query",
        "user-1",
        llm_client=llm,
        retrieval_fn=lambda **_: [_result()],
        reranker=FailingReranker(),
        reranking_enabled=True,
        reranking_top_k=1,
    )

    assert "retrieval order" in llm.user_prompt


def test_reranker_stage_recorded_on_timeout() -> None:
    response = answer_with_rag(
        "query",
        "user-1",
        llm_client=FakeLlm(),
        retrieval_fn=lambda **_: [_result()],
        reranker=SlowReranker(),
        reranking_enabled=True,
        reranking_top_k=1,
        rerank_timeout_s=0.1,
        enable_timing=True,
    )

    stages = {stage["stage"]: stage for stage in response["timing"]["stages"]}
    assert stages["rerank"]["duration_ms"] >= 0
