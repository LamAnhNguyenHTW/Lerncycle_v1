from __future__ import annotations

from rag_pipeline import rag_answer
from rag_pipeline.rag_answer import answer_with_rag
from rag_pipeline.reranker_cache import RerankerResultCache


class FakeLlm:
    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        return "answer"


class CountingReranker:
    def __init__(self) -> None:
        self.calls = 0

    def rerank(self, query, results, top_k):
        self.calls += 1
        return [{**item, "rerank_score": 1.0 / index} for index, item in enumerate(results, start=1)][:top_k]


def _result(chunk_id: str = "chunk-1", content_hash: str = "hash-1") -> dict:
    return {
        "chunk_id": chunk_id,
        "text": f"text {chunk_id}",
        "score": 0.9,
        "source_type": "pdf",
        "source_id": "pdf-1",
        "metadata": {"content_hash": content_hash},
    }


def test_reranker_cache_skips_second_reranker_call(monkeypatch) -> None:
    monkeypatch.setattr(rag_answer, "default_reranker_cache", RerankerResultCache(max_entries=8))
    reranker = CountingReranker()

    for _ in range(2):
        answer_with_rag(
            "query",
            "user-1",
            llm_client=FakeLlm(),
            retrieval_fn=lambda **_: [_result()],
            reranker=reranker,
            reranking_enabled=True,
            reranker_provider="fastembed",
            reranker_model="model",
        )

    assert reranker.calls == 1


def test_reranker_cache_misses_when_candidate_hash_or_model_changes(monkeypatch) -> None:
    monkeypatch.setattr(rag_answer, "default_reranker_cache", RerankerResultCache(max_entries=8))
    reranker = CountingReranker()

    answer_with_rag("query", "user-1", llm_client=FakeLlm(), retrieval_fn=lambda **_: [_result(content_hash="hash-1")], reranker=reranker, reranking_enabled=True, reranker_provider="fastembed", reranker_model="model")
    answer_with_rag("query", "user-1", llm_client=FakeLlm(), retrieval_fn=lambda **_: [_result(content_hash="hash-2")], reranker=reranker, reranking_enabled=True, reranker_provider="fastembed", reranker_model="model")
    answer_with_rag("query", "user-1", llm_client=FakeLlm(), retrieval_fn=lambda **_: [_result(content_hash="hash-2")], reranker=reranker, reranking_enabled=True, reranker_provider="fastembed", reranker_model="other")

    assert reranker.calls == 3
