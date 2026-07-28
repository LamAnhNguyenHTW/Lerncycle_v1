"""Tests for comparison-phase searcher construction."""

from __future__ import annotations

from types import SimpleNamespace

from rag_pipeline.eval.run_comparisons import make_phase_searchers


class _Reranker:
    def rerank(self, query, results, top_k):
        return list(reversed(results))[:top_k]


def test_make_retrieval_and_reranking_searchers_use_fixed_conditions() -> None:
    calls = []

    def retrieval(name):
        def search(query, **kwargs):
            calls.append((name, query, kwargs))
            return [{"chunk_id": "a"}, {"chunk_id": "b"}]

        return search

    config = SimpleNamespace(hybrid_prefetch_limit=30)
    common = dict(
        config=config,
        user_id="eval-user",
        top_k=5,
        store="store",
        embedder="dense-embedder",
        sparse_embedder="sparse-embedder",
        dense_fn=retrieval("dense"),
        sparse_fn=retrieval("sparse"),
        hybrid_fn=retrieval("hybrid"),
    )

    retrieval_searchers = make_phase_searchers("retrieval", **common)
    assert set(retrieval_searchers) == {"dense", "sparse", "hybrid"}
    retrieval_searchers["dense"]("q")
    retrieval_searchers["sparse"]("q")
    retrieval_searchers["hybrid"]("q")

    rerank_searchers = make_phase_searchers(
        "reranking",
        reranker=_Reranker(),
        candidate_k=30,
        base_retrieval="dense",
        **common,
    )
    assert set(rerank_searchers) == {"without_reranking", "cross_encoder_reranking"}
    assert rerank_searchers["without_reranking"]("q")[0]["chunk_id"] == "a"
    assert rerank_searchers["cross_encoder_reranking"]("q")[0]["chunk_id"] == "b"
    assert calls[-2][0] == "dense"
    assert calls[-1][0] == "dense"

    llm_searchers = make_phase_searchers(
        "reranking",
        reranker=_Reranker(),
        candidate_k=20,
        base_retrieval="dense",
        reranked_label="llm_reranking",
        **common,
    )
    assert set(llm_searchers) == {"without_reranking", "llm_reranking"}
    assert llm_searchers["llm_reranking"]("q")[0]["chunk_id"] == "b"

    assert all(call[2]["user_id"] == "eval-user" for call in calls)
    assert all(call[2]["source_types"] == ["pdf"] for call in calls)
