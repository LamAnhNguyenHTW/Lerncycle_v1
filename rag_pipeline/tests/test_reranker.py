from __future__ import annotations

import copy

import pytest

from rag_pipeline.reranker import FastEmbedReranker, NoopReranker, create_reranker


class FakeCrossEncoder:
    def __init__(self, scores: list[float] | None = None, raises: bool = False) -> None:
        self.scores = scores or []
        self.raises = raises
        self.calls = []

    def rerank(self, query: str, documents, **kwargs):
        docs = list(documents)
        self.calls.append({"query": query, "documents": docs, "kwargs": kwargs})
        if self.raises:
            raise RuntimeError("model failed")
        return self.scores


def _result(index: int, **overrides):
    result = {
        "chunk_id": f"chunk-{index}",
        "text": f"Chunk text {index}",
        "score": float(index),
        "source_type": "pdf",
        "source_id": f"source-{index}",
        "page_index": index,
        "title": f"File {index}.pdf",
        "heading": f"Heading {index}",
    }
    result.update(overrides)
    return result


def test_noop_reranker_preserves_order() -> None:
    results = [_result(1), _result(2), _result(3)]

    reranked = NoopReranker().rerank("query", results, top_k=3)

    assert reranked == results


def test_noop_reranker_respects_top_k() -> None:
    results = [_result(1), _result(2), _result(3)]

    reranked = NoopReranker().rerank("query", results, top_k=2)

    assert [item["chunk_id"] for item in reranked] == ["chunk-1", "chunk-2"]


def test_noop_reranker_handles_fewer_results_than_top_k() -> None:
    results = [_result(1)]

    reranked = NoopReranker().rerank("query", results, top_k=5)

    assert reranked == results


def test_fastembed_reranker_reorders_by_scores() -> None:
    results = [_result(1), _result(2), _result(3)]
    client = FakeCrossEncoder([0.1, 0.2, 0.9])

    reranked = FastEmbedReranker(injected_client=client).rerank("query", results, top_k=3)

    assert [item["chunk_id"] for item in reranked] == ["chunk-3", "chunk-2", "chunk-1"]


def test_fastembed_reranker_preserves_original_score_and_rank() -> None:
    results = [_result(1, score=0.7), _result(2, score=0.2)]
    client = FakeCrossEncoder([0.1, 0.9])

    reranked = FastEmbedReranker(injected_client=client).rerank("query", results, top_k=2)

    assert reranked[0]["chunk_id"] == "chunk-2"
    assert reranked[0]["rerank_score"] == 0.9
    assert reranked[0]["original_score"] == 0.2
    assert reranked[0]["original_rank"] == 2


def test_fastembed_reranker_respects_top_k() -> None:
    results = [_result(1), _result(2), _result(3)]
    client = FakeCrossEncoder([0.3, 0.2, 0.1])

    reranked = FastEmbedReranker(injected_client=client).rerank("query", results, top_k=2)

    assert len(reranked) == 2


def test_fastembed_reranker_falls_back_on_exception() -> None:
    results = [_result(1), _result(2), _result(3)]
    client = FakeCrossEncoder(raises=True)

    reranked = FastEmbedReranker(injected_client=client).rerank("query", results, top_k=2)

    assert reranked == results[:2]


def test_fastembed_reranker_falls_back_on_invalid_score_count() -> None:
    results = [_result(1), _result(2), _result(3)]
    client = FakeCrossEncoder([0.5])

    reranked = FastEmbedReranker(injected_client=client).rerank("query", results, top_k=2)

    assert reranked == results[:2]


def test_fastembed_reranker_truncates_candidate_text() -> None:
    long_text = "x" * 1500
    results = [_result(1, text=long_text, heading="H", title="T", page_index=4)]
    client = FakeCrossEncoder([0.5])

    FastEmbedReranker(injected_client=client).rerank("query", results, top_k=1)

    document = client.calls[0]["documents"][0]
    assert "H" in document
    assert "T" in document
    assert "Page: 5" in document
    assert len(document) <= 1100


def test_fastembed_reranker_handles_empty_results() -> None:
    client = FakeCrossEncoder([0.5])

    reranked = FastEmbedReranker(injected_client=client).rerank("query", [], top_k=3)

    assert reranked == []
    assert client.calls == []


def test_fastembed_reranker_does_not_mutate_input_results() -> None:
    results = [_result(1), _result(2)]
    original = copy.deepcopy(results)
    client = FakeCrossEncoder([0.1, 0.9])

    FastEmbedReranker(injected_client=client).rerank("query", results, top_k=2)

    assert results == original


def test_create_reranker_disabled_returns_noop() -> None:
    reranker = create_reranker(
        provider="fastembed",
        model="model",
        enabled=False,
        injected_client=FakeCrossEncoder([0.1]),
    )

    assert isinstance(reranker, NoopReranker)


def test_create_reranker_noop_provider_returns_noop() -> None:
    reranker = create_reranker(provider="noop", model="model", enabled=True)

    assert isinstance(reranker, NoopReranker)


def test_create_reranker_fastembed_returns_fastembed_instance() -> None:
    reranker = create_reranker(
        provider="fastembed",
        model="model",
        enabled=True,
        injected_client=FakeCrossEncoder([0.1]),
    )

    assert isinstance(reranker, FastEmbedReranker)


def test_create_reranker_unknown_provider_raises_value_error() -> None:
    with pytest.raises(ValueError, match="fastembed"):
        create_reranker(provider="unknown", model="model", enabled=True)
