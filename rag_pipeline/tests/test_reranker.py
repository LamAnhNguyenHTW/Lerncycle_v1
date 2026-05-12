from __future__ import annotations

import copy

import pytest

from rag_pipeline.reranker import (
    FastEmbedReranker,
    LlmReranker,
    NoopReranker,
    create_reranker,
)


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


class FakeLlmClient:
    def __init__(self, response: str = "[]", raises: bool = False) -> None:
        self.response = response
        self.raises = raises
        self.calls = []

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        if self.raises:
            raise RuntimeError("llm failed")
        return self.response


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


def test_llm_reranker_reorders_by_scores() -> None:
    results = [_result(1), _result(2), _result(3)]
    client = FakeLlmClient(
        '[{"chunk_id":"chunk-1","score":0.2},{"chunk_id":"chunk-3","score":0.9}]'
    )

    reranked = LlmReranker(llm_client=client).rerank("query", results, top_k=3)

    assert [item["chunk_id"] for item in reranked] == ["chunk-3", "chunk-1", "chunk-2"]


def test_llm_reranker_preserves_original_score_and_rank() -> None:
    results = [_result(1, score=0.7), _result(2, score=0.2)]
    client = FakeLlmClient('[{"chunk_id":"chunk-2","score":0.9,"reason":"best"}]')

    reranked = LlmReranker(llm_client=client).rerank("query", results, top_k=2)

    assert reranked[0]["chunk_id"] == "chunk-2"
    assert reranked[0]["rerank_score"] == 0.9
    assert reranked[0]["original_score"] == 0.2
    assert reranked[0]["original_rank"] == 2
    assert reranked[0]["rerank_reason"] == "best"


def test_llm_reranker_respects_top_k() -> None:
    results = [_result(1), _result(2), _result(3)]
    client = FakeLlmClient(
        '[{"chunk_id":"chunk-1","score":0.8},{"chunk_id":"chunk-2","score":0.7},{"chunk_id":"chunk-3","score":0.6}]'
    )

    reranked = LlmReranker(llm_client=client).rerank("query", results, top_k=2)

    assert len(reranked) == 2


def test_llm_reranker_ignores_unknown_chunk_ids() -> None:
    results = [_result(1), _result(2)]
    client = FakeLlmClient(
        '[{"chunk_id":"unknown","score":1.0},{"chunk_id":"chunk-2","score":0.9}]'
    )

    reranked = LlmReranker(llm_client=client).rerank("query", results, top_k=2)

    assert [item["chunk_id"] for item in reranked] == ["chunk-2", "chunk-1"]


def test_llm_reranker_appends_unscored_candidates_after_scored_ones() -> None:
    results = [_result(1), _result(2), _result(3)]
    client = FakeLlmClient('[{"chunk_id":"chunk-3","score":0.9}]')

    reranked = LlmReranker(llm_client=client).rerank("query", results, top_k=3)

    assert [item["chunk_id"] for item in reranked] == ["chunk-3", "chunk-1", "chunk-2"]


def test_llm_reranker_falls_back_on_malformed_json() -> None:
    results = [_result(1), _result(2)]
    client = FakeLlmClient("not json")

    reranked = LlmReranker(llm_client=client).rerank("query", results, top_k=1)

    assert reranked == results[:1]


def test_llm_reranker_falls_back_on_exception() -> None:
    results = [_result(1), _result(2)]
    client = FakeLlmClient(raises=True)

    reranked = LlmReranker(llm_client=client).rerank("query", results, top_k=1)

    assert reranked == results[:1]


def test_llm_reranker_truncates_candidate_text() -> None:
    results = [_result(1, text="x" * 100)]
    client = FakeLlmClient('[{"chunk_id":"chunk-1","score":0.5}]')

    LlmReranker(llm_client=client, max_candidate_chars=12).rerank("query", results, top_k=1)

    assert "x" * 12 in client.calls[0]["user_prompt"]
    assert "x" * 13 not in client.calls[0]["user_prompt"]


def test_llm_reranker_handles_empty_results() -> None:
    client = FakeLlmClient('[{"chunk_id":"chunk-1","score":0.5}]')

    reranked = LlmReranker(llm_client=client).rerank("query", [], top_k=3)

    assert reranked == []
    assert client.calls == []


def test_llm_reranker_does_not_mutate_input_results() -> None:
    results = [_result(1), _result(2)]
    original = copy.deepcopy(results)
    client = FakeLlmClient('[{"chunk_id":"chunk-2","score":0.9}]')

    LlmReranker(llm_client=client).rerank("query", results, top_k=2)

    assert results == original


def test_llm_reranker_parses_json_code_fence() -> None:
    results = [_result(1), _result(2)]
    client = FakeLlmClient('```json\n[{"chunk_id":"chunk-2","score":0.9}]\n```')

    reranked = LlmReranker(llm_client=client).rerank("query", results, top_k=2)

    assert reranked[0]["chunk_id"] == "chunk-2"


def test_llm_reranker_handles_duplicate_chunk_ids_deterministically() -> None:
    results = [_result(1), _result(2)]
    client = FakeLlmClient(
        '[{"chunk_id":"chunk-2","score":0.7},{"chunk_id":"chunk-2","score":1.0}]'
    )

    reranked = LlmReranker(llm_client=client).rerank("query", results, top_k=2)

    assert reranked[0]["chunk_id"] == "chunk-2"
    assert reranked[0]["rerank_score"] == 0.7


def test_llm_reranker_falls_back_on_non_numeric_score() -> None:
    results = [_result(1), _result(2)]
    client = FakeLlmClient('[{"chunk_id":"chunk-2","score":"0.9"}]')

    reranked = LlmReranker(llm_client=client).rerank("query", results, top_k=2)

    assert reranked == results[:2]


def test_llm_reranker_falls_back_on_score_outside_0_1() -> None:
    results = [_result(1), _result(2)]
    client = FakeLlmClient('[{"chunk_id":"chunk-2","score":1.1}]')

    reranked = LlmReranker(llm_client=client).rerank("query", results, top_k=2)

    assert reranked == results[:2]


def test_llm_reranker_prompt_excludes_raw_metadata() -> None:
    results = [
        _result(
            1,
            metadata={"doc_items": [{"bbox": [1]}], "highlight_areas": [{"x": 1}]},
        )
    ]
    client = FakeLlmClient('[{"chunk_id":"chunk-1","score":0.5}]')

    LlmReranker(llm_client=client).rerank("query", results, top_k=1)

    prompt = client.calls[0]["user_prompt"]
    assert "metadata" not in prompt
    assert "doc_items" not in prompt
    assert "bbox" not in prompt
    assert "highlight_areas" not in prompt


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


def test_create_reranker_llm_returns_llm_reranker() -> None:
    reranker = create_reranker(
        provider="llm",
        model="gpt-4o-mini",
        enabled=True,
        injected_client=FakeLlmClient(),
    )

    assert isinstance(reranker, LlmReranker)


def test_create_reranker_llm_uses_model_from_config() -> None:
    reranker = create_reranker(
        provider="llm",
        model="custom-model",
        enabled=True,
        injected_client=FakeLlmClient(),
    )

    assert isinstance(reranker, LlmReranker)
    assert reranker.model == "custom-model"


def test_create_reranker_unknown_provider_raises_value_error() -> None:
    with pytest.raises(ValueError, match="fastembed.*llm.*noop"):
        create_reranker(provider="unknown", model="model", enabled=True)
