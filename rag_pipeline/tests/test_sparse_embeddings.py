from __future__ import annotations

import builtins

import pytest

from rag_pipeline.sparse_embeddings import SparseEmbedder, SparseVectorData


class FakeSparseEmbedding:
    def __init__(self, indices, values) -> None:
        self.indices = indices
        self.values = values


class FakeFastEmbedModel:
    created_with: list[str] = []

    def __init__(self, model_name: str) -> None:
        self.created_with.append(model_name)

    def embed(self, texts: list[str]):
        for index, text in enumerate(texts):
            yield FakeSparseEmbedding([index, len(text)], [1.0, float(len(text))])


def test_sparse_embedder_returns_empty_list_for_empty_input() -> None:
    embedder = SparseEmbedder(
        provider="fastembed",
        model="Qdrant/bm25",
        fastembed_model_factory=FakeFastEmbedModel,
    )

    assert embedder.embed([]) == []
    assert FakeFastEmbedModel.created_with == []


def test_sparse_embedder_uses_fastembed_bm25() -> None:
    FakeFastEmbedModel.created_with = []
    embedder = SparseEmbedder(
        provider="fastembed",
        model="Qdrant/bm25",
        fastembed_model_factory=FakeFastEmbedModel,
    )

    embedder.embed(["Process Mining"])

    assert FakeFastEmbedModel.created_with == ["Qdrant/bm25"]


def test_sparse_embedder_preserves_order() -> None:
    embedder = SparseEmbedder(
        provider="fastembed",
        model="Qdrant/bm25",
        fastembed_model_factory=FakeFastEmbedModel,
    )

    vectors = embedder.embed(["a", "abcd"])

    assert [vector.indices for vector in vectors] == [[0, 1], [1, 4]]


def test_sparse_embedder_returns_indices_and_values() -> None:
    embedder = SparseEmbedder(
        provider="fastembed",
        model="Qdrant/bm25",
        fastembed_model_factory=FakeFastEmbedModel,
    )

    vectors = embedder.embed(["abc"])

    assert vectors == [SparseVectorData(indices=[0, 3], values=[1.0, 3.0])]


def test_sparse_embedder_raises_for_unknown_provider() -> None:
    embedder = SparseEmbedder(provider="unknown", model="Qdrant/bm25")

    with pytest.raises(RuntimeError, match="Unsupported sparse embedding provider"):
        embedder.embed(["text"])


def test_sparse_embedder_raises_clear_error_when_dependency_missing(
    monkeypatch,
) -> None:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "fastembed":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    embedder = SparseEmbedder(provider="fastembed", model="Qdrant/bm25")

    with pytest.raises(RuntimeError, match="fastembed"):
        embedder.embed(["text"])
