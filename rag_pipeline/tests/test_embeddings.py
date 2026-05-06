from __future__ import annotations

import pytest

from rag_pipeline.embeddings import Embedder


class FakeEmbeddings:
    def __init__(self) -> None:
        self.inputs: list[list[str]] = []

    def create(self, model: str, input: list[str]):
        self.inputs.append(input)
        data = []
        for text in input:
            data.append(type("Embedding", (), {"embedding": [float(len(text)), 1.0]})())
        return type("Response", (), {"data": data})()


class FakeOpenAI:
    def __init__(self) -> None:
        self.embeddings = FakeEmbeddings()


def test_openai_embedder_batches_inputs() -> None:
    client = FakeOpenAI()
    embedder = Embedder("openai", "model", "key", batch_size=2, openai_client=client)

    vectors = embedder.embed(["a", "bb", "ccc"])

    assert client.embeddings.inputs == [["a", "bb"], ["ccc"]]
    assert vectors == [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]


def test_embedder_raises_without_api_key() -> None:
    embedder = Embedder("openai", "model", None)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        embedder.embed(["text"])


def test_gemini_branch_raises_not_implemented() -> None:
    embedder = Embedder("gemini", "model", None, gemini_api_key="key")

    with pytest.raises(NotImplementedError):
        embedder.embed(["text"])


def test_embedder_caches_dimension_after_first_call() -> None:
    embedder = Embedder(
        "openai",
        "model",
        "key",
        openai_client=FakeOpenAI(),
    )

    assert embedder.dimension is None
    embedder.embed(["abcd"])
    assert embedder.dimension == 2


def test_embedder_returns_empty_list_for_empty_input() -> None:
    client = FakeOpenAI()
    embedder = Embedder("openai", "model", "key", openai_client=client)

    assert embedder.embed([]) == []
    assert client.embeddings.inputs == []
