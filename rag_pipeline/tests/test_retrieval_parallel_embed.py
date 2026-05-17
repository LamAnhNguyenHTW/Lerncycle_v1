from __future__ import annotations

import time
from types import SimpleNamespace

from rag_pipeline.retrieval import search_hybrid_chunks
from rag_pipeline.sparse_embeddings import SparseVectorData


class SlowDenseEmbedder:
    def embed(self, texts):
        time.sleep(0.2)
        return [[0.1, 0.2]]


class SlowSparseEmbedder:
    def embed(self, texts):
        time.sleep(0.2)
        return [SparseVectorData(indices=[1], values=[0.5])]


class Store:
    def search_hybrid_chunks(self, *args, **kwargs):
        return [
            SimpleNamespace(
                score=0.9,
                payload={
                    "chunk_id": "chunk-1",
                    "text": "text",
                    "source_type": "pdf",
                    "source_id": "pdf-1",
                    "pdf_id": "pdf-1",
                    "metadata": {},
                },
            )
        ]


def test_hybrid_search_embeds_dense_and_sparse_query_concurrently() -> None:
    started = time.perf_counter()

    search_hybrid_chunks(
        "query",
        "user-1",
        embedder=SlowDenseEmbedder(),
        sparse_embedder=SlowSparseEmbedder(),
        store=Store(),
    )

    assert time.perf_counter() - started <= 0.35
