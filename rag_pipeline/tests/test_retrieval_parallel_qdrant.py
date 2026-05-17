from __future__ import annotations

import time
from types import SimpleNamespace

from rag_pipeline.retrieval import search_hybrid_chunks
from rag_pipeline.sparse_embeddings import SparseVectorData


class DenseEmbedder:
    def embed(self, texts):
        return [[0.1, 0.2]]


class SparseEmbedder:
    def embed(self, texts):
        return [SparseVectorData(indices=[1], values=[0.5])]


class FallbackStore:
    def search_hybrid_chunks(self, *args, **kwargs):
        raise NotImplementedError

    def search_chunks(self, *args, **kwargs):
        time.sleep(0.2)
        return [
            SimpleNamespace(
                score=0.9,
                payload={
                    "chunk_id": "chunk-dense",
                    "text": "dense",
                    "source_type": "pdf",
                    "source_id": "pdf-1",
                    "pdf_id": "pdf-1",
                    "metadata": {},
                },
            )
        ]

    def search_sparse_chunks(self, *args, **kwargs):
        time.sleep(0.2)
        return [
            SimpleNamespace(
                score=0.8,
                payload={
                    "chunk_id": "chunk-sparse",
                    "text": "sparse",
                    "source_type": "pdf",
                    "source_id": "pdf-1",
                    "pdf_id": "pdf-1",
                    "metadata": {},
                },
            )
        ]


def test_hybrid_search_local_fallback_runs_dense_and_sparse_qdrant_calls_concurrently() -> None:
    started = time.perf_counter()

    search_hybrid_chunks(
        "query",
        "user-1",
        embedder=DenseEmbedder(),
        sparse_embedder=SparseEmbedder(),
        store=FallbackStore(),
    )

    assert time.perf_counter() - started <= 0.35
