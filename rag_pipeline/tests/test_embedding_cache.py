from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import time

from rag_pipeline.embedding_cache import EmbeddingCacheKey
from rag_pipeline.embedding_cache import QueryEmbeddingCache
from rag_pipeline.embedding_cache import normalize_query_for_embedding


def _key(query: str = "process mining") -> EmbeddingCacheKey:
    return EmbeddingCacheKey(
        normalized_query=normalize_query_for_embedding(query),
        provider="openai",
        model="text-embedding-3-small",
        embedding_kind="dense",
    )


def test_embedding_cache_hit_skips_underlying_embedder() -> None:
    cache = QueryEmbeddingCache(max_entries=2)
    calls = 0

    def compute() -> list[float]:
        nonlocal calls
        calls += 1
        return [0.1, 0.2]

    assert cache.get_or_compute(_key(), compute) == [0.1, 0.2]
    assert cache.get_or_compute(_key(), compute) == [0.1, 0.2]
    assert calls == 1
    assert cache.hits == 1
    assert cache.misses == 1


def test_embedding_cache_key_ignores_source_scope() -> None:
    assert _key(" Process   Mining ") == _key("process mining")


def test_embedding_cache_lru_eviction() -> None:
    cache = QueryEmbeddingCache(max_entries=2)
    cache.get_or_compute(_key("one"), lambda: [1.0])
    cache.get_or_compute(_key("two"), lambda: [2.0])
    cache.get_or_compute(_key("one"), lambda: [9.0])
    cache.get_or_compute(_key("three"), lambda: [3.0])

    assert cache.get_or_compute(_key("two"), lambda: [22.0]) == [22.0]


def test_embedding_cache_collapses_concurrent_identical_misses() -> None:
    cache = QueryEmbeddingCache(max_entries=2)
    calls = 0

    def compute() -> list[float]:
        nonlocal calls
        calls += 1
        time.sleep(0.1)
        return [0.1]

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: cache.get_or_compute(_key(), compute), range(4)))

    assert results == [[0.1], [0.1], [0.1], [0.1]]
    assert calls == 1
