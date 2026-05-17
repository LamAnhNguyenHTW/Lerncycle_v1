"""Process-local reranker result cache."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import threading
import time
from typing import Any

from rag_pipeline.embedding_cache import normalize_query_for_embedding


@dataclass(frozen=True)
class RerankerCacheKey:
    normalized_query: str
    reranker_provider: str
    reranker_model: str
    candidate_chunk_ids: tuple[str, ...]
    candidate_content_hashes: tuple[str, ...]


class RerankerResultCache:
    def __init__(self, max_entries: int = 512, ttl_s: int = 300) -> None:
        self.max_entries = max_entries
        self.ttl_s = ttl_s
        self.hits = 0
        self.misses = 0
        self._values: OrderedDict[RerankerCacheKey, tuple[float, list[dict[str, Any]]]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: RerankerCacheKey) -> list[dict[str, Any]] | None:
        now = time.monotonic()
        with self._lock:
            entry = self._values.get(key)
            if entry is None:
                self.misses += 1
                return None
            expires_at, value = entry
            if expires_at <= now:
                self._values.pop(key, None)
                self.misses += 1
                return None
            self.hits += 1
            self._values.move_to_end(key)
            return [dict(item) for item in value]

    def set(self, key: RerankerCacheKey, value: list[dict[str, Any]]) -> None:
        expires_at = time.monotonic() + self.ttl_s
        with self._lock:
            self._values[key] = (expires_at, [dict(item) for item in value])
            self._values.move_to_end(key)
            while len(self._values) > self.max_entries:
                self._values.popitem(last=False)


default_reranker_cache = RerankerResultCache()


def make_reranker_cache_key(
    *,
    query: str,
    reranker_provider: str,
    reranker_model: str,
    candidates: list[dict[str, Any]],
) -> RerankerCacheKey:
    return RerankerCacheKey(
        normalized_query=normalize_query_for_embedding(query),
        reranker_provider=reranker_provider,
        reranker_model=reranker_model,
        candidate_chunk_ids=tuple(str(item.get("chunk_id") or "") for item in candidates),
        candidate_content_hashes=tuple(
            str((item.get("metadata") or {}).get("content_hash") or item.get("content_hash") or "")
            for item in candidates
        ),
    )
