"""Process-local query embedding cache with LRU eviction and single-flight misses."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import threading
from typing import Callable


@dataclass(frozen=True)
class EmbeddingCacheKey:
    normalized_query: str
    provider: str
    model: str
    embedding_kind: str


class QueryEmbeddingCache:
    def __init__(self, max_entries: int = 512) -> None:
        self.max_entries = max_entries
        self.hits = 0
        self.misses = 0
        self._values: OrderedDict[EmbeddingCacheKey, list[float]] = OrderedDict()
        self._inflight: dict[EmbeddingCacheKey, threading.Event] = {}
        self._lock = threading.Lock()

    def get_or_compute(
        self,
        key: EmbeddingCacheKey,
        compute: Callable[[], list[float]],
    ) -> list[float]:
        with self._lock:
            cached = self._values.get(key)
            if cached is not None:
                self.hits += 1
                self._values.move_to_end(key)
                return list(cached)
            event = self._inflight.get(key)
            if event is None:
                self.misses += 1
                event = threading.Event()
                self._inflight[key] = event
                owner = True
            else:
                owner = False

        if not owner:
            event.wait()
            with self._lock:
                cached = self._values[key]
                self.hits += 1
                self._values.move_to_end(key)
                return list(cached)

        try:
            value = list(compute())
            with self._lock:
                self._values[key] = value
                self._values.move_to_end(key)
                while len(self._values) > self.max_entries:
                    self._values.popitem(last=False)
                return list(value)
        finally:
            with self._lock:
                self._inflight.pop(key, None)
                event.set()


default_query_embedding_cache = QueryEmbeddingCache()


def normalize_query_for_embedding(query: str) -> str:
    return " ".join(query.strip().lower().split())
