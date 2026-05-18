"""Process-local retrieval result cache."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import threading
import time
from typing import Any


@dataclass(frozen=True)
class RetrievalResultCacheKey:
    user_id: str
    normalized_query: str
    source_types: tuple[str, ...]
    source_ids: tuple[str, ...]
    source_version_hash: str
    retrieval_mode: str
    top_k: int
    candidate_k: int
    embedding_provider: str
    embedding_model: str
    sparse_provider: str
    sparse_model: str
    qdrant_collection: str
    hybrid_strategy: str
    filter_hash: str


class RetrievalResultCache:
    def __init__(self, max_entries: int = 512, ttl_s: int = 300) -> None:
        self.max_entries = max_entries
        self.ttl_s = ttl_s
        self.hits = 0
        self.misses = 0
        self._values: OrderedDict[RetrievalResultCacheKey, tuple[float, list[dict[str, Any]]]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: RetrievalResultCacheKey) -> list[dict[str, Any]] | None:
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

    def set(self, key: RetrievalResultCacheKey, value: list[dict[str, Any]]) -> None:
        expires_at = time.monotonic() + self.ttl_s
        stored = [_cacheable_result(item) for item in value]
        with self._lock:
            self._values[key] = (expires_at, stored)
            self._values.move_to_end(key)
            while len(self._values) > self.max_entries:
                self._values.popitem(last=False)


default_retrieval_result_cache = RetrievalResultCache()


def build_filter_hash(
    *,
    source_types: list[str] | None,
    pdf_ids: list[str] | None,
    source_ids: list[str] | None,
) -> str:
    return _stable_hash(
        {
            "source_types": sorted(source_types or []),
            "pdf_ids": sorted(pdf_ids or []),
            "source_ids": sorted(source_ids or []),
        }
    )


def build_source_version_hash(
    *,
    source_rows: list[dict[str, Any]] | None = None,
    source_ids: list[str] | None = None,
) -> str:
    rows = source_rows or [{"source_id": source_id} for source_id in (source_ids or [])]
    parts = []
    for row in rows:
        source_id = row.get("source_id") or row.get("id") or row.get("pdf_id") or row.get("note_id") or row.get("annotation_id")
        parts.append(
            {
                "source_id": source_id,
                "source_type": row.get("source_type"),
                "content_hash": row.get("content_hash"),
                "updated_at": row.get("updated_at"),
                "index_updated_at": _latest_index_source_update(row.get("index_jobs") or []),
                "chat_memory_updated_at": row.get("chat_memory_updated_at"),
            }
        )
    parts.sort(key=lambda item: str(item.get("source_id") or ""))
    return _stable_hash(parts)


def make_retrieval_cache_key(
    *,
    user_id: str,
    query: str,
    source_types: list[str] | None,
    source_ids: list[str] | None,
    source_version_hash: str,
    retrieval_mode: str,
    top_k: int,
    candidate_k: int,
    config: Any,
    hybrid_strategy: str,
    filter_hash: str,
) -> RetrievalResultCacheKey:
    from rag_pipeline.embedding_cache import normalize_query_for_embedding

    return RetrievalResultCacheKey(
        user_id=user_id,
        normalized_query=normalize_query_for_embedding(query),
        source_types=tuple(sorted(source_types or [])),
        source_ids=tuple(sorted(source_ids or [])),
        source_version_hash=source_version_hash,
        retrieval_mode=retrieval_mode,
        top_k=top_k,
        candidate_k=candidate_k,
        embedding_provider=getattr(config, "embedding_provider", ""),
        embedding_model=getattr(config, "embedding_model", ""),
        sparse_provider=getattr(config, "sparse_provider", ""),
        sparse_model=getattr(config, "sparse_model", ""),
        qdrant_collection=getattr(config, "qdrant_collection", ""),
        hybrid_strategy=hybrid_strategy,
        filter_hash=filter_hash,
    )


def _cacheable_result(result: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(result.get("metadata") or {})
    return {
        "chunk_id": result.get("chunk_id"),
        "text": result.get("text"),
        "score": result.get("score"),
        "source_type": result.get("source_type"),
        "source_id": result.get("source_id"),
        "page_index": result.get("page_index"),
        "title": result.get("title"),
        "heading": result.get("heading"),
        "metadata": metadata,
        "pdf_id": result.get("pdf_id"),
    }


def _latest_index_source_update(index_jobs: list[dict[str, Any]]) -> str | None:
    updates = [
        str(job.get("updated_at"))
        for job in index_jobs
        if job.get("job_kind") == "index_source" and job.get("status") == "completed" and job.get("updated_at")
    ]
    return max(updates) if updates else None


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
