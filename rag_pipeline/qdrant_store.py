"""Qdrant-backed dense retrieval index for RAG chunks."""

from __future__ import annotations

from typing import Any


VECTOR_NAME = "dense"


class QdrantStore:
    """Small adapter around qdrant-client for chunk indexing and search."""

    def __init__(
        self,
        url: str | None,
        api_key: str | None,
        collection_name: str,
        client: Any | None = None,
    ) -> None:
        if not url and client is None:
            raise RuntimeError("QDRANT_URL is required for durable retrieval indexing.")
        self.collection_name = collection_name
        self._client = client or self._create_client(url or "", api_key)

    def ensure_collection(self, dim: int) -> None:
        if self._collection_exists():
            return
        models = _models()
        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                VECTOR_NAME: models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE,
                )
            },
        )

    def upsert_chunks(self, chunks: list[dict[str, Any]]) -> None:
        if not chunks:
            return
        models = _models()
        points = [
            models.PointStruct(
                id=chunk["id"],
                vector={VECTOR_NAME: chunk["embedding"]},
                payload=_payload(chunk),
            )
            for chunk in chunks
        ]
        self._client.upsert(collection_name=self.collection_name, points=points)

    def delete_points_by_source(
        self,
        user_id: str,
        source_type: str,
        source_id: str,
    ) -> None:
        if not self._collection_exists():
            return
        self._client.delete(
            collection_name=self.collection_name,
            points_selector=_filter_selector(
                _filter(
                    {
                        "user_id": user_id,
                        "source_type": source_type,
                        "source_id": source_id,
                    }
                )
            ),
        )

    def search_chunks(
        self,
        query_vector: list[float],
        user_id: str,
        source_types: list[str] | None,
        top_k: int,
    ) -> list[Any]:
        query_filter = _filter({"user_id": user_id}, source_types=source_types)
        if hasattr(self._client, "query_points"):
            result = self._client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                using=VECTOR_NAME,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )
            return getattr(result, "points", result)
        return self._client.search(
            collection_name=self.collection_name,
            query_vector=(VECTOR_NAME, query_vector),
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

    def _collection_exists(self) -> bool:
        if hasattr(self._client, "collection_exists"):
            return bool(self._client.collection_exists(self.collection_name))
        try:
            self._client.get_collection(self.collection_name)
            return True
        except Exception:
            return False

    @staticmethod
    def _create_client(url: str, api_key: str | None) -> Any:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise RuntimeError("qdrant-client is required for Qdrant indexing.") from exc
        return QdrantClient(url=url, api_key=api_key)


def _models() -> Any:
    try:
        from qdrant_client import models
    except ImportError as exc:
        raise RuntimeError("qdrant-client is required for Qdrant indexing.") from exc
    return models


def _filter(
    matches: dict[str, Any],
    source_types: list[str] | None = None,
) -> Any:
    models = _models()
    must = [
        models.FieldCondition(key=key, match=models.MatchValue(value=value))
        for key, value in matches.items()
    ]
    if source_types:
        must.append(
            models.FieldCondition(
                key="source_type",
                match=models.MatchAny(any=source_types),
            )
        )
    return models.Filter(must=must)


def _filter_selector(query_filter: Any) -> Any:
    models = _models()
    if hasattr(models, "FilterSelector"):
        return models.FilterSelector(filter=query_filter)
    return query_filter


def _payload(chunk: dict[str, Any]) -> dict[str, Any]:
    fields = [
        "chunk_id",
        "user_id",
        "source_type",
        "source_id",
        "pdf_id",
        "note_id",
        "annotation_id",
        "document_id",
        "page_index",
        "title",
        "heading",
        "text",
        "metadata",
        "content_hash",
        "chunk_index",
        "created_at",
        "updated_at",
    ]
    payload = {key: chunk.get(key) for key in fields if chunk.get(key) is not None}
    payload.setdefault("chunk_id", chunk.get("id"))
    return {key: value for key, value in payload.items() if value is not None}
