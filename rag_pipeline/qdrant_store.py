"""Qdrant-backed dense retrieval index for RAG chunks."""

from __future__ import annotations

from typing import Any


VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


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

    def ensure_collection(self, dim: int, sparse_enabled: bool = False) -> None:
        if self._collection_exists():
            if sparse_enabled:
                self._ensure_existing_collection_supports_hybrid()
            return
        self._client.create_collection(**self._collection_config(dim, sparse_enabled))

    def recreate_collection_for_hybrid(self, dim: int) -> None:
        """Explicitly recreate the collection for dense+sparse hybrid search."""
        config = self._collection_config(dim, sparse_enabled=True)
        if hasattr(self._client, "recreate_collection"):
            self._client.recreate_collection(**config)
            return
        self._client.delete_collection(collection_name=self.collection_name)
        self._client.create_collection(**config)

    def _collection_config(
        self,
        dim: int,
        sparse_enabled: bool,
    ) -> dict[str, Any]:
        models = _models()
        config: dict[str, Any] = {
            "collection_name": self.collection_name,
            "vectors_config": {
                VECTOR_NAME: models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE,
                )
            },
        }
        if sparse_enabled:
            config["sparse_vectors_config"] = {
                SPARSE_VECTOR_NAME: _sparse_vector_params()
            }
        return config

    def _ensure_existing_collection_supports_hybrid(self) -> None:
        info = self._client.get_collection(self.collection_name)
        params = info.config.params
        vectors = getattr(params, "vectors", {}) or {}
        sparse_vectors = getattr(params, "sparse_vectors", {}) or {}
        if VECTOR_NAME in vectors and SPARSE_VECTOR_NAME in sparse_vectors:
            return
        raise RuntimeError(
            "Qdrant collection exists without required dense+sparse hybrid "
            "vectors. Run QdrantStore.recreate_collection_for_hybrid(dim) "
            "or the explicit reindex maintenance command."
        )

    def upsert_chunks(
        self,
        chunks: list[dict[str, Any]],
        sparse_enabled: bool = False,
    ) -> None:
        if not chunks:
            return
        models = _models()
        points = [
            models.PointStruct(
                id=chunk["id"],
                vector=_point_vectors(chunk, sparse_enabled),
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
        pdf_ids: list[str] | None = None,
    ) -> list[Any]:
        query_filter = _filter({"user_id": user_id}, source_types=source_types, pdf_ids=pdf_ids)
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

    def search_sparse_chunks(
        self,
        query_vector: Any,
        user_id: str,
        source_types: list[str] | None,
        top_k: int,
        pdf_ids: list[str] | None = None,
    ) -> list[Any]:
        models = _models()
        query_filter = _filter({"user_id": user_id}, source_types=source_types, pdf_ids=pdf_ids)
        sparse_query = models.SparseVector(
            indices=list(query_vector.indices),
            values=list(query_vector.values),
        )
        result = self._client.query_points(
            collection_name=self.collection_name,
            query=sparse_query,
            using=SPARSE_VECTOR_NAME,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        return getattr(result, "points", result)

    def search_hybrid_chunks(
        self,
        dense_vector: list[float],
        sparse_vector: Any,
        user_id: str,
        source_types: list[str] | None,
        top_k: int,
        prefetch_limit: int,
        pdf_ids: list[str] | None = None,
    ) -> list[Any]:
        models = _models()
        query_filter = _filter({"user_id": user_id}, source_types=source_types, pdf_ids=pdf_ids)
        sparse_query = models.SparseVector(
            indices=list(sparse_vector.indices),
            values=list(sparse_vector.values),
        )
        if not hasattr(models, "Prefetch") or not hasattr(models, "FusionQuery"):
            raise NotImplementedError("Qdrant server-side hybrid query is unavailable.")
        result = self._client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense_vector,
                    using=VECTOR_NAME,
                    filter=query_filter,
                    limit=prefetch_limit,
                ),
                models.Prefetch(
                    query=sparse_query,
                    using=SPARSE_VECTOR_NAME,
                    filter=query_filter,
                    limit=prefetch_limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        return getattr(result, "points", result)

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
    pdf_ids: list[str] | None = None,
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
    if pdf_ids:
        must.append(
            models.FieldCondition(
                key="pdf_id",
                match=models.MatchAny(any=pdf_ids),
            )
        )
    return models.Filter(must=must)


def _filter_selector(query_filter: Any) -> Any:
    models = _models()
    if hasattr(models, "FilterSelector"):
        return models.FilterSelector(filter=query_filter)
    return query_filter


def _sparse_vector_params() -> Any:
    models = _models()
    modifier = getattr(getattr(models, "Modifier", None), "IDF", None)
    if modifier is None:
        return models.SparseVectorParams()
    return models.SparseVectorParams(modifier=modifier)


def _point_vectors(chunk: dict[str, Any], sparse_enabled: bool) -> dict[str, Any]:
    models = _models()
    dense = chunk.get("embedding")
    if dense is None:
        raise RuntimeError("Qdrant upsert requires a dense embedding.")
    vectors: dict[str, Any] = {VECTOR_NAME: dense}
    if not sparse_enabled:
        return vectors

    sparse = chunk.get("sparse_embedding")
    if sparse is None:
        raise RuntimeError("Qdrant hybrid upsert requires a sparse embedding.")
    if isinstance(sparse, dict):
        indices = sparse["indices"]
        values = sparse["values"]
    else:
        indices = sparse.indices
        values = sparse.values
    vectors[SPARSE_VECTOR_NAME] = models.SparseVector(
        indices=list(indices),
        values=list(values),
    )
    return vectors


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
