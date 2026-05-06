"""Sparse embedding support for hybrid retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class SparseVectorData:
    """Sparse vector data accepted by Qdrant."""

    indices: list[int]
    values: list[float]


class SparseEmbedder:
    """Create sparse vectors with FastEmbed BM25."""

    def __init__(
        self,
        provider: str,
        model: str,
        fastembed_model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self._fastembed_model_factory = fastembed_model_factory
        self._fastembed_model: Any | None = None

    def embed(self, texts: list[str]) -> list[SparseVectorData]:
        """Embed texts as sparse vectors and preserve input order."""
        if not texts:
            return []
        if self.provider != "fastembed":
            raise RuntimeError(
                f"Unsupported sparse embedding provider: {self.provider}"
            )
        model = self._get_fastembed_model()
        return [
            SparseVectorData(
                indices=[int(index) for index in vector.indices],
                values=[float(value) for value in vector.values],
            )
            for vector in model.embed(texts)
        ]

    def _get_fastembed_model(self) -> Any:
        if self._fastembed_model is not None:
            return self._fastembed_model
        factory = self._fastembed_model_factory
        if factory is None:
            try:
                from fastembed import SparseTextEmbedding
            except ImportError as exc:
                raise RuntimeError(
                    "fastembed is required for sparse embeddings. Install "
                    "rag_pipeline requirements with qdrant-client[fastembed]."
                ) from exc
            factory = SparseTextEmbedding
        self._fastembed_model = factory(model_name=self.model)
        return self._fastembed_model
