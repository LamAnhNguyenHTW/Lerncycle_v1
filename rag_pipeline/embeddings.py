"""Provider-switchable durable embedding client."""

from __future__ import annotations

from typing import Any


class Embedder:
    """Create dense embeddings for persisted RAG chunks."""

    def __init__(
        self,
        provider: str,
        model: str,
        openai_api_key: str | None,
        gemini_api_key: str | None = None,
        batch_size: int = 100,
        openai_client: Any | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.openai_api_key = openai_api_key
        self.gemini_api_key = gemini_api_key
        self.batch_size = batch_size
        self.dimension: int | None = None
        self._openai_client = openai_client

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts and preserve input order."""
        if not texts:
            return []
        if self.provider == "openai":
            vectors = self._embed_openai(texts)
        elif self.provider == "gemini":
            vectors = self._embed_gemini(texts)
        else:
            raise RuntimeError(f"Unsupported embedding provider: {self.provider}")

        if len(vectors) != len(texts):
            raise RuntimeError(
                "Embedding provider returned "
                f"{len(vectors)} vectors for {len(texts)} inputs."
            )
        if vectors and self.dimension is None:
            self.dimension = len(vectors[0])
        return vectors

    def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        if not self.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for durable embeddings.")

        client = self._openai_client
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("openai is required for OpenAI embeddings.") from exc
            client = OpenAI(api_key=self.openai_api_key)
            self._openai_client = client

        vectors: list[list[float]] = []
        size = max(1, self.batch_size)
        for start in range(0, len(texts), size):
            batch = texts[start : start + size]
            response = client.embeddings.create(model=self.model, input=batch)
            vectors.extend([item.embedding for item in response.data])
        return vectors

    def _embed_gemini(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("Durable Gemini embeddings are not implemented yet.")
