"""Semantic refinement for overlong or mixed chunks."""

from __future__ import annotations

import logging
import math
from typing import Any, Iterable

from rag_pipeline.text import normalize_content, split_paragraphs, split_sentences


LOGGER = logging.getLogger(__name__)


class SemanticRefiner:
    """Refine chunks with temporary embeddings and deterministic fallback."""

    def __init__(
        self,
        openai_api_key: str | None,
        embedding_provider: str = "openai",
        embedding_model: str = "text-embedding-3-small",
        gemini_api_key: str | None = None,
        gemini_output_dimensionality: int | None = None,
        max_chars: int = 2800,
        similarity_threshold: float = 0.72,
        mixed_paragraph_threshold: int = 3,
    ) -> None:
        """Create a semantic refiner.

        Args:
            openai_api_key: OpenAI API key used only for temporary refinement
                embeddings when embedding_provider is openai.
            embedding_provider: Embedding provider for temporary boundary
                detection. Supported values are openai and gemini.
            embedding_model: Embedding model for boundary detection.
            gemini_api_key: Gemini API key used only when provider is gemini.
            gemini_output_dimensionality: Optional Gemini output dimension.
            max_chars: Chunks longer than this are refined. The final result
                is also forced under this limit.
            similarity_threshold: Adjacent paragraph similarity below this
                starts a new chunk.
            mixed_paragraph_threshold: Multi-paragraph chunks at or above
                this size are checked for semantic boundaries even if they are
                not over max_chars.
        """
        self._openai_api_key = openai_api_key
        self._embedding_provider = embedding_provider.lower()
        self._embedding_model = embedding_model
        self._gemini_api_key = gemini_api_key
        self._gemini_output_dimensionality = gemini_output_dimensionality
        self._max_chars = max_chars
        self._similarity_threshold = similarity_threshold
        self._mixed_paragraph_threshold = mixed_paragraph_threshold

    def refine(
        self,
        content: str,
        chunk_kind: str = "text",
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        """Refine one chunk into one or more semantic chunks.

        Args:
            content: Initial chunk content.
            chunk_kind: Chunk kind from Docling or the source extractor.
            metadata: Optional source metadata used to detect tables.

        Returns:
            Refined chunk content. Falls back to sentence/paragraph splitting
            if embedding-based refinement fails.
        """
        normalized = normalize_content(content)
        if not normalized:
            return []

        if _is_table_chunk(chunk_kind, metadata):
            return self._ensure_max_length(
                self._table_refine(normalized),
            )

        paragraphs = split_paragraphs(normalized)
        should_refine = (
            len(normalized) > self._max_chars
            or len(paragraphs) >= self._mixed_paragraph_threshold
        )
        if not should_refine:
            return [normalized]

        if len(paragraphs) < 2:
            return self._ensure_max_length(self._fallback_refine(normalized))

        if not self._has_embedding_credentials():
            LOGGER.info(
                "Embedding credentials missing; using deterministic refinement."
            )
            return self._ensure_max_length(self._fallback_refine(normalized))

        try:
            return self._ensure_max_length(self._embedding_refine(paragraphs))
        except Exception as exc:
            LOGGER.warning(
                "Semantic refinement failed; falling back to text splitting: %s",
                exc,
            )
            return self._ensure_max_length(self._fallback_refine(normalized))

    def _embedding_refine(self, paragraphs: list[str]) -> list[str]:
        embeddings = self._embed(paragraphs)
        groups: list[list[str]] = [[paragraphs[0]]]

        for index in range(1, len(paragraphs)):
            similarity = _cosine_similarity(
                embeddings[index - 1],
                embeddings[index],
            )
            current_text = "\n\n".join(groups[-1] + [paragraphs[index]])
            if (
                similarity < self._similarity_threshold
                or len(current_text) > self._max_chars
            ):
                groups.append([paragraphs[index]])
            else:
                groups[-1].append(paragraphs[index])

        return [
            normalize_content("\n\n".join(group))
            for group in groups
            if normalize_content("\n\n".join(group))
        ]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not self._has_embedding_credentials():
            raise RuntimeError(
                f"API key missing for embedding provider: "
                f"{self._embedding_provider}"
            )

        if self._embedding_provider == "gemini":
            return self._embed_gemini(texts)
        if self._embedding_provider != "openai":
            raise RuntimeError(
                f"Unsupported embedding provider: {self._embedding_provider}"
            )
        return self._embed_openai(texts)

    def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is required for semantic refinement."
            ) from exc

        client = OpenAI(api_key=self._openai_api_key)
        response = client.embeddings.create(
            model=self._embedding_model,
            input=texts,
            encoding_format="float",
        )
        return [item.embedding for item in response.data]

    def _embed_gemini(self, texts: list[str]) -> list[list[float]]:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "The google-genai package is required for Gemini embeddings."
            ) from exc

        client = genai.Client(api_key=self._gemini_api_key)
        config = types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=self._gemini_output_dimensionality,
        )
        response = client.models.embed_content(
            model=self._embedding_model,
            contents=texts,
            config=config,
        )
        return [embedding.values for embedding in response.embeddings]

    def _has_embedding_credentials(self) -> bool:
        if self._embedding_provider == "gemini":
            return bool(self._gemini_api_key)
        return bool(self._openai_api_key)

    def _fallback_refine(self, content: str) -> list[str]:
        paragraphs = split_paragraphs(content)
        units = paragraphs if len(paragraphs) > 1 else split_sentences(content)
        return _pack_units(units, self._max_chars)

    def _table_refine(self, content: str) -> list[str]:
        rows = [row for row in content.splitlines() if row.strip()]
        units = rows if len(rows) > 1 else [content]
        return _pack_units(units, self._max_chars, separator="\n")

    def _ensure_max_length(self, chunks: list[str]) -> list[str]:
        final_chunks: list[str] = []
        for chunk in chunks:
            if len(chunk) <= self._max_chars:
                final_chunks.append(chunk)
                continue
            final_chunks.extend(
                _pack_units(split_sentences(chunk), self._max_chars)
            )

        return [
            piece
            for chunk in final_chunks
            for piece in _split_oversized_text(chunk, self._max_chars)
            if piece
        ]


def _pack_units(
    units: Iterable[str],
    max_chars: int,
    separator: str = "\n\n",
) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []

    for unit in units:
        normalized_unit = normalize_content(unit)
        if not normalized_unit:
            continue

        if len(normalized_unit) > max_chars:
            if current:
                chunks.append(normalize_content(separator.join(current)))
                current = []
            chunks.extend(_split_oversized_text(normalized_unit, max_chars))
            continue

        candidate = normalize_content(separator.join(current + [unit]))
        if current and len(candidate) > max_chars:
            chunks.append(normalize_content(separator.join(current)))
            current = [normalized_unit]
        else:
            current.append(normalized_unit)

    if current:
        chunks.append(normalize_content(separator.join(current)))

    return [chunk for chunk in chunks if chunk]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _split_oversized_text(content: str, max_chars: int) -> list[str]:
    normalized = normalize_content(content)
    if len(normalized) <= max_chars:
        return [normalized] if normalized else []

    pieces = []
    start = 0
    while start < len(normalized):
        end = min(start + max_chars, len(normalized))
        if end < len(normalized):
            boundary = normalized.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        piece = normalized[start:end].strip()
        if piece:
            pieces.append(piece)
        start = end
        while start < len(normalized) and normalized[start].isspace():
            start += 1
    return pieces


def _is_table_chunk(
    chunk_kind: str,
    metadata: dict[str, Any] | None,
) -> bool:
    if "table" in chunk_kind.lower():
        return True
    if not metadata:
        return False

    serialized = str(metadata).lower()
    if "table" in serialized:
        return True

    content_layer = metadata.get("content_layer")
    label = metadata.get("label")
    return any(
        "table" in str(value).lower()
        for value in (content_layer, label)
        if value is not None
    )
