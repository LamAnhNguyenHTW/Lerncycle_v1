"""Optional cross-encoder reranking for already retrieved RAG candidates.

FastEmbed 0.7.4 exposes the cross-encoder as:
`from fastembed.rerank.cross_encoder import TextCrossEncoder`
with `TextCrossEncoder(model_name=...)` and `rerank(query, documents)`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from typing import Any


logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    chunk_id: str
    score: float
    reason: str | None = None


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, results: list[dict], top_k: int) -> list[dict]:
        """Return reranked copies of already retrieved results."""


class NoopReranker(BaseReranker):
    """Returns first top_k results unchanged, preserving original order."""

    def rerank(self, query: str, results: list[dict], top_k: int) -> list[dict]:
        return results[:top_k]


class FastEmbedReranker(BaseReranker):
    """Cross-encoder reranker backed by a FastEmbed model."""

    def __init__(
        self,
        model: str = "jinaai/jina-reranker-v2-base-multilingual",
        injected_client: Any = None,
    ) -> None:
        self.model = model
        self.client = injected_client or self._create_client(model)

    def rerank(self, query: str, results: list[dict], top_k: int) -> list[dict]:
        if not results:
            return []

        try:
            documents = [_candidate_text(result) for result in results]
            scores = list(self.client.rerank(query, documents))
            if len(scores) != len(results):
                return results[:top_k]
            scored_results = []
            for index, (result, score) in enumerate(zip(results, scores), start=1):
                copied = dict(result)
                copied["rerank_score"] = float(score)
                copied["original_score"] = result.get("score")
                copied["original_rank"] = index
                scored_results.append(copied)
            return sorted(
                scored_results,
                key=lambda item: item["rerank_score"],
                reverse=True,
            )[:top_k]
        except Exception:
            logger.warning("Reranking failed; using original retrieval order.", exc_info=True)
            return results[:top_k]

    def _create_client(self, model: str) -> Any:
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "FastEmbed cross-encoder support is unavailable. "
                "Install a fastembed version exposing fastembed.rerank.cross_encoder.TextCrossEncoder."
            ) from exc
        return TextCrossEncoder(model_name=model)


def create_reranker(
    provider: str,
    model: str,
    enabled: bool,
    injected_client: Any = None,
) -> BaseReranker:
    if not enabled or provider == "noop":
        return NoopReranker()
    if provider == "fastembed":
        return FastEmbedReranker(model=model, injected_client=injected_client)
    raise ValueError("Unknown reranker provider. Valid providers: fastembed, noop")


def _candidate_text(result: dict) -> str:
    parts = []
    heading = result.get("heading")
    title = result.get("title")
    page_index = result.get("page_index")
    if heading:
        parts.append(str(heading))
    if title:
        parts.append(str(title))
    if isinstance(page_index, int):
        parts.append(f"Page: {page_index + 1}")
    text = str(result.get("text") or "")
    parts.append(text[:1000])
    return "\n".join(parts)
