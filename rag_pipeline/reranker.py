"""Optional cross-encoder reranking for already retrieved RAG candidates.

FastEmbed 0.7.4 exposes the cross-encoder as:
`from fastembed.rerank.cross_encoder import TextCrossEncoder`
with `TextCrossEncoder(model_name=...)` and `rerank(query, documents)`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import logging
import re
from typing import Any

from rag_pipeline.llm_client import OpenAILlmClient


logger = logging.getLogger(__name__)
VALID_RERANKER_PROVIDERS = ("fastembed", "llm", "noop")


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


class LlmReranker(BaseReranker):
    """LLM scorer for already retrieved, already scoped RAG candidates."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        llm_client: Any = None,
        max_candidate_chars: int = 700,
    ) -> None:
        self.model = model
        self.llm_client = llm_client or OpenAILlmClient(model=model)
        self.max_candidate_chars = max_candidate_chars

    def rerank(self, query: str, results: list[dict], top_k: int) -> list[dict]:
        if not results:
            return []

        try:
            user_prompt = self._build_user_prompt(query, results)
            response = self.llm_client.complete(
                system_prompt=(
                    "You are a reranker for retrieved learning-material chunks. "
                    "Score each candidate by how factually useful it is for answering "
                    "the user's question. Return only valid JSON. Do not invent chunk "
                    "IDs. Only score candidates you were given."
                ),
                user_prompt=user_prompt,
            )
            scores = self._parse_response(response)
            return self._rank_results(results, scores, top_k)
        except Exception:
            logger.warning(
                "LLM reranking failed; using original retrieval order.",
                exc_info=True,
            )
            return results[:top_k]

    def _build_user_prompt(self, query: str, results: list[dict]) -> str:
        candidates = []
        for index, result in enumerate(results, start=1):
            page = result.get("page_index")
            page_text = str(page + 1) if isinstance(page, int) else ""
            text = str(result.get("text") or "")[: self.max_candidate_chars]
            candidates.append(
                "\n".join(
                    [
                        f"{index}. chunk_id={result.get('chunk_id', '')}",
                        f"source_type={result.get('source_type', '')}",
                        f"title={result.get('title') or ''}",
                        f"heading={result.get('heading') or ''}",
                        f"page={page_text}",
                        f"text={text}",
                    ]
                )
            )
        candidate_block = "\n\n".join(candidates)
        return (
            f"Question:\n{query}\n\n"
            f"Candidates:\n{candidate_block}\n\n"
            "Return JSON array only, no explanation outside JSON:\n"
            '[{"chunk_id": "...", "score": 0.0-1.0, "reason": "one-line reason"}]'
        )

    def _parse_response(self, response: str) -> list[RerankResult]:
        parsed = json.loads(_strip_json_code_fence(response))
        if not isinstance(parsed, list):
            raise ValueError("LLM reranker response must be a JSON array.")

        scores = []
        seen_chunk_ids = set()
        for item in parsed:
            if not isinstance(item, dict):
                raise ValueError("LLM reranker items must be objects.")
            chunk_id = item.get("chunk_id")
            score = item.get("score")
            if not isinstance(chunk_id, str):
                raise ValueError("LLM reranker item chunk_id must be a string.")
            if not isinstance(score, (int, float)) or isinstance(score, bool):
                raise ValueError("LLM reranker item score must be numeric.")
            score_float = float(score)
            if score_float < 0 or score_float > 1:
                raise ValueError("LLM reranker item score must be between 0 and 1.")
            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)
            reason = item.get("reason")
            scores.append(
                RerankResult(
                    chunk_id=chunk_id,
                    score=score_float,
                    reason=reason if isinstance(reason, str) else None,
                )
            )
        return scores

    def _rank_results(
        self,
        results: list[dict],
        scores: list[RerankResult],
        top_k: int,
    ) -> list[dict]:
        by_chunk_id = {}
        original_ranks = {}
        for index, result in enumerate(results, start=1):
            chunk_id = str(result.get("chunk_id"))
            if chunk_id not in by_chunk_id:
                by_chunk_id[chunk_id] = result
                original_ranks[chunk_id] = index
        matched = []
        matched_ids = set()
        for score in scores:
            result = by_chunk_id.get(score.chunk_id)
            if result is None:
                continue
            matched_ids.add(score.chunk_id)
            copied = dict(result)
            copied["rerank_score"] = score.score
            copied["original_score"] = result.get("score")
            copied["original_rank"] = original_ranks[score.chunk_id]
            if score.reason:
                copied["rerank_reason"] = score.reason
            matched.append(copied)

        matched.sort(key=lambda item: item["rerank_score"], reverse=True)
        for index, result in enumerate(results, start=1):
            chunk_id = str(result.get("chunk_id"))
            if chunk_id in matched_ids:
                continue
            copied = dict(result)
            copied["original_score"] = result.get("score")
            copied["original_rank"] = index
            matched.append(copied)
        return matched[:top_k]


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
    if provider == "llm":
        return LlmReranker(model=model, llm_client=injected_client)
    raise ValueError(
        "Unknown reranker provider. Valid providers: "
        + ", ".join(VALID_RERANKER_PROVIDERS)
    )


def _strip_json_code_fence(response: str) -> str:
    stripped = response.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return stripped


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
