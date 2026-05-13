"""FastAPI boundary for server-to-server RAG answers."""

from __future__ import annotations

import logging
import os
from typing import Any, Literal
from uuid import UUID

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_pipeline.config import WorkerConfig
from rag_pipeline.graph_store_factory import create_graph_store
from rag_pipeline.llm_client import OpenAILlmClient
from rag_pipeline.prompt_compaction import compress_conversation_summary
from rag_pipeline.rag_answer import answer_with_rag
from rag_pipeline.reranker import create_reranker


logger = logging.getLogger(__name__)
INTERNAL_API_KEY = os.getenv("RAG_INTERNAL_API_KEY")
if not INTERNAL_API_KEY:
    raise RuntimeError("RAG_INTERNAL_API_KEY is required to start the RAG API service.")

SourceType = Literal["pdf", "note", "annotation_comment"]
MemoryMode = Literal["off", "auto", "on"]
GraphMode = Literal["off", "auto", "on"]
WebMode = Literal["off", "on"]


class RecentMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=2000)


class RagAnswerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2000)
    user_id: str = Field(min_length=1)
    source_types: list[SourceType] | None = None
    top_k: int = Field(default=8, ge=1, le=20)
    pdf_ids: list[str] | None = None
    recent_messages: list[RecentMessage] = Field(default_factory=list, max_length=10)
    reranking_enabled: bool | None = None
    reranking_candidate_k: int | None = Field(default=None, ge=1, le=50)
    reranking_top_k: int | None = Field(default=None, ge=1, le=20)
    session_id: str | None = None
    memory_source_ids: list[str] | None = None
    memory_mode: MemoryMode = "auto"
    include_memory: bool | None = None
    graph_mode: GraphMode = "auto"
    context_summary: str | None = Field(default=None, max_length=4000)
    web_mode: WebMode = "off"
    enable_web_search: bool | None = None
    web_search_query: str | None = Field(default=None, max_length=1000)
    use_intent_classifier: bool | None = None
    use_retrieval_planner: bool | None = None

    @model_validator(mode="after")
    def validate_reranking_bounds(self) -> "RagAnswerRequest":
        if self.reranking_enabled is True:
            candidate_k = self.reranking_candidate_k or 30
            top_k = self.reranking_top_k or 8
            if candidate_k < top_k:
                raise ValueError("reranking_candidate_k must be >= reranking_top_k")
        return self

    @model_validator(mode="after")
    def validate_memory_inputs(self) -> "RagAnswerRequest":
        if self.session_id:
            try:
                UUID(self.session_id)
            except ValueError as exc:
                raise ValueError("session_id must be a valid UUID") from exc
        for source_id in self.memory_source_ids or []:
            try:
                UUID(source_id)
            except ValueError as exc:
                raise ValueError("memory_source_ids must be valid UUIDs") from exc
        return self

    @model_validator(mode="after")
    def normalize_context_summary(self) -> "RagAnswerRequest":
        if self.context_summary is not None:
            stripped = self.context_summary.strip()
            self.context_summary = stripped or None
        return self


class RagAnswerResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
    web_search: dict[str, Any] | None = None
    intent: dict[str, Any] | None = None
    retrieval_plan: dict[str, Any] | None = None


class CompressConversationRequest(BaseModel):
    messages: list[RecentMessage] = Field(default_factory=list, max_length=100)
    existing_summary: str | None = Field(default=None, max_length=3000)
    max_chars: int = Field(default=1500, ge=300, le=4000)


class CompressConversationResponse(BaseModel):
    summary: str


app = FastAPI(title="LearnCycle RAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


def require_internal_auth(authorization: str | None = Header(default=None)) -> None:
    """Validate the internal bearer token used by Next.js."""
    expected = f"Bearer {INTERNAL_API_KEY}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/rag/answer",
    response_model=RagAnswerResponse,
    dependencies=[Depends(require_internal_auth)],
)
def rag_answer(request: RagAnswerRequest) -> dict[str, Any]:
    try:
        config = WorkerConfig.from_env()
        reranking_enabled = (
            request.reranking_enabled
            if request.reranking_enabled is not None
            else config.reranking_enabled
        )
        reranking_candidate_k = request.reranking_candidate_k or config.reranking_candidate_k
        reranking_top_k = request.reranking_top_k or config.reranking_top_k
        if reranking_enabled and reranking_candidate_k < reranking_top_k:
            raise HTTPException(
                status_code=422,
                detail="reranking_candidate_k must be >= reranking_top_k",
            )
        if (
            reranking_enabled
            and config.reranking_provider == "llm"
            and reranking_candidate_k > 30
        ):
            raise HTTPException(
                status_code=422,
                detail="reranking_candidate_k must be <= 30 for llm reranking",
            )
        reranker = create_reranker(
            provider=config.reranking_provider,
            model=config.reranking_model,
            enabled=reranking_enabled,
        )
        graph_store = create_graph_store(config)
        graph_mode = request.graph_mode if config.graph_retrieval_enabled else "off"
        web_mode = "on" if request.enable_web_search is True else request.web_mode
        if not config.web_search_enabled:
            web_mode = "off"
        intent_classifier_enabled = (
            request.use_intent_classifier
            if request.use_intent_classifier is not None
            else config.intent_classifier_enabled
        )
        return answer_with_rag(
            query=request.query.strip(),
            user_id=request.user_id,
            source_types=request.source_types,
            top_k=request.top_k,
            pdf_ids=request.pdf_ids,
            recent_messages=[message.model_dump() for message in request.recent_messages],
            reranker=reranker,
            reranking_enabled=reranking_enabled,
            reranking_candidate_k=reranking_candidate_k,
            reranking_top_k=reranking_top_k,
            session_id=request.session_id,
            memory_source_ids=request.memory_source_ids,
            memory_mode=request.memory_mode,
            chat_memory_retrieval_enabled=config.chat_memory_retrieval_enabled,
            chat_memory_top_k=config.chat_memory_top_k,
            graph_retrieval_enabled=config.graph_retrieval_enabled,
            graph_mode=graph_mode,
            graph_top_k=config.graph_retrieval_top_k,
            graph_store=graph_store,
            context_summary=request.context_summary,
            web_mode=web_mode,
            web_search_enabled=config.web_search_enabled,
            web_search_query=request.web_search_query,
            web_search_top_k=config.web_search_top_k,
            web_search_provider=config.web_search_provider,
            web_search_api_key=config.tavily_api_key,
            web_search_timeout_seconds=config.web_search_timeout_seconds,
            web_search_max_query_chars=config.web_search_max_query_chars,
            web_search_max_context_sources=config.web_search_max_context_sources,
            web_search_max_chars_per_source=config.web_search_max_chars_per_source,
            web_search_max_total_context_chars=config.web_search_max_total_context_chars,
            intent_classifier_enabled=bool(intent_classifier_enabled),
            intent_classifier_config=config,
            retrieval_planner_enabled=config.retrieval_planner_enabled,
            retrieval_planner_config=config,
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("RAG answer generation failed")
        raise HTTPException(
            status_code=500,
            detail="RAG answer generation failed.",
        ) from None


@app.post(
    "/rag/compress",
    response_model=CompressConversationResponse,
    dependencies=[Depends(require_internal_auth)],
)
def rag_compress(request: CompressConversationRequest) -> dict[str, str]:
    try:
        config = WorkerConfig.from_env()
        summary = compress_conversation_summary(
            [message.model_dump() for message in request.messages],
            llm_client=OpenAILlmClient(
                api_key=config.openai_api_key,
                model="gpt-4o-mini",
            ),
            max_chars=request.max_chars,
            existing_summary=request.existing_summary,
        )
        return {"summary": summary}
    except Exception:
        logger.exception("RAG conversation compression failed")
        raise HTTPException(
            status_code=500,
            detail="RAG conversation compression failed.",
        ) from None
