"""FastAPI boundary for server-to-server RAG answers."""

from __future__ import annotations

import logging
import os
import json
import time
from contextlib import asynccontextmanager
from typing import Any, Literal
from uuid import UUID

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_pipeline.config import WorkerConfig
from rag_pipeline.embedding_cache import default_query_embedding_cache
from rag_pipeline.result_cache import default_retrieval_result_cache
from rag_pipeline.reranker_cache import default_reranker_cache
from rag_pipeline.graph_store_factory import create_graph_store
from rag_pipeline.learning_structure.retrieval import get_document_learning_tree
from rag_pipeline.llm_client import OpenAILlmClient
from rag_pipeline.prompt_compaction import compress_conversation_summary
from rag_pipeline.embeddings import Embedder
from rag_pipeline.qdrant_store import QdrantStore
from rag_pipeline.rag_answer import answer_with_rag
from rag_pipeline.rag_answer import stream_answer_with_rag
from rag_pipeline.reranker import create_reranker
from rag_pipeline.retrieval import search_hybrid_chunks
from rag_pipeline.sparse_embeddings import SparseEmbedder
from rag_pipeline.retrieval_tools import build_default_retrieval_tool_registry
from rag_pipeline.revision.generator import generate_flashcards, generate_mock_test


logger = logging.getLogger(__name__)
STARTED_AT = time.monotonic()
INTERNAL_API_KEY = os.getenv("RAG_INTERNAL_API_KEY")
if not INTERNAL_API_KEY:
    raise RuntimeError("RAG_INTERNAL_API_KEY is required to start the RAG API service.")

SourceType = Literal["pdf", "note", "annotation_comment"]
MemoryMode = Literal["off", "auto", "on"]
GraphMode = Literal["off", "auto", "on"]
WebMode = Literal["off", "on"]
ChatMode = Literal["normal", "guided_learning", "feynman"]
ChatLanguage = Literal["de", "en"]
RetrievalMode = Literal["normal", "deep"]

# Fields that must never be provided by the browser or forwarded to the RAG service.
_BROWSER_FORBIDDEN_FIELDS = frozenset({
    "tools", "tool", "tool_args", "tool_registry", "allowed_tools",
    "cypher", "neo4j_query", "agentic_decision", "refinement_action",
    "agentic_tool", "agentic_tool_args", "max_tool_calls",
    "max_refinement_rounds", "raw_tool_calls",
})


def query_embedding_cache_hits() -> int:
    return default_query_embedding_cache.hits


def query_embedding_cache_misses() -> int:
    return default_query_embedding_cache.misses


def retrieval_result_cache_hits() -> int:
    return default_retrieval_result_cache.hits


def retrieval_result_cache_misses() -> int:
    return default_retrieval_result_cache.misses


def reranker_cache_hits() -> int:
    return default_reranker_cache.hits


def reranker_cache_misses() -> int:
    return default_reranker_cache.misses


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
    chat_mode: ChatMode = "normal"
    retrieval_mode: RetrievalMode = "normal"
    use_fast_reranker: bool = False
    use_llm_reranker: bool = False
    active_learning_state: dict[str, Any] | None = None
    active_learning_control: dict[str, Any] | None = None
    chat_language: ChatLanguage | None = None

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

    @model_validator(mode="before")
    @classmethod
    def reject_browser_controlled_tool_fields(cls, values: Any) -> Any:
        """Explicitly reject tool-control fields that must never come from the browser."""
        if isinstance(values, dict):
            forbidden = _BROWSER_FORBIDDEN_FIELDS & values.keys()
            if forbidden:
                raise ValueError(
                    f"Fields not allowed in request: {sorted(forbidden)}"
                )
        return values


class RagAnswerResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
    timing: dict[str, Any] | None = None
    updated_active_learning_state: dict[str, Any] | None = None
    web_search: dict[str, Any] | None = None
    intent: dict[str, Any] | None = None
    retrieval_plan: dict[str, Any] | None = None
    retrieval_tools: dict[str, Any] | None = None
    agentic_retriever: dict[str, Any] | None = None


class CompressConversationRequest(BaseModel):
    messages: list[RecentMessage] = Field(default_factory=list, max_length=100)
    existing_summary: str | None = Field(default=None, max_length=3000)
    max_chars: int = Field(default=1500, ge=300, le=4000)


class CompressConversationResponse(BaseModel):
    summary: str


class RevisionGenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1)
    pdf_ids: list[str] = Field(min_length=1, max_length=20)
    count: int = Field(ge=1, le=50)
    language: ChatLanguage = "de"

    @model_validator(mode="after")
    def validate_uuids(self) -> "RevisionGenerateRequest":
        try:
            UUID(self.user_id)
        except ValueError as exc:
            raise ValueError("user_id must be a valid UUID") from exc
        for pdf_id in self.pdf_ids:
            try:
                UUID(pdf_id)
            except ValueError as exc:
                raise ValueError("pdf_ids must be valid UUIDs") from exc
        return self


class RevisionFlashcardOut(BaseModel):
    front: str
    back: str
    source_chunk_ids: list[str] = Field(default_factory=list)


class RevisionFlashcardsResponse(BaseModel):
    cards: list[RevisionFlashcardOut]


class RevisionMockQuestionOut(BaseModel):
    prompt: str
    choices: list[str]
    correct_index: int
    explanation: str = ""
    source_chunk_ids: list[str] = Field(default_factory=list)


class RevisionMockTestResponse(BaseModel):
    questions: list[RevisionMockQuestionOut]


class LearningTreeResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    label: str
    type: str
    chunk_ids: list[str]
    children: list[dict[str, Any]] = Field(default_factory=list)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _ensure_runtime_state(app)
    yield
    close = getattr(getattr(app.state, "neo4j_driver", None), "close", None)
    if close:
        close()


app = FastAPI(title="LearnCycle RAG API", lifespan=lifespan)
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


def create_neo4j_driver(config: WorkerConfig) -> Any:
    """Create a Neo4j driver for Learning Graph reads."""
    if not config.neo4j_uri or not config.neo4j_user or not config.neo4j_password:
        raise RuntimeError("Neo4j is not configured.")
    from neo4j import GraphDatabase

    return GraphDatabase.driver(
        config.neo4j_uri,
        auth=(config.neo4j_user, config.neo4j_password),
    )


def _ensure_runtime_state(app: FastAPI) -> Any:
    if getattr(app.state, "runtime_ready", False):
        return app.state
    config = WorkerConfig.from_env()
    app.state.runtime_config = config
    app.state.openai_llm_client = _make_openai_llm_client(config)
    app.state.embedder = Embedder(
        provider=config.embedding_provider,
        model=config.embedding_model,
        openai_api_key=config.openai_api_key,
        gemini_api_key=config.gemini_api_key,
        batch_size=config.embedding_batch_size,
    )
    app.state.sparse_embedder = SparseEmbedder(
        provider=config.sparse_provider,
        model=config.sparse_model,
    )
    app.state.qdrant_store = (
        QdrantStore(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key,
            collection_name=config.qdrant_collection,
        )
        if config.qdrant_url
        else None
    )
    app.state.health_cache = {"checked_at": 0.0, "qdrant_ok": False, "openai_ok": False}
    app.state.runtime_ready = True
    return app.state


def _make_openai_llm_client(config: WorkerConfig) -> OpenAILlmClient | None:
    if not config.openai_api_key:
        return None
    try:
        return OpenAILlmClient(api_key=config.openai_api_key, model=config.revision_llm_model)
    except Exception:
        logger.warning("OpenAI LLM client could not be initialized.", exc_info=True)
        return None


def _shared_retrieval_fn(state: Any):
    if state.qdrant_store is None:
        return None

    def retrieve(**kwargs: Any) -> list[dict[str, Any]]:
        return search_hybrid_chunks(
            config=state.runtime_config,
            embedder=state.embedder,
            sparse_embedder=state.sparse_embedder,
            store=state.qdrant_store,
            **kwargs,
        )

    return retrieve


def _cached_health_checks(state: Any) -> dict[str, bool]:
    now = time.monotonic()
    cached = state.health_cache
    if now - float(cached.get("checked_at", 0.0)) < 30:
        return {"qdrant_ok": bool(cached["qdrant_ok"]), "openai_ok": bool(cached["openai_ok"])}
    qdrant_ok = state.qdrant_store is not None
    openai_ok = state.openai_llm_client is not None
    state.health_cache = {
        "checked_at": now,
        "qdrant_ok": qdrant_ok,
        "openai_ok": openai_ok,
    }
    return {"qdrant_ok": qdrant_ok, "openai_ok": openai_ok}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    state = _ensure_runtime_state(app)
    upstream = _cached_health_checks(state)
    return {
        "status": "ok",
        "uptime_s": max(0.0, time.monotonic() - STARTED_AT),
        "qdrant_ok": upstream["qdrant_ok"],
        "openai_ok": upstream["openai_ok"],
        "query_embedding_cache_hits": query_embedding_cache_hits(),
        "query_embedding_cache_misses": query_embedding_cache_misses(),
        "retrieval_result_cache_hits": retrieval_result_cache_hits(),
        "retrieval_result_cache_misses": retrieval_result_cache_misses(),
        "reranker_cache_hits": reranker_cache_hits(),
        "reranker_cache_misses": reranker_cache_misses(),
    }


@app.post(
    "/rag/answer",
    response_model=RagAnswerResponse,
    dependencies=[Depends(require_internal_auth)],
)
def rag_answer(request: RagAnswerRequest, debug_timing: str | None = None) -> dict[str, Any]:
    try:
        state = _ensure_runtime_state(app)
        config = state.runtime_config
        (
            reranking_enabled,
            reranking_provider,
            reranking_candidate_k,
            reranking_top_k,
        ) = _reranking_settings(request, config)
        if reranking_enabled and reranking_candidate_k < reranking_top_k:
            raise HTTPException(
                status_code=422,
                detail="reranking_candidate_k must be >= reranking_top_k",
            )
        if (
            reranking_enabled
            and reranking_provider == "llm"
            and reranking_candidate_k > 30
        ):
            raise HTTPException(
                status_code=422,
                detail="reranking_candidate_k must be <= 30 for llm reranking",
            )
        reranker = create_reranker(
            provider=reranking_provider,
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
        tool_registry = None
        if config.retrieval_tool_registry_enabled:
            tool_registry = build_default_retrieval_tool_registry(
                config,
                dependencies={"graph_store": graph_store},
            )
        response = answer_with_rag(
            query=request.query.strip(),
            user_id=request.user_id,
            source_types=request.source_types,
            top_k=request.top_k,
            pdf_ids=request.pdf_ids,
            recent_messages=[message.model_dump() for message in request.recent_messages],
            llm_client=state.openai_llm_client,
            retrieval_fn=_shared_retrieval_fn(state),
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
            tool_registry=tool_registry,
            agentic_retriever_enabled=config.agentic_retriever_enabled,
            chat_mode=request.chat_mode,
            active_learning_state=request.active_learning_state,
            active_learning_control=request.active_learning_control,
            chat_language=request.chat_language,
            enable_timing=bool(debug_timing == "1" and config.rag_debug_timing_enabled),
            vector_retrieve_timeout_s=config.vector_retrieve_timeout_s,
            graph_retrieve_timeout_s=config.graph_retrieve_timeout_s,
            memory_retrieve_timeout_s=config.memory_retrieve_timeout_s,
            web_retrieve_timeout_s=config.web_retrieve_timeout_s,
            rerank_timeout_s=config.rerank_timeout_s,
            reranker_provider=reranking_provider,
            reranker_model=config.reranking_model,
            reranker_cache_enabled=config.reranker_cache_enabled,
            reranker_cache_max_entries=config.reranker_cache_max_entries,
            reranker_cache_ttl_s=config.reranker_cache_ttl_s,
        )
        if not config.rag_debug_timing_enabled:
            response.pop("timing", None)
        return response
    except HTTPException:
        raise
    except Exception:
        logger.exception("RAG answer generation failed")
        raise HTTPException(
            status_code=500,
            detail="RAG answer generation failed.",
        ) from None


def _sse(event: dict[str, Any]) -> str:
    return (
        f"event: {event.get('event_type', 'message')}\n"
        f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    )


def _reranking_settings(
    request: RagAnswerRequest,
    config: WorkerConfig,
) -> tuple[bool, str, int, int]:
    provider = config.reranking_provider
    candidate_k = request.reranking_candidate_k or config.reranking_candidate_k
    top_k = request.reranking_top_k or config.reranking_top_k
    if request.reranking_enabled is not None:
        enabled = request.reranking_enabled
        if request.use_fast_reranker:
            provider = "fastembed"
        if request.use_llm_reranker:
            provider = "llm"
        if not enabled:
            provider = "noop"
        return enabled, provider, candidate_k, top_k

    if request.retrieval_mode == "normal":
        candidate_k = min(request.reranking_candidate_k or 15, 15)
        provider = config.reranker_normal_mode_default
        if request.use_fast_reranker:
            provider = "fastembed"
        if request.use_llm_reranker:
            provider = "llm"
        enabled = provider != "noop"
    else:
        candidate_k = min(request.reranking_candidate_k or 30, 30)
        provider = "llm" if request.use_llm_reranker else "fastembed"
        enabled = True

    if not enabled:
        provider = "noop"
    return enabled, provider, candidate_k, top_k


def _answer_kwargs(
    request: RagAnswerRequest,
    config: WorkerConfig,
    *,
    reranker: Any,
    graph_store: Any,
    tool_registry: Any,
    debug_timing: bool = False,
) -> dict[str, Any]:
    graph_mode = request.graph_mode if config.graph_retrieval_enabled else "off"
    web_mode = "on" if request.enable_web_search is True else request.web_mode
    if not config.web_search_enabled:
        web_mode = "off"
    intent_classifier_enabled = (
        request.use_intent_classifier
        if request.use_intent_classifier is not None
        else config.intent_classifier_enabled
    )
    reranking_enabled, _provider, reranking_candidate_k, reranking_top_k = _reranking_settings(
        request,
        config,
    )
    state = _ensure_runtime_state(app)
    return {
        "query": request.query.strip(),
        "user_id": request.user_id,
        "source_types": request.source_types,
        "top_k": request.top_k,
        "pdf_ids": request.pdf_ids,
        "recent_messages": [message.model_dump() for message in request.recent_messages],
        "llm_client": state.openai_llm_client,
        "retrieval_fn": _shared_retrieval_fn(state),
        "reranker": reranker,
        "reranking_enabled": reranking_enabled,
        "reranking_candidate_k": reranking_candidate_k,
        "reranking_top_k": reranking_top_k,
        "session_id": request.session_id,
        "memory_source_ids": request.memory_source_ids,
        "memory_mode": request.memory_mode,
        "chat_memory_retrieval_enabled": config.chat_memory_retrieval_enabled,
        "chat_memory_top_k": config.chat_memory_top_k,
        "graph_retrieval_enabled": config.graph_retrieval_enabled,
        "graph_mode": graph_mode,
        "graph_top_k": config.graph_retrieval_top_k,
        "graph_store": graph_store,
        "context_summary": request.context_summary,
        "web_mode": web_mode,
        "web_search_enabled": config.web_search_enabled,
        "web_search_query": request.web_search_query,
        "web_search_top_k": config.web_search_top_k,
        "web_search_provider": config.web_search_provider,
        "web_search_api_key": config.tavily_api_key,
        "web_search_timeout_seconds": config.web_search_timeout_seconds,
        "web_search_max_query_chars": config.web_search_max_query_chars,
        "web_search_max_context_sources": config.web_search_max_context_sources,
        "web_search_max_chars_per_source": config.web_search_max_chars_per_source,
        "web_search_max_total_context_chars": config.web_search_max_total_context_chars,
        "intent_classifier_enabled": bool(intent_classifier_enabled),
        "intent_classifier_config": config,
        "retrieval_planner_enabled": config.retrieval_planner_enabled,
        "retrieval_planner_config": config,
        "tool_registry": tool_registry,
        "agentic_retriever_enabled": config.agentic_retriever_enabled,
        "chat_mode": request.chat_mode,
        "active_learning_state": request.active_learning_state,
        "active_learning_control": request.active_learning_control,
        "chat_language": request.chat_language,
        "enable_timing": debug_timing,
        "vector_retrieve_timeout_s": config.vector_retrieve_timeout_s,
        "graph_retrieve_timeout_s": config.graph_retrieve_timeout_s,
        "memory_retrieve_timeout_s": config.memory_retrieve_timeout_s,
        "web_retrieve_timeout_s": config.web_retrieve_timeout_s,
        "rerank_timeout_s": config.rerank_timeout_s,
        "reranker_provider": _provider,
        "reranker_model": config.reranking_model,
        "reranker_cache_enabled": config.reranker_cache_enabled,
        "reranker_cache_max_entries": config.reranker_cache_max_entries,
        "reranker_cache_ttl_s": config.reranker_cache_ttl_s,
    }


@app.post(
    "/rag/answer/stream",
    dependencies=[Depends(require_internal_auth)],
)
def rag_answer_stream(request: RagAnswerRequest, debug_timing: str | None = None) -> StreamingResponse:
    try:
        state = _ensure_runtime_state(app)
        config = state.runtime_config
        (
            reranking_enabled,
            reranking_provider,
            reranking_candidate_k,
            reranking_top_k,
        ) = _reranking_settings(request, config)
        if reranking_enabled and reranking_candidate_k < reranking_top_k:
            raise HTTPException(
                status_code=422,
                detail="reranking_candidate_k must be >= reranking_top_k",
            )
        reranker = create_reranker(
            provider=reranking_provider,
            model=config.reranking_model,
            enabled=reranking_enabled,
        )
        graph_store = create_graph_store(config)
        tool_registry = None
        if config.retrieval_tool_registry_enabled:
            tool_registry = build_default_retrieval_tool_registry(
                config,
                dependencies={"graph_store": graph_store},
            )
        kwargs = _answer_kwargs(
            request,
            config,
            reranker=reranker,
            graph_store=graph_store,
            tool_registry=tool_registry,
            debug_timing=bool(debug_timing == "1" and config.rag_debug_timing_enabled),
        )

        async def events():
            try:
                async for event in stream_answer_with_rag(**kwargs):
                    if event.get("event_type") == "done" and not config.rag_debug_timing_enabled:
                        event.pop("timing", None)
                    yield _sse(event)
            except Exception:
                logger.exception("RAG streaming answer generation failed")
                yield _sse({"event_type": "error", "message": "RAG answer generation failed."})

        return StreamingResponse(events(), media_type="text/event-stream")
    except HTTPException:
        raise
    except Exception:
        logger.exception("RAG streaming answer generation failed")
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
        state = _ensure_runtime_state(app)
        config = state.runtime_config
        summary = compress_conversation_summary(
            [message.model_dump() for message in request.messages],
            llm_client=state.openai_llm_client
            or OpenAILlmClient(api_key=config.openai_api_key, model="gpt-4o-mini"),
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


def _build_revision_registry_and_llm(config: WorkerConfig) -> tuple[Any, OpenAILlmClient]:
    """Build the retrieval tool registry and LLM client for revision generation.

    The registry is built directly (independent of `RETRIEVAL_TOOL_REGISTRY_ENABLED`)
    because revision generation always relies on retrieval tools regardless of the
    chat-side feature flag.
    """
    graph_store = create_graph_store(config)
    registry = build_default_retrieval_tool_registry(
        config,
        dependencies={"graph_store": graph_store},
    )
    if not config.openai_api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is required for revision generation.",
        )
    state = _ensure_runtime_state(app)
    llm_client = state.openai_llm_client or OpenAILlmClient(
        api_key=config.openai_api_key,
        model=config.revision_llm_model,
    )
    return registry, llm_client


@app.post(
    "/revision/flashcards",
    response_model=RevisionFlashcardsResponse,
    dependencies=[Depends(require_internal_auth)],
)
def revision_flashcards(request: RevisionGenerateRequest) -> dict[str, Any]:
    try:
        state = _ensure_runtime_state(app)
        config = state.runtime_config
        if request.count > config.revision_max_cards_per_deck:
            raise HTTPException(
                status_code=422,
                detail=(
                    "count exceeds REVISION_MAX_CARDS_PER_DECK "
                    f"({config.revision_max_cards_per_deck})"
                ),
            )
        registry, llm_client = _build_revision_registry_and_llm(config)
        batch = generate_flashcards(
            user_id=request.user_id,
            pdf_ids=request.pdf_ids,
            count=request.count,
            language=request.language,
            registry=registry,
            llm_client=llm_client,
            retrieval_top_k=config.revision_retrieval_top_k,
            max_cards=config.revision_max_cards_per_deck,
        )
        return {"cards": [card.model_dump() for card in batch.cards]}
    except HTTPException:
        raise
    except Exception:
        logger.exception("Flashcard generation failed")
        raise HTTPException(status_code=500, detail="Flashcard generation failed.") from None


@app.post(
    "/revision/mocktest",
    response_model=RevisionMockTestResponse,
    dependencies=[Depends(require_internal_auth)],
)
def revision_mocktest(request: RevisionGenerateRequest) -> dict[str, Any]:
    try:
        state = _ensure_runtime_state(app)
        config = state.runtime_config
        if request.count > config.revision_max_questions_per_test:
            raise HTTPException(
                status_code=422,
                detail=(
                    "count exceeds REVISION_MAX_QUESTIONS_PER_TEST "
                    f"({config.revision_max_questions_per_test})"
                ),
            )
        registry, llm_client = _build_revision_registry_and_llm(config)
        batch = generate_mock_test(
            user_id=request.user_id,
            pdf_ids=request.pdf_ids,
            count=request.count,
            language=request.language,
            registry=registry,
            llm_client=llm_client,
            retrieval_top_k=config.revision_retrieval_top_k,
            max_questions=config.revision_max_questions_per_test,
        )
        return {"questions": [q.model_dump() for q in batch.questions]}
    except HTTPException:
        raise
    except Exception:
        logger.exception("Mock test generation failed")
        raise HTTPException(status_code=500, detail="Mock test generation failed.") from None


@app.get(
    "/learning-graph/{source_id}/tree",
    dependencies=[Depends(require_internal_auth)],
)
def learning_graph_tree(source_id: str, user_id: str) -> dict[str, Any]:
    """Return a server-to-server Learning Graph tree for one user/source."""
    driver = None
    try:
        state = _ensure_runtime_state(app)
        config = state.runtime_config
        driver = create_neo4j_driver(config)
        tree = get_document_learning_tree(
            user_id,
            source_id,
            driver=driver,
        )
        if tree is None:
            raise HTTPException(status_code=404, detail="Learning graph not found.")
        if isinstance(tree, dict):
            return tree
        return tree.model_dump()
    except HTTPException:
        raise
    except Exception:
        logger.exception("Learning graph retrieval failed")
        raise HTTPException(status_code=500, detail="Learning graph retrieval failed.") from None
    finally:
        close = getattr(driver, "close", None) if driver is not None else None
        if close:
            close()
