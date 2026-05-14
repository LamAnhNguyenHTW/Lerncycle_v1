"""Retrieval Tool Registry — standardized server-side retrieval tool layer."""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rag_pipeline.web_search import WebSearchOutcome, search_web


logger = logging.getLogger(__name__)

_SENSITIVE_KEYS = frozenset({
    "api_key", "password", "secret", "token", "credential",
    "neo4j_password", "tavily_api_key", "openai_api_key", "gemini_api_key",
    "cypher", "neo4j_query", "stack_trace", "traceback", "raw_error",
    "provider_payload",
})


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RetrievalToolName(str, Enum):
    SEARCH_PDF_CHUNKS = "search_pdf_chunks"
    SEARCH_NOTES = "search_notes"
    SEARCH_ANNOTATIONS = "search_annotations"
    SEARCH_CHAT_MEMORY = "search_chat_memory"
    WEB_SEARCH = "web_search"
    QUERY_KNOWLEDGE_GRAPH = "query_knowledge_graph"


class RetrievalToolStatus(str, Enum):
    SUCCESS = "success"
    EMPTY = "empty"
    SKIPPED = "skipped"
    ERROR = "error"


class RetrievalToolErrorType(str, Enum):
    DISABLED = "disabled"
    NOT_ALLOWED = "not_allowed"
    MISSING_USER_ID = "missing_user_id"
    MISSING_SESSION_ID = "missing_session_id"
    MISSING_CONFIG = "missing_config"
    INVALID_QUERY = "invalid_query"
    INVALID_SCOPE = "invalid_scope"
    PROVIDER_ERROR = "provider_error"
    TIMEOUT = "timeout"
    UNKNOWN_ERROR = "unknown_error"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class RetrievalToolRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: RetrievalToolName
    query: str = Field(max_length=500)
    top_k: int = Field(ge=1, le=50)
    user_id: str
    session_id: str | None = None
    source_types: list[str] | None = None
    source_ids: list[str] | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("query")
    @classmethod
    def normalize_query_field(cls, value: str) -> str:
        return normalize_tool_query(value)


class RetrievalToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    source_type: str
    source_id: str | None = None
    text: str
    score: float | None = None
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalToolOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: RetrievalToolName
    status: RetrievalToolStatus
    results: list[RetrievalToolResult]
    result_count: int
    error_type: RetrievalToolErrorType | None = None
    latency_ms: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def safe_empty_tool_outcome(
    tool: RetrievalToolName,
    error_type: RetrievalToolErrorType | None = None,
    status: RetrievalToolStatus = RetrievalToolStatus.EMPTY,
) -> RetrievalToolOutcome:
    """Return a safe empty outcome — used on guards and failures."""
    return RetrievalToolOutcome(
        tool=tool,
        status=status,
        results=[],
        result_count=0,
        error_type=error_type,
    )


def normalize_tool_query(query: str, max_chars: int = 500) -> str:
    """Trim, collapse whitespace, and cap a tool query."""
    return re.sub(r"\s+", " ", str(query or "")).strip()[:max_chars]


def sanitize_tool_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Remove sensitive keys from metadata recursively."""
    return _sanitize_metadata_value(metadata)


def _sanitize_metadata_value(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = str(key).lower()
            if (
                normalized_key in _SENSITIVE_KEYS
                or "password" in normalized_key
                or "api_key" in normalized_key
                or "secret" in normalized_key
                or "token" in normalized_key
                or "cypher" in normalized_key
                or "traceback" in normalized_key
            ):
                continue
            sanitized[key] = _sanitize_metadata_value(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_metadata_value(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# RetrievalToolSpec dataclass
# ---------------------------------------------------------------------------

@dataclass
class RetrievalToolSpec:
    """Static description of a tool's capabilities.

    Runtime enabled/disabled state is determined by the registry checking
    server-side config at execution time — not stored here.
    """

    name: RetrievalToolName
    description: str
    requires_user_id: bool
    requires_session_id: bool
    requires_config_flag: str | None
    allowed_source_types: list[str]
    execute: Callable[[RetrievalToolRequest], RetrievalToolOutcome]


# ---------------------------------------------------------------------------
# RetrievalToolRegistry
# ---------------------------------------------------------------------------

class RetrievalToolRegistry:
    """Controlled server-side registry for retrieval tools."""

    def __init__(self, config: Any) -> None:
        self._specs: dict[RetrievalToolName, RetrievalToolSpec] = {}
        self._config = config

    def register(self, spec: RetrievalToolSpec) -> None:
        self._specs[spec.name] = spec

    def get(self, tool_name: RetrievalToolName) -> RetrievalToolSpec | None:
        return self._specs.get(tool_name)

    def list_tools(self) -> list[RetrievalToolSpec]:
        return list(self._specs.values())

    def is_allowed(self, tool_name: RetrievalToolName) -> bool:
        allowed = getattr(self._config, "retrieval_tool_allowed_tools", None)
        if allowed is None:
            return True
        return tool_name.value in allowed

    def execute(self, request: RetrievalToolRequest) -> RetrievalToolOutcome:
        """Execute a tool with all access guards and safe error handling."""
        tool_name = request.tool
        start = time.monotonic()

        spec = self._specs.get(tool_name)
        if spec is None:
            logger.warning(
                "Retrieval tool not registered.",
                extra={"tool": tool_name.value, "error_type": "not_allowed"},
            )
            return safe_empty_tool_outcome(
                tool_name, RetrievalToolErrorType.NOT_ALLOWED, RetrievalToolStatus.ERROR
            )

        if not self.is_allowed(tool_name):
            logger.info(
                "Retrieval tool not in allowed_tools list.",
                extra={"tool": tool_name.value, "error_type": "not_allowed"},
            )
            return safe_empty_tool_outcome(
                tool_name, RetrievalToolErrorType.NOT_ALLOWED, RetrievalToolStatus.SKIPPED
            )

        if spec.requires_config_flag:
            flag_value = getattr(self._config, spec.requires_config_flag, False)
            if not flag_value:
                logger.info(
                    "Retrieval tool disabled by config.",
                    extra={"tool": tool_name.value, "error_type": "disabled", "flag": spec.requires_config_flag},
                )
                return safe_empty_tool_outcome(
                    tool_name, RetrievalToolErrorType.DISABLED, RetrievalToolStatus.SKIPPED
                )

        if spec.requires_user_id and not request.user_id:
            logger.warning(
                "Retrieval tool requires user_id.",
                extra={"tool": tool_name.value, "error_type": "missing_user_id"},
            )
            return safe_empty_tool_outcome(
                tool_name, RetrievalToolErrorType.MISSING_USER_ID, RetrievalToolStatus.ERROR
            )

        if spec.requires_session_id and not request.session_id:
            logger.warning(
                "Retrieval tool requires session_id.",
                extra={"tool": tool_name.value, "error_type": "missing_session_id"},
            )
            return safe_empty_tool_outcome(
                tool_name, RetrievalToolErrorType.MISSING_SESSION_ID, RetrievalToolStatus.ERROR
            )

        try:
            outcome = spec.execute(request)
        except Exception:
            logger.warning(
                "Retrieval tool execution raised an exception.",
                extra={"tool": tool_name.value, "error_type": "unknown_error"},
            )
            outcome = safe_empty_tool_outcome(
                tool_name, RetrievalToolErrorType.UNKNOWN_ERROR, RetrievalToolStatus.ERROR
            )

        latency_ms = int((time.monotonic() - start) * 1000)
        timeout_seconds = getattr(self._config, "retrieval_tool_timeout_seconds", None)
        if timeout_seconds is not None and latency_ms > int(timeout_seconds * 1000):
            outcome = safe_empty_tool_outcome(
                tool_name,
                RetrievalToolErrorType.TIMEOUT,
                RetrievalToolStatus.ERROR,
            )

        max_results = getattr(self._config, "retrieval_tool_max_results_per_tool", None)
        if max_results is not None and outcome.results:
            capped_results = outcome.results[: int(max_results)]
            outcome = outcome.model_copy(
                update={
                    "results": capped_results,
                    "result_count": len(capped_results),
                    "metadata": sanitize_tool_metadata(outcome.metadata),
                }
            )
        elif outcome.metadata:
            outcome = outcome.model_copy(update={"metadata": sanitize_tool_metadata(outcome.metadata)})

        logger.info(
            "Retrieval tool executed.",
            extra={
                "tool": tool_name.value,
                "status": outcome.status.value,
                "result_count": outcome.result_count,
                "latency_ms": latency_ms,
                "error_type": outcome.error_type.value if outcome.error_type else None,
            },
        )
        return outcome.model_copy(update={"latency_ms": latency_ms})


# ---------------------------------------------------------------------------
# Internal tool adapters — Phase 4
# ---------------------------------------------------------------------------

def execute_search_pdf_chunks(
    request: RetrievalToolRequest,
    config: Any,
    retrieval_fns: dict[str, Any],
) -> RetrievalToolOutcome:
    """Search PDF chunks via hybrid retrieval, respecting selected PDF scope."""
    try:
        from rag_pipeline.retrieval import search_hybrid_chunks
        material_search = retrieval_fns.get("search_hybrid_chunks") or search_hybrid_chunks
        pdf_ids = request.filters.get("pdf_ids") or request.source_ids or None
        raw = material_search(
            query=request.query,
            user_id=request.user_id,
            source_types=["pdf"],
            top_k=request.top_k,
            pdf_ids=pdf_ids,
            config=config,
        )
        results = [_chunk_to_tool_result(r) for r in raw]
        return RetrievalToolOutcome(
            tool=RetrievalToolName.SEARCH_PDF_CHUNKS,
            status=RetrievalToolStatus.SUCCESS if results else RetrievalToolStatus.EMPTY,
            results=results,
            result_count=len(results),
        )
    except Exception:
        logger.warning("PDF chunk search failed.", extra={"error_type": "provider_error"})
        return safe_empty_tool_outcome(
            RetrievalToolName.SEARCH_PDF_CHUNKS,
            RetrievalToolErrorType.PROVIDER_ERROR,
            RetrievalToolStatus.ERROR,
        )


def execute_search_notes(
    request: RetrievalToolRequest,
    config: Any,
    retrieval_fns: dict[str, Any],
) -> RetrievalToolOutcome:
    """Search note chunks via hybrid retrieval."""
    try:
        from rag_pipeline.retrieval import search_hybrid_chunks
        material_search = retrieval_fns.get("search_hybrid_chunks") or search_hybrid_chunks
        raw = material_search(
            query=request.query,
            user_id=request.user_id,
            source_types=["note"],
            source_ids=request.source_ids,
            top_k=request.top_k,
            pdf_ids=request.filters.get("pdf_ids"),
            config=config,
        )
        results = [_chunk_to_tool_result(r) for r in raw]
        return RetrievalToolOutcome(
            tool=RetrievalToolName.SEARCH_NOTES,
            status=RetrievalToolStatus.SUCCESS if results else RetrievalToolStatus.EMPTY,
            results=results,
            result_count=len(results),
        )
    except Exception:
        logger.warning("Notes search failed.", extra={"error_type": "provider_error"})
        return safe_empty_tool_outcome(
            RetrievalToolName.SEARCH_NOTES,
            RetrievalToolErrorType.PROVIDER_ERROR,
            RetrievalToolStatus.ERROR,
        )


def execute_search_annotations(
    request: RetrievalToolRequest,
    config: Any,
    retrieval_fns: dict[str, Any],
) -> RetrievalToolOutcome:
    """Search annotation_comment chunks via hybrid retrieval."""
    try:
        from rag_pipeline.retrieval import search_hybrid_chunks
        material_search = retrieval_fns.get("search_hybrid_chunks") or search_hybrid_chunks
        raw = material_search(
            query=request.query,
            user_id=request.user_id,
            source_types=["annotation_comment"],
            source_ids=request.source_ids,
            top_k=request.top_k,
            pdf_ids=request.filters.get("pdf_ids"),
            config=config,
        )
        results = [_chunk_to_tool_result(r) for r in raw]
        return RetrievalToolOutcome(
            tool=RetrievalToolName.SEARCH_ANNOTATIONS,
            status=RetrievalToolStatus.SUCCESS if results else RetrievalToolStatus.EMPTY,
            results=results,
            result_count=len(results),
        )
    except Exception:
        logger.warning("Annotations search failed.", extra={"error_type": "provider_error"})
        return safe_empty_tool_outcome(
            RetrievalToolName.SEARCH_ANNOTATIONS,
            RetrievalToolErrorType.PROVIDER_ERROR,
            RetrievalToolStatus.ERROR,
        )


def execute_search_chat_memory(
    request: RetrievalToolRequest,
    config: Any,
    retrieval_fns: dict[str, Any],
) -> RetrievalToolOutcome:
    """Search server-approved chat memory session ids."""
    if not request.session_id:
        return safe_empty_tool_outcome(
            RetrievalToolName.SEARCH_CHAT_MEMORY,
            RetrievalToolErrorType.MISSING_SESSION_ID,
            RetrievalToolStatus.ERROR,
        )
    try:
        from rag_pipeline.retrieval import search_hybrid_chunks
        material_search = retrieval_fns.get("search_hybrid_chunks") or search_hybrid_chunks
        effective_source_ids = []
        for source_id in [request.session_id, *(request.source_ids or [])]:
            if source_id and source_id not in effective_source_ids:
                effective_source_ids.append(source_id)
        raw = material_search(
            query=request.query,
            user_id=request.user_id,
            source_types=["chat_memory"],
            source_ids=effective_source_ids,
            top_k=request.top_k,
            pdf_ids=None,
            config=config,
        )
        results = [_chunk_to_tool_result(r) for r in raw]
        return RetrievalToolOutcome(
            tool=RetrievalToolName.SEARCH_CHAT_MEMORY,
            status=RetrievalToolStatus.SUCCESS if results else RetrievalToolStatus.EMPTY,
            results=results,
            result_count=len(results),
        )
    except Exception:
        logger.warning("Chat memory search failed.", extra={"error_type": "provider_error"})
        return safe_empty_tool_outcome(
            RetrievalToolName.SEARCH_CHAT_MEMORY,
            RetrievalToolErrorType.PROVIDER_ERROR,
            RetrievalToolStatus.ERROR,
        )


# ---------------------------------------------------------------------------
# Web search tool adapter — Phase 5
# ---------------------------------------------------------------------------

def execute_web_search(
    request: RetrievalToolRequest,
    config: Any,
    web_search_fn: Callable[..., Any] | None = None,
) -> RetrievalToolOutcome:
    """Search the web via Tavily; respects WEB_SEARCH_ENABLED guard."""
    if not getattr(config, "web_search_enabled", False):
        return safe_empty_tool_outcome(
            RetrievalToolName.WEB_SEARCH,
            RetrievalToolErrorType.DISABLED,
            RetrievalToolStatus.SKIPPED,
        )
    try:
        active_search = web_search_fn or search_web
        outcome: WebSearchOutcome = active_search(
            query=request.query,
            top_k=request.top_k,
            provider=getattr(config, "web_search_provider", "tavily"),
            api_key=getattr(config, "tavily_api_key", None),
            timeout_seconds=getattr(config, "web_search_timeout_seconds", 15),
            max_query_chars=getattr(config, "web_search_max_query_chars", 300),
        )
        if not isinstance(outcome, WebSearchOutcome):
            outcome = WebSearchOutcome([], "tavily", 0, "provider_error")

        # Map web error types to RetrievalToolErrorType
        error_type: RetrievalToolErrorType | None = None
        status = RetrievalToolStatus.SUCCESS
        if outcome.error_type == "timeout":
            error_type = RetrievalToolErrorType.TIMEOUT
            status = RetrievalToolStatus.ERROR
        elif outcome.error_type in {"missing_api_key", "provider_error"}:
            error_type = RetrievalToolErrorType.PROVIDER_ERROR
            status = RetrievalToolStatus.ERROR
        elif outcome.error_type == "invalid_query":
            error_type = RetrievalToolErrorType.INVALID_QUERY
            status = RetrievalToolStatus.ERROR
        elif outcome.error_type == "empty_results" or not outcome.results:
            status = RetrievalToolStatus.EMPTY

        results = [_chunk_to_tool_result(r) for r in outcome.results]
        return RetrievalToolOutcome(
            tool=RetrievalToolName.WEB_SEARCH,
            status=status,
            results=results,
            result_count=len(results),
            error_type=error_type,
        )
    except Exception:
        logger.warning("Web search tool failed.", extra={"error_type": "provider_error"})
        return safe_empty_tool_outcome(
            RetrievalToolName.WEB_SEARCH,
            RetrievalToolErrorType.PROVIDER_ERROR,
            RetrievalToolStatus.ERROR,
        )


# ---------------------------------------------------------------------------
# Knowledge graph tool adapter — Phase 6
# ---------------------------------------------------------------------------

def execute_query_knowledge_graph(
    request: RetrievalToolRequest,
    config: Any,
    graph_fn: Callable[..., Any] | None = None,
    graph_store: Any = None,
) -> RetrievalToolOutcome:
    """Wrap retrieve_graph_context(); no new Cypher — uses existing implementation."""
    if not getattr(config, "graph_retrieval_enabled", False):
        return safe_empty_tool_outcome(
            RetrievalToolName.QUERY_KNOWLEDGE_GRAPH,
            RetrievalToolErrorType.DISABLED,
            RetrievalToolStatus.SKIPPED,
        )

    neo4j_uri = getattr(config, "neo4j_uri", None)
    neo4j_user = getattr(config, "neo4j_user", None)
    neo4j_password = getattr(config, "neo4j_password", None)
    if not neo4j_uri or not neo4j_user or not neo4j_password:
        return safe_empty_tool_outcome(
            RetrievalToolName.QUERY_KNOWLEDGE_GRAPH,
            RetrievalToolErrorType.MISSING_CONFIG,
            RetrievalToolStatus.ERROR,
        )

    try:
        from rag_pipeline.graph_retrieval import retrieve_graph_context
        active_graph_fn = graph_fn or retrieve_graph_context
        graph_top_k = min(request.top_k, getattr(config, "graph_retrieval_top_k", 8))
        max_chars = getattr(config, "graph_context_max_chars", 6000)

        effective_source_types = (
            None
            if request.source_types == ["knowledge_graph"]
            else request.source_types
        )

        ctx = active_graph_fn(
            query=request.query,
            user_id=request.user_id,
            source_types=effective_source_types,
            source_ids=request.source_ids,
            top_k=graph_top_k,
            max_chars=max_chars,
            graph_store=graph_store,
        )

        context_text = str(ctx.get("context_text") or "").strip()
        sources = ctx.get("sources") or []
        nodes = ctx.get("nodes") or []
        relationships = ctx.get("relationships") or []

        if not context_text:
            return safe_empty_tool_outcome(
                RetrievalToolName.QUERY_KNOWLEDGE_GRAPH,
                status=RetrievalToolStatus.EMPTY,
            )

        # Stable chunk_id from query + context hash
        chunk_hash = hashlib.sha256(
            (request.query + context_text).encode("utf-8")
        ).hexdigest()[:16]
        chunk_id = f"kg:{chunk_hash}"

        node_names = [
            str(n.get("name") or n.get("normalized_name") or "")
            for n in nodes[:10]
            if n.get("name") or n.get("normalized_name")
        ]
        title_entities = ", ".join(node_names[:3]) if node_names else request.query[:60]
        title = f"Knowledge Graph: {title_entities}"

        # Use the snippet from the first source if available
        first_source = sources[0] if sources else {}
        backing_chunk_ids = first_source.get("metadata", {}).get("backing_chunk_ids", [])
        relationship_count = len(relationships)

        result = RetrievalToolResult(
            chunk_id=chunk_id,
            source_type="knowledge_graph",
            source_id=first_source.get("source_id"),
            text=context_text,
            score=first_source.get("score"),
            title=title,
            metadata=sanitize_tool_metadata({
                "provider": "neo4j",
                "node_names": node_names,
                "relationship_count": relationship_count,
                "backing_chunk_ids": backing_chunk_ids[:20],
            }),
        )
        return RetrievalToolOutcome(
            tool=RetrievalToolName.QUERY_KNOWLEDGE_GRAPH,
            status=RetrievalToolStatus.SUCCESS,
            results=[result],
            result_count=1,
        )
    except TimeoutError:
        logger.warning("Graph tool timed out.", extra={"error_type": "timeout"})
        return safe_empty_tool_outcome(
            RetrievalToolName.QUERY_KNOWLEDGE_GRAPH,
            RetrievalToolErrorType.TIMEOUT,
            RetrievalToolStatus.ERROR,
        )
    except Exception:
        logger.warning("Graph tool failed.", extra={"error_type": "provider_error"})
        return safe_empty_tool_outcome(
            RetrievalToolName.QUERY_KNOWLEDGE_GRAPH,
            RetrievalToolErrorType.PROVIDER_ERROR,
            RetrievalToolStatus.ERROR,
        )


# ---------------------------------------------------------------------------
# Default registry factory — Phase 3
# ---------------------------------------------------------------------------

def build_default_retrieval_tool_registry(
    config: Any,
    dependencies: dict[str, Any] | None = None,
) -> RetrievalToolRegistry:
    """Build the default tool registry with all six tools pre-registered."""
    registry = RetrievalToolRegistry(config)
    deps = dependencies or {}
    retrieval_fns: dict[str, Any] = {
        "search_hybrid_chunks": deps.get("search_hybrid_chunks"),
    }
    web_search_fn = deps.get("search_web")
    graph_fn = deps.get("retrieve_graph_context")
    graph_store = deps.get("graph_store")

    registry.register(RetrievalToolSpec(
        name=RetrievalToolName.SEARCH_PDF_CHUNKS,
        description="Search user PDF chunks via hybrid dense+sparse retrieval.",
        requires_user_id=True,
        requires_session_id=False,
        requires_config_flag=None,
        allowed_source_types=["pdf"],
        execute=lambda req: execute_search_pdf_chunks(req, config, retrieval_fns),
    ))
    registry.register(RetrievalToolSpec(
        name=RetrievalToolName.SEARCH_NOTES,
        description="Search user note chunks via hybrid retrieval.",
        requires_user_id=True,
        requires_session_id=False,
        requires_config_flag=None,
        allowed_source_types=["note"],
        execute=lambda req: execute_search_notes(req, config, retrieval_fns),
    ))
    registry.register(RetrievalToolSpec(
        name=RetrievalToolName.SEARCH_ANNOTATIONS,
        description="Search user annotation_comment chunks via hybrid retrieval.",
        requires_user_id=True,
        requires_session_id=False,
        requires_config_flag=None,
        allowed_source_types=["annotation_comment"],
        execute=lambda req: execute_search_annotations(req, config, retrieval_fns),
    ))
    registry.register(RetrievalToolSpec(
        name=RetrievalToolName.SEARCH_CHAT_MEMORY,
        description="Search chat memory summaries scoped to the server-verified session.",
        requires_user_id=True,
        requires_session_id=True,
        requires_config_flag=None,
        allowed_source_types=["chat_memory"],
        execute=lambda req: execute_search_chat_memory(req, config, retrieval_fns),
    ))
    registry.register(RetrievalToolSpec(
        name=RetrievalToolName.WEB_SEARCH,
        description="Search the live web via Tavily when WEB_SEARCH_ENABLED=true.",
        requires_user_id=True,
        requires_session_id=False,
        requires_config_flag="web_search_enabled",
        allowed_source_types=["web"],
        execute=lambda req: execute_web_search(req, config, web_search_fn),
    ))
    registry.register(RetrievalToolSpec(
        name=RetrievalToolName.QUERY_KNOWLEDGE_GRAPH,
        description="Query the Neo4j knowledge graph when GRAPH_RETRIEVAL_ENABLED=true.",
        requires_user_id=True,
        requires_session_id=False,
        requires_config_flag="graph_retrieval_enabled",
        allowed_source_types=["knowledge_graph"],
        execute=lambda req: execute_query_knowledge_graph(req, config, graph_fn, graph_store),
    ))

    return registry


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _chunk_to_tool_result(chunk: dict[str, Any]) -> RetrievalToolResult:
    """Normalize a raw retrieval chunk dict to RetrievalToolResult."""
    raw_meta = chunk.get("metadata") or {}
    return RetrievalToolResult(
        chunk_id=str(chunk.get("chunk_id") or ""),
        source_type=str(chunk.get("source_type") or ""),
        source_id=chunk.get("source_id") or chunk.get("pdf_id") or None,
        text=str(chunk.get("text") or chunk.get("snippet") or ""),
        score=float(chunk["score"]) if isinstance(chunk.get("score"), (int, float)) else None,
        title=chunk.get("title") or None,
        metadata=sanitize_tool_metadata(raw_meta if isinstance(raw_meta, dict) else {}),
    )
