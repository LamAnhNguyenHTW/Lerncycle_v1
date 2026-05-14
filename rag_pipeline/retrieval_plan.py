"""Deterministic retrieval planning and controlled execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
import re
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rag_pipeline.config import WorkerConfig
from rag_pipeline.intent_classifier import RetrievalIntent
from rag_pipeline.retrieval import search_hybrid_chunks
from rag_pipeline.source_types import MATERIAL_SOURCE_TYPES
from rag_pipeline.web_search import WebSearchOutcome, search_web

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from rag_pipeline.retrieval_tools import RetrievalToolRegistry


logger = logging.getLogger(__name__)
PLANNER_VERSION = "v1"
GRAPH_UNAVAILABLE_REASON = "Graph RAG is not implemented yet."
MATERIAL_SOURCE_ORDER = ["pdf", "note", "annotation_comment"]


class RetrievalTool(str, Enum):
    SEARCH_PDF_CHUNKS = "search_pdf_chunks"
    SEARCH_NOTES = "search_notes"
    SEARCH_ANNOTATIONS = "search_annotations"
    SEARCH_CHAT_MEMORY = "search_chat_memory"
    WEB_SEARCH = "web_search"
    QUERY_KNOWLEDGE_GRAPH = "query_knowledge_graph"


class PlanStepStatus(str, Enum):
    ENABLED = "enabled"
    DISABLED = "disabled"
    SKIPPED = "skipped"


class RetrievalPlanStep(BaseModel):
    """One safe retrieval action proposed by the planner."""

    model_config = ConfigDict(extra="forbid")

    tool: RetrievalTool
    query: str = Field(max_length=500)
    top_k: int = Field(ge=1, le=20)
    status: PlanStepStatus = PlanStepStatus.ENABLED
    reason: str | None = Field(default=None, max_length=300)
    source_types: list[str] | None = None
    source_ids: list[str] | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        normalized = normalize_plan_query(value)
        if not normalized:
            raise ValueError("query must not be empty")
        return normalized

    @field_validator("reason")
    @classmethod
    def strip_reason(cls, value: str | None) -> str | None:
        return value.strip()[:300] if value else None


class RetrievalPlan(BaseModel):
    """Validated retrieval plan generated from a RetrievalIntent."""

    model_config = ConfigDict(extra="forbid")

    question_type: str
    steps: list[RetrievalPlanStep] = Field(max_length=20)
    fallback_used: bool = False
    planner_version: str = PLANNER_VERSION
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass
class PlanExecutionOutcome:
    """Result of controlled plan execution."""

    results: list[dict[str, Any]]
    step_outcomes: list[dict[str, Any]]
    total_result_count: int
    fallback_used: bool = False
    registry_used: bool = False
    tool_outcomes: list[dict[str, Any]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.tool_outcomes is None:
            self.tool_outcomes = []


def normalize_plan_query(query: str, max_chars: int = 500) -> str:
    """Trim, collapse whitespace, and cap a plan query."""
    return re.sub(r"\s+", " ", str(query or "")).strip()[:max_chars]


def build_material_query(user_query: str, intent: RetrievalIntent) -> str:
    return normalize_plan_query(user_query)


def build_memory_query(user_query: str, intent: RetrievalIntent) -> str:
    return normalize_plan_query(user_query)


def build_web_query(user_query: str, intent: RetrievalIntent) -> str:
    return normalize_plan_query(user_query)


def build_graph_query(user_query: str, intent: RetrievalIntent) -> str:
    return normalize_plan_query(user_query)


def build_default_plan(
    query: str,
    intent: RetrievalIntent | None,
    config: WorkerConfig,
    selected_pdf_ids: list[str] | None = None,
    allowed_source_types: list[str] | None = None,
) -> RetrievalPlan:
    """Build the safe existing-RAG material fallback plan."""
    allowed = _allowed_material_sources(allowed_source_types)
    steps: list[RetrievalPlanStep] = []
    if "pdf" in allowed:
        steps.append(
            RetrievalPlanStep(
                tool=RetrievalTool.SEARCH_PDF_CHUNKS,
                query=normalize_plan_query(query),
                top_k=config.retrieval_planner_default_top_k,
                source_types=["pdf"],
                source_ids=selected_pdf_ids or None,
                filters={"pdf_ids": selected_pdf_ids or None},
            )
        )
    if "note" in allowed:
        steps.append(
            RetrievalPlanStep(
                tool=RetrievalTool.SEARCH_NOTES,
                query=normalize_plan_query(query),
                top_k=config.retrieval_planner_default_top_k,
                source_types=["note"],
            )
        )
    if "annotation_comment" in allowed:
        steps.append(
            RetrievalPlanStep(
                tool=RetrievalTool.SEARCH_ANNOTATIONS,
                query=normalize_plan_query(query),
                top_k=config.retrieval_planner_default_top_k,
                source_types=["annotation_comment"],
            )
        )
    return RetrievalPlan(
        question_type=_question_type(intent),
        steps=_limit_enabled_steps(steps, config.retrieval_planner_max_steps),
        fallback_used=True,
        metadata={"default_plan": True},
    )


def build_retrieval_plan(
    query: str,
    intent: RetrievalIntent,
    config: WorkerConfig,
    session_id: str | None = None,
    selected_pdf_ids: list[str] | None = None,
    allowed_source_types: list[str] | None = None,
) -> RetrievalPlan:
    """Convert a validated intent into deterministic retrieval steps."""
    if not config.retrieval_planner_enabled:
        return build_default_plan(query, intent, config, selected_pdf_ids, allowed_source_types)

    allowed = _allowed_material_sources(allowed_source_types)
    steps: list[RetrievalPlanStep] = []
    material_query = build_material_query(query, intent)

    if intent.needs_pdf:
        _append_material_step(
            steps,
            enabled="pdf" in allowed,
            include_disabled=config.retrieval_planner_include_disabled_steps,
            tool=RetrievalTool.SEARCH_PDF_CHUNKS,
            query=material_query,
            top_k=config.retrieval_planner_pdf_top_k,
            source_type="pdf",
            source_ids=selected_pdf_ids or None,
            filters={"pdf_ids": selected_pdf_ids or None},
        )
    if intent.needs_notes:
        _append_material_step(
            steps,
            enabled="note" in allowed,
            include_disabled=config.retrieval_planner_include_disabled_steps,
            tool=RetrievalTool.SEARCH_NOTES,
            query=material_query,
            top_k=config.retrieval_planner_notes_top_k,
            source_type="note",
        )
    if intent.needs_annotations:
        _append_material_step(
            steps,
            enabled="annotation_comment" in allowed,
            include_disabled=config.retrieval_planner_include_disabled_steps,
            tool=RetrievalTool.SEARCH_ANNOTATIONS,
            query=material_query,
            top_k=config.retrieval_planner_annotations_top_k,
            source_type="annotation_comment",
        )
    if intent.needs_chat_memory:
        if session_id:
            steps.append(
                RetrievalPlanStep(
                    tool=RetrievalTool.SEARCH_CHAT_MEMORY,
                    query=build_memory_query(query, intent),
                    top_k=config.retrieval_planner_memory_top_k,
                    source_types=["chat_memory"],
                    source_ids=[session_id],
                )
            )
        elif config.retrieval_planner_include_disabled_steps:
            steps.append(_disabled_step(RetrievalTool.SEARCH_CHAT_MEMORY, query, config.retrieval_planner_memory_top_k, "Missing session_id."))
    if intent.needs_web:
        if config.web_search_enabled:
            steps.append(
                RetrievalPlanStep(
                    tool=RetrievalTool.WEB_SEARCH,
                    query=build_web_query(query, intent),
                    top_k=config.retrieval_planner_web_top_k,
                    source_types=["web"],
                )
            )
        elif config.retrieval_planner_include_disabled_steps:
            steps.append(_disabled_step(RetrievalTool.WEB_SEARCH, query, config.retrieval_planner_web_top_k, "Web Search is disabled."))
    if intent.needs_graph:
        graph_retrieval_enabled = getattr(config, "graph_retrieval_enabled", False)
        graph_top_k = getattr(config, "graph_retrieval_top_k", 5)
        if graph_retrieval_enabled:
            steps.append(
                RetrievalPlanStep(
                    tool=RetrievalTool.QUERY_KNOWLEDGE_GRAPH,
                    query=build_graph_query(query, intent),
                    top_k=graph_top_k,
                    source_types=["knowledge_graph"],
                )
            )
        elif config.retrieval_planner_include_disabled_steps:
            steps.append(_disabled_step(
                RetrievalTool.QUERY_KNOWLEDGE_GRAPH,
                build_graph_query(query, intent),
                5,
                GRAPH_UNAVAILABLE_REASON,
            ))

    if not any(step.status == PlanStepStatus.ENABLED for step in steps):
        return build_default_plan(query, intent, config, selected_pdf_ids, allowed_source_types)

    return RetrievalPlan(
        question_type=_question_type(intent),
        steps=_limit_enabled_steps(steps, config.retrieval_planner_max_steps),
        fallback_used=False,
        metadata={"selected_pdf_ids": selected_pdf_ids or [], "allowed_source_types": list(allowed)},
    )


def execute_retrieval_plan(
    plan: RetrievalPlan,
    query: str,
    user_id: str,
    config: WorkerConfig,
    retrieval_fns: dict[str, Callable[..., Any]] | None = None,
    session_id: str | None = None,
    tool_registry: "RetrievalToolRegistry | None" = None,
) -> PlanExecutionOutcome:
    """Execute enabled retrieval steps with safe per-step failure handling."""
    registry_enabled = (
        tool_registry is not None
        and getattr(config, "retrieval_tool_registry_enabled", False)
    )
    if registry_enabled and tool_registry is not None:
        return _execute_plan_via_registry(plan, user_id, config, session_id, tool_registry)

    functions = retrieval_fns or {}
    material_search = functions.get("search_hybrid_chunks") or search_hybrid_chunks
    active_web_search = functions.get("search_web") or search_web
    results: list[dict[str, Any]] = []
    step_outcomes: list[dict[str, Any]] = []
    enabled_count = 0

    for step in plan.steps:
        if step.status != PlanStepStatus.ENABLED:
            step_outcomes.append(_step_outcome(step, 0, None))
            continue
        if enabled_count >= config.retrieval_planner_max_steps:
            step_outcomes.append(_step_outcome(step, 0, "max_steps_exceeded", status="skipped"))
            continue
        enabled_count += 1
        try:
            step_results = _execute_step(step, user_id, config, material_search, active_web_search, session_id)
            results.extend(step_results)
            step_outcomes.append(_step_outcome(step, len(step_results), None))
        except Exception:
            logger.warning("Retrieval plan step failed.", extra={"tool": step.tool.value, "error_type": "tool_error"})
            step_outcomes.append(_step_outcome(step, 0, "tool_error"))
    return PlanExecutionOutcome(results, step_outcomes, len(results), plan.fallback_used)


def _execute_plan_via_registry(
    plan: RetrievalPlan,
    user_id: str,
    config: WorkerConfig,
    session_id: str | None,
    tool_registry: "RetrievalToolRegistry",
) -> PlanExecutionOutcome:
    """Execute plan steps through the RetrievalToolRegistry."""
    from rag_pipeline.retrieval_tools import RetrievalToolName, RetrievalToolRequest

    results: list[dict[str, Any]] = []
    step_outcomes: list[dict[str, Any]] = []
    tool_outcomes_meta: list[dict[str, Any]] = []
    enabled_count = 0

    for step in plan.steps:
        if step.status != PlanStepStatus.ENABLED:
            step_outcomes.append(_step_outcome(step, 0, None))
            continue
        if enabled_count >= config.retrieval_planner_max_steps:
            step_outcomes.append(_step_outcome(step, 0, "max_steps_exceeded", status="skipped"))
            continue
        enabled_count += 1
        try:
            tool_req = RetrievalToolRequest(
                tool=RetrievalToolName(step.tool.value),
                query=step.query,
                top_k=step.top_k,
                user_id=user_id,
                session_id=session_id,
                source_types=step.source_types,
                source_ids=step.source_ids,
                filters=step.filters,
                metadata=step.metadata,
            )
            outcome = tool_registry.execute(tool_req)
            step_results = [r.model_dump() for r in outcome.results]
            results.extend(step_results)
            step_outcomes.append({
                "tool": step.tool.value,
                "status": outcome.status.value,
                "top_k": step.top_k,
                "reason": step.reason,
                "result_count": outcome.result_count,
                "error_type": outcome.error_type.value if outcome.error_type else None,
                "latency_ms": outcome.latency_ms,
            })
            tool_outcomes_meta.append({
                "tool": step.tool.value,
                "status": outcome.status.value,
                "result_count": outcome.result_count,
                "latency_ms": outcome.latency_ms,
                "error_type": outcome.error_type.value if outcome.error_type else None,
            })
        except Exception:
            logger.warning(
                "Registry plan step failed.",
                extra={"tool": step.tool.value, "error_type": "tool_error"},
            )
            step_outcomes.append(_step_outcome(step, 0, "tool_error"))

    outcome_obj = PlanExecutionOutcome(results, step_outcomes, len(results), plan.fallback_used)
    outcome_obj.registry_used = True
    outcome_obj.tool_outcomes = tool_outcomes_meta
    return outcome_obj


def _execute_step(
    step: RetrievalPlanStep,
    user_id: str,
    config: WorkerConfig,
    material_search: Callable[..., list[dict[str, Any]]],
    active_web_search: Callable[..., Any],
    session_id: str | None,
) -> list[dict[str, Any]]:
    if step.tool == RetrievalTool.QUERY_KNOWLEDGE_GRAPH:
        return []
    if step.tool == RetrievalTool.WEB_SEARCH:
        if not config.web_search_enabled:
            return []
        outcome = active_web_search(
            query=step.query,
            top_k=step.top_k,
            provider=config.web_search_provider,
            api_key=config.tavily_api_key,
            timeout_seconds=config.web_search_timeout_seconds,
            max_query_chars=config.web_search_max_query_chars,
        )
        if isinstance(outcome, WebSearchOutcome):
            return outcome.results
        return outcome if isinstance(outcome, list) else []
    if step.tool == RetrievalTool.SEARCH_CHAT_MEMORY:
        if not session_id:
            return []
        return material_search(
            query=step.query,
            user_id=user_id,
            source_types=["chat_memory"],
            source_ids=[session_id],
            top_k=step.top_k,
            pdf_ids=None,
            config=config,
        )
    source_types = step.source_types or _source_types_for_tool(step.tool)
    return material_search(
        query=step.query,
        user_id=user_id,
        source_types=source_types,
        top_k=step.top_k,
        pdf_ids=step.filters.get("pdf_ids") if step.tool == RetrievalTool.SEARCH_PDF_CHUNKS else None,
        config=config,
    )


def _append_material_step(
    steps: list[RetrievalPlanStep],
    *,
    enabled: bool,
    include_disabled: bool,
    tool: RetrievalTool,
    query: str,
    top_k: int,
    source_type: str,
    source_ids: list[str] | None = None,
    filters: dict[str, Any] | None = None,
) -> None:
    if enabled:
        steps.append(
            RetrievalPlanStep(
                tool=tool,
                query=query,
                top_k=top_k,
                source_types=[source_type],
                source_ids=source_ids,
                filters=filters or {},
            )
        )
    elif include_disabled:
        steps.append(_disabled_step(tool, query, top_k, f"{source_type} is not allowed by source filters."))


def _disabled_step(tool: RetrievalTool, query: str, top_k: int, reason: str) -> RetrievalPlanStep:
    return RetrievalPlanStep(tool=tool, query=query, top_k=top_k, status=PlanStepStatus.DISABLED, reason=reason)


def _limit_enabled_steps(steps: list[RetrievalPlanStep], max_steps: int) -> list[RetrievalPlanStep]:
    limited: list[RetrievalPlanStep] = []
    enabled_count = 0
    for step in steps:
        if step.status == PlanStepStatus.ENABLED:
            if enabled_count >= max_steps:
                continue
            enabled_count += 1
        limited.append(step)
    return limited


def _allowed_material_sources(allowed_source_types: list[str] | None) -> set[str]:
    requested = allowed_source_types or list(MATERIAL_SOURCE_TYPES)
    return {source_type for source_type in requested if source_type in MATERIAL_SOURCE_ORDER}


def _source_types_for_tool(tool: RetrievalTool) -> list[str]:
    return {
        RetrievalTool.SEARCH_PDF_CHUNKS: ["pdf"],
        RetrievalTool.SEARCH_NOTES: ["note"],
        RetrievalTool.SEARCH_ANNOTATIONS: ["annotation_comment"],
    }.get(tool, [])


def _question_type(intent: RetrievalIntent | None) -> str:
    if intent is None:
        return "document_grounded"
    return str(getattr(intent.question_type, "value", intent.question_type))


def _step_outcome(
    step: RetrievalPlanStep,
    result_count: int,
    error_type: str | None,
    status: str | None = None,
) -> dict[str, Any]:
    return {
        "tool": step.tool.value,
        "status": status or step.status.value,
        "top_k": step.top_k,
        "reason": step.reason,
        "result_count": result_count,
        "error_type": error_type,
    }
