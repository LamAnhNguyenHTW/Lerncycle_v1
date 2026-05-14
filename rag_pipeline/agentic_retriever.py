"""Controlled Agentic Retriever over the retrieval tool registry."""

from __future__ import annotations

import json
import logging
import re
import time
from enum import Enum
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rag_pipeline.retrieval_plan import PlanExecutionOutcome, RetrievalPlan, RetrievalPlanStep
from rag_pipeline.retrieval_plan import RetrievalTool as PlanTool
from rag_pipeline.retrieval_plan import execute_retrieval_plan
from rag_pipeline.retrieval_tools import RetrievalToolName, RetrievalToolRequest
from rag_pipeline.retrieval_tools import sanitize_tool_metadata


logger = logging.getLogger(__name__)


class RetrievalQualityStatus(str, Enum):
    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"
    EMPTY = "empty"
    ERROR = "error"


class RefinementDecision(str, Enum):
    NONE = "none"
    RETRY_SAME_TOOLS = "retry_same_tools"
    ADD_TOOL = "add_tool"
    BROADEN_QUERY = "broaden_query"
    FALLBACK_TO_DEFAULT = "fallback_to_default"


class RetrievalQualityAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: RetrievalQualityStatus
    total_result_count: int = Field(ge=0)
    avg_score: float | None = None
    missing_aspects: list[str] = Field(default_factory=list)
    recommended_decision: RefinementDecision
    reason: str | None = Field(default=None, max_length=300)


class RefinementAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: RefinementDecision
    tool: RetrievalToolName | None = None
    query: str | None = Field(default=None, max_length=500)
    top_k: int | None = Field(default=None, ge=1, le=50)
    reason: str | None = Field(default=None, max_length=300)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return normalize_refinement_query(value)

    @model_validator(mode="after")
    def require_tool_and_query_for_execution(self) -> "RefinementAction":
        if self.decision in {
            RefinementDecision.ADD_TOOL,
            RefinementDecision.RETRY_SAME_TOOLS,
            RefinementDecision.BROADEN_QUERY,
        }:
            if self.tool is None:
                raise ValueError("tool is required for executable refinement decisions")
            if not self.query:
                raise ValueError("query is required for executable refinement decisions")
        return self


class AgenticRetrievalOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: list[dict[str, Any]]
    initial_plan_executed: bool
    refinement_used: bool
    refinement_rounds: int = Field(ge=0)
    tool_call_count: int = Field(ge=0)
    quality: RetrievalQualityAssessment
    tool_outcomes: list[dict[str, Any]]
    fallback_used: bool = False
    error_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def sanitize_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return sanitize_agentic_metadata(value)


def normalize_refinement_query(query: str, max_chars: int = 500) -> str:
    return re.sub(r"\s+", " ", str(query or "")).strip()[:max_chars]


def sanitize_agentic_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    cleaned = sanitize_tool_metadata(metadata)
    return _drop_raw_content(cleaned)


def _drop_raw_content(value: Any) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            normalized = str(key).lower()
            if normalized in {"text", "content", "prompt", "raw_llm_output", "raw_results"}:
                continue
            result[key] = _drop_raw_content(item)
        return result
    if isinstance(value, list):
        return [_drop_raw_content(item) for item in value]
    return value


def safe_agentic_outcome_from_plan_execution(
    execution_outcome: PlanExecutionOutcome,
    quality: RetrievalQualityAssessment | None = None,
    *,
    fallback_used: bool = False,
    error_type: str | None = None,
) -> AgenticRetrievalOutcome:
    assessment = quality or RetrievalQualityAssessment(
        status=RetrievalQualityStatus.SUFFICIENT if execution_outcome.total_result_count else RetrievalQualityStatus.EMPTY,
        total_result_count=execution_outcome.total_result_count,
        avg_score=_avg_score(execution_outcome.results),
        missing_aspects=[],
        recommended_decision=RefinementDecision.NONE,
    )
    return AgenticRetrievalOutcome(
        results=execution_outcome.results,
        initial_plan_executed=True,
        refinement_used=False,
        refinement_rounds=0,
        tool_call_count=len(execution_outcome.step_outcomes),
        quality=assessment,
        tool_outcomes=_safe_tool_outcomes(execution_outcome.tool_outcomes or execution_outcome.step_outcomes),
        fallback_used=fallback_used,
        error_type=error_type,
    )


def assess_retrieval_quality_heuristic(
    query: str,
    intent: Any,
    plan: RetrievalPlan,
    execution_outcome: PlanExecutionOutcome,
    config: Any,
) -> RetrievalQualityAssessment:
    total = execution_outcome.total_result_count
    avg_score = _avg_score(execution_outcome.results)
    missing = _missing_aspects(intent, plan, execution_outcome)
    if any(step.get("error_type") == "tool_error" for step in execution_outcome.step_outcomes):
        status = RetrievalQualityStatus.ERROR
        decision = RefinementDecision.FALLBACK_TO_DEFAULT
    elif total == 0:
        status = RetrievalQualityStatus.EMPTY
        decision = RefinementDecision.RETRY_SAME_TOOLS
    elif total < getattr(config, "agentic_retriever_min_total_results", 3):
        status = RetrievalQualityStatus.INSUFFICIENT
        decision = RefinementDecision.BROADEN_QUERY
    elif avg_score is not None and avg_score < getattr(config, "agentic_retriever_min_avg_score", 0.25):
        status = RetrievalQualityStatus.INSUFFICIENT
        decision = RefinementDecision.BROADEN_QUERY
    elif missing:
        status = RetrievalQualityStatus.INSUFFICIENT
        decision = RefinementDecision.ADD_TOOL
    else:
        status = RetrievalQualityStatus.SUFFICIENT
        decision = RefinementDecision.NONE
    if "internal_material" in missing and status != RetrievalQualityStatus.SUFFICIENT:
        decision = RefinementDecision.BROADEN_QUERY
    return RetrievalQualityAssessment(
        status=status,
        total_result_count=total,
        avg_score=avg_score,
        missing_aspects=missing,
        recommended_decision=decision,
        reason="heuristic assessment",
    )


def assess_retrieval_quality_llm(
    query: str,
    intent: Any,
    plan: RetrievalPlan,
    execution_outcome: PlanExecutionOutcome,
    llm_client: Any = None,
    config: Any = None,
) -> RetrievalQualityAssessment:
    try:
        if llm_client is None:
            raise RuntimeError("llm_client unavailable")
        prompt = _quality_prompt(query, intent, plan, execution_outcome)
        raw = llm_client.complete(
            system_prompt="Return compact JSON for retrieval quality. No chain-of-thought.",
            user_prompt=prompt,
        )
        data = json.loads(raw)
        return RetrievalQualityAssessment(**data)
    except Exception:
        if config is None or getattr(config, "agentic_retriever_llm_fallback_to_heuristic", True):
            return assess_retrieval_quality_heuristic(query, intent, plan, execution_outcome, config or object())
        return RetrievalQualityAssessment(
            status=RetrievalQualityStatus.ERROR,
            total_result_count=execution_outcome.total_result_count,
            avg_score=_avg_score(execution_outcome.results),
            missing_aspects=[],
            recommended_decision=RefinementDecision.NONE,
            reason="llm assessment failed",
        )


def assess_retrieval_quality(
    mode: str,
    query: str,
    intent: Any,
    plan: RetrievalPlan,
    execution_outcome: PlanExecutionOutcome,
    config: Any,
    llm_client: Any = None,
) -> RetrievalQualityAssessment:
    if mode == "llm":
        return assess_retrieval_quality_llm(query, intent, plan, execution_outcome, llm_client, config)
    return assess_retrieval_quality_heuristic(query, intent, plan, execution_outcome, config)


def build_refinement_action_heuristic(
    query: str,
    intent: Any,
    plan: RetrievalPlan,
    quality: RetrievalQualityAssessment,
    registry: Any,
    config: Any,
    session_id: str | None = None,
    selected_pdf_ids: list[str] | None = None,
    allowed_source_types: list[str] | None = None,
) -> RefinementAction:
    if quality.status == RetrievalQualityStatus.SUFFICIENT or getattr(config, "agentic_retriever_max_refinement_rounds", 1) == 0:
        return RefinementAction(decision=RefinementDecision.NONE, reason="quality sufficient")
    tool = _tool_for_missing_aspect(quality.missing_aspects, registry, config, session_id, allowed_source_types)
    if tool is None:
        tool = _first_plan_tool(plan) or RetrievalToolName.SEARCH_PDF_CHUNKS
    decision = quality.recommended_decision
    if decision == RefinementDecision.NONE:
        decision = RefinementDecision.BROADEN_QUERY
    refined_query = _broaden_query(query) if decision == RefinementDecision.BROADEN_QUERY else query
    try:
        return RefinementAction(
            decision=decision,
            tool=tool,
            query=refined_query,
            top_k=min(10, getattr(config, "retrieval_tool_max_results_per_tool", 10)),
            reason="heuristic refinement",
        )
    except Exception:
        return RefinementAction(decision=RefinementDecision.NONE, reason="invalid heuristic refinement")


def build_refinement_action_llm(
    query: str,
    intent: Any,
    plan: RetrievalPlan,
    quality: RetrievalQualityAssessment,
    registry: Any,
    config: Any,
    llm_client: Any = None,
    session_id: str | None = None,
    selected_pdf_ids: list[str] | None = None,
    allowed_source_types: list[str] | None = None,
) -> RefinementAction:
    try:
        if llm_client is None:
            raise RuntimeError("llm_client unavailable")
        allowed_tools = [spec.name.value for spec in registry.list_tools() if registry.is_allowed(spec.name)]
        raw = llm_client.complete(
            system_prompt="Suggest one safe retrieval refinement as JSON only.",
            user_prompt=json.dumps({
                "query": query[:500],
                "intent": _intent_summary(intent),
                "plan": [{"tool": step.tool.value, "status": step.status.value} for step in plan.steps],
                "quality": quality.model_dump(mode="json"),
                "allowed_tools": allowed_tools,
            }),
        )
        action = RefinementAction(**json.loads(raw))
        return validate_refinement_action(
            action, registry, config, session_id, selected_pdf_ids, allowed_source_types
        )
    except Exception:
        if getattr(config, "agentic_retriever_llm_fallback_to_heuristic", True):
            return build_refinement_action_heuristic(
                query, intent, plan, quality, registry, config,
                session_id, selected_pdf_ids, allowed_source_types,
            )
        return RefinementAction(decision=RefinementDecision.NONE, reason="llm refinement failed")


def build_refinement_action(
    mode: str,
    query: str,
    intent: Any,
    plan: RetrievalPlan,
    quality: RetrievalQualityAssessment,
    registry: Any,
    config: Any,
    llm_client: Any = None,
    session_id: str | None = None,
    selected_pdf_ids: list[str] | None = None,
    allowed_source_types: list[str] | None = None,
) -> RefinementAction:
    if mode == "llm":
        return build_refinement_action_llm(
            query, intent, plan, quality, registry, config, llm_client,
            session_id, selected_pdf_ids, allowed_source_types,
        )
    return build_refinement_action_heuristic(
        query, intent, plan, quality, registry, config,
        session_id, selected_pdf_ids, allowed_source_types,
    )


def validate_refinement_action(
    action: RefinementAction,
    registry: Any,
    config: Any,
    session_id: str | None = None,
    selected_pdf_ids: list[str] | None = None,
    allowed_source_types: list[str] | None = None,
    tool_call_count: int = 0,
) -> RefinementAction:
    if action.decision == RefinementDecision.NONE:
        return action
    if tool_call_count >= getattr(config, "agentic_retriever_max_tool_calls", 8):
        return _rejected("tool budget exceeded")
    if action.tool is None or registry.get(action.tool) is None or not registry.is_allowed(action.tool):
        return _rejected("tool not allowed")
    if action.tool == RetrievalToolName.WEB_SEARCH and not getattr(config, "web_search_enabled", False):
        return _rejected("web disabled")
    if action.tool == RetrievalToolName.QUERY_KNOWLEDGE_GRAPH and not getattr(config, "graph_retrieval_enabled", False):
        return _rejected("graph disabled")
    if action.tool == RetrievalToolName.SEARCH_CHAT_MEMORY and not session_id:
        return _rejected("missing session")
    if not action.query:
        return _rejected("empty query")
    query = normalize_refinement_query(action.query)
    if not query:
        return _rejected("empty query")
    if action.tool == RetrievalToolName.QUERY_KNOWLEDGE_GRAPH and _looks_like_cypher(query):
        return _rejected("raw cypher rejected")
    if not _tool_allowed_by_sources(action.tool, allowed_source_types):
        return _rejected("source type not allowed")
    return action.model_copy(update={"query": query})


def run_controlled_agentic_retrieval(
    query: str,
    intent: Any,
    plan: RetrievalPlan,
    user_id: str,
    config: Any,
    tool_registry: Any,
    plan_executor_fn: Callable[..., PlanExecutionOutcome] | None = None,
    quality_assessor_fn: Callable[..., RetrievalQualityAssessment] | None = None,
    refinement_builder_fn: Callable[..., RefinementAction] | None = None,
    llm_client: Any = None,
    session_id: str | None = None,
    selected_pdf_ids: list[str] | None = None,
    allowed_source_types: list[str] | None = None,
) -> AgenticRetrievalOutcome:
    start = time.monotonic()
    if tool_registry is None or not getattr(config, "retrieval_tool_registry_enabled", False):
        fallback = (plan_executor_fn or execute_retrieval_plan)(
            plan=plan, query=query, user_id=user_id, config=config, session_id=session_id,
            tool_registry=tool_registry,
        )
        return safe_agentic_outcome_from_plan_execution(fallback, fallback_used=True, error_type="registry_unavailable")
    try:
        executor = plan_executor_fn or execute_retrieval_plan
        initial = executor(
            plan=plan, query=query, user_id=user_id, config=config, session_id=session_id,
            tool_registry=tool_registry,
        )
        assessor = quality_assessor_fn or (
            lambda **kwargs: assess_retrieval_quality(
                getattr(config, "agentic_retriever_quality_assessment_mode", "heuristic"), **kwargs
            )
        )
        quality = assessor(
            query=query, intent=intent, plan=plan, execution_outcome=initial,
            config=config, llm_client=llm_client,
        )
        tool_call_count = len(initial.tool_outcomes or initial.step_outcomes)
        results = list(initial.results)
        tool_outcomes = _safe_tool_outcomes(initial.tool_outcomes or initial.step_outcomes)
        refinement_used = False
        refinement_rounds = 0
        if (
            quality.status != RetrievalQualityStatus.SUFFICIENT
            and getattr(config, "agentic_retriever_max_refinement_rounds", 1) > 0
            and tool_call_count < getattr(config, "agentic_retriever_max_tool_calls", 8)
            and (time.monotonic() - start) < getattr(config, "agentic_retriever_max_latency_seconds", 30)
        ):
            builder = refinement_builder_fn or (
                lambda **kwargs: build_refinement_action(
                    getattr(config, "agentic_retriever_refinement_mode", "heuristic"), **kwargs
                )
            )
            action = builder(
                query=query, intent=intent, plan=plan, quality=quality, registry=tool_registry,
                config=config, llm_client=llm_client, session_id=session_id,
                selected_pdf_ids=selected_pdf_ids, allowed_source_types=allowed_source_types,
            )
            action = validate_refinement_action(
                action, tool_registry, config, session_id, selected_pdf_ids,
                allowed_source_types, tool_call_count,
            )
            if action.decision != RefinementDecision.NONE and action.tool is not None and action.query:
                request = RetrievalToolRequest(
                    tool=action.tool,
                    query=action.query,
                    top_k=action.top_k or 5,
                    user_id=user_id,
                    session_id=session_id,
                    source_types=(
                        allowed_source_types
                        if action.tool == RetrievalToolName.QUERY_KNOWLEDGE_GRAPH
                        else _source_types_for_tool(action.tool)
                    ),
                    source_ids=(
                        selected_pdf_ids
                        if action.tool in {
                            RetrievalToolName.SEARCH_PDF_CHUNKS,
                            RetrievalToolName.QUERY_KNOWLEDGE_GRAPH,
                        }
                        else None
                    ),
                    filters={"pdf_ids": selected_pdf_ids} if action.tool == RetrievalToolName.SEARCH_PDF_CHUNKS else {},
                )
                outcome = tool_registry.execute(request)
                tool_call_count += 1
                tool_outcomes.append(_safe_tool_outcome(outcome.model_dump(mode="json")))
                if outcome.results:
                    refinement_used = True
                    refinement_rounds = 1
                    results.extend([result.model_dump() for result in outcome.results])
                    final_execution = PlanExecutionOutcome(results, tool_outcomes, len(results), initial.fallback_used)
                    quality = assess_retrieval_quality_heuristic(query, intent, plan, final_execution, config)
        logger.info(
            "Controlled agentic retrieval completed.",
            extra={
                "quality_status": quality.status.value,
                "missing_aspects": quality.missing_aspects,
                "refinement_used": refinement_used,
                "tool_call_count": tool_call_count,
                "fallback_used": False,
            },
        )
        return AgenticRetrievalOutcome(
            results=results,
            initial_plan_executed=True,
            refinement_used=refinement_used,
            refinement_rounds=refinement_rounds,
            tool_call_count=tool_call_count,
            quality=quality,
            tool_outcomes=tool_outcomes,
            fallback_used=False,
            metadata={
                "quality_mode": getattr(config, "agentic_retriever_quality_assessment_mode", "heuristic"),
                "refinement_mode": getattr(config, "agentic_retriever_refinement_mode", "heuristic"),
            },
        )
    except Exception:
        logger.warning("Controlled agentic retrieval failed.", extra={"error_type": "agentic_error"})
        fallback = (plan_executor_fn or execute_retrieval_plan)(
            plan=plan, query=query, user_id=user_id, config=config, session_id=session_id,
            tool_registry=tool_registry,
        )
        return safe_agentic_outcome_from_plan_execution(fallback, fallback_used=True, error_type="agentic_error")


def _avg_score(results: list[dict[str, Any]]) -> float | None:
    scores = [float(r["score"]) for r in results if isinstance(r.get("score"), (int, float))]
    return sum(scores) / len(scores) if scores else None


def _missing_aspects(intent: Any, plan: RetrievalPlan, outcome: PlanExecutionOutcome) -> list[str]:
    missing: list[str] = []
    by_tool = {step.get("tool"): step for step in outcome.step_outcomes}
    if getattr(intent, "needs_web", False) and _tool_weak(by_tool.get("web_search")):
        missing.append("web")
    if getattr(intent, "needs_chat_memory", False) and _tool_weak(by_tool.get("search_chat_memory")):
        missing.append("chat_memory")
    if getattr(intent, "needs_graph", False) and _tool_weak(by_tool.get("query_knowledge_graph")):
        missing.append("knowledge_graph")
    material_tools = {"search_pdf_chunks", "search_notes", "search_annotations"}
    if any(getattr(intent, flag, False) for flag in ["needs_pdf", "needs_notes", "needs_annotations"]):
        material_counts = [by_tool.get(tool, {}).get("result_count", 0) for tool in material_tools]
        if not any(material_counts):
            missing.append("internal_material")
    return missing


def _tool_weak(step: dict[str, Any] | None) -> bool:
    return step is None or step.get("status") in {"empty", "skipped", "error", "disabled"} or step.get("result_count", 0) == 0


def _tool_for_missing_aspect(
    missing: list[str],
    registry: Any,
    config: Any,
    session_id: str | None,
    allowed_source_types: list[str] | None,
) -> RetrievalToolName | None:
    candidates = [
        ("web", RetrievalToolName.WEB_SEARCH),
        ("chat_memory", RetrievalToolName.SEARCH_CHAT_MEMORY),
        ("knowledge_graph", RetrievalToolName.QUERY_KNOWLEDGE_GRAPH),
        ("internal_material", RetrievalToolName.SEARCH_PDF_CHUNKS),
    ]
    for aspect, tool in candidates:
        if aspect in missing and registry.get(tool) is not None and registry.is_allowed(tool):
            if tool == RetrievalToolName.WEB_SEARCH and not getattr(config, "web_search_enabled", False):
                continue
            if tool == RetrievalToolName.SEARCH_CHAT_MEMORY and not session_id:
                continue
            if tool == RetrievalToolName.QUERY_KNOWLEDGE_GRAPH and not getattr(config, "graph_retrieval_enabled", False):
                continue
            if not _tool_allowed_by_sources(tool, allowed_source_types):
                continue
            return tool
    return None


def _first_plan_tool(plan: RetrievalPlan) -> RetrievalToolName | None:
    for step in plan.steps:
        try:
            return RetrievalToolName(step.tool.value)
        except ValueError:
            continue
    return None


def _broaden_query(query: str) -> str:
    return normalize_refinement_query(f"{query} Definition Zusammenhang Beispiele Kernaussagen")


def _rejected(reason: str) -> RefinementAction:
    return RefinementAction(decision=RefinementDecision.NONE, reason=reason)


def _looks_like_cypher(query: str) -> bool:
    normalized = query.lower()
    return "match (" in normalized or " return " in normalized or "merge (" in normalized


def _tool_allowed_by_sources(tool: RetrievalToolName, allowed_source_types: list[str] | None) -> bool:
    if allowed_source_types is None:
        return True
    mapping = {
        RetrievalToolName.SEARCH_PDF_CHUNKS: "pdf",
        RetrievalToolName.SEARCH_NOTES: "note",
        RetrievalToolName.SEARCH_ANNOTATIONS: "annotation_comment",
        RetrievalToolName.SEARCH_CHAT_MEMORY: "chat_memory",
        RetrievalToolName.WEB_SEARCH: "web",
        RetrievalToolName.QUERY_KNOWLEDGE_GRAPH: "knowledge_graph",
    }
    source_type = mapping.get(tool)
    return source_type in allowed_source_types or source_type in {"chat_memory", "web", "knowledge_graph"}


def _source_types_for_tool(tool: RetrievalToolName) -> list[str]:
    return {
        RetrievalToolName.SEARCH_PDF_CHUNKS: ["pdf"],
        RetrievalToolName.SEARCH_NOTES: ["note"],
        RetrievalToolName.SEARCH_ANNOTATIONS: ["annotation_comment"],
        RetrievalToolName.SEARCH_CHAT_MEMORY: ["chat_memory"],
        RetrievalToolName.WEB_SEARCH: ["web"],
        RetrievalToolName.QUERY_KNOWLEDGE_GRAPH: ["pdf", "note", "annotation_comment"],
    }.get(tool, [])


def _safe_tool_outcomes(outcomes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_safe_tool_outcome(outcome) for outcome in outcomes]


def _safe_tool_outcome(outcome: dict[str, Any]) -> dict[str, Any]:
    return {
        "tool": outcome.get("tool"),
        "status": outcome.get("status"),
        "result_count": outcome.get("result_count", 0),
        "latency_ms": outcome.get("latency_ms"),
        "error_type": outcome.get("error_type"),
    }


def _quality_prompt(query: str, intent: Any, plan: RetrievalPlan, outcome: PlanExecutionOutcome) -> str:
    snippets = [
        {
            "title": result.get("title"),
            "source_type": result.get("source_type"),
            "snippet": str(result.get("text") or "")[:240],
        }
        for result in outcome.results[:5]
    ]
    payload = {
        "query": query[:500],
        "intent": _intent_summary(intent),
        "plan": [{"tool": step.tool.value, "status": step.status.value} for step in plan.steps],
        "tool_outcomes": _safe_tool_outcomes(outcome.tool_outcomes or outcome.step_outcomes),
        "snippets": sanitize_agentic_metadata({"items": snippets})["items"],
    }
    return json.dumps(payload, ensure_ascii=True)


def _intent_summary(intent: Any) -> dict[str, Any]:
    return {
        "needs_pdf": bool(getattr(intent, "needs_pdf", False)),
        "needs_notes": bool(getattr(intent, "needs_notes", False)),
        "needs_annotations": bool(getattr(intent, "needs_annotations", False)),
        "needs_chat_memory": bool(getattr(intent, "needs_chat_memory", False)),
        "needs_graph": bool(getattr(intent, "needs_graph", False)),
        "needs_web": bool(getattr(intent, "needs_web", False)),
    }
