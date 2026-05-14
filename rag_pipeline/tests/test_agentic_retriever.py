from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from rag_pipeline.agentic_retriever import (
    AgenticRetrievalOutcome,
    RefinementAction,
    RefinementDecision,
    RetrievalQualityAssessment,
    RetrievalQualityStatus,
    assess_retrieval_quality,
    assess_retrieval_quality_heuristic,
    assess_retrieval_quality_llm,
    build_refinement_action_heuristic,
    build_refinement_action_llm,
    normalize_refinement_query,
    run_controlled_agentic_retrieval,
    safe_agentic_outcome_from_plan_execution,
    sanitize_agentic_metadata,
    validate_refinement_action,
)
from rag_pipeline.intent_classifier import RetrievalIntent
from rag_pipeline.retrieval_plan import PlanExecutionOutcome, RetrievalPlan, RetrievalPlanStep
from rag_pipeline.retrieval_tools import (
    RetrievalToolName,
    RetrievalToolOutcome,
    RetrievalToolRegistry,
    RetrievalToolResult,
    RetrievalToolSpec,
    RetrievalToolStatus,
    safe_empty_tool_outcome,
)


def _config(**overrides: Any) -> Any:
    values = {
        "retrieval_tool_registry_enabled": True,
        "retrieval_tool_max_results_per_tool": 20,
        "agentic_retriever_quality_assessment_mode": "heuristic",
        "agentic_retriever_refinement_mode": "heuristic",
        "agentic_retriever_llm_fallback_to_heuristic": True,
        "agentic_retriever_max_refinement_rounds": 1,
        "agentic_retriever_max_tool_calls": 8,
        "agentic_retriever_min_total_results": 3,
        "agentic_retriever_min_avg_score": 0.25,
        "agentic_retriever_max_latency_seconds": 30,
        "web_search_enabled": False,
        "graph_retrieval_enabled": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _intent(**overrides: Any) -> RetrievalIntent:
    values = {
        "question_type": "document_grounded",
        "needs_pdf": True,
        "needs_notes": False,
        "needs_annotations": False,
        "needs_chat_memory": False,
        "needs_graph": False,
        "needs_web": False,
        "confidence": 0.9,
        "reasoning_summary": "test",
    }
    values.update(overrides)
    return RetrievalIntent(**values)


def _result(score: float = 0.8, chunk_id: str = "c1") -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "source_type": "pdf",
        "source_id": "pdf-1",
        "text": "short text",
        "score": score,
        "title": "PDF",
        "metadata": {},
    }


def _plan(tool: str = "search_pdf_chunks") -> RetrievalPlan:
    return RetrievalPlan(
        question_type="document_grounded",
        steps=[RetrievalPlanStep(tool=tool, query="Process Mining", top_k=5)],
    )


def _execution(results: list[dict[str, Any]] | None = None, tool: str = "search_pdf_chunks") -> PlanExecutionOutcome:
    rows = results if results is not None else [_result(), _result(0.7, "c2"), _result(0.6, "c3")]
    return PlanExecutionOutcome(
        results=rows,
        step_outcomes=[{"tool": tool, "status": "success", "result_count": len(rows), "error_type": None}],
        total_result_count=len(rows),
        registry_used=True,
        tool_outcomes=[{"tool": tool, "status": "success", "result_count": len(rows), "error_type": None}],
    )


def _registry(config: Any, tool: RetrievalToolName = RetrievalToolName.SEARCH_PDF_CHUNKS) -> RetrievalToolRegistry:
    registry = RetrievalToolRegistry(config)

    def execute(req):
        return RetrievalToolOutcome(
            tool=req.tool,
            status=RetrievalToolStatus.SUCCESS,
            results=[RetrievalToolResult(chunk_id="refined", source_type="pdf", text="refined", score=0.9)],
            result_count=1,
        )

    registry.register(RetrievalToolSpec(
        name=tool,
        description="test",
        requires_user_id=True,
        requires_session_id=False,
        requires_config_flag=None,
        allowed_source_types=["pdf"],
        execute=execute,
    ))
    return registry


class FakeLlm:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict[str, str]] = []

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        return self.response


def test_quality_assessment_schema_valid() -> None:
    assessment = RetrievalQualityAssessment(
        status="sufficient",
        total_result_count=3,
        avg_score=0.7,
        recommended_decision="none",
    )
    assert assessment.status == RetrievalQualityStatus.SUFFICIENT


def test_refinement_action_requires_tool_when_add_tool() -> None:
    with pytest.raises(ValidationError):
        RefinementAction(decision="add_tool", query="q")


def test_refinement_action_none_decision_allows_null_tool_and_query() -> None:
    action = RefinementAction(decision="none")
    assert action.tool is None


def test_agentic_outcome_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        AgenticRetrievalOutcome(
            results=[],
            initial_plan_executed=True,
            refinement_used=False,
            refinement_rounds=0,
            tool_call_count=0,
            quality=RetrievalQualityAssessment(status="empty", total_result_count=0, recommended_decision="none"),
            tool_outcomes=[],
            unexpected=True,
        )


def test_agentic_metadata_sanitizer_removes_raw_content_and_secrets() -> None:
    result = sanitize_agentic_metadata({
        "prompt": "raw prompt",
        "api_key": "secret",
        "nested": {"content": "full text", "safe": "ok"},
    })
    assert result == {"nested": {"safe": "ok"}}


def test_safe_agentic_outcome_from_plan_execution() -> None:
    outcome = safe_agentic_outcome_from_plan_execution(_execution())
    assert outcome.initial_plan_executed is True
    assert outcome.tool_call_count == 1


def test_quality_assessment_sufficient_when_results_good() -> None:
    assessment = assess_retrieval_quality_heuristic("q", _intent(), _plan(), _execution(), _config())
    assert assessment.status == RetrievalQualityStatus.SUFFICIENT
    assert assessment.recommended_decision == RefinementDecision.NONE


def test_quality_assessment_empty_when_no_results() -> None:
    assessment = assess_retrieval_quality_heuristic("q", _intent(), _plan(), _execution([]), _config())
    assert assessment.status == RetrievalQualityStatus.EMPTY


def test_quality_assessment_insufficient_when_avg_score_low() -> None:
    assessment = assess_retrieval_quality_heuristic(
        "q", _intent(), _plan(), _execution([_result(0.1), _result(0.1, "c2"), _result(0.1, "c3")]), _config()
    )
    assert assessment.status == RetrievalQualityStatus.INSUFFICIENT


def test_quality_assessment_detects_missing_web() -> None:
    assessment = assess_retrieval_quality_heuristic("q", _intent(needs_web=True), _plan(), _execution(), _config())
    assert "web" in assessment.missing_aspects


def test_llm_quality_assessment_returns_valid_assessment_from_fake_llm() -> None:
    llm = FakeLlm('{"status":"sufficient","total_result_count":3,"avg_score":0.8,"missing_aspects":[],"recommended_decision":"none"}')
    assessment = assess_retrieval_quality_llm("q", _intent(), _plan(), _execution(), llm, _config())
    assert assessment.status == RetrievalQualityStatus.SUFFICIENT


def test_llm_quality_assessment_rejects_invalid_json_and_falls_back() -> None:
    assessment = assess_retrieval_quality_llm("q", _intent(), _plan(), _execution(), FakeLlm("not-json"), _config())
    assert assessment.reason == "heuristic assessment"


def test_assess_retrieval_quality_selector_uses_llm_mode() -> None:
    llm = FakeLlm('{"status":"empty","total_result_count":0,"missing_aspects":[],"recommended_decision":"none"}')
    assessment = assess_retrieval_quality("llm", "q", _intent(), _plan(), _execution([]), _config(), llm)
    assert assessment.status == RetrievalQualityStatus.EMPTY


def test_heuristic_refinement_action_none_when_quality_sufficient() -> None:
    action = build_refinement_action_heuristic(
        "q", _intent(), _plan(),
        RetrievalQualityAssessment(status="sufficient", total_result_count=3, recommended_decision="none"),
        _registry(_config()), _config(),
    )
    assert action.decision == RefinementDecision.NONE


def test_heuristic_refinement_action_broadens_query_for_weak_material_results() -> None:
    quality = RetrievalQualityAssessment(
        status="insufficient", total_result_count=1, missing_aspects=["internal_material"],
        recommended_decision="broaden_query",
    )
    action = build_refinement_action_heuristic("Process Mining", _intent(), _plan(), quality, _registry(_config()), _config())
    assert action.decision == RefinementDecision.BROADEN_QUERY
    assert "Kernaussagen" in action.query


def test_llm_refinement_builder_returns_valid_action_from_fake_llm() -> None:
    cfg = _config()
    action = build_refinement_action_llm(
        "q", _intent(), _plan(),
        RetrievalQualityAssessment(status="empty", total_result_count=0, recommended_decision="retry_same_tools"),
        _registry(cfg), cfg,
        llm_client=FakeLlm('{"decision":"retry_same_tools","tool":"search_pdf_chunks","query":"q refined","top_k":5}'),
    )
    assert action.tool == RetrievalToolName.SEARCH_PDF_CHUNKS


def test_validate_refinement_rejects_web_when_disabled() -> None:
    cfg = _config(web_search_enabled=False)
    registry = _registry(cfg, RetrievalToolName.WEB_SEARCH)
    action = RefinementAction(decision="add_tool", tool="web_search", query="q")
    assert validate_refinement_action(action, registry, cfg).decision == RefinementDecision.NONE


def test_validate_refinement_rejects_raw_cypher() -> None:
    cfg = _config(graph_retrieval_enabled=True)
    registry = _registry(cfg, RetrievalToolName.QUERY_KNOWLEDGE_GRAPH)
    action = RefinementAction(decision="add_tool", tool="query_knowledge_graph", query="MATCH (n) RETURN n")
    assert validate_refinement_action(action, registry, cfg).decision == RefinementDecision.NONE


def test_agentic_retriever_executes_initial_plan() -> None:
    calls: list[dict[str, Any]] = []

    def executor(**kwargs):
        calls.append(kwargs)
        return _execution([])

    outcome = run_controlled_agentic_retrieval(
        "q", _intent(), _plan(), "user-1", _config(agentic_retriever_max_refinement_rounds=0),
        _registry(_config()), plan_executor_fn=executor,
    )
    assert calls
    assert outcome.initial_plan_executed is True


def test_agentic_retriever_refines_when_quality_insufficient() -> None:
    cfg = _config()
    outcome = run_controlled_agentic_retrieval("q", _intent(), _plan(), "user-1", cfg, _registry(cfg), plan_executor_fn=lambda **_: _execution([]))
    assert outcome.refinement_used is True
    assert any(result["chunk_id"] == "refined" for result in outcome.results)


def test_agentic_retriever_falls_back_when_registry_disabled() -> None:
    cfg = _config(retrieval_tool_registry_enabled=False)
    outcome = run_controlled_agentic_retrieval("q", _intent(), _plan(), "user-1", cfg, None, plan_executor_fn=lambda **_: _execution())
    assert outcome.fallback_used is True
    assert outcome.error_type == "registry_unavailable"


def test_agentic_retriever_metadata_safe() -> None:
    outcome = run_controlled_agentic_retrieval("q", _intent(), _plan(), "user-1", _config(), _registry(_config()), plan_executor_fn=lambda **_: _execution())
    assert "raw_results" not in str(outcome.metadata)


def test_normalize_refinement_query() -> None:
    assert normalize_refinement_query("  a   b  ") == "a b"
    assert len(normalize_refinement_query("x" * 600)) == 500
