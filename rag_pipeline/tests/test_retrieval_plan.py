from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from rag_pipeline.intent_classifier import RetrievalIntent
from rag_pipeline.retrieval_plan import PlanExecutionOutcome
from rag_pipeline.retrieval_plan import PlanStepStatus
from rag_pipeline.retrieval_plan import RetrievalPlan
from rag_pipeline.retrieval_plan import RetrievalPlanStep
from rag_pipeline.retrieval_plan import RetrievalTool
from rag_pipeline.retrieval_plan import build_retrieval_plan
from rag_pipeline.retrieval_plan import execute_retrieval_plan
from rag_pipeline.retrieval_plan import normalize_plan_query


def _config(**overrides):
    values = {
        "retrieval_planner_enabled": True,
        "retrieval_planner_default_top_k": 6,
        "retrieval_planner_pdf_top_k": 6,
        "retrieval_planner_notes_top_k": 4,
        "retrieval_planner_annotations_top_k": 4,
        "retrieval_planner_memory_top_k": 3,
        "retrieval_planner_web_top_k": 5,
        "retrieval_planner_max_steps": 5,
        "retrieval_planner_include_disabled_steps": True,
        "web_search_enabled": False,
        "web_search_provider": "tavily",
        "web_search_timeout_seconds": 15,
        "web_search_max_query_chars": 300,
        "tavily_api_key": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _intent(**overrides):
    values = {
        "question_type": "document_grounded",
        "needs_pdf": True,
        "needs_notes": False,
        "needs_annotations": False,
        "needs_chat_memory": False,
        "needs_graph": False,
        "needs_web": False,
        "confidence": 0.8,
        "reasoning_summary": "test",
    }
    values.update(overrides)
    return RetrievalIntent(**values)


def test_retrieval_plan_step_schema_valid() -> None:
    step = RetrievalPlanStep(tool="search_pdf_chunks", query=" Was ist   BPMN? ", top_k=3)

    assert step.tool == RetrievalTool.SEARCH_PDF_CHUNKS
    assert step.query == "Was ist BPMN?"


def test_retrieval_plan_rejects_invalid_tool() -> None:
    with pytest.raises(ValidationError):
        RetrievalPlanStep(tool="unsafe_tool", query="Frage", top_k=3)


def test_retrieval_plan_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        RetrievalPlanStep(tool="search_pdf_chunks", query="Frage", top_k=3, unsafe=True)


def test_retrieval_plan_rejects_top_k_above_limit() -> None:
    with pytest.raises(ValidationError):
        RetrievalPlanStep(tool="search_pdf_chunks", query="Frage", top_k=21)


def test_normalize_plan_query_trims_collapses_and_caps() -> None:
    assert normalize_plan_query("  Was\n\nist\tBPMN?  ") == "Was ist BPMN?"
    assert len(normalize_plan_query("x" * 600)) == 500


def test_plan_adds_ordered_steps_and_respects_scope() -> None:
    plan = build_retrieval_plan(
        "Was ist der Zusammenhang?",
        _intent(needs_notes=True, needs_annotations=True, needs_chat_memory=True, needs_graph=True),
        _config(),
        session_id="session-1",
        selected_pdf_ids=["pdf-1"],
        allowed_source_types=["pdf", "note", "annotation_comment"],
    )

    assert [step.tool.value for step in plan.steps] == [
        "search_pdf_chunks",
        "search_notes",
        "search_annotations",
        "search_chat_memory",
        "query_knowledge_graph",
    ]
    assert plan.steps[0].source_ids == ["pdf-1"]
    assert plan.steps[-1].status == PlanStepStatus.DISABLED


def test_plan_chat_memory_includes_related_memory_source_ids() -> None:
    plan = build_retrieval_plan(
        "Was hatten wir besprochen?",
        _intent(needs_pdf=False, needs_chat_memory=True),
        _config(),
        session_id="session-current",
        memory_source_ids=["session-old", "session-current"],
    )

    memory_step = next(step for step in plan.steps if step.tool == RetrievalTool.SEARCH_CHAT_MEMORY)
    assert memory_step.source_ids == ["session-current", "session-old"]


def test_plan_skips_memory_and_web_when_unavailable() -> None:
    plan = build_retrieval_plan(
        "Was hatten wir aktuell besprochen?",
        _intent(needs_pdf=False, needs_chat_memory=True, needs_web=True),
        _config(web_search_enabled=False),
        session_id=None,
        allowed_source_types=["pdf"],
    )

    assert not any(step.status == PlanStepStatus.ENABLED and step.tool in {RetrievalTool.SEARCH_CHAT_MEMORY, RetrievalTool.WEB_SEARCH} for step in plan.steps)
    assert plan.fallback_used is True


def test_plan_respects_allowed_source_types() -> None:
    plan = build_retrieval_plan(
        "Was steht in meinen Notizen?",
        _intent(needs_pdf=True, needs_notes=True),
        _config(),
        allowed_source_types=["note"],
    )

    assert not any(step.status == PlanStepStatus.ENABLED and step.source_types == ["pdf"] for step in plan.steps)
    assert any(step.status == PlanStepStatus.ENABLED and step.source_types == ["note"] for step in plan.steps)


def test_plan_execution_outcome_schema_valid() -> None:
    outcome = PlanExecutionOutcome(results=[], step_outcomes=[{"tool": "search_pdf_chunks", "status": "enabled", "result_count": 0}], total_result_count=0)

    assert outcome.total_result_count == 0
    assert "text" not in outcome.step_outcomes[0]


def test_execute_plan_runs_enabled_steps_with_user_id_and_session() -> None:
    calls = []
    plan = RetrievalPlan(
        question_type="conversation_memory",
        steps=[
            RetrievalPlanStep(tool="search_pdf_chunks", query="BPMN", top_k=2, source_types=["pdf"]),
            RetrievalPlanStep(tool="search_chat_memory", query="vorhin BPMN", top_k=1, source_types=["chat_memory"], source_ids=["session-1"]),
        ],
    )

    def search(**kwargs):
        calls.append(kwargs)
        return [{"chunk_id": f"chunk-{len(calls)}", "text": "x", "source_type": kwargs["source_types"][0]}]

    outcome = execute_retrieval_plan(
        plan,
        "BPMN",
        "user-1",
        _config(),
        retrieval_fns={"search_hybrid_chunks": search},
        session_id="session-1",
    )

    assert calls[0]["user_id"] == "user-1"
    assert calls[1]["source_ids"] == ["session-1"]
    assert [result["chunk_id"] for result in outcome.results] == ["chunk-1", "chunk-2"]


def test_execute_plan_never_runs_graph_or_disabled_steps() -> None:
    calls = []
    plan = RetrievalPlan(
        question_type="concept_relationship",
        steps=[
            RetrievalPlanStep(tool="query_knowledge_graph", query="BPMN PM", top_k=5, status="disabled"),
            RetrievalPlanStep(tool="search_notes", query="BPMN", top_k=2, status="skipped"),
        ],
    )

    outcome = execute_retrieval_plan(
        plan,
        "BPMN",
        "user-1",
        _config(),
        retrieval_fns={"search_hybrid_chunks": lambda **kwargs: calls.append(kwargs) or []},
    )

    assert calls == []
    assert outcome.results == []


def test_execute_plan_tool_error_is_safe() -> None:
    plan = RetrievalPlan(question_type="document_grounded", steps=[RetrievalPlanStep(tool="search_pdf_chunks", query="BPMN", top_k=2)])

    outcome = execute_retrieval_plan(
        plan,
        "BPMN",
        "user-1",
        _config(),
        retrieval_fns={"search_hybrid_chunks": lambda **_: (_ for _ in ()).throw(RuntimeError("secret stack"))},
    )

    assert outcome.step_outcomes[0]["error_type"] == "tool_error"
    assert "secret" not in str(outcome.step_outcomes[0])
