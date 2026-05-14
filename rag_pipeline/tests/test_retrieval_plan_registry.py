"""Tests for execute_retrieval_plan with RetrievalToolRegistry integration (Phase 8)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from rag_pipeline.intent_classifier import RetrievalIntent
from rag_pipeline.retrieval_plan import (
    PlanExecutionOutcome,
    PlanStepStatus,
    RetrievalPlan,
    RetrievalPlanStep,
    RetrievalTool,
    build_retrieval_plan,
    execute_retrieval_plan,
)
from rag_pipeline.retrieval_tools import (
    RetrievalToolErrorType,
    RetrievalToolName,
    RetrievalToolOutcome,
    RetrievalToolRequest,
    RetrievalToolSpec,
    RetrievalToolStatus,
    RetrievalToolRegistry,
    build_default_retrieval_tool_registry,
    safe_empty_tool_outcome,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config(**overrides: Any) -> Any:
    defaults: dict[str, Any] = {
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
        "graph_retrieval_enabled": False,
        "graph_retrieval_top_k": 5,
        "graph_context_max_chars": 6000,
        "neo4j_uri": None,
        "neo4j_user": None,
        "neo4j_password": None,
        "retrieval_tool_registry_enabled": True,
        "retrieval_tool_allowed_tools": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _intent(**overrides: Any) -> RetrievalIntent:
    defaults: dict[str, Any] = {
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
    defaults.update(overrides)
    return RetrievalIntent(**defaults)


def _simple_plan(tools: list[RetrievalTool] | None = None) -> RetrievalPlan:
    tool_list = tools or [RetrievalTool.SEARCH_PDF_CHUNKS]
    steps = [
        RetrievalPlanStep(tool=t, query="test query", top_k=5)
        for t in tool_list
    ]
    return RetrievalPlan(question_type="document_grounded", steps=steps)


def _registry_with_fake_search(
    config: Any,
    results: list[dict[str, Any]] | None = None,
    tool_name: RetrievalToolName = RetrievalToolName.SEARCH_PDF_CHUNKS,
) -> RetrievalToolRegistry:
    """Build a registry where the given tool returns fake results."""
    fake_results = results or []

    def fake_execute(req: RetrievalToolRequest) -> RetrievalToolOutcome:
        from rag_pipeline.retrieval_tools import RetrievalToolResult
        tool_results = [
            RetrievalToolResult(
                chunk_id=r.get("chunk_id", "c1"),
                source_type=r.get("source_type", "pdf"),
                text=r.get("text", ""),
                score=r.get("score"),
            )
            for r in fake_results
        ]
        return RetrievalToolOutcome(
            tool=tool_name,
            status=RetrievalToolStatus.SUCCESS if tool_results else RetrievalToolStatus.EMPTY,
            results=tool_results,
            result_count=len(tool_results),
        )

    registry = RetrievalToolRegistry(config)
    registry.register(RetrievalToolSpec(
        name=tool_name,
        description="fake",
        requires_user_id=True,
        requires_session_id=tool_name == RetrievalToolName.SEARCH_CHAT_MEMORY,
        requires_config_flag=None,
        allowed_source_types=["pdf"],
        execute=fake_execute,
    ))
    return registry


# ---------------------------------------------------------------------------
# Phase 8 tests
# ---------------------------------------------------------------------------

class TestPlanExecutorWithRegistry:
    def test_uses_tool_registry_when_enabled(self) -> None:
        cfg = _config()
        plan = _simple_plan()
        registry = _registry_with_fake_search(cfg, [{"chunk_id": "x", "text": "result"}])
        outcome = execute_retrieval_plan(
            plan=plan, query="test", user_id="u1", config=cfg, tool_registry=registry
        )
        assert outcome.registry_used is True

    def test_preserves_plan_order_with_registry(self) -> None:
        cfg = _config()
        tool_order: list[str] = []

        def make_execute(name: RetrievalToolName) -> Any:
            def execute(req: RetrievalToolRequest) -> RetrievalToolOutcome:
                tool_order.append(name.value)
                return safe_empty_tool_outcome(name)
            return execute

        registry = RetrievalToolRegistry(cfg)
        for name in [RetrievalToolName.SEARCH_PDF_CHUNKS, RetrievalToolName.SEARCH_NOTES]:
            registry.register(RetrievalToolSpec(
                name=name, description="", requires_user_id=True,
                requires_session_id=False, requires_config_flag=None,
                allowed_source_types=[], execute=make_execute(name),
            ))

        plan = _simple_plan([RetrievalTool.SEARCH_PDF_CHUNKS, RetrievalTool.SEARCH_NOTES])
        execute_retrieval_plan(plan=plan, query="q", user_id="u", config=cfg, tool_registry=registry)
        assert tool_order == ["search_pdf_chunks", "search_notes"]

    def test_passes_user_id_to_tool_request(self) -> None:
        cfg = _config()
        received: list[str] = []

        def execute(req: RetrievalToolRequest) -> RetrievalToolOutcome:
            received.append(req.user_id)
            return safe_empty_tool_outcome(RetrievalToolName.SEARCH_PDF_CHUNKS)

        registry = RetrievalToolRegistry(cfg)
        registry.register(RetrievalToolSpec(
            name=RetrievalToolName.SEARCH_PDF_CHUNKS, description="",
            requires_user_id=True, requires_session_id=False, requires_config_flag=None,
            allowed_source_types=[], execute=execute,
        ))
        execute_retrieval_plan(
            plan=_simple_plan(), query="q", user_id="verified-user-id",
            config=cfg, tool_registry=registry,
        )
        assert received == ["verified-user-id"]

    def test_passes_session_id_to_memory_tool(self) -> None:
        cfg = _config()
        received: list[str | None] = []

        def execute(req: RetrievalToolRequest) -> RetrievalToolOutcome:
            received.append(req.session_id)
            return safe_empty_tool_outcome(RetrievalToolName.SEARCH_CHAT_MEMORY)

        registry = RetrievalToolRegistry(cfg)
        registry.register(RetrievalToolSpec(
            name=RetrievalToolName.SEARCH_CHAT_MEMORY, description="",
            requires_user_id=True, requires_session_id=False, requires_config_flag=None,
            allowed_source_types=[], execute=execute,
        ))

        step = RetrievalPlanStep(
            tool=RetrievalTool.SEARCH_CHAT_MEMORY,
            query="q",
            top_k=3,
            source_types=["chat_memory"],
            source_ids=["session-xyz"],
        )
        plan = RetrievalPlan(question_type="document_grounded", steps=[step])
        execute_retrieval_plan(
            plan=plan, query="q", user_id="u",
            config=cfg, tool_registry=registry, session_id="session-xyz",
        )
        assert received == ["session-xyz"]

    def test_passes_selected_pdf_scope_to_pdf_tool(self) -> None:
        cfg = _config()
        received: list[list[str] | None] = []

        def execute(req: RetrievalToolRequest) -> RetrievalToolOutcome:
            received.append(req.source_ids)
            return safe_empty_tool_outcome(RetrievalToolName.SEARCH_PDF_CHUNKS)

        registry = RetrievalToolRegistry(cfg)
        registry.register(RetrievalToolSpec(
            name=RetrievalToolName.SEARCH_PDF_CHUNKS, description="",
            requires_user_id=True, requires_session_id=False, requires_config_flag=None,
            allowed_source_types=[], execute=execute,
        ))
        step = RetrievalPlanStep(
            tool=RetrievalTool.SEARCH_PDF_CHUNKS, query="q", top_k=5,
            source_types=["pdf"], source_ids=["pdf-1", "pdf-2"],
            filters={"pdf_ids": ["pdf-1", "pdf-2"]},
        )
        plan = RetrievalPlan(question_type="document_grounded", steps=[step])
        execute_retrieval_plan(plan=plan, query="q", user_id="u", config=cfg, tool_registry=registry)
        assert received[0] == ["pdf-1", "pdf-2"]

    def test_records_tool_outcomes_in_plan_execution_outcome(self) -> None:
        cfg = _config()
        registry = _registry_with_fake_search(cfg)
        outcome = execute_retrieval_plan(
            plan=_simple_plan(), query="q", user_id="u", config=cfg, tool_registry=registry
        )
        assert len(outcome.tool_outcomes) == 1
        assert outcome.tool_outcomes[0]["tool"] == "search_pdf_chunks"
        assert "status" in outcome.tool_outcomes[0]
        assert "result_count" in outcome.tool_outcomes[0]

    def test_registry_tool_error_safe(self) -> None:
        cfg = _config()

        def boom(req: RetrievalToolRequest) -> RetrievalToolOutcome:
            raise RuntimeError("db failure")

        registry = RetrievalToolRegistry(cfg)
        registry.register(RetrievalToolSpec(
            name=RetrievalToolName.SEARCH_PDF_CHUNKS, description="",
            requires_user_id=True, requires_session_id=False, requires_config_flag=None,
            allowed_source_types=[], execute=boom,
        ))
        # Should not raise — tool error is caught safely
        outcome = execute_retrieval_plan(
            plan=_simple_plan(), query="q", user_id="u", config=cfg, tool_registry=registry
        )
        assert isinstance(outcome, PlanExecutionOutcome)

    def test_plan_executor_graph_runs_only_when_graph_retrieval_enabled(self) -> None:
        cfg = _config(graph_retrieval_enabled=True, neo4j_uri="bolt://x", neo4j_user="neo4j", neo4j_password="pw")
        graph_called: list[bool] = []

        def execute(req: RetrievalToolRequest) -> RetrievalToolOutcome:
            graph_called.append(True)
            return safe_empty_tool_outcome(RetrievalToolName.QUERY_KNOWLEDGE_GRAPH)

        registry = RetrievalToolRegistry(cfg)
        registry.register(RetrievalToolSpec(
            name=RetrievalToolName.QUERY_KNOWLEDGE_GRAPH, description="",
            requires_user_id=True, requires_session_id=False,
            requires_config_flag="graph_retrieval_enabled",
            allowed_source_types=[], execute=execute,
        ))

        step = RetrievalPlanStep(
            tool=RetrievalTool.QUERY_KNOWLEDGE_GRAPH, query="q", top_k=5
        )
        plan = RetrievalPlan(question_type="concept_map", steps=[step])
        execute_retrieval_plan(plan=plan, query="q", user_id="u", config=cfg, tool_registry=registry)
        assert graph_called == [True]

    def test_plan_executor_graph_skipped_when_graph_retrieval_disabled(self) -> None:
        cfg = _config(graph_retrieval_enabled=False)
        graph_called: list[bool] = []

        def execute(req: RetrievalToolRequest) -> RetrievalToolOutcome:
            graph_called.append(True)
            return safe_empty_tool_outcome(RetrievalToolName.QUERY_KNOWLEDGE_GRAPH)

        registry = RetrievalToolRegistry(cfg)
        registry.register(RetrievalToolSpec(
            name=RetrievalToolName.QUERY_KNOWLEDGE_GRAPH, description="",
            requires_user_id=True, requires_session_id=False,
            requires_config_flag="graph_retrieval_enabled",
            allowed_source_types=[], execute=execute,
        ))

        step = RetrievalPlanStep(
            tool=RetrievalTool.QUERY_KNOWLEDGE_GRAPH, query="q", top_k=5
        )
        plan = RetrievalPlan(question_type="concept_map", steps=[step])
        # Plan still succeeds — graph step is safely skipped by the registry's config check
        outcome = execute_retrieval_plan(
            plan=plan, query="q", user_id="u", config=cfg, tool_registry=registry
        )
        assert graph_called == []
        assert isinstance(outcome, PlanExecutionOutcome)

    def test_plan_executor_falls_back_when_registry_disabled(self) -> None:
        """Without registry, executor uses existing material_search path."""
        cfg = _config(retrieval_tool_registry_enabled=False)
        search_called: list[bool] = []

        def fake_search(**kwargs: Any) -> list[dict[str, Any]]:
            search_called.append(True)
            return []

        # Registry provided but disabled by config — should fall through to old path
        registry = build_default_retrieval_tool_registry(cfg, {"search_hybrid_chunks": fake_search})
        execute_retrieval_plan(
            plan=_simple_plan(),
            query="q",
            user_id="u",
            config=cfg,
            retrieval_fns={"search_hybrid_chunks": fake_search},
            tool_registry=registry,
        )
        # The old path calls fake_search directly
        assert search_called


class TestBuildRetrievalPlanGraph:
    def test_graph_step_enabled_when_graph_retrieval_enabled(self) -> None:
        cfg = _config(graph_retrieval_enabled=True, graph_retrieval_top_k=8)
        intent = _intent(needs_graph=True)
        plan = build_retrieval_plan("relationship question", intent, cfg)
        graph_steps = [s for s in plan.steps if s.tool == RetrievalTool.QUERY_KNOWLEDGE_GRAPH]
        assert len(graph_steps) == 1
        assert graph_steps[0].status == PlanStepStatus.ENABLED

    def test_graph_step_disabled_when_graph_retrieval_disabled(self) -> None:
        cfg = _config(graph_retrieval_enabled=False)
        intent = _intent(needs_graph=True)
        plan = build_retrieval_plan("relationship question", intent, cfg)
        graph_steps = [s for s in plan.steps if s.tool == RetrievalTool.QUERY_KNOWLEDGE_GRAPH]
        assert len(graph_steps) == 1
        assert graph_steps[0].status == PlanStepStatus.DISABLED
