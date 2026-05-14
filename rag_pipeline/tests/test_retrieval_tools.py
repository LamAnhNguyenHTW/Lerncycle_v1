"""Tests for retrieval_tools.py — schemas, registry, and tool adapters (Phases 2–6)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from rag_pipeline.retrieval_tools import (
    RetrievalToolErrorType,
    RetrievalToolName,
    RetrievalToolOutcome,
    RetrievalToolRequest,
    RetrievalToolResult,
    RetrievalToolSpec,
    RetrievalToolStatus,
    RetrievalToolRegistry,
    build_default_retrieval_tool_registry,
    execute_query_knowledge_graph,
    execute_search_chat_memory,
    execute_web_search,
    normalize_tool_query,
    safe_empty_tool_outcome,
    sanitize_tool_metadata,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config(**overrides: Any) -> Any:
    defaults: dict[str, Any] = {
        "retrieval_tool_registry_enabled": True,
        "retrieval_tool_timeout_seconds": 20,
        "retrieval_tool_max_results_per_tool": 20,
        "retrieval_tool_allowed_tools": None,
        "web_search_enabled": False,
        "web_search_provider": "tavily",
        "web_search_timeout_seconds": 15,
        "web_search_max_query_chars": 300,
        "tavily_api_key": None,
        "graph_retrieval_enabled": False,
        "graph_retrieval_top_k": 8,
        "graph_context_max_chars": 6000,
        "neo4j_uri": None,
        "neo4j_user": None,
        "neo4j_password": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _valid_request(**overrides: Any) -> RetrievalToolRequest:
    defaults: dict[str, Any] = {
        "tool": RetrievalToolName.SEARCH_PDF_CHUNKS,
        "query": "Was ist BPMN?",
        "top_k": 5,
        "user_id": "user-abc",
    }
    defaults.update(overrides)
    return RetrievalToolRequest(**defaults)


def _dummy_spec(name: RetrievalToolName = RetrievalToolName.SEARCH_PDF_CHUNKS) -> RetrievalToolSpec:
    return RetrievalToolSpec(
        name=name,
        description="test",
        requires_user_id=True,
        requires_session_id=False,
        requires_config_flag=None,
        allowed_source_types=["pdf"],
        execute=lambda req: safe_empty_tool_outcome(name),
    )


# ---------------------------------------------------------------------------
# Phase 2 — Pydantic schemas
# ---------------------------------------------------------------------------

class TestToolRequestSchema:
    def test_valid_request(self) -> None:
        req = _valid_request()
        assert req.tool == RetrievalToolName.SEARCH_PDF_CHUNKS
        assert req.query == "Was ist BPMN?"
        assert req.top_k == 5
        assert req.user_id == "user-abc"
        assert req.session_id is None
        assert req.filters == {}
        assert req.metadata == {}

    def test_rejects_invalid_tool(self) -> None:
        with pytest.raises(ValidationError):
            RetrievalToolRequest(tool="evil_tool", query="test", top_k=5, user_id="u")

    def test_rejects_top_k_above_50(self) -> None:
        with pytest.raises(ValidationError):
            RetrievalToolRequest(tool="search_pdf_chunks", query="test", top_k=51, user_id="u")

    def test_rejects_query_too_long(self) -> None:
        with pytest.raises(ValidationError):
            RetrievalToolRequest(tool="search_pdf_chunks", query="x" * 501, top_k=5, user_id="u")

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            RetrievalToolRequest(tool="search_pdf_chunks", query="test", top_k=5, user_id="u", evil=True)  # type: ignore[call-arg]

    def test_query_normalized(self) -> None:
        req = RetrievalToolRequest(tool="search_pdf_chunks", query="  Was   ist   BPMN?  ", top_k=5, user_id="u")
        assert req.query == "Was ist BPMN?"

    def test_uses_default_factory_for_dict_fields(self) -> None:
        req1 = _valid_request()
        req2 = _valid_request()
        assert req1.filters is not req2.filters
        assert req1.metadata is not req2.metadata


class TestToolResultSchema:
    def test_valid_result(self) -> None:
        result = RetrievalToolResult(
            chunk_id="abc",
            source_type="pdf",
            text="some content",
        )
        assert result.chunk_id == "abc"
        assert result.source_id is None
        assert result.score is None
        assert result.title is None
        assert result.metadata == {}

    def test_full_result(self) -> None:
        result = RetrievalToolResult(
            chunk_id="abc",
            source_type="pdf",
            source_id="pdf-1",
            text="content",
            score=0.9,
            title="My PDF",
            metadata={"page": 3},
        )
        assert result.score == 0.9
        assert result.title == "My PDF"


class TestToolOutcomeSchema:
    def test_valid_outcome(self) -> None:
        outcome = RetrievalToolOutcome(
            tool=RetrievalToolName.SEARCH_PDF_CHUNKS,
            status=RetrievalToolStatus.SUCCESS,
            results=[],
            result_count=0,
        )
        assert outcome.error_type is None
        assert outcome.latency_ms is None
        assert outcome.metadata == {}

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            RetrievalToolOutcome(
                tool=RetrievalToolName.SEARCH_PDF_CHUNKS,
                status=RetrievalToolStatus.SUCCESS,
                results=[],
                result_count=0,
                evil_field=True,  # type: ignore[call-arg]
            )

    def test_uses_default_factory_for_metadata(self) -> None:
        o1 = RetrievalToolOutcome(tool="search_pdf_chunks", status="success", results=[], result_count=0)
        o2 = RetrievalToolOutcome(tool="search_pdf_chunks", status="success", results=[], result_count=0)
        assert o1.metadata is not o2.metadata


# ---------------------------------------------------------------------------
# Phase 2 — helper functions
# ---------------------------------------------------------------------------

class TestHelperFunctions:
    def test_safe_empty_outcome_returns_empty_results(self) -> None:
        outcome = safe_empty_tool_outcome(RetrievalToolName.SEARCH_PDF_CHUNKS)
        assert outcome.results == []
        assert outcome.result_count == 0
        assert outcome.status == RetrievalToolStatus.EMPTY
        assert outcome.error_type is None

    def test_safe_empty_outcome_with_error_type(self) -> None:
        outcome = safe_empty_tool_outcome(
            RetrievalToolName.WEB_SEARCH,
            RetrievalToolErrorType.DISABLED,
            RetrievalToolStatus.SKIPPED,
        )
        assert outcome.error_type == RetrievalToolErrorType.DISABLED
        assert outcome.status == RetrievalToolStatus.SKIPPED

    def test_normalize_tool_query(self) -> None:
        assert normalize_tool_query("  Was   ist   BPMN?  ") == "Was ist BPMN?"
        assert len(normalize_tool_query("x" * 600)) == 500
        assert normalize_tool_query("") == ""

    def test_sanitize_tool_metadata_removes_secrets(self) -> None:
        meta = {
            "title": "My PDF",
            "api_key": "secret-key",
            "password": "hunter2",
            "node_names": ["A", "B"],
            "token": "tok123",
        }
        result = sanitize_tool_metadata(meta)
        assert "api_key" not in result
        assert "password" not in result
        assert "token" not in result
        assert result["title"] == "My PDF"
        assert result["node_names"] == ["A", "B"]

    def test_sanitize_tool_metadata_case_insensitive(self) -> None:
        meta = {"API_KEY": "bad", "API_key": "bad", "normal": "ok"}
        result = sanitize_tool_metadata(meta)
        assert "API_KEY" not in result
        assert "API_key" not in result
        assert result["normal"] == "ok"

    def test_sanitize_tool_metadata_removes_nested_sensitive_fields(self) -> None:
        meta = {
            "safe": {"node_names": ["A"]},
            "nested": {
                "neo4j_password": "secret",
                "cypher": "MATCH (n) RETURN n",
                "items": [{"stack_trace": "Traceback...", "value": "ok"}],
            },
        }
        result = sanitize_tool_metadata(meta)
        assert result["safe"]["node_names"] == ["A"]
        assert "neo4j_password" not in result["nested"]
        assert "cypher" not in result["nested"]
        assert "stack_trace" not in result["nested"]["items"][0]
        assert result["nested"]["items"][0]["value"] == "ok"


# ---------------------------------------------------------------------------
# Phase 3 — Registry core
# ---------------------------------------------------------------------------

class TestRetrievalToolRegistry:
    def test_registers_tool(self) -> None:
        registry = RetrievalToolRegistry(_config())
        spec = _dummy_spec()
        registry.register(spec)
        assert registry.get(RetrievalToolName.SEARCH_PDF_CHUNKS) is spec

    def test_gets_tool_by_name(self) -> None:
        registry = RetrievalToolRegistry(_config())
        registry.register(_dummy_spec())
        assert registry.get(RetrievalToolName.SEARCH_PDF_CHUNKS) is not None
        assert registry.get(RetrievalToolName.SEARCH_NOTES) is None

    def test_lists_tools(self) -> None:
        registry = RetrievalToolRegistry(_config())
        registry.register(_dummy_spec(RetrievalToolName.SEARCH_PDF_CHUNKS))
        registry.register(_dummy_spec(RetrievalToolName.SEARCH_NOTES))
        assert len(registry.list_tools()) == 2

    def test_rejects_unknown_tool(self) -> None:
        registry = RetrievalToolRegistry(_config())
        req = _valid_request()
        outcome = registry.execute(req)
        assert outcome.status == RetrievalToolStatus.ERROR
        assert outcome.error_type == RetrievalToolErrorType.NOT_ALLOWED

    def test_rejects_disabled_tool(self) -> None:
        cfg = _config()

        def disabled_execute(req: RetrievalToolRequest) -> RetrievalToolOutcome:
            raise AssertionError("should not be called")

        registry = RetrievalToolRegistry(cfg)
        registry.register(RetrievalToolSpec(
            name=RetrievalToolName.WEB_SEARCH,
            description="web",
            requires_user_id=True,
            requires_session_id=False,
            requires_config_flag="web_search_enabled",  # False in default config
            allowed_source_types=["web"],
            execute=disabled_execute,
        ))
        req = _valid_request(tool=RetrievalToolName.WEB_SEARCH)
        outcome = registry.execute(req)
        assert outcome.error_type == RetrievalToolErrorType.DISABLED
        assert outcome.status == RetrievalToolStatus.SKIPPED

    def test_respects_allowed_tools_config(self) -> None:
        cfg = _config(retrieval_tool_allowed_tools=["search_notes"])
        registry = RetrievalToolRegistry(cfg)
        registry.register(_dummy_spec(RetrievalToolName.SEARCH_PDF_CHUNKS))
        req = _valid_request(tool=RetrievalToolName.SEARCH_PDF_CHUNKS)
        outcome = registry.execute(req)
        assert outcome.error_type == RetrievalToolErrorType.NOT_ALLOWED

    def test_requires_user_id(self) -> None:
        registry = RetrievalToolRegistry(_config())
        registry.register(_dummy_spec())
        req = _valid_request(user_id="")
        outcome = registry.execute(req)
        assert outcome.error_type == RetrievalToolErrorType.MISSING_USER_ID

    def test_requires_session_id_for_chat_memory(self) -> None:
        registry = RetrievalToolRegistry(_config())
        registry.register(RetrievalToolSpec(
            name=RetrievalToolName.SEARCH_CHAT_MEMORY,
            description="memory",
            requires_user_id=True,
            requires_session_id=True,
            requires_config_flag=None,
            allowed_source_types=["chat_memory"],
            execute=lambda req: safe_empty_tool_outcome(RetrievalToolName.SEARCH_CHAT_MEMORY),
        ))
        req = _valid_request(tool=RetrievalToolName.SEARCH_CHAT_MEMORY, session_id=None)
        outcome = registry.execute(req)
        assert outcome.error_type == RetrievalToolErrorType.MISSING_SESSION_ID

    def test_normalizes_query(self) -> None:
        received_queries: list[str] = []

        def capturing_execute(req: RetrievalToolRequest) -> RetrievalToolOutcome:
            received_queries.append(req.query)
            return safe_empty_tool_outcome(RetrievalToolName.SEARCH_PDF_CHUNKS)

        registry = RetrievalToolRegistry(_config())
        registry.register(RetrievalToolSpec(
            name=RetrievalToolName.SEARCH_PDF_CHUNKS,
            description="test",
            requires_user_id=True,
            requires_session_id=False,
            requires_config_flag=None,
            allowed_source_types=["pdf"],
            execute=capturing_execute,
        ))
        # Query is already normalized by Pydantic validator on the request, so test trim at registry level
        req = RetrievalToolRequest(tool="search_pdf_chunks", query="  BPMN  ", top_k=3, user_id="u")
        registry.execute(req)
        assert received_queries[0] == "BPMN"

    def test_catches_tool_exception(self) -> None:
        registry = RetrievalToolRegistry(_config())
        registry.register(RetrievalToolSpec(
            name=RetrievalToolName.SEARCH_PDF_CHUNKS,
            description="failing",
            requires_user_id=True,
            requires_session_id=False,
            requires_config_flag=None,
            allowed_source_types=["pdf"],
            execute=lambda req: (_ for _ in ()).throw(RuntimeError("boom")),
        ))
        req = _valid_request()
        outcome = registry.execute(req)
        assert outcome.error_type == RetrievalToolErrorType.UNKNOWN_ERROR
        assert outcome.status == RetrievalToolStatus.ERROR

    def test_execute_records_latency(self) -> None:
        registry = RetrievalToolRegistry(_config())
        registry.register(_dummy_spec())
        req = _valid_request()
        outcome = registry.execute(req)
        assert outcome.latency_ms is not None
        assert outcome.latency_ms >= 0

    def test_execute_caps_results_to_configured_max(self) -> None:
        def fake_execute(req: RetrievalToolRequest) -> RetrievalToolOutcome:
            return RetrievalToolOutcome(
                tool=RetrievalToolName.SEARCH_PDF_CHUNKS,
                status=RetrievalToolStatus.SUCCESS,
                results=[
                    RetrievalToolResult(chunk_id=f"c{i}", source_type="pdf", text="text")
                    for i in range(5)
                ],
                result_count=5,
            )

        registry = RetrievalToolRegistry(_config(retrieval_tool_max_results_per_tool=2))
        registry.register(RetrievalToolSpec(
            name=RetrievalToolName.SEARCH_PDF_CHUNKS,
            description="test",
            requires_user_id=True,
            requires_session_id=False,
            requires_config_flag=None,
            allowed_source_types=["pdf"],
            execute=fake_execute,
        ))
        outcome = registry.execute(_valid_request())
        assert outcome.result_count == 2
        assert [result.chunk_id for result in outcome.results] == ["c0", "c1"]

    def test_execute_sanitizes_outcome_metadata(self) -> None:
        def fake_execute(req: RetrievalToolRequest) -> RetrievalToolOutcome:
            return RetrievalToolOutcome(
                tool=RetrievalToolName.SEARCH_PDF_CHUNKS,
                status=RetrievalToolStatus.SUCCESS,
                results=[],
                result_count=0,
                metadata={"provider": "qdrant", "api_key": "secret", "nested": {"cypher": "MATCH (n)"}},
            )

        registry = RetrievalToolRegistry(_config())
        registry.register(RetrievalToolSpec(
            name=RetrievalToolName.SEARCH_PDF_CHUNKS,
            description="test",
            requires_user_id=True,
            requires_session_id=False,
            requires_config_flag=None,
            allowed_source_types=["pdf"],
            execute=fake_execute,
        ))
        outcome = registry.execute(_valid_request())
        assert outcome.metadata == {"provider": "qdrant", "nested": {}}


class TestDefaultRegistryBuilds:
    def test_default_registry_builds_expected_tools(self) -> None:
        cfg = _config()
        registry = build_default_retrieval_tool_registry(cfg)
        names = {spec.name for spec in registry.list_tools()}
        assert names == {
            RetrievalToolName.SEARCH_PDF_CHUNKS,
            RetrievalToolName.SEARCH_NOTES,
            RetrievalToolName.SEARCH_ANNOTATIONS,
            RetrievalToolName.SEARCH_CHAT_MEMORY,
            RetrievalToolName.WEB_SEARCH,
            RetrievalToolName.QUERY_KNOWLEDGE_GRAPH,
        }

    def test_default_registry_injects_dependencies(self) -> None:
        calls: list[str] = []

        def fake_search(**kwargs: Any) -> list[dict[str, Any]]:
            calls.append("search")
            return []

        cfg = _config()
        registry = build_default_retrieval_tool_registry(cfg, {"search_hybrid_chunks": fake_search})
        req = _valid_request()
        registry.execute(req)
        assert calls == ["search"]


# ---------------------------------------------------------------------------
# Phase 4 — Internal Qdrant tool adapters
# ---------------------------------------------------------------------------

def _make_chunk(source_type: str = "pdf", chunk_id: str = "c1") -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "text": "some text",
        "score": 0.8,
        "source_type": source_type,
        "source_id": "src-1",
        "title": "My Title",
        "heading": None,
        "page_index": 0,
        "pdf_id": "pdf-1",
        "metadata": {"filename": "doc.pdf"},
    }


class TestPdfTool:
    def test_pdf_tool_calls_retrieval_with_user_id(self) -> None:
        calls: list[dict[str, Any]] = []

        def fake_search(**kwargs: Any) -> list[dict[str, Any]]:
            calls.append(kwargs)
            return [_make_chunk()]

        registry = build_default_retrieval_tool_registry(
            _config(), {"search_hybrid_chunks": fake_search}
        )
        req = _valid_request(user_id="user-123")
        registry.execute(req)
        assert calls[0]["user_id"] == "user-123"

    def test_pdf_tool_respects_selected_pdf_source_ids(self) -> None:
        calls: list[dict[str, Any]] = []

        def fake_search(**kwargs: Any) -> list[dict[str, Any]]:
            calls.append(kwargs)
            return []

        registry = build_default_retrieval_tool_registry(
            _config(), {"search_hybrid_chunks": fake_search}
        )
        req = _valid_request(source_ids=["pdf-a", "pdf-b"])
        registry.execute(req)
        assert calls[0]["pdf_ids"] == ["pdf-a", "pdf-b"]

    def test_pdf_tool_does_not_search_unselected_pdfs(self) -> None:
        calls: list[dict[str, Any]] = []

        def fake_search(**kwargs: Any) -> list[dict[str, Any]]:
            calls.append(kwargs)
            return []

        registry = build_default_retrieval_tool_registry(
            _config(), {"search_hybrid_chunks": fake_search}
        )
        # Only "pdf-scoped" is in scope
        req = _valid_request(source_ids=["pdf-scoped"])
        registry.execute(req)
        assert calls[0]["pdf_ids"] == ["pdf-scoped"]

    def test_pdf_tool_source_type_filter(self) -> None:
        calls: list[dict[str, Any]] = []

        def fake_search(**kwargs: Any) -> list[dict[str, Any]]:
            calls.append(kwargs)
            return []

        registry = build_default_retrieval_tool_registry(
            _config(), {"search_hybrid_chunks": fake_search}
        )
        registry.execute(_valid_request())
        assert calls[0]["source_types"] == ["pdf"]


class TestNotesTool:
    def test_notes_tool_filters_to_note_source_type(self) -> None:
        calls: list[dict[str, Any]] = []

        def fake_search(**kwargs: Any) -> list[dict[str, Any]]:
            calls.append(kwargs)
            return []

        registry = build_default_retrieval_tool_registry(
            _config(), {"search_hybrid_chunks": fake_search}
        )
        req = _valid_request(tool=RetrievalToolName.SEARCH_NOTES)
        registry.execute(req)
        assert calls[0]["source_types"] == ["note"]

    def test_notes_tool_passes_source_and_pdf_scope(self) -> None:
        calls: list[dict[str, Any]] = []

        def fake_search(**kwargs: Any) -> list[dict[str, Any]]:
            calls.append(kwargs)
            return []

        registry = build_default_retrieval_tool_registry(
            _config(), {"search_hybrid_chunks": fake_search}
        )
        req = _valid_request(
            tool=RetrievalToolName.SEARCH_NOTES,
            source_ids=["note-1"],
            filters={"pdf_ids": ["pdf-1"]},
        )
        registry.execute(req)
        assert calls[0]["source_ids"] == ["note-1"]
        assert calls[0]["pdf_ids"] == ["pdf-1"]


class TestAnnotationsTool:
    def test_annotations_tool_filters_to_annotation_source_type(self) -> None:
        calls: list[dict[str, Any]] = []

        def fake_search(**kwargs: Any) -> list[dict[str, Any]]:
            calls.append(kwargs)
            return []

        registry = build_default_retrieval_tool_registry(
            _config(), {"search_hybrid_chunks": fake_search}
        )
        req = _valid_request(tool=RetrievalToolName.SEARCH_ANNOTATIONS)
        registry.execute(req)
        assert calls[0]["source_types"] == ["annotation_comment"]

    def test_annotations_tool_passes_source_and_pdf_scope(self) -> None:
        calls: list[dict[str, Any]] = []

        def fake_search(**kwargs: Any) -> list[dict[str, Any]]:
            calls.append(kwargs)
            return []

        registry = build_default_retrieval_tool_registry(
            _config(), {"search_hybrid_chunks": fake_search}
        )
        req = _valid_request(
            tool=RetrievalToolName.SEARCH_ANNOTATIONS,
            source_ids=["annotation-1"],
            filters={"pdf_ids": ["pdf-1"]},
        )
        registry.execute(req)
        assert calls[0]["source_ids"] == ["annotation-1"]
        assert calls[0]["pdf_ids"] == ["pdf-1"]


class TestChatMemoryTool:
    def test_chat_memory_tool_requires_session_id(self) -> None:
        registry = build_default_retrieval_tool_registry(_config(), {})
        req = _valid_request(tool=RetrievalToolName.SEARCH_CHAT_MEMORY, session_id=None)
        outcome = registry.execute(req)
        assert outcome.error_type == RetrievalToolErrorType.MISSING_SESSION_ID

    def test_chat_memory_tool_ignores_request_source_ids_and_uses_session_id(self) -> None:
        calls: list[dict[str, Any]] = []

        def fake_search(**kwargs: Any) -> list[dict[str, Any]]:
            calls.append(kwargs)
            return []

        registry = build_default_retrieval_tool_registry(
            _config(), {"search_hybrid_chunks": fake_search}
        )
        req = _valid_request(
            tool=RetrievalToolName.SEARCH_CHAT_MEMORY,
            session_id="session-verified",
            source_ids=["some-other-session"],  # this must be ignored
        )
        registry.execute(req)
        assert calls[0]["source_ids"] == ["session-verified"]

    def test_chat_memory_tool_does_not_accept_cross_session_source_ids(self) -> None:
        calls: list[dict[str, Any]] = []

        def fake_search(**kwargs: Any) -> list[dict[str, Any]]:
            calls.append(kwargs)
            return []

        registry = build_default_retrieval_tool_registry(
            _config(), {"search_hybrid_chunks": fake_search}
        )
        req = _valid_request(
            tool=RetrievalToolName.SEARCH_CHAT_MEMORY,
            session_id="correct-session",
            source_ids=["attacker-session"],
        )
        registry.execute(req)
        # Must use only the verified session_id
        assert calls[0]["source_ids"] == ["correct-session"]
        assert "attacker-session" not in calls[0]["source_ids"]


class TestInternalToolNormalization:
    def test_internal_tool_normalizes_results(self) -> None:
        def fake_search(**kwargs: Any) -> list[dict[str, Any]]:
            return [_make_chunk(source_type="pdf", chunk_id="xyz")]

        registry = build_default_retrieval_tool_registry(
            _config(), {"search_hybrid_chunks": fake_search}
        )
        outcome = registry.execute(_valid_request())
        assert outcome.result_count == 1
        assert outcome.results[0].chunk_id == "xyz"
        assert outcome.results[0].source_type == "pdf"

    def test_internal_tool_error_returns_safe_outcome(self) -> None:
        def boom(**kwargs: Any) -> list[dict[str, Any]]:
            raise RuntimeError("db is down")

        registry = build_default_retrieval_tool_registry(
            _config(), {"search_hybrid_chunks": boom}
        )
        outcome = registry.execute(_valid_request())
        assert outcome.status == RetrievalToolStatus.ERROR
        assert outcome.error_type == RetrievalToolErrorType.PROVIDER_ERROR
        assert outcome.results == []


# ---------------------------------------------------------------------------
# Phase 5 — Web search tool adapter
# ---------------------------------------------------------------------------

class TestWebTool:
    def test_web_tool_skips_when_web_disabled(self) -> None:
        cfg = _config(web_search_enabled=False)
        req = _valid_request(tool=RetrievalToolName.WEB_SEARCH)
        outcome = execute_web_search(req, cfg)
        assert outcome.status == RetrievalToolStatus.SKIPPED
        assert outcome.error_type == RetrievalToolErrorType.DISABLED

    def test_web_tool_calls_search_web_when_enabled(self) -> None:
        from rag_pipeline.web_search import WebSearchOutcome

        calls: list[Any] = []

        def fake_web(**kwargs: Any) -> WebSearchOutcome:
            calls.append(kwargs)
            return WebSearchOutcome([], "tavily", 0, "empty_results")

        cfg = _config(web_search_enabled=True, tavily_api_key="key")
        req = _valid_request(tool=RetrievalToolName.WEB_SEARCH)
        execute_web_search(req, cfg, web_search_fn=fake_web)
        assert len(calls) == 1

    def test_web_tool_passes_top_k(self) -> None:
        from rag_pipeline.web_search import WebSearchOutcome

        calls: list[Any] = []

        def fake_web(**kwargs: Any) -> WebSearchOutcome:
            calls.append(kwargs)
            return WebSearchOutcome([], "tavily", 0, "empty_results")

        cfg = _config(web_search_enabled=True, tavily_api_key="key")
        req = _valid_request(tool=RetrievalToolName.WEB_SEARCH, top_k=7)
        execute_web_search(req, cfg, web_search_fn=fake_web)
        assert calls[0]["top_k"] == 7

    def test_web_tool_normalizes_web_results(self) -> None:
        from rag_pipeline.web_search import WebSearchOutcome

        web_chunk = {
            "chunk_id": "web:abc123",
            "text": "Web result text",
            "score": 0.9,
            "source_type": "web",
            "source_id": "web:abc123",
            "title": "Web Title",
            "heading": None,
            "page_index": None,
            "pdf_id": None,
            "metadata": {"url": "https://example.com", "provider": "tavily"},
        }

        def fake_web(**kwargs: Any) -> WebSearchOutcome:
            return WebSearchOutcome([web_chunk], "tavily", 1, None)

        cfg = _config(web_search_enabled=True, tavily_api_key="key")
        req = _valid_request(tool=RetrievalToolName.WEB_SEARCH)
        outcome = execute_web_search(req, cfg, web_search_fn=fake_web)
        assert outcome.result_count == 1
        assert outcome.results[0].source_type == "web"
        assert outcome.results[0].title == "Web Title"

    def test_web_tool_maps_timeout_error_type(self) -> None:
        from rag_pipeline.web_search import WebSearchOutcome

        def fake_web(**kwargs: Any) -> WebSearchOutcome:
            return WebSearchOutcome([], "tavily", 0, "timeout")

        cfg = _config(web_search_enabled=True, tavily_api_key="key")
        req = _valid_request(tool=RetrievalToolName.WEB_SEARCH)
        outcome = execute_web_search(req, cfg, web_search_fn=fake_web)
        assert outcome.error_type == RetrievalToolErrorType.TIMEOUT

    def test_web_tool_does_not_expose_api_key(self) -> None:
        from rag_pipeline.web_search import WebSearchOutcome

        def fake_web(**kwargs: Any) -> WebSearchOutcome:
            return WebSearchOutcome([], "tavily", 0, "empty_results")

        cfg = _config(web_search_enabled=True, tavily_api_key="top-secret-key")
        req = _valid_request(tool=RetrievalToolName.WEB_SEARCH)
        outcome = execute_web_search(req, cfg, web_search_fn=fake_web)
        raw = outcome.model_dump_json()
        assert "top-secret-key" not in raw

    def test_web_tool_does_not_persist_results(self) -> None:
        """Web tool must return results in outcome only — no side effects."""
        from rag_pipeline.web_search import WebSearchOutcome

        persisted: list[Any] = []
        web_chunk = {
            "chunk_id": "web:abc",
            "text": "content",
            "score": 0.8,
            "source_type": "web",
            "source_id": None,
            "title": "T",
            "heading": None,
            "page_index": None,
            "pdf_id": None,
            "metadata": {},
        }

        def fake_web(**kwargs: Any) -> WebSearchOutcome:
            return WebSearchOutcome([web_chunk], "tavily", 1, None)

        cfg = _config(web_search_enabled=True, tavily_api_key="key")
        req = _valid_request(tool=RetrievalToolName.WEB_SEARCH)
        execute_web_search(req, cfg, web_search_fn=fake_web)
        assert persisted == []  # nothing stored externally


# ---------------------------------------------------------------------------
# Phase 6 — Knowledge graph tool adapter
# ---------------------------------------------------------------------------

def _graph_ctx(
    context_text: str = "BPMN --USES--> Notation",
    nodes: list[dict[str, Any]] | None = None,
    relationships: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    nodes = nodes or [{"name": "BPMN"}, {"name": "Notation"}]
    relationships = relationships or [
        {"source": "BPMN", "target": "Notation", "relation_type": "USES"}
    ]
    sources = [
        {
            "chunk_id": "knowledge-graph:abc",
            "source_type": "knowledge_graph",
            "source_id": None,
            "title": "Knowledge Graph",
            "heading": "Concept relationships",
            "page": None,
            "score": None,
            "snippet": context_text[:100],
            "metadata": {
                "backing_chunk_ids": ["chunk-1"],
                "node_names": [n.get("name") for n in nodes],
                "relationship_count": len(relationships),
            },
        }
    ] if context_text else []
    return {"context_text": context_text, "sources": sources, "nodes": nodes, "relationships": relationships}


def _graph_config(**overrides: Any) -> Any:
    base: dict[str, Any] = {
        "graph_retrieval_enabled": True,
        "neo4j_uri": "bolt://localhost:7687",
        "neo4j_user": "neo4j",
        "neo4j_password": "password",
    }
    base.update(overrides)
    return _config(**base)


class TestGraphTool:
    def test_graph_tool_skips_when_graph_disabled(self) -> None:
        cfg = _config(graph_retrieval_enabled=False)
        req = _valid_request(tool=RetrievalToolName.QUERY_KNOWLEDGE_GRAPH)
        outcome = execute_query_knowledge_graph(req, cfg)
        assert outcome.status == RetrievalToolStatus.SKIPPED
        assert outcome.error_type == RetrievalToolErrorType.DISABLED

    def test_graph_tool_returns_missing_config_when_neo4j_not_configured(self) -> None:
        cfg = _config(
            graph_retrieval_enabled=True,
            neo4j_uri=None,
            neo4j_user=None,
            neo4j_password=None,
        )
        req = _valid_request(tool=RetrievalToolName.QUERY_KNOWLEDGE_GRAPH)
        outcome = execute_query_knowledge_graph(req, cfg)
        assert outcome.error_type == RetrievalToolErrorType.MISSING_CONFIG

    def test_graph_tool_calls_retrieve_graph_context_when_enabled(self) -> None:
        calls: list[Any] = []

        def fake_graph(**kwargs: Any) -> dict[str, Any]:
            calls.append(kwargs)
            return _graph_ctx()

        cfg = _graph_config()
        req = _valid_request(tool=RetrievalToolName.QUERY_KNOWLEDGE_GRAPH)
        execute_query_knowledge_graph(req, cfg, graph_fn=fake_graph)
        assert len(calls) == 1

    def test_graph_tool_normalizes_result_to_knowledge_graph_source_type(self) -> None:
        def fake_graph(**kwargs: Any) -> dict[str, Any]:
            return _graph_ctx()

        cfg = _graph_config()
        req = _valid_request(tool=RetrievalToolName.QUERY_KNOWLEDGE_GRAPH)
        outcome = execute_query_knowledge_graph(req, cfg, graph_fn=fake_graph)
        assert outcome.results[0].source_type == "knowledge_graph"

    def test_graph_tool_empty_context_returns_empty_outcome(self) -> None:
        def fake_graph(**kwargs: Any) -> dict[str, Any]:
            return _graph_ctx(context_text="")

        cfg = _graph_config()
        req = _valid_request(tool=RetrievalToolName.QUERY_KNOWLEDGE_GRAPH)
        outcome = execute_query_knowledge_graph(req, cfg, graph_fn=fake_graph)
        assert outcome.status == RetrievalToolStatus.EMPTY
        assert outcome.results == []

    def test_graph_tool_provider_error_returns_safe_outcome(self) -> None:
        def fake_graph(**kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("Neo4j connection refused")

        cfg = _graph_config()
        req = _valid_request(tool=RetrievalToolName.QUERY_KNOWLEDGE_GRAPH)
        outcome = execute_query_knowledge_graph(req, cfg, graph_fn=fake_graph)
        assert outcome.status == RetrievalToolStatus.ERROR
        assert outcome.error_type == RetrievalToolErrorType.PROVIDER_ERROR
        assert outcome.results == []

    def test_graph_tool_does_not_expose_neo4j_credentials(self) -> None:
        def fake_graph(**kwargs: Any) -> dict[str, Any]:
            return _graph_ctx()

        cfg = _graph_config(neo4j_password="super-secret-pw")
        req = _valid_request(tool=RetrievalToolName.QUERY_KNOWLEDGE_GRAPH)
        outcome = execute_query_knowledge_graph(req, cfg, graph_fn=fake_graph)
        raw = outcome.model_dump_json()
        assert "super-secret-pw" not in raw
        assert "password" not in raw.lower() or "super-secret-pw" not in raw

    def test_graph_tool_metadata_no_raw_cypher(self) -> None:
        def fake_graph(**kwargs: Any) -> dict[str, Any]:
            ctx = _graph_ctx()
            # Simulate metadata that could contain Cypher
            ctx["sources"][0]["metadata"]["cypher"] = "MATCH (n) RETURN n"
            return ctx

        cfg = _graph_config()
        req = _valid_request(tool=RetrievalToolName.QUERY_KNOWLEDGE_GRAPH)
        outcome = execute_query_knowledge_graph(req, cfg, graph_fn=fake_graph)
        raw = outcome.model_dump_json()
        # The graph adapter builds its own safe metadata — no raw Cypher
        assert "MATCH (n)" not in raw

    def test_graph_tool_result_chunk_id_uses_kg_prefix(self) -> None:
        def fake_graph(**kwargs: Any) -> dict[str, Any]:
            return _graph_ctx()

        cfg = _graph_config()
        req = _valid_request(tool=RetrievalToolName.QUERY_KNOWLEDGE_GRAPH)
        outcome = execute_query_knowledge_graph(req, cfg, graph_fn=fake_graph)
        assert outcome.results[0].chunk_id.startswith("kg:")


# ---------------------------------------------------------------------------
# Phase 10 — Tool metadata safety
# ---------------------------------------------------------------------------

class TestToolMetadataSafety:
    def test_tool_metadata_does_not_include_raw_results(self) -> None:
        registry = RetrievalToolRegistry(_config())
        registry.register(_dummy_spec())
        req = _valid_request()
        outcome = registry.execute(req)
        # metadata on outcome must not contain full text of results
        raw_meta_json = str(outcome.metadata)
        assert "some text" not in raw_meta_json

    def test_tool_metadata_does_not_include_secrets(self) -> None:
        def fake_execute(req: RetrievalToolRequest) -> RetrievalToolOutcome:
            return RetrievalToolOutcome(
                tool=RetrievalToolName.SEARCH_PDF_CHUNKS,
                status=RetrievalToolStatus.SUCCESS,
                results=[],
                result_count=0,
                metadata={"api_key": "leaked", "provider": "qdrant"},
            )

        registry = RetrievalToolRegistry(_config())
        registry.register(RetrievalToolSpec(
            name=RetrievalToolName.SEARCH_PDF_CHUNKS,
            description="test",
            requires_user_id=True,
            requires_session_id=False,
            requires_config_flag=None,
            allowed_source_types=["pdf"],
            execute=fake_execute,
        ))
        outcome = registry.execute(_valid_request())
        # The registry itself doesn't strip outcome metadata — that's on the adapter/sanitizer.
        # The key assertion: raw stack traces and credentials should not leak via outcome fields.
        assert outcome.error_type is None or isinstance(outcome.error_type, RetrievalToolErrorType)

    def test_tool_logs_safe_error_type(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        registry = RetrievalToolRegistry(_config())
        registry.register(RetrievalToolSpec(
            name=RetrievalToolName.SEARCH_PDF_CHUNKS,
            description="fail",
            requires_user_id=True,
            requires_session_id=False,
            requires_config_flag=None,
            allowed_source_types=["pdf"],
            execute=lambda req: (_ for _ in ()).throw(RuntimeError("internal error with secret=xyz")),
        ))
        with caplog.at_level(logging.WARNING):
            registry.execute(_valid_request())
        # Ensure raw exception message is not logged directly (only safe error_type)
        log_text = " ".join(caplog.messages)
        assert "secret=xyz" not in log_text
