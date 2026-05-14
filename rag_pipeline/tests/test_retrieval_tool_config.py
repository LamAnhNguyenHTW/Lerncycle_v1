"""Tests for retrieval tool registry configuration fields (Phase 1)."""

from __future__ import annotations

import pytest

from rag_pipeline.config import WorkerConfig


def _base_env(monkeypatch) -> None:
    monkeypatch.setattr("rag_pipeline.config._load_dotenv", lambda: None)
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")


def test_retrieval_tool_registry_config_defaults(monkeypatch) -> None:
    _base_env(monkeypatch)
    monkeypatch.delenv("RETRIEVAL_TOOL_REGISTRY_ENABLED", raising=False)
    monkeypatch.delenv("RETRIEVAL_TOOL_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("RETRIEVAL_TOOL_MAX_RESULTS_PER_TOOL", raising=False)
    monkeypatch.delenv("RETRIEVAL_TOOL_ALLOWED_TOOLS", raising=False)

    config = WorkerConfig.from_env()

    assert config.retrieval_tool_registry_enabled is False
    assert config.retrieval_tool_timeout_seconds == 20
    assert config.retrieval_tool_max_results_per_tool == 20
    assert config.retrieval_tool_allowed_tools is None


def test_retrieval_tool_registry_config_env_overrides(monkeypatch) -> None:
    _base_env(monkeypatch)
    monkeypatch.setenv("RETRIEVAL_TOOL_REGISTRY_ENABLED", "true")
    monkeypatch.setenv("RETRIEVAL_TOOL_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("RETRIEVAL_TOOL_MAX_RESULTS_PER_TOOL", "10")
    monkeypatch.setenv("RETRIEVAL_TOOL_ALLOWED_TOOLS", "search_pdf_chunks,web_search")

    config = WorkerConfig.from_env()

    assert config.retrieval_tool_registry_enabled is True
    assert config.retrieval_tool_timeout_seconds == 30
    assert config.retrieval_tool_max_results_per_tool == 10
    assert config.retrieval_tool_allowed_tools == ["search_pdf_chunks", "web_search"]


def test_retrieval_tool_registry_enabled_true_parses_true(monkeypatch) -> None:
    _base_env(monkeypatch)
    for value in ["true", "True", "1", "yes", "on"]:
        monkeypatch.setenv("RETRIEVAL_TOOL_REGISTRY_ENABLED", value)
        assert WorkerConfig.from_env().retrieval_tool_registry_enabled is True


def test_retrieval_tool_invalid_timeout_rejected(monkeypatch) -> None:
    _base_env(monkeypatch)
    monkeypatch.setenv("RETRIEVAL_TOOL_TIMEOUT_SECONDS", "200")
    with pytest.raises(ValueError, match="RETRIEVAL_TOOL_TIMEOUT_SECONDS"):
        WorkerConfig.from_env()


def test_retrieval_tool_timeout_too_low_rejected(monkeypatch) -> None:
    _base_env(monkeypatch)
    monkeypatch.setenv("RETRIEVAL_TOOL_TIMEOUT_SECONDS", "1")
    with pytest.raises(ValueError, match="RETRIEVAL_TOOL_TIMEOUT_SECONDS"):
        WorkerConfig.from_env()


def test_retrieval_tool_invalid_max_results_rejected(monkeypatch) -> None:
    _base_env(monkeypatch)
    monkeypatch.setenv("RETRIEVAL_TOOL_MAX_RESULTS_PER_TOOL", "100")
    with pytest.raises(ValueError, match="RETRIEVAL_TOOL_MAX_RESULTS_PER_TOOL"):
        WorkerConfig.from_env()


def test_retrieval_tool_invalid_allowed_tool_rejected(monkeypatch) -> None:
    _base_env(monkeypatch)
    monkeypatch.setenv("RETRIEVAL_TOOL_ALLOWED_TOOLS", "search_pdf_chunks,evil_tool")
    with pytest.raises(ValueError, match="unknown tools"):
        WorkerConfig.from_env()


def test_retrieval_tool_allowed_tools_none_when_unset(monkeypatch) -> None:
    _base_env(monkeypatch)
    monkeypatch.delenv("RETRIEVAL_TOOL_ALLOWED_TOOLS", raising=False)
    config = WorkerConfig.from_env()
    assert config.retrieval_tool_allowed_tools is None


def test_retrieval_tool_allowed_tools_all_valid(monkeypatch) -> None:
    _base_env(monkeypatch)
    monkeypatch.setenv(
        "RETRIEVAL_TOOL_ALLOWED_TOOLS",
        "search_pdf_chunks,search_notes,search_annotations,search_chat_memory,web_search,query_knowledge_graph",
    )
    config = WorkerConfig.from_env()
    assert len(config.retrieval_tool_allowed_tools) == 6  # type: ignore[arg-type]
