from __future__ import annotations

import pytest

from rag_pipeline.config import WorkerConfig


def _base_env(monkeypatch) -> None:
    monkeypatch.setattr("rag_pipeline.config._load_dotenv", lambda: None)
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")


def test_agentic_retriever_config_defaults(monkeypatch) -> None:
    _base_env(monkeypatch)
    config = WorkerConfig.from_env()
    assert config.agentic_retriever_enabled is False
    assert config.agentic_retriever_mode == "controlled"
    assert config.agentic_retriever_quality_assessment_mode == "heuristic"
    assert config.agentic_retriever_refinement_mode == "heuristic"
    assert config.agentic_retriever_max_refinement_rounds == 1
    assert config.agentic_retriever_max_tool_calls == 8


def test_agentic_retriever_config_env_overrides(monkeypatch) -> None:
    _base_env(monkeypatch)
    monkeypatch.setenv("AGENTIC_RETRIEVER_ENABLED", "true")
    monkeypatch.setenv("AGENTIC_RETRIEVER_QUALITY_ASSESSMENT_MODE", "llm")
    monkeypatch.setenv("AGENTIC_RETRIEVER_REFINEMENT_MODE", "llm")
    monkeypatch.setenv("AGENTIC_RETRIEVER_MAX_REFINEMENT_ROUNDS", "2")
    monkeypatch.setenv("AGENTIC_RETRIEVER_MAX_TOOL_CALLS", "12")
    monkeypatch.setenv("AGENTIC_RETRIEVER_MIN_AVG_SCORE", "0.4")
    config = WorkerConfig.from_env()
    assert config.agentic_retriever_enabled is True
    assert config.agentic_retriever_quality_assessment_mode == "llm"
    assert config.agentic_retriever_refinement_mode == "llm"
    assert config.agentic_retriever_max_refinement_rounds == 2
    assert config.agentic_retriever_max_tool_calls == 12
    assert config.agentic_retriever_min_avg_score == 0.4


def test_agentic_retriever_enabled_true_parses_true(monkeypatch) -> None:
    _base_env(monkeypatch)
    for value in ["true", "True", "1", "yes", "on"]:
        monkeypatch.setenv("AGENTIC_RETRIEVER_ENABLED", value)
        assert WorkerConfig.from_env().agentic_retriever_enabled is True


def test_agentic_retriever_invalid_mode_rejected(monkeypatch) -> None:
    _base_env(monkeypatch)
    monkeypatch.setenv("AGENTIC_RETRIEVER_MODE", "autonomous")
    with pytest.raises(ValueError, match="AGENTIC_RETRIEVER_MODE"):
        WorkerConfig.from_env()


def test_agentic_retriever_invalid_quality_assessment_mode_rejected(monkeypatch) -> None:
    _base_env(monkeypatch)
    monkeypatch.setenv("AGENTIC_RETRIEVER_QUALITY_ASSESSMENT_MODE", "bad")
    with pytest.raises(ValueError, match="QUALITY_ASSESSMENT"):
        WorkerConfig.from_env()


def test_agentic_retriever_invalid_refinement_mode_rejected(monkeypatch) -> None:
    _base_env(monkeypatch)
    monkeypatch.setenv("AGENTIC_RETRIEVER_REFINEMENT_MODE", "bad")
    with pytest.raises(ValueError, match="REFINEMENT_MODE"):
        WorkerConfig.from_env()


def test_agentic_retriever_invalid_max_refinement_rounds_rejected(monkeypatch) -> None:
    _base_env(monkeypatch)
    monkeypatch.setenv("AGENTIC_RETRIEVER_MAX_REFINEMENT_ROUNDS", "5")
    with pytest.raises(ValueError, match="MAX_REFINEMENT_ROUNDS"):
        WorkerConfig.from_env()


def test_agentic_retriever_invalid_max_tool_calls_rejected(monkeypatch) -> None:
    _base_env(monkeypatch)
    monkeypatch.setenv("AGENTIC_RETRIEVER_MAX_TOOL_CALLS", "0")
    with pytest.raises(ValueError, match="MAX_TOOL_CALLS"):
        WorkerConfig.from_env()


def test_agentic_retriever_invalid_min_avg_score_rejected(monkeypatch) -> None:
    _base_env(monkeypatch)
    monkeypatch.setenv("AGENTIC_RETRIEVER_MIN_AVG_SCORE", "2")
    with pytest.raises(ValueError, match="MIN_AVG_SCORE"):
        WorkerConfig.from_env()
