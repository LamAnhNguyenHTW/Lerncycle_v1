from __future__ import annotations

import pytest

from rag_pipeline.config import WorkerConfig


def test_config_defaults(monkeypatch) -> None:
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")
    monkeypatch.delenv("QDRANT_COLLECTION", raising=False)
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_BATCH_SIZE", raising=False)
    monkeypatch.delenv("SPARSE_PROVIDER", raising=False)
    monkeypatch.delenv("SPARSE_MODEL", raising=False)
    monkeypatch.delenv("SPARSE_VECTOR_NAME", raising=False)
    monkeypatch.delenv("SPARSE_ENABLED", raising=False)
    monkeypatch.delenv("HYBRID_FUSION", raising=False)
    monkeypatch.delenv("HYBRID_PREFETCH_LIMIT", raising=False)
    monkeypatch.delenv("HYBRID_TOP_K", raising=False)

    config = WorkerConfig.from_env()

    assert config.qdrant_collection == "learncycle_chunks"
    assert config.embedding_provider == "openai"
    assert config.embedding_model == "text-embedding-3-small"
    assert config.embedding_batch_size == 100
    assert config.sparse_provider == "fastembed"
    assert config.sparse_model == "Qdrant/bm25"
    assert config.sparse_vector_name == "sparse"
    assert config.sparse_enabled is True
    assert config.hybrid_fusion == "rrf"
    assert config.hybrid_prefetch_limit == 30
    assert config.hybrid_top_k == 10


def test_config_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")
    monkeypatch.setenv("QDRANT_COLLECTION", "custom_chunks")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv("EMBEDDING_MODEL", "model-x")
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "12")
    monkeypatch.setenv("SPARSE_PROVIDER", "custom")
    monkeypatch.setenv("SPARSE_MODEL", "custom-sparse")
    monkeypatch.setenv("SPARSE_VECTOR_NAME", "custom_sparse")
    monkeypatch.setenv("SPARSE_ENABLED", "false")
    monkeypatch.setenv("HYBRID_FUSION", "local_rrf")
    monkeypatch.setenv("HYBRID_PREFETCH_LIMIT", "42")
    monkeypatch.setenv("HYBRID_TOP_K", "7")

    config = WorkerConfig.from_env()

    assert config.qdrant_collection == "custom_chunks"
    assert config.embedding_provider == "gemini"
    assert config.embedding_model == "model-x"
    assert config.embedding_batch_size == 12
    assert config.sparse_provider == "custom"
    assert config.sparse_model == "custom-sparse"
    assert config.sparse_vector_name == "custom_sparse"
    assert config.sparse_enabled is False
    assert config.hybrid_fusion == "local_rrf"
    assert config.hybrid_prefetch_limit == 42
    assert config.hybrid_top_k == 7


def test_sparse_enabled_false_values(monkeypatch) -> None:
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")

    for value in ["false", "False", "0", "no", "off"]:
        monkeypatch.setenv("SPARSE_ENABLED", value)
        assert WorkerConfig.from_env().sparse_enabled is False


def test_reranking_config_defaults(monkeypatch) -> None:
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")
    monkeypatch.delenv("RERANKING_ENABLED", raising=False)
    monkeypatch.setenv("RERANKING_ENABLED", "false")
    monkeypatch.delenv("RERANKING_PROVIDER", raising=False)
    monkeypatch.setenv("RERANKING_PROVIDER", "fastembed")
    monkeypatch.delenv("RERANKING_MODEL", raising=False)
    monkeypatch.setenv("RERANKING_MODEL", "jinaai/jina-reranker-v2-base-multilingual")
    monkeypatch.delenv("RERANKING_CANDIDATE_K", raising=False)
    monkeypatch.setenv("RERANKING_CANDIDATE_K", "30")
    monkeypatch.delenv("RERANKING_TOP_K", raising=False)
    monkeypatch.setenv("RERANKING_TOP_K", "8")

    config = WorkerConfig.from_env()

    assert config.reranking_enabled is False
    assert config.reranking_provider == "fastembed"
    assert config.reranking_model == "jinaai/jina-reranker-v2-base-multilingual"
    assert config.reranking_candidate_k == 30
    assert config.reranking_top_k == 8


def test_reranking_config_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")
    monkeypatch.setenv("RERANKING_ENABLED", "true")
    monkeypatch.setenv("RERANKING_PROVIDER", "noop")
    monkeypatch.setenv("RERANKING_MODEL", "custom-reranker")
    monkeypatch.setenv("RERANKING_CANDIDATE_K", "40")
    monkeypatch.setenv("RERANKING_TOP_K", "12")

    config = WorkerConfig.from_env()

    assert config.reranking_enabled is True
    assert config.reranking_provider == "noop"
    assert config.reranking_model == "custom-reranker"
    assert config.reranking_candidate_k == 40
    assert config.reranking_top_k == 12


def test_reranking_config_allows_llm_provider(monkeypatch) -> None:
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")
    monkeypatch.setenv("RERANKING_PROVIDER", "llm")
    monkeypatch.setenv("RERANKING_CANDIDATE_K", "20")

    config = WorkerConfig.from_env()

    assert config.reranking_provider == "llm"


def test_llm_reranking_candidate_k_above_30_rejected(monkeypatch) -> None:
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")
    monkeypatch.setenv("RERANKING_PROVIDER", "llm")
    monkeypatch.setenv("RERANKING_CANDIDATE_K", "31")

    with pytest.raises(ValueError, match="RERANKING_CANDIDATE_K.*30"):
        WorkerConfig.from_env()


def test_reranking_config_parses_enabled_false(monkeypatch) -> None:
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")
    monkeypatch.setenv("RERANKING_ENABLED", "false")

    assert WorkerConfig.from_env().reranking_enabled is False


def test_reranking_config_parses_enabled_true(monkeypatch) -> None:
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")
    monkeypatch.setenv("RERANKING_ENABLED", "true")

    assert WorkerConfig.from_env().reranking_enabled is True


def test_chat_memory_config_defaults(monkeypatch) -> None:
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")
    for name in [
        "CHAT_MEMORY_ENABLED",
        "CHAT_MEMORY_SUMMARY_THRESHOLD",
        "CHAT_MEMORY_SUMMARY_INTERVAL",
        "CHAT_MEMORY_KEEP_RECENT",
        "CHAT_MEMORY_MAX_SUMMARY_CHARS",
        "CHAT_MEMORY_RETRIEVAL_ENABLED",
        "CHAT_MEMORY_DEFAULT_INCLUDED",
        "CHAT_MEMORY_TOP_K",
        "CHAT_MEMORY_SOURCE_TYPE",
    ]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CHAT_MEMORY_ENABLED", "false")
    monkeypatch.setenv("CHAT_MEMORY_RETRIEVAL_ENABLED", "false")
    monkeypatch.setenv("CHAT_MEMORY_DEFAULT_INCLUDED", "false")

    config = WorkerConfig.from_env()

    assert config.chat_memory_enabled is False
    assert config.chat_memory_summary_threshold == 8
    assert config.chat_memory_summary_interval == 4
    assert config.chat_memory_keep_recent == 4
    assert config.chat_memory_max_summary_chars == 2500
    assert config.chat_memory_retrieval_enabled is False
    assert config.chat_memory_default_included is False
    assert config.chat_memory_top_k == 2
    assert config.chat_memory_source_type == "chat_memory"


def test_chat_memory_config_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")
    monkeypatch.setenv("CHAT_MEMORY_ENABLED", "true")
    monkeypatch.setenv("CHAT_MEMORY_SUMMARY_THRESHOLD", "10")
    monkeypatch.setenv("CHAT_MEMORY_SUMMARY_INTERVAL", "3")
    monkeypatch.setenv("CHAT_MEMORY_KEEP_RECENT", "2")
    monkeypatch.setenv("CHAT_MEMORY_MAX_SUMMARY_CHARS", "3000")
    monkeypatch.setenv("CHAT_MEMORY_RETRIEVAL_ENABLED", "true")
    monkeypatch.setenv("CHAT_MEMORY_DEFAULT_INCLUDED", "true")
    monkeypatch.setenv("CHAT_MEMORY_TOP_K", "3")
    monkeypatch.setenv("CHAT_MEMORY_SOURCE_TYPE", "chat_memory")

    config = WorkerConfig.from_env()

    assert config.chat_memory_enabled is True
    assert config.chat_memory_summary_threshold == 10
    assert config.chat_memory_summary_interval == 3
    assert config.chat_memory_keep_recent == 2
    assert config.chat_memory_max_summary_chars == 3000
    assert config.chat_memory_retrieval_enabled is True
    assert config.chat_memory_default_included is True
    assert config.chat_memory_top_k == 3


def test_chat_memory_invalid_keep_recent_rejected(monkeypatch) -> None:
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")
    monkeypatch.setenv("CHAT_MEMORY_SUMMARY_THRESHOLD", "4")
    monkeypatch.setenv("CHAT_MEMORY_KEEP_RECENT", "4")

    with pytest.raises(ValueError, match="KEEP_RECENT"):
        WorkerConfig.from_env()


def test_chat_memory_invalid_summary_threshold_rejected(monkeypatch) -> None:
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")
    monkeypatch.setenv("CHAT_MEMORY_SUMMARY_THRESHOLD", "1")

    with pytest.raises(ValueError, match="SUMMARY_THRESHOLD"):
        WorkerConfig.from_env()


def test_chat_memory_invalid_top_k_rejected(monkeypatch) -> None:
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")
    monkeypatch.setenv("CHAT_MEMORY_TOP_K", "11")

    with pytest.raises(ValueError, match="CHAT_MEMORY_TOP_K"):
        WorkerConfig.from_env()
