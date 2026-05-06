from __future__ import annotations

from rag_pipeline.config import WorkerConfig


def test_config_defaults(monkeypatch) -> None:
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")
    monkeypatch.delenv("QDRANT_COLLECTION", raising=False)
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_BATCH_SIZE", raising=False)

    config = WorkerConfig.from_env()

    assert config.qdrant_collection == "learncycle_chunks"
    assert config.embedding_provider == "openai"
    assert config.embedding_model == "text-embedding-3-small"
    assert config.embedding_batch_size == 100


def test_config_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")
    monkeypatch.setenv("QDRANT_COLLECTION", "custom_chunks")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv("EMBEDDING_MODEL", "model-x")
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "12")

    config = WorkerConfig.from_env()

    assert config.qdrant_collection == "custom_chunks"
    assert config.embedding_provider == "gemini"
    assert config.embedding_model == "model-x"
    assert config.embedding_batch_size == 12
