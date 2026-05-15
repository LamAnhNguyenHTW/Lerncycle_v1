from __future__ import annotations

import pytest

import rag_pipeline.config as config_module
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
    monkeypatch.setattr("rag_pipeline.config._load_dotenv", lambda: None)
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
    monkeypatch.setenv("GRAPH_ENABLED", "false")
    monkeypatch.setenv("GRAPH_EXTRACTION_ENABLED", "false")
    monkeypatch.setenv("GRAPH_RETRIEVAL_ENABLED", "false")
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


def test_graph_config_defaults(monkeypatch) -> None:
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")
    for name in [
        "GRAPH_ENABLED",
        "GRAPH_EXTRACTION_ENABLED",
        "GRAPH_RETRIEVAL_ENABLED",
        "GRAPH_STORE_PROVIDER",
        "NEO4J_URI",
        "NEO4J_USER",
        "NEO4J_PASSWORD",
        "NEO4J_DATABASE",
        "GRAPH_MAX_NODES_PER_CHUNK",
        "GRAPH_MAX_EDGES_PER_CHUNK",
        "GRAPH_RETRIEVAL_TOP_K",
        "GRAPH_CONTEXT_MAX_CHARS",
        "GRAPH_SOURCE_TYPE",
    ]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GRAPH_ENABLED", "false")
    monkeypatch.setenv("GRAPH_EXTRACTION_ENABLED", "false")
    monkeypatch.setenv("GRAPH_RETRIEVAL_ENABLED", "false")

    config = WorkerConfig.from_env()

    assert config.graph_enabled is False
    assert config.graph_extraction_enabled is False
    assert config.graph_retrieval_enabled is False
    assert config.graph_store_provider == "neo4j"
    assert config.neo4j_database == "neo4j"
    assert config.graph_max_nodes_per_chunk == 12
    assert config.graph_max_edges_per_chunk == 20
    assert config.graph_retrieval_top_k == 8
    assert config.graph_context_max_chars == 6000
    assert config.graph_source_type == "knowledge_graph"


def test_graph_config_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")
    monkeypatch.setenv("GRAPH_ENABLED", "true")
    monkeypatch.setenv("GRAPH_EXTRACTION_ENABLED", "true")
    monkeypatch.setenv("GRAPH_RETRIEVAL_ENABLED", "true")
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "password")
    monkeypatch.setenv("NEO4J_DATABASE", "graph")
    monkeypatch.setenv("GRAPH_MAX_NODES_PER_CHUNK", "10")
    monkeypatch.setenv("GRAPH_MAX_EDGES_PER_CHUNK", "15")
    monkeypatch.setenv("GRAPH_RETRIEVAL_TOP_K", "6")
    monkeypatch.setenv("GRAPH_CONTEXT_MAX_CHARS", "3000")

    config = WorkerConfig.from_env()

    assert config.graph_enabled is True
    assert config.graph_extraction_enabled is True
    assert config.graph_retrieval_enabled is True
    assert config.neo4j_uri == "bolt://localhost:7687"
    assert config.neo4j_user == "neo4j"
    assert config.neo4j_password == "password"
    assert config.neo4j_database == "graph"
    assert config.graph_max_nodes_per_chunk == 10
    assert config.graph_max_edges_per_chunk == 15
    assert config.graph_retrieval_top_k == 6
    assert config.graph_context_max_chars == 3000


def test_graph_enabled_true_parses_true(monkeypatch) -> None:
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")
    monkeypatch.setenv("GRAPH_ENABLED", "true")
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "password")

    assert WorkerConfig.from_env().graph_enabled is True


def test_graph_requires_neo4j_credentials_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")
    monkeypatch.setenv("GRAPH_RETRIEVAL_ENABLED", "true")
    monkeypatch.setenv("NEO4J_URI", "")
    monkeypatch.setenv("NEO4J_USER", "")
    monkeypatch.setenv("NEO4J_PASSWORD", "")

    with pytest.raises(RuntimeError, match="NEO4J_URI"):
        WorkerConfig.from_env()


def test_graph_provider_rejects_non_neo4j(monkeypatch) -> None:
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")
    monkeypatch.setenv("GRAPH_STORE_PROVIDER", "other")

    with pytest.raises(ValueError, match="GRAPH_STORE_PROVIDER"):
        WorkerConfig.from_env()


def test_graph_invalid_top_k_rejected(monkeypatch) -> None:
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")
    monkeypatch.setenv("GRAPH_RETRIEVAL_TOP_K", "31")

    with pytest.raises(ValueError, match="GRAPH_RETRIEVAL_TOP_K"):
        WorkerConfig.from_env()


def test_graph_invalid_context_max_chars_rejected(monkeypatch) -> None:
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")
    monkeypatch.setenv("GRAPH_CONTEXT_MAX_CHARS", "999")

    with pytest.raises(ValueError, match="GRAPH_CONTEXT_MAX_CHARS"):
        WorkerConfig.from_env()


def test_learning_graph_config_defaults(monkeypatch) -> None:
    monkeypatch.setattr(config_module, "_load_dotenv", lambda: None)
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")
    for name in [
        "LEARNING_GRAPH_EXTRACTION_ENABLED",
        "LEARNING_GRAPH_MAX_CHUNKS_PER_GROUP",
        "LEARNING_GRAPH_MIN_CONFIDENCE",
        "LEARNING_GRAPH_MAX_TOPICS_PER_DOC",
        "LEARNING_GRAPH_MIN_CHUNK_COVERAGE",
    ]:
        monkeypatch.delenv(name, raising=False)

    config = WorkerConfig.from_env()

    assert config.learning_graph_extraction_enabled is False
    assert config.learning_graph_max_chunks_per_group == 8
    assert config.learning_graph_min_confidence == 0.5
    assert config.learning_graph_max_topics_per_doc == 30
    assert config.learning_graph_min_chunk_coverage == 0.35


def test_learning_graph_config_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")
    monkeypatch.setenv("LEARNING_GRAPH_EXTRACTION_ENABLED", "true")
    monkeypatch.setenv("LEARNING_GRAPH_MAX_CHUNKS_PER_GROUP", "12")
    monkeypatch.setenv("LEARNING_GRAPH_MIN_CONFIDENCE", "0.7")
    monkeypatch.setenv("LEARNING_GRAPH_MAX_TOPICS_PER_DOC", "40")
    monkeypatch.setenv("LEARNING_GRAPH_MIN_CHUNK_COVERAGE", "0.45")

    config = WorkerConfig.from_env()

    assert config.learning_graph_extraction_enabled is True
    assert config.learning_graph_max_chunks_per_group == 12
    assert config.learning_graph_min_confidence == 0.7
    assert config.learning_graph_max_topics_per_doc == 40
    assert config.learning_graph_min_chunk_coverage == 0.45


def test_learning_graph_invalid_bounds_rejected(monkeypatch) -> None:
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")
    monkeypatch.setenv("LEARNING_GRAPH_MAX_CHUNKS_PER_GROUP", "0")
    with pytest.raises(ValueError, match="LEARNING_GRAPH_MAX_CHUNKS_PER_GROUP"):
        WorkerConfig.from_env()
    monkeypatch.setenv("LEARNING_GRAPH_MAX_CHUNKS_PER_GROUP", "8")
    monkeypatch.setenv("LEARNING_GRAPH_MIN_CONFIDENCE", "1.1")
    with pytest.raises(ValueError, match="LEARNING_GRAPH_MIN_CONFIDENCE"):
        WorkerConfig.from_env()
    monkeypatch.setenv("LEARNING_GRAPH_MIN_CONFIDENCE", "0.5")
    monkeypatch.setenv("LEARNING_GRAPH_MAX_TOPICS_PER_DOC", "0")
    with pytest.raises(ValueError, match="LEARNING_GRAPH_MAX_TOPICS_PER_DOC"):
        WorkerConfig.from_env()
    monkeypatch.setenv("LEARNING_GRAPH_MAX_TOPICS_PER_DOC", "30")
    monkeypatch.setenv("LEARNING_GRAPH_MIN_CHUNK_COVERAGE", "-0.1")
    with pytest.raises(ValueError, match="LEARNING_GRAPH_MIN_CHUNK_COVERAGE"):
        WorkerConfig.from_env()


def test_web_search_config_defaults(monkeypatch) -> None:
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")
    for name in [
        "WEB_SEARCH_PROVIDER",
        "WEB_SEARCH_TOP_K",
        "WEB_SEARCH_TIMEOUT_SECONDS",
        "WEB_SEARCH_MAX_QUERY_CHARS",
        "WEB_SEARCH_SOURCE_TYPE",
    ]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("WEB_SEARCH_ENABLED", "false")
    monkeypatch.setenv("TAVILY_API_KEY", "")

    config = WorkerConfig.from_env()

    assert config.web_search_enabled is False
    assert config.web_search_provider == "tavily"
    assert config.web_search_top_k == 5
    assert config.web_search_timeout_seconds == 15
    assert config.web_search_max_query_chars == 300
    assert config.web_search_source_type == "web"
    assert config.tavily_api_key is None


def test_web_search_config_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")
    monkeypatch.setenv("WEB_SEARCH_ENABLED", "true")
    monkeypatch.setenv("WEB_SEARCH_TOP_K", "7")
    monkeypatch.setenv("WEB_SEARCH_TIMEOUT_SECONDS", "20")
    monkeypatch.setenv("WEB_SEARCH_MAX_QUERY_CHARS", "500")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-key")

    config = WorkerConfig.from_env()

    assert config.web_search_enabled is True
    assert config.web_search_top_k == 7
    assert config.web_search_timeout_seconds == 20
    assert config.web_search_max_query_chars == 500
    assert config.tavily_api_key == "tvly-key"


def test_web_search_invalid_provider_rejected(monkeypatch) -> None:
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "other")

    with pytest.raises(ValueError, match="WEB_SEARCH_PROVIDER"):
        WorkerConfig.from_env()


def test_web_search_invalid_top_k_rejected(monkeypatch) -> None:
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")
    monkeypatch.setenv("WEB_SEARCH_TOP_K", "11")

    with pytest.raises(ValueError, match="WEB_SEARCH_TOP_K"):
        WorkerConfig.from_env()


def test_web_search_invalid_timeout_rejected(monkeypatch) -> None:
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")
    monkeypatch.setenv("WEB_SEARCH_TIMEOUT_SECONDS", "2")

    with pytest.raises(ValueError, match="WEB_SEARCH_TIMEOUT_SECONDS"):
        WorkerConfig.from_env()


def test_web_search_invalid_max_query_chars_rejected(monkeypatch) -> None:
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")
    monkeypatch.setenv("WEB_SEARCH_MAX_QUERY_CHARS", "49")

    with pytest.raises(ValueError, match="WEB_SEARCH_MAX_QUERY_CHARS"):
        WorkerConfig.from_env()


def test_web_search_invalid_source_type_rejected(monkeypatch) -> None:
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")
    monkeypatch.setenv("WEB_SEARCH_SOURCE_TYPE", "internet")

    with pytest.raises(ValueError, match="WEB_SEARCH_SOURCE_TYPE"):
        WorkerConfig.from_env()


def test_missing_tavily_api_key_does_not_crash_config_loading(monkeypatch) -> None:
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")
    monkeypatch.setenv("WEB_SEARCH_ENABLED", "true")
    monkeypatch.setenv("TAVILY_API_KEY", "")

    assert WorkerConfig.from_env().tavily_api_key is None


def test_intent_classifier_config_defaults(monkeypatch) -> None:
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")
    monkeypatch.setenv("INTENT_CLASSIFIER_ENABLED", "false")
    for name in [
        "INTENT_CLASSIFIER_PROVIDER",
        "INTENT_CLASSIFIER_MODEL",
        "INTENT_CLASSIFIER_TIMEOUT_SECONDS",
        "INTENT_CLASSIFIER_MAX_RECENT_MESSAGES",
        "INTENT_CLASSIFIER_MAX_MESSAGE_CHARS",
        "INTENT_CLASSIFIER_FALLBACK_ENABLED",
    ]:
        monkeypatch.delenv(name, raising=False)

    config = WorkerConfig.from_env()

    assert config.intent_classifier_enabled is False
    assert config.intent_classifier_provider == "openai"
    assert config.intent_classifier_model == "gpt-4.1-mini"
    assert config.intent_classifier_timeout_seconds == 10
    assert config.intent_classifier_max_recent_messages == 4
    assert config.intent_classifier_max_message_chars == 1000
    assert config.intent_classifier_fallback_enabled is True


def test_intent_classifier_config_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")
    monkeypatch.setenv("INTENT_CLASSIFIER_ENABLED", "true")
    monkeypatch.setenv("INTENT_CLASSIFIER_MODEL", "gpt-test")
    monkeypatch.setenv("INTENT_CLASSIFIER_TIMEOUT_SECONDS", "12")
    monkeypatch.setenv("INTENT_CLASSIFIER_MAX_RECENT_MESSAGES", "2")
    monkeypatch.setenv("INTENT_CLASSIFIER_MAX_MESSAGE_CHARS", "500")
    monkeypatch.setenv("INTENT_CLASSIFIER_FALLBACK_ENABLED", "false")

    config = WorkerConfig.from_env()

    assert config.intent_classifier_enabled is True
    assert config.intent_classifier_model == "gpt-test"
    assert config.intent_classifier_timeout_seconds == 12
    assert config.intent_classifier_max_recent_messages == 2
    assert config.intent_classifier_max_message_chars == 500
    assert config.intent_classifier_fallback_enabled is False


def test_intent_classifier_invalid_provider_rejected(monkeypatch) -> None:
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")
    monkeypatch.setenv("INTENT_CLASSIFIER_PROVIDER", "other")

    with pytest.raises(ValueError, match="INTENT_CLASSIFIER_PROVIDER"):
        WorkerConfig.from_env()


def test_intent_classifier_invalid_bounds_rejected(monkeypatch) -> None:
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")
    monkeypatch.setenv("INTENT_CLASSIFIER_TIMEOUT_SECONDS", "2")
    with pytest.raises(ValueError, match="INTENT_CLASSIFIER_TIMEOUT_SECONDS"):
        WorkerConfig.from_env()
    monkeypatch.setenv("INTENT_CLASSIFIER_TIMEOUT_SECONDS", "10")
    monkeypatch.setenv("INTENT_CLASSIFIER_MAX_RECENT_MESSAGES", "11")
    with pytest.raises(ValueError, match="INTENT_CLASSIFIER_MAX_RECENT_MESSAGES"):
        WorkerConfig.from_env()
    monkeypatch.setenv("INTENT_CLASSIFIER_MAX_RECENT_MESSAGES", "4")
    monkeypatch.setenv("INTENT_CLASSIFIER_MAX_MESSAGE_CHARS", "199")
    with pytest.raises(ValueError, match="INTENT_CLASSIFIER_MAX_MESSAGE_CHARS"):
        WorkerConfig.from_env()


def test_retrieval_planner_config_defaults(monkeypatch) -> None:
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")
    monkeypatch.setenv("RETRIEVAL_PLANNER_ENABLED", "false")

    config = WorkerConfig.from_env()

    assert config.retrieval_planner_enabled is False
    assert config.retrieval_planner_default_top_k == 6
    assert config.retrieval_planner_memory_top_k == 3
    assert config.retrieval_planner_include_disabled_steps is True


def test_retrieval_planner_config_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")
    monkeypatch.setenv("RETRIEVAL_PLANNER_ENABLED", "true")
    monkeypatch.setenv("RETRIEVAL_PLANNER_DEFAULT_TOP_K", "7")
    monkeypatch.setenv("RETRIEVAL_PLANNER_MEMORY_TOP_K", "4")
    monkeypatch.setenv("RETRIEVAL_PLANNER_MAX_STEPS", "3")
    monkeypatch.setenv("RETRIEVAL_PLANNER_INCLUDE_DISABLED_STEPS", "false")

    config = WorkerConfig.from_env()

    assert config.retrieval_planner_enabled is True
    assert config.retrieval_planner_default_top_k == 7
    assert config.retrieval_planner_memory_top_k == 4
    assert config.retrieval_planner_max_steps == 3
    assert config.retrieval_planner_include_disabled_steps is False


def test_retrieval_planner_invalid_top_k_rejected(monkeypatch) -> None:
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")
    monkeypatch.setenv("RETRIEVAL_PLANNER_WEB_TOP_K", "0")

    with pytest.raises(ValueError, match="RETRIEVAL_PLANNER_WEB_TOP_K"):
        WorkerConfig.from_env()


def test_retrieval_planner_invalid_max_steps_rejected(monkeypatch) -> None:
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")
    monkeypatch.setenv("RETRIEVAL_PLANNER_MAX_STEPS", "11")

    with pytest.raises(ValueError, match="RETRIEVAL_PLANNER_MAX_STEPS"):
        WorkerConfig.from_env()
