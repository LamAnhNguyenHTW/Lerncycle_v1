"""Environment configuration for the RAG worker."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class WorkerConfig:
    """Runtime configuration loaded from environment variables."""

    supabase_url: str
    supabase_service_role_key: str
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    openai_api_key: str | None = None
    gemini_api_key: str | None = None
    gemini_output_dimensionality: int | None = None
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_collection: str = "learncycle_chunks"
    embedding_batch_size: int = 100
    sparse_provider: str = "fastembed"
    sparse_model: str = "Qdrant/bm25"
    sparse_vector_name: str = "sparse"
    sparse_enabled: bool = True
    hybrid_fusion: str = "rrf"
    hybrid_prefetch_limit: int = 30
    hybrid_top_k: int = 10
    reranking_enabled: bool = False
    reranking_provider: str = "fastembed"
    reranking_model: str = "jinaai/jina-reranker-v2-base-multilingual"
    reranking_candidate_k: int = 30
    reranking_top_k: int = 8
    chat_memory_enabled: bool = False
    chat_memory_summary_threshold: int = 8
    chat_memory_summary_interval: int = 4
    chat_memory_keep_recent: int = 4
    chat_memory_max_summary_chars: int = 2500
    chat_memory_retrieval_enabled: bool = False
    chat_memory_default_included: bool = False
    chat_memory_top_k: int = 2
    chat_memory_source_type: str = "chat_memory"
    graph_enabled: bool = False
    graph_extraction_enabled: bool = False
    graph_retrieval_enabled: bool = False
    graph_store_provider: str = "neo4j"
    neo4j_uri: str | None = None
    neo4j_user: str | None = None
    neo4j_password: str | None = None
    neo4j_database: str = "neo4j"
    graph_max_nodes_per_chunk: int = 12
    graph_max_edges_per_chunk: int = 20
    graph_extraction_concurrency: int = 8
    graph_retrieval_top_k: int = 8
    graph_context_max_chars: int = 6000
    graph_source_type: str = "knowledge_graph"
    web_search_enabled: bool = False
    web_search_provider: str = "tavily"
    web_search_top_k: int = 5
    web_search_timeout_seconds: int = 15
    web_search_max_query_chars: int = 300
    web_search_max_context_sources: int = 5
    web_search_max_chars_per_source: int = 1000
    web_search_max_total_context_chars: int = 4000
    web_search_source_type: str = "web"
    tavily_api_key: str | None = None
    intent_classifier_enabled: bool = False
    intent_classifier_provider: str = "openai"
    intent_classifier_model: str = "gpt-4.1-mini"
    intent_classifier_timeout_seconds: int = 10
    intent_classifier_max_recent_messages: int = 4
    intent_classifier_max_message_chars: int = 1000
    intent_classifier_fallback_enabled: bool = True
    retrieval_planner_enabled: bool = False
    retrieval_planner_default_top_k: int = 6
    retrieval_planner_pdf_top_k: int = 6
    retrieval_planner_notes_top_k: int = 4
    retrieval_planner_annotations_top_k: int = 4
    retrieval_planner_memory_top_k: int = 3
    retrieval_planner_web_top_k: int = 5
    retrieval_planner_max_steps: int = 5
    retrieval_planner_include_disabled_steps: bool = True
    chunking_strategy: str = "docling_hybrid_semantic_refinement"
    chunking_version: str = "v1"

    @classmethod
    def from_env(cls) -> "WorkerConfig":
        """Create a config object from environment variables.

        Returns:
            WorkerConfig with required Supabase/OpenAI settings and optional
            Qdrant settings.

        Raises:
            RuntimeError: If a required environment variable is missing.
        """
        _load_dotenv()

        required = {
            "SUPABASE_URL": os.getenv("SUPABASE_URL"),
            "SUPABASE_SERVICE_ROLE_KEY": os.getenv(
                "SUPABASE_SERVICE_ROLE_KEY"
            ),
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise RuntimeError(
                "Missing required environment variables: "
                + ", ".join(missing)
            )

        valid_reranking_providers = {"fastembed", "llm", "noop"}
        reranking_provider = os.getenv("RERANKING_PROVIDER", "fastembed")
        if reranking_provider not in valid_reranking_providers:
            raise ValueError(
                "RERANKING_PROVIDER must be one of: fastembed, llm, noop"
            )
        reranking_candidate_k = _optional_int(os.getenv("RERANKING_CANDIDATE_K")) or 30
        reranking_top_k = _optional_int(os.getenv("RERANKING_TOP_K")) or 8
        if not 1 <= reranking_candidate_k <= 50:
            raise ValueError("RERANKING_CANDIDATE_K must be between 1 and 50")
        if reranking_provider == "llm" and reranking_candidate_k > 30:
            raise ValueError(
                "RERANKING_CANDIDATE_K must be <= 30 when RERANKING_PROVIDER=llm"
            )
        if not 1 <= reranking_top_k <= 20:
            raise ValueError("RERANKING_TOP_K must be between 1 and 20")
        chat_memory_summary_threshold = (
            _optional_int(os.getenv("CHAT_MEMORY_SUMMARY_THRESHOLD")) or 8
        )
        chat_memory_summary_interval = (
            _optional_int(os.getenv("CHAT_MEMORY_SUMMARY_INTERVAL")) or 4
        )
        chat_memory_keep_recent = (
            _optional_int(os.getenv("CHAT_MEMORY_KEEP_RECENT")) or 4
        )
        chat_memory_max_summary_chars = (
            _optional_int(os.getenv("CHAT_MEMORY_MAX_SUMMARY_CHARS")) or 2500
        )
        chat_memory_top_k = _optional_int(os.getenv("CHAT_MEMORY_TOP_K")) or 2
        chat_memory_source_type = os.getenv(
            "CHAT_MEMORY_SOURCE_TYPE",
            "chat_memory",
        )
        if chat_memory_summary_threshold < 2:
            raise ValueError("CHAT_MEMORY_SUMMARY_THRESHOLD must be >= 2")
        if chat_memory_summary_interval < 1:
            raise ValueError("CHAT_MEMORY_SUMMARY_INTERVAL must be >= 1")
        if chat_memory_keep_recent < 1:
            raise ValueError("CHAT_MEMORY_KEEP_RECENT must be >= 1")
        if chat_memory_keep_recent >= chat_memory_summary_threshold:
            raise ValueError(
                "CHAT_MEMORY_KEEP_RECENT must be less than CHAT_MEMORY_SUMMARY_THRESHOLD"
            )
        if not 500 <= chat_memory_max_summary_chars <= 10000:
            raise ValueError(
                "CHAT_MEMORY_MAX_SUMMARY_CHARS must be between 500 and 10000"
            )
        if not 1 <= chat_memory_top_k <= 10:
            raise ValueError("CHAT_MEMORY_TOP_K must be between 1 and 10")
        if chat_memory_source_type != "chat_memory":
            raise ValueError("CHAT_MEMORY_SOURCE_TYPE must be chat_memory")
        graph_enabled = _optional_bool(os.getenv("GRAPH_ENABLED"), False)
        graph_extraction_enabled = _optional_bool(
            os.getenv("GRAPH_EXTRACTION_ENABLED"),
            False,
        )
        graph_retrieval_enabled = _optional_bool(
            os.getenv("GRAPH_RETRIEVAL_ENABLED"),
            False,
        )
        graph_store_provider = os.getenv("GRAPH_STORE_PROVIDER", "neo4j")
        neo4j_uri = os.getenv("NEO4J_URI")
        neo4j_user = os.getenv("NEO4J_USER")
        neo4j_password = os.getenv("NEO4J_PASSWORD")
        graph_max_nodes_per_chunk = (
            _optional_int(os.getenv("GRAPH_MAX_NODES_PER_CHUNK")) or 12
        )
        graph_max_edges_per_chunk = (
            _optional_int(os.getenv("GRAPH_MAX_EDGES_PER_CHUNK")) or 20
        )
        graph_retrieval_top_k = (
            _optional_int(os.getenv("GRAPH_RETRIEVAL_TOP_K")) or 8
        )
        graph_context_max_chars = (
            _optional_int(os.getenv("GRAPH_CONTEXT_MAX_CHARS")) or 6000
        )
        graph_source_type = os.getenv("GRAPH_SOURCE_TYPE", "knowledge_graph")
        graph_extraction_concurrency = (
            _optional_int(os.getenv("GRAPH_EXTRACTION_CONCURRENCY")) or 8
        )
        if graph_store_provider != "neo4j":
            raise ValueError("GRAPH_STORE_PROVIDER must be neo4j")
        if not 1 <= graph_max_nodes_per_chunk <= 50:
            raise ValueError("GRAPH_MAX_NODES_PER_CHUNK must be between 1 and 50")
        if not 1 <= graph_max_edges_per_chunk <= 100:
            raise ValueError("GRAPH_MAX_EDGES_PER_CHUNK must be between 1 and 100")
        if not 1 <= graph_retrieval_top_k <= 30:
            raise ValueError("GRAPH_RETRIEVAL_TOP_K must be between 1 and 30")
        if not 1000 <= graph_context_max_chars <= 20000:
            raise ValueError("GRAPH_CONTEXT_MAX_CHARS must be between 1000 and 20000")
        if graph_source_type != "knowledge_graph":
            raise ValueError("GRAPH_SOURCE_TYPE must be knowledge_graph")
        if graph_enabled or graph_extraction_enabled or graph_retrieval_enabled:
            missing_graph = [
                name
                for name, value in {
                    "NEO4J_URI": neo4j_uri,
                    "NEO4J_USER": neo4j_user,
                    "NEO4J_PASSWORD": neo4j_password,
                }.items()
                if not value
            ]
            if missing_graph:
                raise RuntimeError(
                    "Missing Neo4j environment variables: "
                    + ", ".join(missing_graph)
                )
        web_search_enabled = _optional_bool(os.getenv("WEB_SEARCH_ENABLED"), False)
        web_search_provider = os.getenv("WEB_SEARCH_PROVIDER", "tavily")
        web_search_top_k = _optional_int(os.getenv("WEB_SEARCH_TOP_K")) or 5
        web_search_timeout_seconds = (
            _optional_int(os.getenv("WEB_SEARCH_TIMEOUT_SECONDS")) or 15
        )
        web_search_max_query_chars = (
            _optional_int(os.getenv("WEB_SEARCH_MAX_QUERY_CHARS")) or 300
        )
        web_search_max_context_sources = (
            _optional_int(os.getenv("WEB_SEARCH_MAX_CONTEXT_SOURCES")) or 5
        )
        web_search_max_chars_per_source = (
            _optional_int(os.getenv("WEB_SEARCH_MAX_CHARS_PER_SOURCE")) or 1000
        )
        web_search_max_total_context_chars = (
            _optional_int(os.getenv("WEB_SEARCH_MAX_TOTAL_CONTEXT_CHARS")) or 4000
        )
        web_search_source_type = os.getenv("WEB_SEARCH_SOURCE_TYPE", "web")
        if web_search_provider != "tavily":
            raise ValueError("WEB_SEARCH_PROVIDER must be tavily")
        if not 1 <= web_search_top_k <= 10:
            raise ValueError("WEB_SEARCH_TOP_K must be between 1 and 10")
        if not 3 <= web_search_timeout_seconds <= 60:
            raise ValueError("WEB_SEARCH_TIMEOUT_SECONDS must be between 3 and 60")
        if not 50 <= web_search_max_query_chars <= 1000:
            raise ValueError("WEB_SEARCH_MAX_QUERY_CHARS must be between 50 and 1000")
        if not 1 <= web_search_max_context_sources <= 10:
            raise ValueError("WEB_SEARCH_MAX_CONTEXT_SOURCES must be between 1 and 10")
        if not 200 <= web_search_max_chars_per_source <= 3000:
            raise ValueError("WEB_SEARCH_MAX_CHARS_PER_SOURCE must be between 200 and 3000")
        if not 1000 <= web_search_max_total_context_chars <= 10000:
            raise ValueError("WEB_SEARCH_MAX_TOTAL_CONTEXT_CHARS must be between 1000 and 10000")
        if web_search_source_type != "web":
            raise ValueError("WEB_SEARCH_SOURCE_TYPE must be web")
        intent_classifier_provider = os.getenv("INTENT_CLASSIFIER_PROVIDER", "openai")
        intent_classifier_timeout_seconds = (
            _optional_int(os.getenv("INTENT_CLASSIFIER_TIMEOUT_SECONDS")) or 10
        )
        intent_classifier_max_recent_messages = (
            _optional_int(os.getenv("INTENT_CLASSIFIER_MAX_RECENT_MESSAGES")) or 4
        )
        intent_classifier_max_message_chars = (
            _optional_int(os.getenv("INTENT_CLASSIFIER_MAX_MESSAGE_CHARS")) or 1000
        )
        if intent_classifier_provider != "openai":
            raise ValueError("INTENT_CLASSIFIER_PROVIDER must be openai")
        if not 3 <= intent_classifier_timeout_seconds <= 60:
            raise ValueError("INTENT_CLASSIFIER_TIMEOUT_SECONDS must be between 3 and 60")
        if not 0 <= intent_classifier_max_recent_messages <= 10:
            raise ValueError("INTENT_CLASSIFIER_MAX_RECENT_MESSAGES must be between 0 and 10")
        if not 200 <= intent_classifier_max_message_chars <= 4000:
            raise ValueError("INTENT_CLASSIFIER_MAX_MESSAGE_CHARS must be between 200 and 4000")
        retrieval_planner_default_top_k = _int_or_default("RETRIEVAL_PLANNER_DEFAULT_TOP_K", 6)
        retrieval_planner_pdf_top_k = _int_or_default("RETRIEVAL_PLANNER_PDF_TOP_K", 6)
        retrieval_planner_notes_top_k = _int_or_default("RETRIEVAL_PLANNER_NOTES_TOP_K", 4)
        retrieval_planner_annotations_top_k = _int_or_default("RETRIEVAL_PLANNER_ANNOTATIONS_TOP_K", 4)
        retrieval_planner_memory_top_k = _int_or_default("RETRIEVAL_PLANNER_MEMORY_TOP_K", 3)
        retrieval_planner_web_top_k = _int_or_default("RETRIEVAL_PLANNER_WEB_TOP_K", 5)
        retrieval_planner_max_steps = _int_or_default("RETRIEVAL_PLANNER_MAX_STEPS", 5)
        for name, value in {
            "RETRIEVAL_PLANNER_DEFAULT_TOP_K": retrieval_planner_default_top_k,
            "RETRIEVAL_PLANNER_PDF_TOP_K": retrieval_planner_pdf_top_k,
            "RETRIEVAL_PLANNER_NOTES_TOP_K": retrieval_planner_notes_top_k,
            "RETRIEVAL_PLANNER_ANNOTATIONS_TOP_K": retrieval_planner_annotations_top_k,
            "RETRIEVAL_PLANNER_MEMORY_TOP_K": retrieval_planner_memory_top_k,
            "RETRIEVAL_PLANNER_WEB_TOP_K": retrieval_planner_web_top_k,
        }.items():
            if not 1 <= value <= 20:
                raise ValueError(f"{name} must be between 1 and 20")
        if not 1 <= retrieval_planner_max_steps <= 10:
            raise ValueError("RETRIEVAL_PLANNER_MAX_STEPS must be between 1 and 10")

        return cls(
            supabase_url=required["SUPABASE_URL"] or "",
            supabase_service_role_key=(
                required["SUPABASE_SERVICE_ROLE_KEY"] or ""
            ),
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "openai"),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL",
                "text-embedding-3-small",
            ),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            gemini_output_dimensionality=_optional_int(
                os.getenv("GEMINI_OUTPUT_DIMENSIONALITY")
            ),
            qdrant_url=os.getenv("QDRANT_URL"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY"),
            qdrant_collection=os.getenv(
                "QDRANT_COLLECTION",
                "learncycle_chunks",
            ),
            embedding_batch_size=_optional_int(
                os.getenv("EMBEDDING_BATCH_SIZE")
            )
            or 100,
            sparse_provider=os.getenv("SPARSE_PROVIDER", "fastembed"),
            sparse_model=os.getenv("SPARSE_MODEL", "Qdrant/bm25"),
            sparse_vector_name=os.getenv("SPARSE_VECTOR_NAME", "sparse"),
            sparse_enabled=_optional_bool(os.getenv("SPARSE_ENABLED"), True),
            hybrid_fusion=os.getenv("HYBRID_FUSION", "rrf"),
            hybrid_prefetch_limit=_optional_int(
                os.getenv("HYBRID_PREFETCH_LIMIT")
            )
            or 30,
            hybrid_top_k=_optional_int(os.getenv("HYBRID_TOP_K")) or 10,
            reranking_enabled=_optional_bool(os.getenv("RERANKING_ENABLED"), False),
            reranking_provider=reranking_provider,
            reranking_model=os.getenv(
                "RERANKING_MODEL",
                "jinaai/jina-reranker-v2-base-multilingual",
            ),
            reranking_candidate_k=reranking_candidate_k,
            reranking_top_k=reranking_top_k,
            chat_memory_enabled=_optional_bool(
                os.getenv("CHAT_MEMORY_ENABLED"),
                False,
            ),
            chat_memory_summary_threshold=chat_memory_summary_threshold,
            chat_memory_summary_interval=chat_memory_summary_interval,
            chat_memory_keep_recent=chat_memory_keep_recent,
            chat_memory_max_summary_chars=chat_memory_max_summary_chars,
            chat_memory_retrieval_enabled=_optional_bool(
                os.getenv("CHAT_MEMORY_RETRIEVAL_ENABLED"),
                False,
            ),
            chat_memory_default_included=_optional_bool(
                os.getenv("CHAT_MEMORY_DEFAULT_INCLUDED"),
                False,
            ),
            chat_memory_top_k=chat_memory_top_k,
            chat_memory_source_type=chat_memory_source_type,
            graph_enabled=graph_enabled,
            graph_extraction_enabled=graph_extraction_enabled,
            graph_retrieval_enabled=graph_retrieval_enabled,
            graph_store_provider=graph_store_provider,
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j"),
            graph_max_nodes_per_chunk=graph_max_nodes_per_chunk,
            graph_max_edges_per_chunk=graph_max_edges_per_chunk,
            graph_extraction_concurrency=graph_extraction_concurrency,
            graph_retrieval_top_k=graph_retrieval_top_k,
            graph_context_max_chars=graph_context_max_chars,
            graph_source_type=graph_source_type,
            web_search_enabled=web_search_enabled,
            web_search_provider=web_search_provider,
            web_search_top_k=web_search_top_k,
            web_search_timeout_seconds=web_search_timeout_seconds,
            web_search_max_query_chars=web_search_max_query_chars,
            web_search_max_context_sources=web_search_max_context_sources,
            web_search_max_chars_per_source=web_search_max_chars_per_source,
            web_search_max_total_context_chars=web_search_max_total_context_chars,
            web_search_source_type=web_search_source_type,
            tavily_api_key=os.getenv("TAVILY_API_KEY") or None,
            intent_classifier_enabled=_optional_bool(
                os.getenv("INTENT_CLASSIFIER_ENABLED"),
                False,
            ),
            intent_classifier_provider=intent_classifier_provider,
            intent_classifier_model=os.getenv("INTENT_CLASSIFIER_MODEL", "gpt-4.1-mini"),
            intent_classifier_timeout_seconds=intent_classifier_timeout_seconds,
            intent_classifier_max_recent_messages=intent_classifier_max_recent_messages,
            intent_classifier_max_message_chars=intent_classifier_max_message_chars,
            intent_classifier_fallback_enabled=_optional_bool(
                os.getenv("INTENT_CLASSIFIER_FALLBACK_ENABLED"),
                True,
            ),
            retrieval_planner_enabled=_optional_bool(
                os.getenv("RETRIEVAL_PLANNER_ENABLED"),
                False,
            ),
            retrieval_planner_default_top_k=retrieval_planner_default_top_k,
            retrieval_planner_pdf_top_k=retrieval_planner_pdf_top_k,
            retrieval_planner_notes_top_k=retrieval_planner_notes_top_k,
            retrieval_planner_annotations_top_k=retrieval_planner_annotations_top_k,
            retrieval_planner_memory_top_k=retrieval_planner_memory_top_k,
            retrieval_planner_web_top_k=retrieval_planner_web_top_k,
            retrieval_planner_max_steps=retrieval_planner_max_steps,
            retrieval_planner_include_disabled_steps=_optional_bool(
                os.getenv("RETRIEVAL_PLANNER_INCLUDE_DISABLED_STEPS"),
                True,
            ),
            chunking_strategy=os.getenv(
                "RAG_CHUNKING_STRATEGY",
                "docling_hybrid_semantic_refinement",
            ),
            chunking_version=os.getenv("RAG_CHUNKING_VERSION", "v1"),
        )


def _load_dotenv() -> None:
    """Load rag_pipeline/.env when python-dotenv is installed."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(env_path)


def _optional_int(value: str | None) -> int | None:
    """Parse optional integer environment values."""
    if not value:
        return None
    return int(value)


def _int_or_default(name: str, default: int) -> int:
    parsed = _optional_int(os.getenv(name))
    return default if parsed is None else parsed


def _optional_bool(value: str | None, default: bool) -> bool:
    """Parse optional boolean environment values."""
    if value is None or value == "":
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}
