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


def _optional_bool(value: str | None, default: bool) -> bool:
    """Parse optional boolean environment values."""
    if value is None or value == "":
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}
