"""Index an evaluation corpus under three chunking strategies.

Builds one Qdrant collection per strategy from the *same* documents and the
*same* embedding models, so the chunking comparison changes only the chunker:

  - fixed_size        -> rag_pipeline.fixed_chunking (evaluation-only baseline)
  - docling           -> Docling HybridChunker, no semantic refinement
  - docling_semantic  -> Docling HybridChunker + SemanticRefiner (production)

Per strategy it records indexing duration, chunk count, chunk-length stats, and
the exact configuration used. Nothing here touches the production worker or its
default collection; strategy collections are named `eval_*`.

Usage (real run — needs OPENAI_API_KEY + QDRANT_URL, and Docling for the
docling* strategies):

    python -m rag_pipeline.eval.index_corpus \
        --manifest rag_pipeline/eval/corpus/manifest.json \
        --strategy all --user-id <eval-user> \
        --out rag_pipeline/eval/results/index_stats.json

    # cheap chunk-only inspection, no embeddings / no Qdrant:
    python -m rag_pipeline.eval.index_corpus --manifest ... --strategy fixed_size --dry-run
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from rag_pipeline.fixed_chunking import (
    CHUNKING_STRATEGY_NAME as FIXED_STRATEGY_NAME,
    CHUNKING_STRATEGY_VERSION as FIXED_STRATEGY_VERSION,
    DEFAULT_MAX_CHARS,
    DEFAULT_OVERLAP_CHARS,
    chunk_fixed_size,
)
from rag_pipeline.models import RagChunk, SourceRef
from rag_pipeline.text import build_content_hash, normalize_content


FIXED_SIZE = "fixed_size"
DOCLING = "docling"
DOCLING_SEMANTIC = "docling_semantic"
STRATEGIES = (FIXED_SIZE, DOCLING, DOCLING_SEMANTIC)

STRATEGY_COLLECTIONS = {
    FIXED_SIZE: "eval_fixed_size",
    DOCLING: "eval_docling",
    DOCLING_SEMANTIC: "eval_docling_semantic",
}


@dataclass
class CorpusDocument:
    """One evaluation source document."""

    source_id: str
    pdf_path: Path
    title: str | None = None
    language: str | None = None
    source_type: str = "pdf"


@dataclass
class ChunkStats:
    """Chunk-length statistics for one strategy run."""

    chunk_count: int
    mean_length: float
    min_length: int
    max_length: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_count": self.chunk_count,
            "mean_length": self.mean_length,
            "min_length": self.min_length,
            "max_length": self.max_length,
        }


@dataclass
class IndexRunResult:
    """Recorded outcome of indexing the corpus under one strategy."""

    strategy: str
    collection: str
    stats: ChunkStats
    indexing_seconds: float
    documents_indexed: int
    config: dict[str, Any]
    per_document: list[dict[str, Any]] = field(default_factory=list)
    extraction_errors: list[dict[str, Any]] = field(default_factory=list)
    extraction_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "collection": self.collection,
            "indexing_seconds": self.indexing_seconds,
            "documents_indexed": self.documents_indexed,
            "dry_run": self.dry_run,
            "stats": self.stats.to_dict(),
            "config": self.config,
            "per_document": self.per_document,
            "extraction_errors": self.extraction_errors,
            "extraction_diagnostics": self.extraction_diagnostics,
        }


class _NoOpRefiner:
    """Refiner stub for pure Docling: returns the chunk text unchanged.

    Matches the SemanticRefiner.refine signature used by process_pdf, so the
    Docling chunk boundaries are preserved without any semantic splitting.
    """

    def refine(
        self,
        content: str,
        chunk_kind: str = "text",
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        normalized = normalize_content(content)
        return [normalized] if normalized else []


def compute_chunk_stats(chunks: list[RagChunk]) -> ChunkStats:
    """Compute count and length statistics over chunk contents."""
    lengths = [len(chunk.content) for chunk in chunks]
    if not lengths:
        return ChunkStats(chunk_count=0, mean_length=0.0, min_length=0, max_length=0)
    return ChunkStats(
        chunk_count=len(lengths),
        mean_length=sum(lengths) / len(lengths),
        min_length=min(lengths),
        max_length=max(lengths),
    )


def build_fixed_size_chunks(
    document: CorpusDocument,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
    page_texts: list[str] | None = None,
) -> list[RagChunk]:
    """Chunk a document with the fixed-size baseline, preserving page attribution.

    Chunks are built per page so `page_index` stays meaningful for page-labeled
    ground truth. `page_texts` (0-based per page) can be injected for tests; when
    omitted the pages are read from the PDF via pypdfium2.
    """
    pages = page_texts if page_texts is not None else _read_pdf_pages(document.pdf_path)
    source = SourceRef(
        user_id="",  # filled in at index time; hashing uses source_type/source_id
        source_type=document.source_type,
        source_id=document.source_id,
        pdf_id=document.source_id if document.source_type == "pdf" else None,
    )
    chunks: list[RagChunk] = []
    chunk_index = 0
    for page_index, page_text in enumerate(pages):
        for piece in chunk_fixed_size(
            page_text, max_chars=max_chars, overlap_chars=overlap_chars
        ):
            content_hash = build_content_hash(
                piece, source, FIXED_STRATEGY_NAME, FIXED_STRATEGY_VERSION
            )
            chunks.append(
                RagChunk(
                    source=source,
                    content=piece,
                    content_hash=content_hash,
                    page_index=page_index,
                    heading_path=[],
                    chunk_kind="text",
                    metadata={"chunk_index": chunk_index, "strategy": FIXED_SIZE},
                )
            )
            chunk_index += 1
    return chunks


def build_docling_chunks(
    document: CorpusDocument,
    *,
    semantic_refinement: bool,
    openai_api_key: str | None,
    embedding_provider: str = "openai",
    embedding_model: str = "text-embedding-3-small",
    diagnostics: dict[str, Any] | None = None,
) -> list[RagChunk]:
    """Chunk a document with Docling, optionally with semantic refinement.

    Imported lazily so the fixed-size path and unit tests never require Docling.
    """
    from rag_pipeline.docling_ingestion import process_pdf
    from rag_pipeline.refinement import SemanticRefiner

    source = SourceRef(
        user_id="",
        source_type=document.source_type,
        source_id=document.source_id,
        pdf_id=document.source_id if document.source_type == "pdf" else None,
    )
    if semantic_refinement:
        refiner: Any = SemanticRefiner(
            openai_api_key=openai_api_key,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
        )
        strategy_name = "docling_hybrid_semantic_refinement"
    else:
        refiner = _NoOpRefiner()
        strategy_name = "docling_hybrid"

    chunks, docling_version, coverage = process_pdf(
        pdf_path=document.pdf_path,
        source=source,
        refiner=refiner,
        chunking_strategy=strategy_name,
        chunking_version="eval",
    )
    if diagnostics is not None:
        diagnostics.update(coverage)
        diagnostics["docling_version"] = docling_version
    for index, chunk in enumerate(chunks):
        chunk.metadata.setdefault("chunk_index", index)
        chunk.metadata["strategy"] = (
            DOCLING_SEMANTIC if semantic_refinement else DOCLING
        )
    return chunks


def build_chunks(
    strategy: str,
    document: CorpusDocument,
    *,
    openai_api_key: str | None = None,
    fixed_max_chars: int = DEFAULT_MAX_CHARS,
    fixed_overlap_chars: int = DEFAULT_OVERLAP_CHARS,
    embedding_model: str = "text-embedding-3-small",
    diagnostics: dict[str, Any] | None = None,
) -> list[RagChunk]:
    """Dispatch to the strategy-specific chunker."""
    if strategy == FIXED_SIZE:
        return build_fixed_size_chunks(
            document, max_chars=fixed_max_chars, overlap_chars=fixed_overlap_chars
        )
    if strategy in (DOCLING, DOCLING_SEMANTIC):
        return build_docling_chunks(
            document,
            semantic_refinement=(strategy == DOCLING_SEMANTIC),
            openai_api_key=openai_api_key,
            embedding_model=embedding_model,
            diagnostics=diagnostics,
        )
    raise ValueError(f"Unknown chunking strategy: {strategy}")


def _chunk_to_qdrant_payload(
    chunk: RagChunk,
    user_id: str,
    dense_vector: list[float],
    sparse_vector: Any,
    point_id: str,
) -> dict[str, Any]:
    heading = " > ".join(chunk.heading_path) if chunk.heading_path else None
    return {
        "id": point_id,
        "embedding": dense_vector,
        "sparse_embedding": sparse_vector,
        "chunk_id": point_id,
        "user_id": user_id,
        "source_type": chunk.source.source_type,
        "source_id": chunk.source.source_id,
        "pdf_id": chunk.source.pdf_id,
        "note_id": chunk.source.note_id,
        "annotation_id": chunk.source.annotation_id,
        "page_index": chunk.page_index,
        "heading": heading,
        "title": None,
        "text": chunk.content,
        "metadata": chunk.metadata,
        "content_hash": chunk.content_hash,
        "chunk_index": chunk.metadata.get("chunk_index"),
    }


def index_corpus(
    strategy: str,
    documents: list[CorpusDocument],
    *,
    user_id: str,
    embedder: Any = None,
    sparse_embedder: Any = None,
    store: Any = None,
    collection_name: str | None = None,
    openai_api_key: str | None = None,
    fixed_max_chars: int = DEFAULT_MAX_CHARS,
    fixed_overlap_chars: int = DEFAULT_OVERLAP_CHARS,
    embedding_model: str = "text-embedding-3-small",
    sparse_model: str = "Qdrant/bm25",
    dry_run: bool = False,
    continue_on_extraction_error: bool = False,
    point_id_factory: Callable[[str, str, int], str] | None = None,
) -> IndexRunResult:
    """Chunk, embed, and upsert the corpus under one strategy.

    With `dry_run=True` chunks are built and measured but nothing is embedded or
    upserted (no OpenAI, no Qdrant). Otherwise `embedder`, `sparse_embedder`, and
    `store` must be provided; the caller owns their construction and credentials.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
    collection = collection_name or STRATEGY_COLLECTIONS[strategy]
    make_point_id = point_id_factory or _default_point_id

    start = time.perf_counter()
    all_chunks: list[RagChunk] = []
    per_document: list[dict[str, Any]] = []
    extraction_errors: list[dict[str, Any]] = []
    extraction_diagnostics: list[dict[str, Any]] = []

    for document in documents:
        diagnostics: dict[str, Any] = {}
        try:
            doc_chunks = build_chunks(
                strategy,
                document,
                openai_api_key=openai_api_key,
                fixed_max_chars=fixed_max_chars,
                fixed_overlap_chars=fixed_overlap_chars,
                embedding_model=embedding_model,
                diagnostics=diagnostics,
            )
        except Exception as exc:
            if not continue_on_extraction_error:
                raise
            extraction_errors.append(
                {
                    "strategy": strategy,
                    "source_id": document.source_id,
                    "stage": "extraction",
                    "error_type": type(exc).__name__,
                }
            )
            continue
        if diagnostics:
            extraction_diagnostics.append(
                _sanitize_coverage_diagnostics(
                    diagnostics,
                    strategy=strategy,
                    source_id=document.source_id,
                )
            )
        all_chunks.extend(doc_chunks)
        doc_stats = compute_chunk_stats(doc_chunks)
        per_document.append(
            {
                "source_id": document.source_id,
                "language": document.language,
                **doc_stats.to_dict(),
            }
        )

    if not dry_run:
        _embed_and_upsert(
            all_chunks,
            user_id=user_id,
            embedder=embedder,
            sparse_embedder=sparse_embedder,
            store=store,
            make_point_id=make_point_id,
        )

    indexing_seconds = time.perf_counter() - start
    stats = compute_chunk_stats(all_chunks)
    config = {
        "strategy": strategy,
        "collection": collection,
        "embedding_model": embedding_model,
        "sparse_model": sparse_model,
        "user_id": user_id,
        "document_count": len(documents),
        "dry_run": dry_run,
    }
    if strategy == FIXED_SIZE:
        config["fixed_max_chars"] = fixed_max_chars
        config["fixed_overlap_chars"] = fixed_overlap_chars

    return IndexRunResult(
        strategy=strategy,
        collection=collection,
        stats=stats,
        indexing_seconds=indexing_seconds,
        documents_indexed=len(per_document),
        config=config,
        per_document=per_document,
        extraction_errors=extraction_errors,
        extraction_diagnostics=extraction_diagnostics,
        dry_run=dry_run,
    )


def _sanitize_coverage_diagnostics(
    diagnostics: dict[str, Any],
    *,
    strategy: str,
    source_id: str,
) -> dict[str, Any]:
    """Keep page coverage while excluding raw conversion exception messages."""
    retry_errors = diagnostics.get("retry_errors")
    retry_error_keys = (
        sorted(str(key) for key in retry_errors)
        if isinstance(retry_errors, dict)
        else []
    )
    return {
        "strategy": strategy,
        "source_id": source_id,
        "total_pages": int(diagnostics.get("total_pages") or 0),
        "chunked_pages": list(diagnostics.get("chunked_pages") or []),
        "missing_pages": list(diagnostics.get("missing_pages") or []),
        "fallback_pages": list(diagnostics.get("fallback_pages") or []),
        "retry_error_keys": retry_error_keys,
        "page_batch_size": int(diagnostics.get("page_batch_size") or 0),
    }


def _embed_and_upsert(
    chunks: list[RagChunk],
    *,
    user_id: str,
    embedder: Any,
    sparse_embedder: Any,
    store: Any,
    make_point_id: Callable[[str, str, int], str],
) -> None:
    if embedder is None or sparse_embedder is None or store is None:
        raise ValueError(
            "index_corpus requires embedder, sparse_embedder, and store unless dry_run=True."
        )
    if not chunks:
        return
    texts = [chunk.content for chunk in chunks]
    dense_vectors = embedder.embed(texts)
    sparse_vectors = sparse_embedder.embed(texts)
    if len(dense_vectors) != len(chunks) or len(sparse_vectors) != len(chunks):
        raise RuntimeError("Embedding count does not match chunk count.")

    dim = getattr(embedder, "dimension", None) or len(dense_vectors[0])
    store.ensure_collection(dim, sparse_enabled=True)

    payloads = []
    for index, chunk in enumerate(chunks):
        point_id = make_point_id(
            chunk.metadata.get("strategy", ""), chunk.source.source_id, index
        )
        payloads.append(
            _chunk_to_qdrant_payload(
                chunk, user_id, dense_vectors[index], sparse_vectors[index], point_id
            )
        )
    store.upsert_chunks(payloads, sparse_enabled=True)


def _default_point_id(strategy: str, source_id: str, index: int) -> str:
    from uuid import uuid5, NAMESPACE_URL

    return str(uuid5(NAMESPACE_URL, f"eval:{strategy}:{source_id}:{index}"))


def _read_pdf_pages(pdf_path: Path) -> list[str]:
    """Return normalized per-page text for a PDF via pypdfium2."""
    try:
        import pypdfium2
    except ImportError as exc:  # pragma: no cover - exercised only on real runs
        raise RuntimeError("pypdfium2 is required to read corpus PDFs.") from exc

    pdf = pypdfium2.PdfDocument(str(pdf_path))
    pages: list[str] = []
    try:
        for page_index in range(len(pdf)):
            page = pdf[page_index]
            text_page = page.get_textpage()
            try:
                pages.append(normalize_content(text_page.get_text_range()))
            finally:
                text_page.close()
                page.close()
    finally:
        pdf.close()
    return pages


def load_manifest(path: str | Path) -> list[CorpusDocument]:
    """Load a corpus manifest JSON into CorpusDocument objects.

    Manifest format: a list (or {'documents': [...]}) of objects with keys
    source_id, pdf_path, and optional title/language/source_type. pdf_path is
    resolved relative to the manifest file's directory when not absolute.
    """
    manifest_path = Path(path)
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = raw["documents"] if isinstance(raw, dict) else raw
    documents: list[CorpusDocument] = []
    for item in items:
        pdf_path = Path(item["pdf_path"])
        if not pdf_path.is_absolute():
            pdf_path = (manifest_path.parent / pdf_path).resolve()
        documents.append(
            CorpusDocument(
                source_id=str(item["source_id"]),
                pdf_path=pdf_path,
                title=item.get("title"),
                language=item.get("language"),
                source_type=item.get("source_type", "pdf"),
            )
        )
    return documents


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--strategy",
        default="all",
        choices=[*STRATEGIES, "all"],
    )
    parser.add_argument("--user-id", default="eval-user")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--continue-on-extraction-error",
        action="store_true",
        help="Record sanitized per-document extraction errors and continue.",
    )
    parser.add_argument("--fixed-max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--fixed-overlap-chars", type=int, default=DEFAULT_OVERLAP_CHARS)
    args = parser.parse_args(argv)

    documents = load_manifest(args.manifest)
    strategies = list(STRATEGIES) if args.strategy == "all" else [args.strategy]

    embedder = sparse_embedder = store = None
    embedding_model = "text-embedding-3-small"
    sparse_model = "Qdrant/bm25"
    openai_api_key = None
    config = None
    if not args.dry_run:
        from rag_pipeline.config import WorkerConfig
        from rag_pipeline.embeddings import Embedder
        from rag_pipeline.sparse_embeddings import SparseEmbedder

        config = WorkerConfig.from_env()
        embedding_model = config.embedding_model
        sparse_model = config.sparse_model
        openai_api_key = config.openai_api_key
        embedder = Embedder(
            provider=config.embedding_provider,
            model=config.embedding_model,
            openai_api_key=config.openai_api_key,
            gemini_api_key=config.gemini_api_key,
            batch_size=config.embedding_batch_size,
        )
        sparse_embedder = SparseEmbedder(
            provider=config.sparse_provider, model=config.sparse_model
        )

    results = []
    for strategy in strategies:
        store = None
        if not args.dry_run and config is not None:
            from rag_pipeline.qdrant_store import QdrantStore

            store = QdrantStore(
                url=config.qdrant_url,
                api_key=config.qdrant_api_key,
                collection_name=STRATEGY_COLLECTIONS[strategy],
            )
        result = index_corpus(
            strategy,
            documents,
            user_id=args.user_id,
            embedder=embedder,
            sparse_embedder=sparse_embedder,
            store=store,
            openai_api_key=openai_api_key,
            fixed_max_chars=args.fixed_max_chars,
            fixed_overlap_chars=args.fixed_overlap_chars,
            embedding_model=embedding_model,
            sparse_model=sparse_model,
            dry_run=args.dry_run,
            continue_on_extraction_error=args.continue_on_extraction_error,
        )
        results.append(result.to_dict())

    output = {
        "index_runs": results,
        "extraction_errors": [
            error
            for result in results
            for error in result.get("extraction_errors", [])
        ],
        "extraction_diagnostics": [
            diagnostic
            for result in results
            for diagnostic in result.get("extraction_diagnostics", [])
        ],
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
