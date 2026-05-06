"""Docling PDF ingestion and hybrid chunking."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rag_pipeline.models import RagChunk, SourceRef
from rag_pipeline.refinement import SemanticRefiner
from rag_pipeline.text import build_content_hash, normalize_content


LOGGER = logging.getLogger(__name__)


def process_pdf(
    pdf_path: Path,
    source: SourceRef,
    refiner: SemanticRefiner,
    chunking_strategy: str,
    chunking_version: str,
) -> tuple[list[RagChunk], str | None, dict[str, Any]]:
    """Convert and chunk a PDF with Docling.

    Args:
        pdf_path: Local path to the PDF.
        source: Source reference for the PDF.
        refiner: Semantic refiner for oversized chunks.
        chunking_strategy: Strategy name for hash and metadata.
        chunking_version: Strategy version for hash and metadata.

    Returns:
        Tuple of final chunks, detected Docling version, and coverage metadata.

    Raises:
        RuntimeError: If Docling is not installed.
    """
    try:
        import docling
        from docling.chunking import HybridChunker
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.settings import settings
        from docling.document_converter import DocumentConverter
        from docling.document_converter import PdfFormatOption
    except ImportError as exc:
        raise RuntimeError(
            "Docling is required for PDF ingestion. Install rag_pipeline "
            "requirements in the Python worker environment."
        ) from exc

    chunker = HybridChunker()
    docling_version = getattr(docling, "__version__", None)
    total_pages = _pdf_page_count(pdf_path)

    previous_page_batch_size = settings.perf.page_batch_size
    try:
        settings.perf.page_batch_size = 1
        converter = DocumentConverter()
        document = converter.convert(str(pdf_path)).document
        final_chunks = _chunks_from_document(
            document=document,
            chunker=chunker,
            source=source,
            refiner=refiner,
            chunking_strategy=chunking_strategy,
            chunking_version=chunking_version,
            metadata_patch={"docling_pass": "full"},
        )

        chunked_pages = _chunked_pages(final_chunks)
        expected_pages = set(range(total_pages))
        missing_pages = sorted(expected_pages - chunked_pages)
        fallback_pages: list[int] = []
        retry_errors: dict[str, str] = {}

        if missing_pages:
            LOGGER.warning(
                "Docling full conversion missed pages: %s",
                [page + 1 for page in missing_pages],
            )

        full_page_converter = DocumentConverter()

        lightweight_options = PdfPipelineOptions(
            do_ocr=False,
            do_table_structure=False,
            force_backend_text=True,
        )
        lightweight_converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=lightweight_options,
                )
            },
        )

        for page_index in missing_pages:
            page_no = page_index + 1
            retry_chunks = _convert_single_page(
                converter=full_page_converter,
                pdf_path=pdf_path,
                page_no=page_no,
                chunker=chunker,
                source=source,
                refiner=refiner,
                chunking_strategy=chunking_strategy,
                chunking_version=chunking_version,
                docling_pass="single_page_full",
                retry_errors=retry_errors,
            )

            if not retry_chunks:
                retry_chunks = _convert_single_page(
                    converter=lightweight_converter,
                    pdf_path=pdf_path,
                    page_no=page_no,
                    chunker=chunker,
                    source=source,
                    refiner=refiner,
                    chunking_strategy=chunking_strategy,
                    chunking_version=chunking_version,
                    docling_pass="single_page_lightweight",
                    retry_errors=retry_errors,
                )

            if retry_chunks:
                final_chunks.extend(retry_chunks)
                continue

            text = _extract_page_text(pdf_path, page_index)
            if text:
                fallback_pages.append(page_no)
                final_chunks.extend(
                    _chunks_from_text(
                        text=text,
                        source=source,
                        refiner=refiner,
                        chunking_strategy=chunking_strategy,
                        chunking_version=chunking_version,
                        page_index=page_index,
                        metadata={
                            "docling_pass": "pypdfium_text_fallback",
                            "fallback_for_page": page_no,
                        },
                    )
                )

        final_chunks = _dedupe_chunks(final_chunks)
        coverage = {
            "total_pages": total_pages,
            "chunked_pages": sorted(
                page + 1 for page in _chunked_pages(final_chunks)
            ),
            "missing_pages": sorted(
                page + 1 for page in expected_pages - _chunked_pages(final_chunks)
            ),
            "fallback_pages": fallback_pages,
            "retry_errors": retry_errors,
            "page_batch_size": 1,
        }
        return final_chunks, docling_version, coverage
    finally:
        settings.perf.page_batch_size = previous_page_batch_size


def _convert_single_page(
    converter: Any,
    pdf_path: Path,
    page_no: int,
    chunker: Any,
    source: SourceRef,
    refiner: SemanticRefiner,
    chunking_strategy: str,
    chunking_version: str,
    docling_pass: str,
    retry_errors: dict[str, str],
) -> list[RagChunk]:
    try:
        retry_document = converter.convert(
            str(pdf_path),
            page_range=(page_no, page_no),
        ).document
        return _chunks_from_document(
            document=retry_document,
            chunker=chunker,
            source=source,
            refiner=refiner,
            chunking_strategy=chunking_strategy,
            chunking_version=chunking_version,
            metadata_patch={
                "docling_pass": docling_pass,
                "fallback_for_page": page_no,
            },
        )
    except Exception as exc:
        retry_errors[f"{page_no}:{docling_pass}"] = str(exc)
        return []


def _chunks_from_document(
    document: Any,
    chunker: Any,
    source: SourceRef,
    refiner: SemanticRefiner,
    chunking_strategy: str,
    chunking_version: str,
    metadata_patch: dict[str, Any],
) -> list[RagChunk]:
    chunks: list[RagChunk] = []
    for raw_chunk in chunker.chunk(dl_doc=document):
        text = normalize_content(chunker.contextualize(chunk=raw_chunk))
        if not text:
            continue

        metadata = _chunk_metadata(raw_chunk)
        metadata.update(metadata_patch)
        chunk_kind = metadata.get("chunk_kind", "text")
        for refined in refiner.refine(text, str(chunk_kind), metadata):
            content_hash = build_content_hash(
                refined,
                source,
                chunking_strategy,
                chunking_version,
            )
            chunks.append(
                RagChunk(
                    source=source,
                    content=refined,
                    content_hash=content_hash,
                    page_index=_page_index(metadata),
                    heading_path=_heading_path(metadata),
                    chunk_kind=str(chunk_kind),
                    metadata=metadata,
                )
            )
    return chunks


def _chunks_from_text(
    text: str,
    source: SourceRef,
    refiner: SemanticRefiner,
    chunking_strategy: str,
    chunking_version: str,
    page_index: int,
    metadata: dict[str, Any],
) -> list[RagChunk]:
    chunks = []
    for refined in refiner.refine(text, "text", metadata):
        content_hash = build_content_hash(
            refined,
            source,
            chunking_strategy,
            chunking_version,
        )
        chunks.append(
            RagChunk(
                source=source,
                content=refined,
                content_hash=content_hash,
                page_index=page_index,
                heading_path=[],
                chunk_kind="text",
                metadata=metadata,
            )
        )
    return chunks


def _chunk_metadata(chunk: Any) -> dict[str, Any]:
    metadata = getattr(chunk, "meta", None)
    if metadata is None:
        return {}
    if hasattr(metadata, "model_dump"):
        dumped = metadata.model_dump(mode="json")
        return dumped if isinstance(dumped, dict) else {}
    if isinstance(metadata, dict):
        return metadata
    return {"raw_meta": str(metadata)}


def _heading_path(metadata: dict[str, Any]) -> list[str]:
    headings = metadata.get("headings")
    if isinstance(headings, list):
        return [str(heading) for heading in headings if str(heading).strip()]
    heading = metadata.get("heading")
    if isinstance(heading, str) and heading.strip():
        return [heading.strip()]
    return []


def _page_index(metadata: dict[str, Any]) -> int | None:
    page_index = metadata.get("page_index")
    if isinstance(page_index, int):
        return page_index

    doc_items = metadata.get("doc_items")
    if not isinstance(doc_items, list):
        return None

    for item in doc_items:
        if not isinstance(item, dict):
            continue
        prov = item.get("prov")
        if not isinstance(prov, list):
            continue
        for entry in prov:
            if not isinstance(entry, dict):
                continue
            page_no = entry.get("page_no")
            if isinstance(page_no, int):
                return max(page_no - 1, 0)
    return None


def _chunked_pages(chunks: list[RagChunk]) -> set[int]:
    return {
        chunk.page_index
        for chunk in chunks
        if chunk.page_index is not None
    }


def _dedupe_chunks(chunks: list[RagChunk]) -> list[RagChunk]:
    seen: set[str] = set()
    deduped: list[RagChunk] = []
    for chunk in chunks:
        if chunk.content_hash in seen:
            continue
        seen.add(chunk.content_hash)
        deduped.append(chunk)
    return deduped


def _pdf_page_count(pdf_path: Path) -> int:
    try:
        import pypdfium2
    except ImportError as exc:
        raise RuntimeError("pypdfium2 is required to verify PDF page coverage.") from exc

    pdf = pypdfium2.PdfDocument(str(pdf_path))
    try:
        return len(pdf)
    finally:
        pdf.close()


def _extract_page_text(pdf_path: Path, page_index: int) -> str:
    try:
        import pypdfium2
    except ImportError:
        return ""

    try:
        pdf = pypdfium2.PdfDocument(str(pdf_path))
        page = pdf[page_index]
        text_page = page.get_textpage()
        try:
            return normalize_content(text_page.get_text_range())
        finally:
            text_page.close()
            page.close()
            pdf.close()
    except Exception as exc:
        LOGGER.warning(
            "pypdfium text fallback failed for page %s: %s",
            page_index + 1,
            exc,
        )
        return ""
