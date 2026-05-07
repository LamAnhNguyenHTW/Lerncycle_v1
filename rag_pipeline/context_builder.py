"""Build compact LLM context and source citations from retrieval results."""

from __future__ import annotations

from typing import Any


SOURCE_LABELS = {
    "pdf": "PDF",
    "note": "Note",
    "annotation_comment": "Annotation",
}


def normalize_source(result: dict[str, Any]) -> dict[str, Any]:
    """Return a frontend-safe citation shape for one retrieval result."""
    source_type = result.get("source_type")
    metadata = result.get("metadata") or {}
    page_index = result.get("page_index")
    page = page_index + 1 if isinstance(page_index, int) else None
    text = str(result.get("text") or "").strip()

    title = result.get("title")
    safe_metadata: dict[str, Any] = {}
    if source_type == "pdf":
        filename = metadata.get("filename") or title
        if filename is not None:
            safe_metadata["filename"] = filename
        title = title or filename
    elif source_type == "annotation_comment":
        title = "Annotation"

    return {
        "chunk_id": result.get("chunk_id"),
        "source_type": source_type,
        "source_id": result.get("source_id"),
        "title": title,
        "heading": result.get("heading"),
        "page": page,
        "score": result.get("score"),
        "snippet": _snippet(text),
        "metadata": safe_metadata,
    }


def build_rag_context(
    results: list[dict[str, Any]],
    max_chunks: int = 8,
    max_chars: int = 12000,
) -> dict[str, Any]:
    """Build ranked source blocks and citations for a RAG answer."""
    unique_results = _dedupe_by_chunk_id(results)[:max_chunks]
    context_blocks: list[str] = []
    sources: list[dict[str, Any]] = []
    total_chars = 0

    for result in unique_results:
        source = normalize_source(result)
        block = _format_context_block(len(context_blocks) + 1, result, source)
        block_len = len(block) + (2 if context_blocks else 0)
        if context_blocks and total_chars + block_len > max_chars:
            break
        if not context_blocks and block_len > max_chars:
            block = block[:max_chars].rstrip()
            block_len = len(block)
        context_blocks.append(block)
        sources.append(source)
        total_chars += block_len

    return {"context_text": "\n\n".join(context_blocks), "sources": sources}


def _dedupe_by_chunk_id(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[Any] = set()
    unique: list[dict[str, Any]] = []
    for result in results:
        chunk_id = result.get("chunk_id")
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        unique.append(result)
    return unique


def _format_context_block(index: int, result: dict[str, Any], source: dict[str, Any]) -> str:
    source_type = source.get("source_type")
    label = SOURCE_LABELS.get(str(source_type), str(source_type or "Unknown"))
    text = str(result.get("text") or "").strip()
    lines = [f"[Source {index}]", f"Type: {label}"]

    if source_type == "pdf":
        filename = source.get("metadata", {}).get("filename")
        lines.append(f"File: {filename}")
    elif source_type == "note":
        lines.append(f"Title: {source.get('title')}")

    if source.get("page") is not None:
        lines.append(f"Page: {source.get('page')}")
    if source.get("heading") is not None:
        lines.append(f"Heading: {source.get('heading')}")

    lines.extend(["Content:", text])
    return "\n".join(lines)


def _snippet(text: str, max_length: int = 200) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= max_length:
        return collapsed
    return collapsed[: max_length - 1].rstrip() + "..."
