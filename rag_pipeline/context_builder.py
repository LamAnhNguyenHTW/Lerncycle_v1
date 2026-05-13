"""Build compact LLM context and source citations from retrieval results."""

from __future__ import annotations

from typing import Any


SOURCE_LABELS = {
    "pdf": "PDF",
    "note": "Note",
    "annotation_comment": "Annotation",
    "chat_memory": "Chat Memory",
    "knowledge_graph": "Knowledge Graph",
    "web": "Web",
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
        filename = _first_text(title, metadata.get("filename"), _origin_filename(metadata), "PDF")
        safe_metadata["filename"] = filename
        title = filename
    elif source_type == "annotation_comment":
        title = "Annotation"
        filename = _first_text(metadata.get("filename"), _origin_filename(metadata))
        if filename is not None:
            safe_metadata["filename"] = filename
    elif source_type == "note":
        title = _first_text(title, metadata.get("title"), "Note")
    elif source_type == "chat_memory":
        title = "Chat Memory"
        safe_metadata["session_id"] = metadata.get("session_id")
        safe_metadata["memory_kind"] = metadata.get("memory_kind")
        safe_metadata = {key: value for key, value in safe_metadata.items() if value}
        page = None
    elif source_type == "knowledge_graph":
        title = "Knowledge Graph"
        safe_metadata["backing_chunk_ids"] = metadata.get("backing_chunk_ids")
        safe_metadata["node_names"] = metadata.get("node_names")
        safe_metadata["relationship_count"] = metadata.get("relationship_count")
        safe_metadata = {key: value for key, value in safe_metadata.items() if value}
    elif source_type == "web":
        title = _first_text(title, metadata.get("title"), metadata.get("url"), "Web Source")
        safe_metadata["url"] = metadata.get("url")
        safe_metadata["provider"] = metadata.get("provider")
        safe_metadata["published_date"] = metadata.get("published_date")
        safe_metadata["retrieved_at"] = metadata.get("retrieved_at")
        safe_metadata["rank"] = metadata.get("rank")
        safe_metadata = {key: value for key, value in safe_metadata.items() if value is not None}
        page = None

    return {
        "chunk_id": result.get("chunk_id"),
        "source_type": source_type,
        "source_id": result.get("source_id"),
        "title": title,
        "heading": result.get("heading"),
        "page": page,
        "score": result.get("score"),
        "snippet": _snippet(text, 200 if source_type == "chat_memory" else 200),
        "metadata": safe_metadata,
    }


def _origin_filename(metadata: dict[str, Any]) -> Any:
    origin = metadata.get("origin")
    if not isinstance(origin, dict):
        return None
    return origin.get("filename")


def _first_text(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value
    return None


def build_rag_context(
    results: list[dict[str, Any]],
    max_chunks: int = 8,
    max_chars: int = 12000,
    web_max_sources: int = 5,
    web_max_chars_per_source: int = 1000,
    web_max_total_chars: int = 4000,
) -> dict[str, Any]:
    """Build ranked source blocks and citations for a RAG answer."""
    unique_results = _dedupe_by_chunk_id(results)[:max_chunks]
    context_blocks: list[str] = []
    sources: list[dict[str, Any]] = []
    total_chars = 0
    web_sources = 0
    web_chars = 0

    for result in unique_results:
        block_result = result
        if result.get("source_type") == "web":
            if web_sources >= web_max_sources or web_chars >= web_max_total_chars:
                continue
            remaining_web_chars = max(0, web_max_total_chars - web_chars)
            web_text_limit = min(web_max_chars_per_source, remaining_web_chars)
            block_result = {**result, "text": str(result.get("text") or "")[:web_text_limit].rstrip()}
            web_sources += 1
        source = normalize_source(block_result)
        block = _format_context_block(len(context_blocks) + 1, block_result, source)
        block_len = len(block) + (2 if context_blocks else 0)
        if context_blocks and total_chars + block_len > max_chars:
            break
        if not context_blocks and block_len > max_chars:
            block = block[:max_chars].rstrip()
            block_len = len(block)
        context_blocks.append(block)
        sources.append(source)
        total_chars += block_len
        if result.get("source_type") == "web":
            web_chars += len(str(block_result.get("text") or ""))

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
    elif source_type == "chat_memory":
        lines.append("Role: Session-scoped learning history summary")
    elif source_type == "knowledge_graph":
        lines.append("Role: Concept relationship context backed by chunks")
    elif source_type == "web":
        metadata = source.get("metadata", {})
        lines.append("Role: External web source")
        lines.append(f"Title: {source.get('title')}")
        if metadata.get("url"):
            lines.append(f"URL: {metadata.get('url')}")
        if metadata.get("retrieved_at"):
            lines.append(f"Retrieved at: {metadata.get('retrieved_at')}")

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
    return collapsed[: max_length - 3].rstrip() + "..."
