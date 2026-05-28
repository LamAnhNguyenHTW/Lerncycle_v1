"""Compact document primers for realtime voice session orientation."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from rag_pipeline.models import RagChunk

MAX_LLM_CHUNKS = 8
MAX_CHUNK_PROMPT_CHARS = 900
MAX_SUMMARY_CHARS = 700
MAX_LIST_ITEMS = 12
STOPWORDS = {
    "aber",
    "also",
    "and",
    "auf",
    "aus",
    "bei",
    "das",
    "der",
    "die",
    "ein",
    "eine",
    "for",
    "ist",
    "mit",
    "oder",
    "the",
    "und",
    "von",
    "was",
    "with",
    "werden",
}


class DocumentPrimer(BaseModel):
    """Validated compact orientation data for one indexed source."""

    model_config = ConfigDict(extra="ignore")

    title: str | None = None
    summary: str
    main_topics: list[str] = Field(default_factory=list)
    key_terms: list[str] = Field(default_factory=list)
    learning_objectives: list[str] = Field(default_factory=list)
    page_ranges: list[dict[str, int]] = Field(default_factory=list)
    content_hash: str

    @field_validator("summary")
    @classmethod
    def _summary_must_be_compact(cls, value: str) -> str:
        value = _clean_text(value)
        if not value:
            raise ValueError("summary must not be empty")
        return value[:MAX_SUMMARY_CHARS]

    @field_validator("main_topics", "key_terms", "learning_objectives")
    @classmethod
    def _clean_list(cls, value: list[str]) -> list[str]:
        return _dedupe([_clean_text(item) for item in value if _clean_text(item)])[:MAX_LIST_ITEMS]

    @field_validator("page_ranges")
    @classmethod
    def _clean_page_ranges(cls, value: list[dict[str, int]]) -> list[dict[str, int]]:
        cleaned: list[dict[str, int]] = []
        for item in value:
            start = int(item.get("start", 0) or 0)
            end = int(item.get("end", start) or start)
            if start > 0 and end >= start:
                cleaned.append({"start": start, "end": end})
        return cleaned[:8]


def build_document_primer(
    chunks: list[RagChunk],
    *,
    llm_client: Any | None = None,
) -> DocumentPrimer:
    """Build a compact primer from existing chunks with LLM fallback."""
    if not chunks:
        raise ValueError("Cannot build a document primer without chunks.")
    content_hash = document_primer_content_hash(chunks)
    if llm_client is not None:
        try:
            parsed = _parse_llm_primer(llm_client.complete(
                system_prompt=_primer_system_prompt(),
                user_prompt=_primer_user_prompt(chunks),
            ))
            return parsed.model_copy(update={"content_hash": content_hash})
        except (RuntimeError, ValueError, ValidationError, json.JSONDecodeError):
            pass
    return _fallback_primer(chunks, content_hash)


def document_primer_content_hash(chunks: list[RagChunk]) -> str:
    """Hash the source chunk content to avoid unnecessary regeneration."""
    digest = hashlib.sha256()
    for chunk in sorted(chunks, key=lambda item: (item.page_index is None, item.page_index or 0, item.content_hash)):
        digest.update(chunk.content_hash.encode("utf-8"))
        digest.update(b"\0")
        digest.update(chunk.content.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def primer_to_row(primer: DocumentPrimer, chunks: list[RagChunk]) -> dict[str, Any]:
    """Convert a primer into the Supabase row shape."""
    source = chunks[0].source
    return {
        "user_id": source.user_id,
        "source_type": source.source_type,
        "source_id": source.source_id,
        "pdf_id": source.pdf_id,
        "title": primer.title,
        "summary": primer.summary,
        "main_topics": primer.main_topics,
        "key_terms": primer.key_terms,
        "learning_objectives": primer.learning_objectives,
        "page_ranges": primer.page_ranges,
        "content_hash": primer.content_hash,
    }


def _parse_llm_primer(raw: str) -> DocumentPrimer:
    data = json.loads(_extract_json_object(raw))
    return DocumentPrimer.model_validate({**data, "content_hash": ""})


def _fallback_primer(chunks: list[RagChunk], content_hash: str) -> DocumentPrimer:
    title = _source_title(chunks)
    headings = _heading_topics(chunks)
    terms = _frequent_terms(chunks)
    pages = _page_ranges(chunks)
    representative = _representative_chunks(chunks, limit=3)
    topic_phrase = ", ".join(headings[:4] or terms[:4] or ["die ausgewählten Inhalte"])
    summary_parts = [
        f"{title or 'Das Dokument'} behandelt {topic_phrase}.",
        "Wichtige Orientierungspunkte sind "
        + ", ".join((terms[:5] or headings[:5] or ["zentrale Begriffe"])[:5])
        + ".",
    ]
    if representative:
        summary_parts.append(f"Erster Kontext: {_clean_text(representative[0].content)[:180]}.")
    return DocumentPrimer(
        title=title,
        summary=" ".join(summary_parts)[:MAX_SUMMARY_CHARS],
        main_topics=headings[:MAX_LIST_ITEMS],
        key_terms=terms[:MAX_LIST_ITEMS],
        learning_objectives=[
            f"{item} in eigenen Worten erklären"
            for item in (headings[:3] or terms[:3])
        ],
        page_ranges=pages,
        content_hash=content_hash,
    )


def _primer_system_prompt() -> str:
    return (
        "Create a compact JSON document primer for realtime voice tutoring. "
        "Return only JSON with title, summary, main_topics, key_terms, "
        "learning_objectives, page_ranges. Do not copy long passages."
    )


def _primer_user_prompt(chunks: list[RagChunk]) -> str:
    lines = []
    for index, chunk in enumerate(_representative_chunks(chunks, MAX_LLM_CHUNKS), start=1):
        heading = " > ".join(chunk.heading_path)
        page = f"page {(chunk.page_index or 0) + 1}" if chunk.page_index is not None else "unknown page"
        lines.append(
            f"Chunk {index} ({page}; {heading or 'no heading'}): "
            f"{_clean_text(chunk.content)[:MAX_CHUNK_PROMPT_CHARS]}"
        )
    return "\n".join(lines)


def _extract_json_object(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("LLM primer response did not contain JSON.")
    return text[start : end + 1]


def _source_title(chunks: list[RagChunk]) -> str | None:
    for chunk in chunks:
        metadata = chunk.metadata or {}
        for key in ("filename", "title", "source_title"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return chunks[0].source.source_id if chunks else None


def _heading_topics(chunks: list[RagChunk]) -> list[str]:
    topics: list[str] = []
    for chunk in chunks:
        topics.extend(chunk.heading_path)
    return _dedupe([topic for topic in topics if topic])[:MAX_LIST_ITEMS]


def _frequent_terms(chunks: list[RagChunk]) -> list[str]:
    words = re.findall(r"[A-Za-zÄÖÜäöüß][A-Za-zÄÖÜäöüß-]{3,}", " ".join(chunk.content for chunk in chunks))
    counts = Counter(
        word.strip("-").lower()
        for word in words
        if word.lower() not in STOPWORDS
    )
    return [term for term, _count in counts.most_common(MAX_LIST_ITEMS)]


def _page_ranges(chunks: list[RagChunk]) -> list[dict[str, int]]:
    pages = sorted({chunk.page_index + 1 for chunk in chunks if chunk.page_index is not None})
    if not pages:
        return []
    return [{"start": pages[0], "end": pages[-1]}]


def _representative_chunks(chunks: list[RagChunk], limit: int) -> list[RagChunk]:
    return sorted(
        chunks,
        key=lambda chunk: (
            chunk.page_index is None,
            chunk.page_index if chunk.page_index is not None else 10**9,
            -len(chunk.heading_path),
        ),
    )[:limit]


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result
