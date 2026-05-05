"""Tests for RAG text normalization and hashing."""

from rag_pipeline.models import SourceRef
from rag_pipeline.refinement import SemanticRefiner
from rag_pipeline.text import (
    annotation_to_text,
    build_content_hash,
    normalize_content,
    tiptap_to_text,
)
from rag_pipeline.source_ingestion import chunks_from_annotation


def test_tiptap_to_text_extracts_nested_text() -> None:
    content = {
        "type": "doc",
        "content": [
            {
                "type": "heading",
                "content": [{"type": "text", "text": "Chapter 1"}],
            },
            {
                "type": "paragraph",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "text", "text": " world"},
                ],
            },
        ],
    }

    assert tiptap_to_text(content) == "Chapter 1\nHello world"


def test_annotation_to_text_keeps_quote_and_comment() -> None:
    text = annotation_to_text("Important quote", "My comment")

    assert text == "Quote: Important quote\nComment: My comment"


def test_content_hash_includes_context() -> None:
    source = SourceRef(
        user_id="user-1",
        source_type="pdf",
        source_id="source-1",
        pdf_id="source-1",
    )
    other_source = SourceRef(
        user_id="user-1",
        source_type="note",
        source_id="source-1",
        note_id="source-1",
    )

    first = build_content_hash("Same text", source, "strategy", "v1")
    second = build_content_hash("Same text", other_source, "strategy", "v1")

    assert first != second


def test_normalize_content_compacts_whitespace() -> None:
    assert normalize_content(" A   B \n\n\n C ") == "A B\n\nC"


def test_semantic_refinement_falls_back_without_openai_package() -> None:
    refiner = SemanticRefiner(openai_api_key="test", max_chars=30)
    content = (
        "First sentence. Second sentence. Third sentence. "
        "Fourth sentence."
    )

    chunks = refiner.refine(content)

    assert len(chunks) > 1
    assert all(len(chunk) <= 40 for chunk in chunks)


def test_refinement_without_openai_key_uses_fallback() -> None:
    refiner = SemanticRefiner(openai_api_key=None, max_chars=40)
    content = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."

    chunks = refiner.refine(content)

    assert chunks == ["First paragraph.\n\nSecond paragraph.", "Third paragraph."]


def test_refinement_enforces_max_chars_for_long_words() -> None:
    refiner = SemanticRefiner(openai_api_key=None, max_chars=10)

    chunks = refiner.refine("abcdefghijklmnopqrstuv")

    assert chunks == ["abcdefghij", "klmnopqrst", "uv"]


def test_table_chunks_skip_semantic_refinement() -> None:
    refiner = SemanticRefiner(openai_api_key="unused", max_chars=28)
    content = "Header | Value\nAlpha | One\nBeta | Two"

    chunks = refiner.refine(
        content,
        chunk_kind="table",
        metadata={"label": "table"},
    )

    assert chunks == ["Header | Value\nAlpha | One", "Beta | Two"]


def test_annotation_chunks_use_annotation_source_id() -> None:
    refiner = SemanticRefiner(openai_api_key="test", max_chars=200)

    chunks = chunks_from_annotation(
        user_id="user-1",
        annotation_id="annotation-1",
        pdf_id="pdf-1",
        quote="Highlighted text",
        comment="Comment text",
        metadata={"page_index": 2},
        refiner=refiner,
        chunking_strategy="strategy",
        chunking_version="v1",
    )

    assert chunks[0].source.source_type == "annotation_comment"
    assert chunks[0].source.source_id == "annotation-1"
    assert chunks[0].source.pdf_id == "pdf-1"
    assert chunks[0].page_index == 2
