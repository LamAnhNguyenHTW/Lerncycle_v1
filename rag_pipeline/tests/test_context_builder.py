from __future__ import annotations

from rag_pipeline.context_builder import build_rag_context, normalize_source


def _result(**overrides):
    base = {
        "chunk_id": "chunk-1",
        "text": "Process Mining ist eine Datenanalysetechnologie.",
        "score": 0.83,
        "source_type": "pdf",
        "source_id": "source-1",
        "page_index": 9,
        "title": "GPAA_SoSe2026_2.pdf",
        "heading": "Definition",
        "metadata": {"filename": "GPAA_SoSe2026_2.pdf"},
    }
    base.update(overrides)
    return base


def test_context_builder_formats_pdf_sources() -> None:
    context = build_rag_context([_result()])

    assert "Type: PDF" in context["context_text"]
    assert "File: GPAA_SoSe2026_2.pdf" in context["context_text"]
    assert context["sources"][0]["metadata"] == {"filename": "GPAA_SoSe2026_2.pdf"}


def test_context_builder_formats_note_sources() -> None:
    context = build_rag_context(
        [_result(source_type="note", title="My note", page_index=None, metadata={})]
    )

    assert "Type: Note" in context["context_text"]
    assert "Title: My note" in context["context_text"]
    assert context["sources"][0]["page"] is None


def test_context_builder_formats_annotation_sources() -> None:
    context = build_rag_context(
        [
            _result(
                source_type="annotation_comment",
                title=None,
                text="Quote: Process Mining.\nComment: Wichtig",
                metadata={},
            )
        ]
    )

    assert "Type: Annotation" in context["context_text"]
    assert "Quote: Process Mining." in context["context_text"]
    assert context["sources"][0]["title"] == "Annotation"


def test_context_builder_converts_page_index_to_display_page() -> None:
    context = build_rag_context([_result(page_index=9)])

    assert "Page: 10" in context["context_text"]
    assert context["sources"][0]["page"] == 10
    assert "page_index" not in context["sources"][0]


def test_context_builder_deduplicates_chunk_ids() -> None:
    context = build_rag_context([_result(), _result(text="duplicate")])

    assert len(context["sources"]) == 1
    assert context["context_text"].count("[Source 1]") == 1


def test_context_builder_respects_max_chunks() -> None:
    results = [_result(chunk_id=f"chunk-{idx}", source_id=f"source-{idx}") for idx in range(3)]

    context = build_rag_context(results, max_chunks=2)

    assert [source["chunk_id"] for source in context["sources"]] == ["chunk-0", "chunk-1"]


def test_context_builder_respects_max_chars() -> None:
    results = [
        _result(chunk_id="chunk-1", source_id="source-1", text="A" * 80),
        _result(chunk_id="chunk-2", source_id="source-2", text="B" * 80),
    ]

    context = build_rag_context(results, max_chars=170)

    assert "A" * 20 in context["context_text"]
    assert "B" * 20 not in context["context_text"]
    assert [source["chunk_id"] for source in context["sources"]] == ["chunk-1"]


def test_context_builder_omits_large_docling_metadata() -> None:
    context = build_rag_context(
        [
            _result(
                metadata={
                    "filename": "file.pdf",
                    "bbox": [1, 2, 3],
                    "charspan": [0, 2],
                    "highlight_areas": [{"x": 1}],
                    "doc_items": [{"raw": True}],
                }
            )
        ]
    )

    text = context["context_text"]
    assert "bbox" not in text
    assert "charspan" not in text
    assert "highlight_areas" not in text
    assert "doc_items" not in text
    assert context["sources"][0]["metadata"] == {"filename": "GPAA_SoSe2026_2.pdf"}


def test_normalize_source_pdf() -> None:
    source = normalize_source(_result())

    assert source == {
        "chunk_id": "chunk-1",
        "source_type": "pdf",
        "source_id": "source-1",
        "title": "GPAA_SoSe2026_2.pdf",
        "heading": "Definition",
        "page": 10,
        "score": 0.83,
        "snippet": "Process Mining ist eine Datenanalysetechnologie.",
        "metadata": {"filename": "GPAA_SoSe2026_2.pdf"},
    }


def test_normalize_source_pdf_uses_origin_filename() -> None:
    source = normalize_source(_result(title=None, metadata={"origin": {"filename": "GPAA.pdf"}}))

    assert source["title"] == "GPAA.pdf"
    assert source["metadata"] == {"filename": "GPAA.pdf"}


def test_normalize_source_pdf_uses_direct_metadata_filename() -> None:
    source = normalize_source(_result(title=None, metadata={"filename": "direct.pdf"}))

    assert source["title"] == "direct.pdf"
    assert source["metadata"] == {"filename": "direct.pdf"}


def test_normalize_source_pdf_falls_back_to_pdf_title_when_no_filename() -> None:
    source = normalize_source(_result(title=None, metadata={}))

    assert source["title"] == "PDF"
    assert source["metadata"] == {"filename": "PDF"}


def test_normalize_source_pdf_does_not_return_untitled_when_filename_exists() -> None:
    source = normalize_source(_result(title=None, metadata={"origin": {"filename": "GPAA.pdf"}}))

    assert source["title"] is not None
    assert source["title"] != "Untitled source"


def test_normalize_source_annotation_keeps_annotation_title_and_filename_metadata() -> None:
    source = normalize_source(
        _result(source_type="annotation_comment", metadata={"origin": {"filename": "GPAA.pdf"}})
    )

    assert source["title"] == "Annotation"
    assert source["metadata"] == {"filename": "GPAA.pdf"}


def test_normalize_source_annotation_no_filename_has_empty_metadata() -> None:
    source = normalize_source(_result(source_type="annotation_comment", metadata={}))

    assert source["title"] == "Annotation"
    assert source["metadata"] == {}


def test_normalize_source_note_uses_note_title_fallback() -> None:
    source = normalize_source(_result(source_type="note", title=None, metadata={"title": "My notes"}))

    assert source["title"] == "My notes"
    assert source["metadata"] == {}


def test_normalize_source_note_falls_back_to_note_label() -> None:
    source = normalize_source(_result(source_type="note", title=None, metadata={}))

    assert source["title"] == "Note"
    assert source["metadata"] == {}


def test_normalize_source_note() -> None:
    source = normalize_source(
        _result(source_type="note", title="Note", page_index=None, metadata={"filename": "x"})
    )

    assert source["page"] is None
    assert source["metadata"] == {}


def test_normalize_source_annotation_comment() -> None:
    source = normalize_source(_result(source_type="annotation_comment", title="Ignored"))

    assert source["title"] == "Annotation"
    assert source["metadata"] == {"filename": "GPAA_SoSe2026_2.pdf"}


def test_normalize_source_chat_memory() -> None:
    source = normalize_source(
        _result(
            source_type="chat_memory",
            source_id="session-1",
            title="Ignored",
            heading="Learning history",
            text="We discussed Process Mining.",
            metadata={
                "session_id": "session-1",
                "memory_kind": "rolling_summary",
                "raw_messages": ["secret"],
            },
        )
    )

    assert source["title"] == "Chat Memory"
    assert source["page"] is None
    assert source["heading"] == "Learning history"
    assert source["metadata"] == {
        "session_id": "session-1",
        "memory_kind": "rolling_summary",
    }


def test_chat_memory_snippet_is_truncated_to_200_chars() -> None:
    source = normalize_source(_result(source_type="chat_memory", text="x" * 250))

    assert len(source["snippet"]) <= 200
    assert source["snippet"].endswith("...")


def test_source_shape_does_not_include_large_metadata() -> None:
    source = normalize_source(_result(metadata={"filename": "file.pdf", "bbox": [1]}))

    assert "bbox" not in source["metadata"]
    assert set(source) == {
        "chunk_id",
        "source_type",
        "source_id",
        "title",
        "heading",
        "page",
        "score",
        "snippet",
        "metadata",
    }


def test_source_shape_keeps_chunk_id_and_source_id() -> None:
    source = normalize_source(
        _result(title=None, heading=None, score=None, page_index=None, metadata={})
    )

    assert source["chunk_id"] == "chunk-1"
    assert source["source_id"] == "source-1"
    assert source["title"] == "PDF"
    assert source["heading"] is None
