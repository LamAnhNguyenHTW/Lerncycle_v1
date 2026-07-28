"""Tests for the corpus indexer (fixed-size path + stats + upsert wiring).

Docling-dependent strategies are not exercised here; they require Docling and
real PDFs and belong to the real (non-smoke) run.
"""

from __future__ import annotations

from rag_pipeline.eval.index_corpus import (
    FIXED_SIZE,
    STRATEGY_COLLECTIONS,
    CorpusDocument,
    build_fixed_size_chunks,
    compute_chunk_stats,
    index_corpus,
)


def _document() -> CorpusDocument:
    return CorpusDocument(
        source_id="pdf-1",
        pdf_path="unused-in-tests.pdf",
        language="de",
        source_type="pdf",
    )


def _page_texts() -> list[str]:
    page0 = " ".join(f"seite0wort{i}" for i in range(120))
    page1 = " ".join(f"seite1wort{i}" for i in range(60))
    return [page0, page1]


def test_fixed_size_chunks_preserve_page_attribution() -> None:
    chunks = build_fixed_size_chunks(
        _document(), max_chars=200, overlap_chars=40, page_texts=_page_texts()
    )

    assert chunks
    assert {chunk.page_index for chunk in chunks} == {0, 1}
    assert all(chunk.source.source_id == "pdf-1" for chunk in chunks)
    assert all(chunk.source.pdf_id == "pdf-1" for chunk in chunks)
    assert all(len(chunk.content) <= 200 for chunk in chunks)
    # chunk_index is monotonic across pages
    indices = [chunk.metadata["chunk_index"] for chunk in chunks]
    assert indices == sorted(indices)


def test_fixed_size_chunks_are_deterministic() -> None:
    first = build_fixed_size_chunks(_document(), page_texts=_page_texts())
    second = build_fixed_size_chunks(_document(), page_texts=_page_texts())

    assert [c.content_hash for c in first] == [c.content_hash for c in second]


def test_compute_chunk_stats() -> None:
    chunks = build_fixed_size_chunks(
        _document(), max_chars=200, overlap_chars=40, page_texts=_page_texts()
    )
    stats = compute_chunk_stats(chunks)

    lengths = [len(c.content) for c in chunks]
    assert stats.chunk_count == len(chunks)
    assert stats.min_length == min(lengths)
    assert stats.max_length == max(lengths)
    assert abs(stats.mean_length - sum(lengths) / len(lengths)) < 1e-9


def test_compute_chunk_stats_empty() -> None:
    stats = compute_chunk_stats([])
    assert stats.chunk_count == 0
    assert stats.mean_length == 0.0
    assert stats.min_length == 0
    assert stats.max_length == 0


class _FakeEmbedder:
    def __init__(self) -> None:
        self.dimension = 3
        self.embedded: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embedded.append(texts)
        return [[0.1, 0.2, 0.3] for _ in texts]


class _FakeSparse:
    def embed(self, texts: list[str]):
        return [object() for _ in texts]


class _FakeStore:
    def __init__(self) -> None:
        self.ensured = None
        self.upserted = None

    def ensure_collection(self, dim, sparse_enabled=False):
        self.ensured = (dim, sparse_enabled)

    def upsert_chunks(self, chunks, sparse_enabled=False):
        self.upserted = (chunks, sparse_enabled)


def test_index_corpus_embeds_and_upserts_into_named_collection(monkeypatch) -> None:
    document = _document()
    monkeypatch.setattr(
        "rag_pipeline.eval.index_corpus._read_pdf_pages",
        lambda _path: _page_texts(),
    )
    embedder = _FakeEmbedder()
    sparse = _FakeSparse()
    store = _FakeStore()

    result = index_corpus(
        FIXED_SIZE,
        [document],
        user_id="eval-user",
        embedder=embedder,
        sparse_embedder=sparse,
        store=store,
        fixed_max_chars=200,
        fixed_overlap_chars=40,
    )

    assert result.strategy == FIXED_SIZE
    assert result.collection == STRATEGY_COLLECTIONS[FIXED_SIZE]
    assert result.stats.chunk_count > 0
    assert result.config["fixed_max_chars"] == 200
    assert store.ensured == (3, True)
    payloads, sparse_enabled = store.upserted
    assert sparse_enabled is True
    assert len(payloads) == result.stats.chunk_count
    assert all(p["user_id"] == "eval-user" for p in payloads)
    assert all(p["text"] for p in payloads)
    # point ids are stable/unique
    ids = [p["id"] for p in payloads]
    assert len(ids) == len(set(ids))


def test_index_corpus_dry_run_skips_embedding(monkeypatch) -> None:
    monkeypatch.setattr(
        "rag_pipeline.eval.index_corpus._read_pdf_pages",
        lambda _path: _page_texts(),
    )
    result = index_corpus(
        FIXED_SIZE,
        [_document()],
        user_id="eval-user",
        dry_run=True,
        fixed_max_chars=200,
        fixed_overlap_chars=40,
    )

    assert result.dry_run is True
    assert result.stats.chunk_count > 0
    assert result.per_document[0]["source_id"] == "pdf-1"


def test_index_corpus_records_extraction_errors_without_raw_messages(
    monkeypatch,
) -> None:
    good = _document()
    bad = CorpusDocument(
        source_id="pdf-bad",
        pdf_path="secret-local-path.pdf",
        language="en",
    )

    def fake_build(strategy, document, **_kwargs):
        if document.source_id == "pdf-bad":
            raise RuntimeError("secret-local-path.pdf api_key=must-not-leak")
        return build_fixed_size_chunks(
            document, page_texts=["synthetic page text"], max_chars=200
        )

    monkeypatch.setattr("rag_pipeline.eval.index_corpus.build_chunks", fake_build)

    result = index_corpus(
        FIXED_SIZE,
        [good, bad],
        user_id="eval-user",
        dry_run=True,
        continue_on_extraction_error=True,
    )
    serialized = str(result.to_dict())

    assert result.documents_indexed == 1
    assert result.extraction_errors == [
        {
            "strategy": FIXED_SIZE,
            "source_id": "pdf-bad",
            "stage": "extraction",
            "error_type": "RuntimeError",
        }
    ]
    assert "must-not-leak" not in serialized
    assert "secret-local-path.pdf" not in serialized


def test_index_corpus_records_docling_coverage_diagnostics(monkeypatch) -> None:
    def fake_build(_strategy, document, **kwargs):
        kwargs["diagnostics"].update(
            {
                "total_pages": 3,
                "chunked_pages": [1, 2, 3],
                "missing_pages": [],
                "fallback_pages": [2],
                "retry_errors": {
                    "2:single_page_full": "local path and internal details"
                },
                "page_batch_size": 1,
            }
        )
        return build_fixed_size_chunks(
            document, page_texts=["synthetic page text"], max_chars=200
        )

    monkeypatch.setattr("rag_pipeline.eval.index_corpus.build_chunks", fake_build)

    result = index_corpus(
        FIXED_SIZE,
        [_document()],
        user_id="eval-user",
        dry_run=True,
    )
    serialized = str(result.to_dict())

    assert result.extraction_diagnostics == [
        {
            "strategy": FIXED_SIZE,
            "source_id": "pdf-1",
            "total_pages": 3,
            "chunked_pages": [1, 2, 3],
            "missing_pages": [],
            "fallback_pages": [2],
            "retry_error_keys": ["2:single_page_full"],
            "page_batch_size": 1,
        }
    ]
    assert "local path and internal details" not in serialized
