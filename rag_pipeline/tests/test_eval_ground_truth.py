"""Tests for the evaluation ground-truth schema and loader."""

from __future__ import annotations

import json
import hashlib
from dataclasses import replace

import pytest

from rag_pipeline.eval.ground_truth import (
    GroundTruthQuery,
    is_hit_relevant,
    load_ground_truth,
    parse_query,
    validate_pilot_ground_truth,
)


def _valid_raw(**overrides):
    data = {
        "query_id": "q1",
        "question": "Was ist X?",
        "language": "de",
        "question_type": "factual",
        "expected_pages": [3],
    }
    data.update(overrides)
    return data


def test_parse_valid_query() -> None:
    query = parse_query(_valid_raw(expected_phrases=["X"], reference_answer="X ist Y"))

    assert query.query_id == "q1"
    assert query.expected_pages == [3]
    assert query.expected_phrases == ["X"]
    assert query.reference_answer == "X ist Y"
    assert query.has_relevance_labels()


def test_missing_required_field_raises() -> None:
    with pytest.raises(ValueError):
        parse_query({"query_id": "q1", "language": "de", "question_type": "factual"})


def test_invalid_language_raises() -> None:
    with pytest.raises(ValueError):
        parse_query(_valid_raw(language="fr"))


def test_invalid_question_type_raises() -> None:
    with pytest.raises(ValueError):
        parse_query(_valid_raw(question_type="opinion"))


def test_query_without_relevance_labels_raises() -> None:
    with pytest.raises(ValueError):
        parse_query(
            {
                "query_id": "q1",
                "question": "Was ist X?",
                "language": "de",
                "question_type": "factual",
            }
        )


def test_one_based_pages_must_be_positive() -> None:
    with pytest.raises(ValueError):
        parse_query(_valid_raw(expected_pages=[0]))


def test_load_ground_truth_and_duplicate_detection(tmp_path) -> None:
    good = tmp_path / "queries.json"
    good.write_text(
        json.dumps({"queries": [_valid_raw(), _valid_raw(query_id="q2")]}),
        encoding="utf-8",
    )
    queries = load_ground_truth(good)
    assert [q.query_id for q in queries] == ["q1", "q2"]

    dup = tmp_path / "dup.json"
    dup.write_text(json.dumps([_valid_raw(), _valid_raw()]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_ground_truth(dup)


def test_is_hit_relevant_by_source_id() -> None:
    query = parse_query(_valid_raw(expected_pages=[], expected_source_ids=["pdf-1"]))
    assert is_hit_relevant({"source_id": "pdf-1"}, query)
    assert not is_hit_relevant({"source_id": "pdf-2"}, query)


def test_is_hit_relevant_by_page_offset() -> None:
    query = parse_query(_valid_raw(expected_pages=[3]))
    # 1-based label 3 matches 0-based page_index 2.
    assert is_hit_relevant({"page_index": 2}, query)
    assert not is_hit_relevant({"page_index": 5}, query)


def test_is_hit_relevant_by_phrase_case_insensitive() -> None:
    query = parse_query(_valid_raw(expected_pages=[], expected_phrases=["Process Mining"]))
    assert is_hit_relevant({"text": "a short process mining note"}, query)
    assert not is_hit_relevant({"text": "unrelated"}, query)


def test_example_queries_file_is_valid() -> None:
    from pathlib import Path

    example = (
        Path(__file__).resolve().parents[1] / "eval" / "example_queries.json"
    )
    queries = load_ground_truth(example)
    assert len(queries) >= 5
    assert all(isinstance(q, GroundTruthQuery) for q in queries)


PILOT_FIELDS = {
    "query_id": "pilot-q1",
    "question": "Was ist X?",
    "language": "de",
    "question_type": "fact",
    "expected_source_ids": ["source-1"],
    "expected_pages": [2],
    "expected_phrases": ["X"],
    "reference_answer": "X ist im Dokument auf Seite 2 definiert.",
    "relevance_notes": "Quelle und Seite sind eindeutig.",
    "difficulty": "easy",
    "requires_multiple_chunks": False,
    "requires_multiple_documents": False,
    "extraction_risk": False,
    "manual_review_required": False,
}


def _write_pilot_files(tmp_path, query_overrides=None, *, extra_pdf=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "source.pdf").write_bytes(b"pdf fixture")
    if extra_pdf:
        (tmp_path / "unknown.pdf").write_bytes(b"pdf fixture")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "source_id": "source-1",
                        "pdf_path": "source.pdf",
                        "title": "Source",
                        "language": "de",
                        "source_type": "pdf",
                        "page_count": 3,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    query = dict(PILOT_FIELDS)
    query.update(query_overrides or {})
    queries = tmp_path / "queries.json"
    queries.write_text(json.dumps({"queries": [query]}), encoding="utf-8")
    return queries, manifest


def test_validate_pilot_ground_truth_accepts_complete_query(tmp_path) -> None:
    queries, manifest = _write_pilot_files(tmp_path)

    loaded = validate_pilot_ground_truth(
        queries, manifest, verify_pdf_page_counts=False
    )

    assert loaded[0].difficulty == "easy"
    assert loaded[0].relevance_notes == "Quelle und Seite sind eindeutig."


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"question_type": "factual"}, "question_type"),
        ({"difficulty": "extreme"}, "difficulty"),
        ({"expected_source_ids": ["unknown"]}, "unknown source_id"),
        ({"expected_pages": [4]}, "outside"),
        ({"reference_answer": ""}, "reference_answer"),
    ],
)
def test_validate_pilot_ground_truth_rejects_invalid_fields(
    tmp_path, overrides, message
) -> None:
    queries, manifest = _write_pilot_files(tmp_path, overrides)

    with pytest.raises(ValueError, match=message):
        validate_pilot_ground_truth(
            queries, manifest, verify_pdf_page_counts=False
        )


def test_validate_pilot_ground_truth_requires_every_pilot_field(tmp_path) -> None:
    queries, manifest = _write_pilot_files(tmp_path)
    raw = json.loads(queries.read_text(encoding="utf-8"))
    del raw["queries"][0]["extraction_risk"]
    queries.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="extraction_risk"):
        validate_pilot_ground_truth(
            queries, manifest, verify_pdf_page_counts=False
        )


def test_unanswerable_query_requires_empty_relevance_labels(tmp_path) -> None:
    overrides = {
        "question_type": "unanswerable",
        "expected_source_ids": [],
        "expected_pages": [],
        "expected_phrases": [],
        "reference_answer": "Diese Information ist im Korpus nicht enthalten.",
    }
    queries, manifest = _write_pilot_files(tmp_path, overrides)
    loaded = validate_pilot_ground_truth(
        queries, manifest, verify_pdf_page_counts=False
    )
    assert not loaded[0].has_relevance_labels()

    invalid_queries, invalid_manifest = _write_pilot_files(
        tmp_path / "invalid",
        {**overrides, "expected_pages": [1]},
    )
    with pytest.raises(ValueError, match="unanswerable"):
        validate_pilot_ground_truth(
            invalid_queries,
            invalid_manifest,
            verify_pdf_page_counts=False,
        )


def test_manifest_rejects_absolute_paths_and_unknown_pdf_files(tmp_path) -> None:
    queries, manifest = _write_pilot_files(tmp_path, extra_pdf=True)
    with pytest.raises(ValueError, match="unknown PDF"):
        validate_pilot_ground_truth(
            queries, manifest, verify_pdf_page_counts=False
        )

    raw = json.loads(manifest.read_text(encoding="utf-8"))
    raw["documents"][0]["pdf_path"] = str((tmp_path / "source.pdf").resolve())
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    (tmp_path / "unknown.pdf").unlink()
    with pytest.raises(ValueError, match="absolute local path|relative"):
        validate_pilot_ground_truth(
            queries, manifest, verify_pdf_page_counts=False
        )


def test_pilot_validation_rejects_secret_keys(tmp_path) -> None:
    queries, manifest = _write_pilot_files(tmp_path)
    raw = json.loads(queries.read_text(encoding="utf-8"))
    raw["api_key"] = "must-not-appear-in-error"
    queries.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="secret") as exc_info:
        validate_pilot_ground_truth(
            queries, manifest, verify_pdf_page_counts=False
        )
    assert "must-not-appear-in-error" not in str(exc_info.value)


def test_repository_pilot_set_has_required_distribution() -> None:
    from collections import Counter
    from pathlib import Path

    eval_dir = Path(__file__).resolve().parents[1] / "eval"
    queries = validate_pilot_ground_truth(
        eval_dir / "queries_pilot.json",
        eval_dir / "corpus" / "manifest.json",
    )

    assert len(queries) == 15
    source_counts = Counter(
        query.expected_source_ids[0]
        if query.expected_source_ids
        else "unanswerable"
        for query in queries
    )
    assert source_counts == {
        "gpaa-geschaeftsprozessmanagement": 3,
        "pwl-logistik-wise-2025": 3,
        "verteilte-anwendungen-http": 3,
        "se-prozess-uebersicht-2023": 1,
        "self-rag-iclr-2024": 4,
        "unanswerable": 1,
    }
    assert any(query.question_type == "unanswerable" for query in queries)
    assert any(query.requires_multiple_chunks for query in queries)
    assert any(len(query.expected_pages) > 1 for query in queries)
    assert any(query.question_type == "relational" for query in queries)
    assert any(query.question_type == "table_or_figure" for query in queries)
    assert any(query.language == "en" for query in queries)
    risk_ids = {query.query_id for query in queries if query.extraction_risk}
    assert len(risk_ids) <= 5
    assert {
        "pilot-pwl-03",
        "pilot-http-02",
        "pilot-selfrag-02",
        "pilot-selfrag-03",
    } <= risk_ids


def test_repository_review_lists_every_pilot_query() -> None:
    from pathlib import Path

    eval_dir = Path(__file__).resolve().parents[1] / "eval"
    queries = validate_pilot_ground_truth(
        eval_dir / "queries_pilot.json",
        eval_dir / "corpus" / "manifest.json",
    )
    review = (eval_dir / "GROUND_TRUTH_REVIEW.md").read_text(encoding="utf-8")
    review_ids = {
        line.removeprefix("## ").strip()
        for line in review.splitlines()
        if line.startswith("## pilot-")
    }

    assert review_ids == {query.query_id for query in queries}
    for query in queries:
        pages = (
            ", ".join(str(page) for page in query.expected_pages)
            if query.expected_pages
            else "keine"
        )
        assert f"- **Frage:** {query.question}" in review
        assert f"- **Seiten:** {pages}" in review
        assert f"- **Referenzantwort:** {query.reference_answer}" in review
        assert (
            f"- **extraction_risk:** `{str(query.extraction_risk).lower()}`"
            in review
        )
        assert (
            "- **manual_review_required:** "
            f"`{str(query.manual_review_required).lower()}`"
            in review
        )


def test_repository_final_set_has_target_distribution_and_preserves_pilot() -> None:
    from collections import Counter
    from pathlib import Path

    eval_dir = Path(__file__).resolve().parents[1] / "eval"
    final_queries = validate_pilot_ground_truth(
        eval_dir / "queries_final.json",
        eval_dir / "corpus" / "manifest.json",
    )
    pilot_queries = validate_pilot_ground_truth(
        eval_dir / "queries_pilot.json",
        eval_dir / "corpus" / "manifest.json",
    )

    assert len(final_queries) == 60
    assert Counter(query.language for query in final_queries) == {"de": 45, "en": 15}
    assert Counter(query.question_type for query in final_queries) == {
        "fact": 10,
        "semantic_paraphrase": 10,
        "terminology": 8,
        "contextual": 8,
        "relational": 8,
        "multi_hop": 8,
        "table_or_figure": 5,
        "unanswerable": 3,
    }
    assert Counter(
        query.expected_source_ids[0]
        if query.expected_source_ids
        else "unanswerable"
        for query in final_queries
    ) == {
        "gpaa-geschaeftsprozessmanagement": 11,
        "pwl-logistik-wise-2025": 14,
        "verteilte-anwendungen-http": 12,
        "se-prozess-uebersicht-2023": 8,
        "self-rag-iclr-2024": 12,
        "unanswerable": 3,
    }
    assert Counter(
        query.query_id.split("-")[1] for query in final_queries
    ) == {
        "gpm": 12,
        "pwl": 15,
        "http": 12,
        "se": 9,
        "selfrag": 12,
    }

    final_by_id = {query.query_id: query for query in final_queries}
    assert all(
        replace(
            final_by_id[query.query_id],
            manual_review_required=query.manual_review_required,
        ) == query
        for query in pilot_queries
    )
    assert sum(bool(query.manual_review_required) for query in final_queries) == 0
    assert sum(query.extraction_risk for query in final_queries) <= 5


def test_repository_final_review_lists_every_query() -> None:
    from pathlib import Path

    eval_dir = Path(__file__).resolve().parents[1] / "eval"
    queries = validate_pilot_ground_truth(
        eval_dir / "queries_final.json",
        eval_dir / "corpus" / "manifest.json",
    )
    review = (eval_dir / "GROUND_TRUTH_FINAL_REVIEW.md").read_text(
        encoding="utf-8"
    )
    review_ids = {
        line.removeprefix("## ").strip()
        for line in review.splitlines()
        if line.startswith(("## pilot-", "## final-"))
    }

    assert review_ids == {query.query_id for query in queries}
    for query in queries:
        assert f"- **Frage:** {query.question}" in review
        assert f"- **Referenzantwort:** {query.reference_answer}" in review
    assert "- [ ]" not in review
    assert review.count("- **manual_review_required:** `false`") == 60


def test_repository_final_ground_truth_freeze_manifest_matches_files() -> None:
    from pathlib import Path

    eval_dir = Path(__file__).resolve().parents[1] / "eval"
    freeze = json.loads(
        (eval_dir / "ground_truth_freeze.json").read_text(encoding="utf-8")
    )
    query_bytes = (eval_dir / freeze["query_file"]).read_bytes()
    review_bytes = (eval_dir / freeze["review_file"]).read_bytes()

    assert freeze["freeze_id"] == "ground-truth-final-v1"
    assert freeze["status"] == "frozen"
    assert freeze["query_count"] == 60
    assert freeze["query_sha256"] == hashlib.sha256(query_bytes).hexdigest()
    assert freeze["review_sha256"] == hashlib.sha256(review_bytes).hexdigest()
