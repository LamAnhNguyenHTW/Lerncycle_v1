"""Tests for labeled-query evaluation and JSON report export."""

from __future__ import annotations

import json

import pytest

from rag_pipeline.eval.ground_truth import parse_query
from rag_pipeline.evaluate_retrieval import (
    build_eval_report,
    eval_config_with_caches_disabled,
    evaluate_labeled_queries,
    write_report,
)


def _queries():
    return [
        parse_query(
            {
                "query_id": "q1",
                "question": "Was ist Process Mining?",
                "language": "de",
                "question_type": "factual",
                "expected_source_ids": ["pdf-1"],
            }
        ),
        parse_query(
            {
                "query_id": "q3",
                "question": "Welche Information fehlt im Korpus?",
                "language": "de",
                "question_type": "unanswerable",
                "expected_source_ids": [],
                "expected_pages": [],
                "expected_phrases": [],
                "reference_answer": "Diese Information ist im Korpus nicht enthalten.",
            }
        ),
        parse_query(
            {
                "query_id": "q2",
                "question": "How does BM25 work?",
                "language": "en",
                "question_type": "factual",
                "expected_source_ids": ["pdf-2"],
            }
        ),
    ]


def test_evaluate_labeled_queries_structure_and_breakdowns() -> None:
    searchers = {
        "hybrid": lambda question: (
            [{"source_id": "pdf-1"}] if "Process" in question else [{"source_id": "x"}]
        ),
    }

    results = evaluate_labeled_queries(_queries(), searchers, k=5)

    hybrid = results["hybrid"]
    assert set(hybrid) == {
        "aggregate",
        "by_language",
        "by_question_type",
        "per_query",
        "negative_controls",
    }
    assert hybrid["aggregate"]["evaluated_queries"] == 2
    assert hybrid["aggregate"]["total_queries"] == 3
    assert hybrid["aggregate"]["negative_control_queries"] == 1
    assert hybrid["aggregate"]["hit_at_1"] == 0.5
    assert "latency_ms" in hybrid["aggregate"]
    assert hybrid["by_language"]["de"]["hit_at_1"] == 1.0
    assert hybrid["by_language"]["en"]["hit_at_1"] == 0.0
    assert len(hybrid["per_query"]) == 2
    assert all("latency_ms" in row for row in hybrid["per_query"])
    assert hybrid["per_query"][0]["retrieved_hits"] == [
        {
            "rank": 1,
            "source_id": "pdf-1",
            "page": None,
            "chunk_id": None,
            "score": None,
        }
    ]
    assert len(hybrid["negative_controls"]) == 1
    negative = hybrid["negative_controls"][0]
    assert negative["query_id"] == "q3"
    assert negative["language"] == "de"
    assert negative["question_type"] == "unanswerable"
    assert negative["num_retrieved"] == 1
    assert negative["returned_any"] is True
    assert negative["metric_status"] == "not_applicable_no_relevance_labels"
    assert "latency_ms" in negative
    assert negative["retrieved_hits"][0]["rank"] == 1


def test_evaluate_labeled_queries_without_latency() -> None:
    searchers = {"dense": lambda _q: [{"source_id": "pdf-1"}]}

    results = evaluate_labeled_queries(_queries(), searchers, k=5, capture_latency=False)

    assert "latency_ms" not in results["dense"]["aggregate"]
    assert all("latency_ms" not in row for row in results["dense"]["per_query"])
    assert all(
        "latency_ms" not in row for row in results["dense"]["negative_controls"]
    )


def test_build_and_write_report_roundtrip(tmp_path) -> None:
    searchers = {"hybrid": lambda _q: [{"source_id": "pdf-1"}]}
    modes = evaluate_labeled_queries(_queries(), searchers, k=5)
    report = build_eval_report(
        mode_results=modes,
        strategy="docling_semantic",
        collection="eval_docling_semantic",
        config={"embedding_model": "text-embedding-3-small", "top_k": 5, "caches_disabled": True},
        k=5,
        ground_truth_path="rag_pipeline/eval/example_queries.json",
    )

    out = write_report(report, tmp_path / "report.json")
    loaded = json.loads(out.read_text(encoding="utf-8"))

    assert loaded["strategy"] == "docling_semantic"
    assert loaded["collection"] == "eval_docling_semantic"
    assert loaded["k"] == 5
    assert loaded["config"]["caches_disabled"] is True
    assert "modes" in loaded and "hybrid" in loaded["modes"]
    assert "generated_at" in loaded


def test_report_contains_no_secret_keys(tmp_path) -> None:
    # Reports must never carry credentials; build_eval_report writes config as-is,
    # so callers pass only non-secret settings. This guards the example path.
    searchers = {"hybrid": lambda _q: [{"source_id": "pdf-1"}]}
    modes = evaluate_labeled_queries(_queries(), searchers, k=5)
    report = build_eval_report(
        mode_results=modes,
        strategy="fixed_size",
        collection="eval_fixed_size",
        config={"embedding_model": "text-embedding-3-small", "top_k": 5},
    )
    serialized = json.dumps(report).lower()

    for forbidden in ("api_key", "service_role", "password", "secret", "qdrant_api"):
        assert forbidden not in serialized


@pytest.mark.parametrize(
    'config',
    [
        {'openai_api_key': 'must-not-be-written'},
        {'nested': {'service_role_key': 'must-not-be-written'}},
        {'qdrant_url': 'https://user:must-not-be-written@example.test'},
    ],
)
def test_report_rejects_secret_configuration(config) -> None:
    with pytest.raises(ValueError, match='non-secret') as exc_info:
        build_eval_report(
            mode_results={},
            strategy='fixed_size',
            collection='eval_fixed_size',
            config=config,
        )

    assert 'must-not-be-written' not in str(exc_info.value)


def test_eval_config_disables_caches() -> None:
    from types import SimpleNamespace
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class _Cfg:
        query_embedding_cache_enabled: bool = True
        retrieval_result_cache_enabled: bool = True
        reranker_cache_enabled: bool = True
        other: str = "keep"

    disabled = eval_config_with_caches_disabled(_Cfg())

    assert disabled.query_embedding_cache_enabled is False
    assert disabled.retrieval_result_cache_enabled is False
    assert disabled.reranker_cache_enabled is False
    assert disabled.other == "keep"
