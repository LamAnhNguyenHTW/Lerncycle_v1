"""Tests for the reproducible three-collection pilot runner."""

from __future__ import annotations

import json

from rag_pipeline.config import WorkerConfig
from rag_pipeline.eval.ground_truth import parse_query
from rag_pipeline.eval.index_corpus import STRATEGIES, STRATEGY_COLLECTIONS
from rag_pipeline.eval.run_pilot import run_pilot_retrieval, write_pilot_report


def _queries():
    return [
        parse_query(
            {
                "query_id": "q1",
                "question": "answerable",
                "language": "de",
                "question_type": "fact",
                "expected_source_ids": ["pdf-1"],
            }
        ),
        parse_query(
            {
                "query_id": "q2",
                "question": "not in corpus",
                "language": "de",
                "question_type": "unanswerable",
                "expected_source_ids": [],
                "expected_pages": [],
                "expected_phrases": [],
            }
        ),
    ]


def test_run_pilot_uses_same_hybrid_no_rerank_config_for_all_collections(
    tmp_path,
) -> None:
    config = WorkerConfig(
        supabase_url="https://example.test",
        supabase_service_role_key="must-not-leak",
        qdrant_url="http://localhost:6333",
        qdrant_api_key="must-not-leak",
        openai_api_key="must-not-leak",
        query_embedding_cache_enabled=True,
        retrieval_result_cache_enabled=True,
        reranker_cache_enabled=True,
        reranking_enabled=True,
    )
    calls = []

    def fake_hybrid(question, **kwargs):
        calls.append((question, kwargs))
        return [{"source_id": "pdf-1", "page_index": 0}]

    report = run_pilot_retrieval(
        _queries(),
        base_config=config,
        user_id="eval-pilot-v1",
        top_k=5,
        embedder=object(),
        sparse_embedder=object(),
        store_factory=lambda collection: {"collection": collection},
        hybrid_fn=fake_hybrid,
        extraction_errors=[
            {
                "strategy": "docling",
                "source_id": "pdf-2",
                "stage": "extraction",
                "error_type": "RuntimeError",
            }
        ],
        extraction_diagnostics=[
            {
                "strategy": "docling",
                "source_id": "pdf-2",
                "fallback_pages": [2],
                "missing_pages": [],
            }
        ],
    )
    output = write_pilot_report(report, tmp_path / "pilot.json")
    serialized = output.read_text(encoding="utf-8")
    loaded = json.loads(serialized)

    assert loaded["query_count"] == 2
    assert loaded["metric_evaluable_queries"] == 1
    assert loaded["negative_control_queries"] == 1
    assert loaded["interpretation_status"] == "pilot_only_no_strategy_decision"
    assert set(loaded["runs"]) == set(STRATEGIES)
    assert loaded["retrieval_configuration"]["mode"] == "hybrid"
    assert loaded["retrieval_configuration"]["reranking_enabled"] is False
    assert loaded["retrieval_configuration"]["caches_disabled"] is True
    assert loaded["extraction_errors"][0]["source_id"] == "pdf-2"
    assert loaded["extraction_diagnostics"][0]["fallback_pages"] == [2]
    assert len(calls) == len(STRATEGIES) * 2
    for strategy, run in loaded["runs"].items():
        assert run["collection"] == STRATEGY_COLLECTIONS[strategy]
        assert run["modes"]["hybrid"]["aggregate"]["total_queries"] == 2
    for _question, kwargs in calls:
        assert kwargs["config"].query_embedding_cache_enabled is False
        assert kwargs["config"].retrieval_result_cache_enabled is False
        assert kwargs["config"].reranking_enabled is False
    assert "must-not-leak" not in serialized
