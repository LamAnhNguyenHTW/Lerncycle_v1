"""Repeated retrieval evaluation with warm-up and deterministic order rotation."""

from __future__ import annotations

import statistics
from typing import Any

from rag_pipeline.eval.ground_truth import GroundTruthQuery
from rag_pipeline.eval.metrics import percentile
from rag_pipeline.evaluate_retrieval import SearchFn, evaluate_labeled_queries


METRIC_KEYS = (
    "hit_at_1",
    "hit_at_3",
    "hit_at_5",
    "mrr",
    "precision_at_5",
    "recall_at_5",
    "ndcg_at_5",
)


def _rotate(items: list[Any], offset: int) -> list[Any]:
    if not items:
        return []
    position = offset % len(items)
    return items[position:] + items[:position]


def _series(values: list[float]) -> dict[str, float | int]:
    return {
        "mean": statistics.fmean(values) if values else 0.0,
        "stddev": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
        "count": len(values),
    }


def _latency_summary(values: list[float]) -> dict[str, float | int]:
    summary = _series(values)
    summary["p50"] = percentile(values, 50)
    summary["p95"] = percentile(values, 95)
    return summary


def _breakdown_summary(
    mode_results: list[dict[str, Any]],
    field: str,
) -> dict[str, dict[str, dict[str, float | int]]]:
    groups = sorted(
        {
            group
            for result in mode_results
            for group in result.get(field, {})
        }
    )
    return {
        group: {
            metric: _series(
                [
                    float(result[field][group].get(metric, 0.0))
                    for result in mode_results
                    if group in result.get(field, {})
                ]
            )
            for metric in METRIC_KEYS
        }
        for group in groups
    }


def run_repeated_evaluation(
    queries: list[GroundTruthQuery],
    searchers: dict[str, SearchFn],
    *,
    repetitions: int = 3,
    k: int = 5,
    warmup: bool = True,
) -> dict[str, Any]:
    """Run variants repeatedly with rotated variant and query order.

    One warm-up request per variant is excluded from all metrics. Retrieval
    caches must be disabled by the caller. Raw per-repetition results remain in
    the report for page-level audits; summary metrics contain mean and standard
    deviation across repetitions.
    """
    if repetitions < 1:
        raise ValueError("repetitions must be at least 1")
    if not queries:
        raise ValueError("queries must not be empty")
    if not searchers:
        raise ValueError("searchers must not be empty")

    labels = list(searchers)
    if warmup:
        warmup_question = queries[0].question
        for label in labels:
            searchers[label](warmup_question)

    runs: list[dict[str, Any]] = []
    collected: dict[str, list[dict[str, Any]]] = {label: [] for label in labels}
    for repetition in range(repetitions):
        variant_order = _rotate(labels, repetition)
        query_order = _rotate(queries, repetition)
        run_results: dict[str, Any] = {}
        for label in variant_order:
            result = evaluate_labeled_queries(
                query_order,
                {label: searchers[label]},
                k=k,
                capture_latency=True,
            )[label]
            run_results[label] = result
            collected[label].append(result)
        runs.append(
            {
                "repetition": repetition + 1,
                "variant_order": variant_order,
                "query_order": [query.query_id for query in query_order],
                "results": run_results,
            }
        )

    variants: dict[str, Any] = {}
    for label, results in collected.items():
        latencies = [
            float(row["latency_ms"])
            for result in results
            for row in result["per_query"] + result["negative_controls"]
            if "latency_ms" in row
        ]
        variants[label] = {
            "metrics": {
                metric: _series(
                    [float(result["aggregate"].get(metric, 0.0)) for result in results]
                )
                for metric in METRIC_KEYS
            },
            "latency_ms": _latency_summary(latencies),
            "by_language": _breakdown_summary(results, "by_language"),
            "by_question_type": _breakdown_summary(results, "by_question_type"),
        }

    return {
        "repetitions": repetitions,
        "warmup_enabled": warmup,
        "order_rotation": "cyclic",
        "query_count": len(queries),
        "variants": variants,
        "runs": runs,
    }
