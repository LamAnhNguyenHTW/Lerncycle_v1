"""Derive query-subset metrics from stored per-query evaluation results.

No retrieval, embedding, database, or model-provider calls are performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag_pipeline.eval.ground_truth import load_ground_truth
from rag_pipeline.eval.metrics import aggregate_metrics, breakdown_by, percentile
from rag_pipeline.eval.repeated import METRIC_KEYS
from rag_pipeline.evaluate_retrieval import write_report


def _series(values: list[float]) -> dict[str, float | int]:
    return {
        "mean": statistics.fmean(values) if values else 0.0,
        "stddev": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
        "count": len(values),
    }


def _latency(values: list[float]) -> dict[str, float | int]:
    result = _series(values)
    result["p50"] = percentile(values, 50)
    result["p95"] = percentile(values, 95)
    return result


def _filter_result(
    result: dict[str, Any], query_ids: set[str], *, k: int
) -> dict[str, Any]:
    per_query = [
        row for row in result.get("per_query", []) if row.get("query_id") in query_ids
    ]
    negative_controls = [
        row
        for row in result.get("negative_controls", [])
        if row.get("query_id") in query_ids
    ]
    latencies = [
        float(row["latency_ms"])
        for row in per_query + negative_controls
        if "latency_ms" in row
    ]
    aggregate = aggregate_metrics(per_query, latencies, k=k)
    aggregate["total_queries"] = len(per_query) + len(negative_controls)
    aggregate["negative_control_queries"] = len(negative_controls)
    return {
        "aggregate": aggregate,
        "by_language": breakdown_by(per_query, "language", k=k),
        "by_question_type": breakdown_by(per_query, "question_type", k=k),
        "per_query": per_query,
        "negative_controls": negative_controls,
    }


def _breakdown_series(
    results: list[dict[str, Any]], field: str
) -> dict[str, dict[str, dict[str, float | int]]]:
    groups = sorted({group for result in results for group in result[field]})
    return {
        group: {
            metric: _series(
                [
                    float(result[field][group].get(metric, 0.0))
                    for result in results
                    if group in result[field]
                ]
            )
            for metric in METRIC_KEYS
        }
        for group in groups
    }


def derive_repeated_subset(
    report: dict[str, Any], query_ids: set[str], *, k: int = 5
) -> dict[str, Any]:
    """Recompute repeated aggregates for exactly the selected query IDs."""
    evaluation = report.get("evaluation")
    if not isinstance(evaluation, dict) or not isinstance(evaluation.get("runs"), list):
        raise ValueError("Expected a repeated comparison report with evaluation.runs.")
    source_runs = evaluation["runs"]
    if not source_runs:
        raise ValueError("Repeated comparison report contains no runs.")
    variants = list(source_runs[0].get("results", {}))
    if not variants:
        raise ValueError("Repeated comparison report contains no variants.")

    collected: dict[str, list[dict[str, Any]]] = {name: [] for name in variants}
    run_aggregates: list[dict[str, Any]] = []
    observed_ids: set[str] = set()
    for source_run in source_runs:
        current: dict[str, Any] = {}
        for name in variants:
            filtered = _filter_result(source_run["results"][name], query_ids, k=k)
            collected[name].append(filtered)
            current[name] = filtered["aggregate"]
            observed_ids.update(row["query_id"] for row in filtered["per_query"])
            observed_ids.update(row["query_id"] for row in filtered["negative_controls"])
        run_aggregates.append(
            {"repetition": source_run.get("repetition"), "variants": current}
        )
    missing = query_ids - observed_ids
    if missing:
        raise ValueError(f"Source report is missing {len(missing)} selected query IDs.")

    summaries: dict[str, Any] = {}
    for name, results in collected.items():
        latencies = [
            float(row["latency_ms"])
            for result in results
            for row in result["per_query"] + result["negative_controls"]
            if "latency_ms" in row
        ]
        summaries[name] = {
            "metrics": {
                metric: _series(
                    [float(result["aggregate"].get(metric, 0.0)) for result in results]
                )
                for metric in METRIC_KEYS
            },
            "latency_ms": _latency(latencies),
            "by_language": _breakdown_series(results, "by_language"),
            "by_question_type": _breakdown_series(results, "by_question_type"),
        }

    first = collected[variants[0]][0]
    return {
        "repetitions": len(source_runs),
        "query_count": len(query_ids),
        "answerable_query_count": len(first["per_query"]),
        "negative_control_query_count": len(first["negative_controls"]),
        "variants": summaries,
        "run_aggregates": run_aggregates,
    }


def derive_graph_subset(
    report: dict[str, Any], query_ids: set[str]
) -> dict[str, Any]:
    """Filter stored Concept-Graph observations to selected query IDs."""
    runs = report.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("Expected a Concept-Graph report with runs.")
    rows = [
        row
        for run in runs
        for row in run.get("results", [])
        if row.get("query_id") in query_ids
    ]
    observed_ids = sorted({str(row["query_id"]) for row in rows})
    added_ids = sorted(
        {str(row["query_id"]) for row in rows if row.get("added_relevant_pages")}
    )
    return {
        "questions": len(observed_ids),
        "query_ids": observed_ids,
        "observations": len(rows),
        "graph_context_available_observations": sum(
            bool(row.get("graph_context_available")) for row in rows
        ),
        "observations_with_added_relevant_pages": sum(
            bool(row.get("added_relevant_pages")) for row in rows
        ),
        "questions_with_added_relevant_pages": len(added_ids),
        "query_ids_with_added_relevant_pages": added_ids,
        "relevant_graph_relationships": sum(
            int(row.get("relevant_graph_relationships", 0)) for row in rows
        ),
        "vector_latency_ms": _latency(
            [float(row["vector_latency_ms"]) for row in rows]
        ),
        "graph_latency_ms": _latency(
            [float(row["graph_latency_ms"]) for row in rows]
        ),
    }


def _metadata(path: Path) -> dict[str, str]:
    return {
        "file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def build_robustness_report(
    *,
    queries_path: Path,
    repeated_reports: dict[str, Path],
    graph_report: Path,
    query_id_prefix: str = "final-",
    k: int = 5,
) -> dict[str, Any]:
    queries = load_ground_truth(queries_path)
    selected = [query for query in queries if query.query_id.startswith(query_id_prefix)]
    query_ids = {query.query_id for query in selected}
    if not query_ids:
        raise ValueError("Query prefix selected no ground-truth questions.")

    phases: dict[str, Any] = {}
    sources: dict[str, Any] = {}
    for phase, path in repeated_reports.items():
        raw = json.loads(path.read_text(encoding="utf-8"))
        phases[phase] = derive_repeated_subset(raw, query_ids, k=k)
        sources[phase] = _metadata(path)
    graph_raw = json.loads(graph_report.read_text(encoding="utf-8"))
    phases["concept_graph"] = derive_graph_subset(graph_raw, query_ids)
    sources["concept_graph"] = _metadata(graph_report)

    answerable = sum(query.has_relevance_labels() for query in selected)
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_kind": "stored_results_robustness_subset",
        "derivation": "local_only_no_retrieval_embedding_database_or_provider_calls",
        "ground_truth": {
            "file": queries_path.name,
            "sha256": hashlib.sha256(queries_path.read_bytes()).hexdigest(),
            "query_id_prefix": query_id_prefix,
            "query_count": len(selected),
            "answerable_query_count": answerable,
            "negative_control_query_count": len(selected) - answerable,
        },
        "source_reports": sources,
        "phases": phases,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--chunking", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--cross-reranking", type=Path, required=True)
    parser.add_argument("--llm-reranking", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--query-id-prefix", default="final-")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_robustness_report(
        queries_path=args.queries,
        repeated_reports={
            "document_processing_chunking": args.chunking,
            "retrieval": args.retrieval,
            "cross_encoder_reranking": args.cross_reranking,
            "llm_reranking": args.llm_reranking,
        },
        graph_report=args.graph,
        query_id_prefix=args.query_id_prefix,
    )
    write_report(report, args.out)
    print(args.out.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
