"""Retrieval metrics for the evaluation harness.

Relevance is predicate-based (a hit matches a query's source/page/phrase
labels; see ground_truth.is_hit_relevant). Because the global set of relevant
chunks is unknown under this labeling, Recall@k is defined over the *labeled
target units* (expected pages, else expected source ids, else a binary
phrase-coverage fallback). All definitions are documented per function so the
thesis can report them exactly.
"""

from __future__ import annotations

import math
from typing import Any

from rag_pipeline.eval.ground_truth import GroundTruthQuery, is_hit_relevant


def _target_units(query: GroundTruthQuery) -> tuple[str, set[str]]:
    """Return the (unit_kind, unit_set) used as the relevant set for recall.

    Priority: pages > source ids > phrases. Pages/source ids give a countable
    relevant set. Phrase-only queries have no enumerable relevant set, so recall
    degrades to binary coverage (1 unit) — flagged via unit_kind 'binary'.
    """
    if query.expected_pages:
        return "page", {str(page) for page in query.expected_pages}
    if query.expected_source_ids:
        return "source_id", {str(sid) for sid in query.expected_source_ids}
    return "binary", {"__any_relevant__"}


def _covered_units(
    hit: dict[str, Any],
    unit_kind: str,
    query: GroundTruthQuery,
) -> set[str]:
    """Return the target units a single hit covers, for recall accounting."""
    if unit_kind == "page":
        page_index = hit.get("page_index")
        if page_index is None:
            return set()
        return {str(int(page_index) + 1)}
    if unit_kind == "source_id":
        source_id = hit.get("source_id")
        if source_id is None:
            return set()
        return {str(source_id)}
    # binary: any relevant hit covers the single synthetic unit
    return {"__any_relevant__"} if is_hit_relevant(hit, query) else set()


def _dcg(relevances: list[int]) -> float:
    """Discounted cumulative gain with binary gains and log2(rank+1) discount."""
    return sum(
        rel / math.log2(position + 2)  # position is 0-based → rank = position+1
        for position, rel in enumerate(relevances)
    )


def evaluate_query(
    query: GroundTruthQuery,
    hits: list[dict[str, Any]],
    *,
    k: int = 5,
) -> dict[str, Any]:
    """Compute per-query retrieval metrics for one ranked hit list.

    Metric definitions (all binary relevance via is_hit_relevant):
      - hit_at_1/3/5: 1.0 if any relevant hit is within the top N, else 0.0.
      - reciprocal_rank: 1/rank of the first relevant hit (0 if none).
      - precision_at_k: relevant hits in top-k divided by k (textbook Precision@k).
      - recall_at_k: labeled target units covered by top-k divided by all target
        units (pages, else source ids, else binary phrase coverage).
      - ndcg_at_k: DCG@k of the ranking divided by the ideal DCG of the same
        retrieved list (all relevant hits moved to the front).

    Args:
        query: Labeled query.
        hits: Ranked retrieval results (best first), each a normalized hit dict.
        k: Cutoff for @k metrics.

    Returns:
        Dict of metric name to value plus bookkeeping counts.
    """
    top_k = hits[:k]
    relevances = [1 if is_hit_relevant(hit, query) else 0 for hit in top_k]

    first_relevant_rank = 0
    for rank, hit in enumerate(hits, start=1):
        if is_hit_relevant(hit, query):
            first_relevant_rank = rank
            break

    unit_kind, target_units = _target_units(query)
    covered: set[str] = set()
    for hit in top_k:
        covered |= _covered_units(hit, unit_kind, query) & target_units
    recall_at_k = len(covered) / len(target_units) if target_units else 0.0

    relevant_in_topk = sum(relevances)
    precision_at_k = relevant_in_topk / k if k > 0 else 0.0

    dcg = _dcg(relevances)
    ideal_relevances = [1] * min(k, relevant_in_topk)
    idcg = _dcg(ideal_relevances)
    ndcg_at_k = dcg / idcg if idcg > 0 else 0.0

    return {
        "query_id": query.query_id,
        "language": query.language,
        "question_type": query.question_type,
        "num_retrieved": len(hits),
        "relevant_in_topk": relevant_in_topk,
        "hit_at_1": 1.0 if 0 < first_relevant_rank <= 1 else 0.0,
        "hit_at_3": 1.0 if 0 < first_relevant_rank <= 3 else 0.0,
        "hit_at_5": 1.0 if 0 < first_relevant_rank <= 5 else 0.0,
        "reciprocal_rank": (1.0 / first_relevant_rank) if first_relevant_rank else 0.0,
        f"precision_at_{k}": precision_at_k,
        f"recall_at_{k}": recall_at_k,
        f"ndcg_at_{k}": ndcg_at_k,
    }


def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile (pct in [0, 100]); 0.0 for empty input."""
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered[int(rank)])
    weight = rank - low
    return float(ordered[low] * (1 - weight) + ordered[high] * weight)


_METRIC_KEYS = (
    "hit_at_1",
    "hit_at_3",
    "hit_at_5",
    "reciprocal_rank",
    "precision_at_5",
    "recall_at_5",
    "ndcg_at_5",
)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate_metrics(
    per_query: list[dict[str, Any]],
    latencies_ms: list[float] | None = None,
    *,
    k: int = 5,
) -> dict[str, Any]:
    """Aggregate per-query metrics into means plus latency percentiles.

    `mrr` is the mean reciprocal rank. Precision/recall/ndcg keys follow the
    cutoff k. Latency percentiles are omitted when no latencies are provided.
    """
    keys = [
        "hit_at_1",
        "hit_at_3",
        "hit_at_5",
        "precision_at_" + str(k),
        "recall_at_" + str(k),
        "ndcg_at_" + str(k),
    ]
    summary: dict[str, Any] = {
        "evaluated_queries": len(per_query),
    }
    for key in keys:
        summary[key] = _mean([float(row.get(key, 0.0)) for row in per_query])
    summary["mrr"] = _mean([float(row.get("reciprocal_rank", 0.0)) for row in per_query])

    if latencies_ms:
        summary["latency_ms"] = {
            "mean": _mean(latencies_ms),
            "p50": percentile(latencies_ms, 50),
            "p95": percentile(latencies_ms, 95),
            "min": min(latencies_ms),
            "max": max(latencies_ms),
            "count": len(latencies_ms),
        }
    return summary


def breakdown_by(
    per_query: list[dict[str, Any]],
    field: str,
    *,
    k: int = 5,
) -> dict[str, dict[str, Any]]:
    """Aggregate metrics grouped by a query field (e.g. 'language', 'question_type')."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in per_query:
        groups.setdefault(str(row.get(field, "unknown")), []).append(row)
    return {
        group: aggregate_metrics(rows, k=k)
        for group, rows in sorted(groups.items())
    }
