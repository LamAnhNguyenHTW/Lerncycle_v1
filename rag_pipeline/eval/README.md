# Retrieval / Chunking Evaluation Harness

Offline evaluation tooling for LearnCycle's RAG pipeline. **Evaluation only** —
nothing in `rag_pipeline/eval/` is imported by the production worker, RAG API,
or Next.js app. All strategy collections are named `eval_*` and never touch the
production `learncycle_chunks` collection.

> No strategy, retrieval mode, or configuration is claimed to be better here.
> This directory only *produces* measurements. Conclusions belong in the thesis
> after real runs against the labeled corpus.

## Components

| File | Purpose |
|---|---|
| `ground_truth.py` | Ground-truth query schema + loader; predicate-based relevance (`is_hit_relevant`) |
| `metrics.py` | Hit@1/3/5, MRR, Precision@5, Recall@5, nDCG@5, latency percentiles, per-field breakdowns |
| `index_corpus.py` | Chunk + embed + upsert the corpus under 3 strategies into separate collections; records chunk stats |
| `repeated.py` | Warm-up, cyclic order rotation, repeated measurements, mean/stddev aggregation |
| `run_comparisons.py` | Repeated chunking, retrieval, Cross-Encoder, and opt-in LLM-reranker comparison runner |
| `index_graph.py` | Isolated Concept-Graph indexing for the evaluation user |
| `graph_comparison.py` | Dense vector support vs. additional production Concept-Graph context |
| `derive_robustness.py` | Recompute a query-subset analysis locally from stored per-query reports; no retrieval or provider calls |
| `../evaluate_retrieval.py` | `evaluate_labeled_queries`, `build_eval_report`, `write_report`, `eval_config_with_caches_disabled` |
| `example_queries.json` | Example ground-truth file (schema reference) |
| `example_manifest.json` | Example corpus manifest (schema reference) |
| `queries_final.json` | Manually approved, frozen 60-question ground truth (`ground-truth-final-v1`) |
| `ground_truth_freeze.json` | Freeze identifier, approval state, file names, and SHA-256 checksums |
| `results/` | Output JSON reports (git-tracked via `.gitkeep`; real reports committed for reproducibility) |

## Chunking strategies (separate collections)

| Strategy | Collection | Chunker |
|---|---|---|
| `fixed_size` | `eval_fixed_size` | `rag_pipeline/fixed_chunking.py`, chunked per page (keeps `page_index`) |
| `docling` | `eval_docling` | Docling HybridChunker, no semantic refinement (pass-through refiner) |
| `docling_semantic` | `eval_docling_semantic` | Docling HybridChunker + `SemanticRefiner` (production chunker) |

All three use the **same documents** and the **same embedding models**
(`text-embedding-3-small` dense, `Qdrant/bm25` sparse). Only the chunker varies.

## Ground-truth format

A JSON list, or `{"queries": [...]}`. Relevance is labeled by source id, page,
and/or phrase — never by chunk id — so the same file is reusable across all
chunking strategies. Required fields per query:

| Field | Meaning |
|---|---|
| `query_id` | Stable unique id, used in result files |
| `question` | Query sent to retrieval |
| `language` | `de` or `en` (enables per-language breakdown) |
| `question_type` | `factual` \| `contextual` \| `table` \| `relational` \| `cross_document` |
| `expected_source_ids` | Source ids whose chunks are relevant |
| `expected_pages` | 1-based page numbers that are relevant (matched against 0-based `page_index`) |
| `expected_phrases` | Case-insensitive substrings a relevant chunk should contain |
| `reference_answer` | Gold answer (for later answer-quality / LLM-judge runs; unused by retrieval metrics) |
| `notes` | Optional annotation |

At least one of `expected_source_ids` / `expected_pages` / `expected_phrases`
must be present. See `example_queries.json`.

When source and page labels are both present, a hit must match both. Phrases
are supplementary and are used as the relevance fallback only when neither a
source nor a page is labeled.

### Strict corpus ground truth

queries_pilot.json is the 15-question, manually reviewable pilot set for the
five local corpus PDFs. It adds these required annotation fields:
relevance_notes, difficulty, requires_multiple_chunks,
requires_multiple_documents, extraction_risk, and manual_review_required.

The pilot accepts the question types fact, semantic_paraphrase, terminology,
contextual, relational, multi_hop, table_or_figure, and unanswerable.
Unanswerable queries must have empty source, page, and phrase labels.

validate_pilot_ground_truth() checks the pilot against corpus/manifest.json:
required fields, unique IDs, source IDs, relative PDF paths, actual page counts,
page bounds, unknown corpus PDFs, secret-like keys, credentialed URLs, and the
unanswerable structure. GROUND_TRUTH_REVIEW.md is the human sign-off checklist.

`queries_final.json` is the manually approved 60-question set used for final
measurements. Its sign-off is recorded in `GROUND_TRUTH_FINAL_REVIEW.md`, and
`ground_truth_freeze.json` fixes the version as `ground-truth-final-v1` with
SHA-256 checksums for both files. Any content change requires a new freeze ID,
renewed manual review, and new checksums. The 15-question `queries_pilot.json`
remains available only for reproducing the earlier pilot measurements.

## Metric definitions (report exactly these in the thesis)

- **Hit@N**: 1 if any relevant hit is within the top N, else 0 (averaged over queries).
- **MRR**: mean of 1/rank of the first relevant hit.
- **Precision@k**: relevant hits in top-k divided by k (textbook definition).
- **Recall@k**: labeled target units covered by top-k divided by all target units.
  Target units are pages (if `expected_pages` set), else source ids
  (if `expected_source_ids` set), else a single binary phrase-coverage unit.
- **nDCG@k**: DCG@k (binary gains, log2 discount) divided by the ideal DCG of the
  same retrieved list (all relevant hits moved to the front).
- **Latency**: wall-clock ms per query; reported as mean / p50 / p95 / min / max.

Breakdowns are produced by `language` and by `question_type`.

## Usage

### 1. Assemble the corpus

Put PDFs under `rag_pipeline/eval/corpus/` (git-ignored — do not commit copyrighted
material) and write a manifest (see `example_manifest.json`). Use the `source_id`
values in your ground-truth `expected_source_ids`.

### 2. Index under each strategy

```bash
# Cheap chunk-only inspection (no OpenAI, no Qdrant): counts + length stats only
python -m rag_pipeline.eval.index_corpus \
    --manifest rag_pipeline/eval/corpus/manifest.json \
    --strategy fixed_size --dry-run

# Real indexing into eval_* collections (needs OPENAI_API_KEY + QDRANT_URL;
# docling* strategies also need Docling installed):
python -m rag_pipeline.eval.index_corpus \
    --manifest rag_pipeline/eval/corpus/manifest.json \
    --strategy all --user-id eval-pilot-v1 \
    --continue-on-extraction-error \
    --out rag_pipeline/eval/results/2026-07-13_index_stats.json
```

The index report separates whole-document `extraction_errors` from Docling
`extraction_diagnostics` (chunked, missing, fallback, and retry pages).
Raw exception messages and local paths are never persisted.

### 3. Run the fixed-config pilot

The pilot runner executes the same query set against all three collections with
one fixed configuration: hybrid dense+sparse, Top-5, no reranking, all caches
disabled.

```bash
python -m rag_pipeline.eval.run_pilot \
    --queries rag_pipeline/eval/queries_pilot.json \
    --index-stats rag_pipeline/eval/results/2026-07-13_index_stats.json \
    --user-id eval-pilot-v1 --top-k 5 \
    --out rag_pipeline/eval/results/2026-07-13_pilot_retrieval.json
```

All questions are executed. Answerable questions contribute to Hit@1/3/5, MRR,
Recall@5, Precision@5, and nDCG@5. An `unanswerable` query has no relevant set,
so it is reported separately under `negative_controls` and is not folded into
relevance metrics. Latency includes all executed questions and is also stored
per query.

### 4. Run custom retrieval evaluation

Retrieval evaluation is scripted via `evaluate_retrieval.evaluate_labeled_queries`
plus `create_searchers` pointed at the strategy collection. Real runs **must**
disable caches:

```python
from rag_pipeline.config import WorkerConfig
from rag_pipeline.qdrant_store import QdrantStore
from rag_pipeline.eval.ground_truth import load_ground_truth
from rag_pipeline.evaluate_retrieval import (
    create_searchers, evaluate_labeled_queries, build_eval_report, write_report,
    eval_config_with_caches_disabled,
)

cfg = eval_config_with_caches_disabled(WorkerConfig.from_env())
store = QdrantStore(url=cfg.qdrant_url, api_key=cfg.qdrant_api_key,
                    collection_name="eval_docling_semantic")
queries = load_ground_truth("rag_pipeline/eval/corpus/queries.json")
searchers = create_searchers(user_id="eval-user", top_k=5, mode="all",
                             retrieval_kwargs={"config": cfg, "store": store})
modes = evaluate_labeled_queries(queries, searchers, k=5)
report = build_eval_report(mode_results=modes, strategy="docling_semantic",
                           collection="eval_docling_semantic",
                           config={"embedding_model": cfg.embedding_model,
                                   "top_k": 5, "caches_disabled": True})
write_report(report, "rag_pipeline/eval/results/2026-07-13_docling_semantic.json")
```

### 5. Run the final repeated comparisons

All final comparisons use an excluded warm-up, three repetitions, cyclic
variant/query order, and disabled caches.

```bash
python -m rag_pipeline.eval.run_comparisons --phase chunking \
  --queries rag_pipeline/eval/queries_final.json --repetitions 3 \
  --out rag_pipeline/eval/results/chunking_comparison.json

python -m rag_pipeline.eval.run_comparisons --phase retrieval \
  --queries rag_pipeline/eval/queries_final.json \
  --collection-strategy fixed_size --repetitions 3 \
  --out rag_pipeline/eval/results/retrieval_comparison.json

python -m rag_pipeline.eval.run_comparisons --phase reranking \
  --queries rag_pipeline/eval/queries_final.json \
  --collection-strategy fixed_size --base-retrieval dense --repetitions 3 \
  --reranker-model jinaai/jina-reranker-v2-base-multilingual \
  --out rag_pipeline/eval/results/reranking_comparison.json
```

The reranking command above uses the local FastEmbed Cross-Encoder. An LLM
reranking run is deliberately opt-in because it transmits each query and the
retrieved candidate texts to the configured provider:

```bash
# Run only after explicit authorization for this separate data transfer.
python -m rag_pipeline.eval.run_comparisons --phase reranking \
  --queries rag_pipeline/eval/queries_final.json \
  --collection-strategy fixed_size --base-retrieval dense --repetitions 3 \
  --reranker-provider llm --reranker-model gpt-4o-mini \
  --out rag_pipeline/eval/results/reranking_llm_comparison.json
```

Concept-Graph indexing sends evaluation chunk text to the configured LLM
provider. It must only be run after explicit authorization for that data
transfer. The graph is isolated by `user_id=eval-pilot-v1`. Failed chunks can
be retried without deleting or retransmitting successful chunks:

```bash
python -m rag_pipeline.eval.index_graph --strategy fixed_size \
  --manifest rag_pipeline/eval/corpus/manifest.json \
  --retry-report rag_pipeline/eval/results/previous_graph_index.json \
  --out rag_pipeline/eval/results/graph_index_retry.json
```

### 6. Derive the 45-question robustness analysis locally

The following command filters stored rows to query IDs beginning with
`final-`. It calls neither OpenAI nor Qdrant, Neo4j, an embedder, or a
reranker. Source-report SHA-256 checksums are stored in the derived report.

```bash
python -m rag_pipeline.eval.derive_robustness \\
  --queries rag_pipeline/eval/queries_final.json \\
  --chunking rag_pipeline/eval/results/chunking_comparison_final_v1_20260721.json \\
  --retrieval rag_pipeline/eval/results/retrieval_comparison_fixed_size_final_v1_20260721.json \\
  --cross-reranking rag_pipeline/eval/results/reranking_cross_encoder_dense_fixed_size_final_v1_20260721.json \\
  --llm-reranking rag_pipeline/eval/results/reranking_llm_dense_fixed_size_final_v1_20260721.json \\
  --graph rag_pipeline/eval/results/concept_graph_comparison_fixed_size_final_v1_corrected_20260721.json \\
  --out rag_pipeline/eval/results/robustness_45_final_v1_20260722.json
```

## Guarantees

- **No production changes.** The worker's default pipeline and collection are
  untouched.
- **Reproducible config.** Every report embeds `strategy`, `collection`, `k`,
  ground-truth path, and non-secret config.
- **Caches off for real runs** via `eval_config_with_caches_disabled`.
- **No secrets in reports.** `build_eval_report` accepts only non-secret settings
  (model names, top_k, flags) and rejects secret-like keys or credentialed URLs
  before export. Never pass API keys or service-role credentials.
