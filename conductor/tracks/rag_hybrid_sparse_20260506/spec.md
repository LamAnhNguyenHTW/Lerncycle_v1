# Track: Sparse Vectors & Hybrid Search for the RAG Pipeline

## Overview
The existing RAG pipeline ([rag_pipeline/](../../../rag_pipeline/)) chunks PDFs, notes, and
annotation comments into Supabase `rag_chunks`, generates dense OpenAI
`text-embedding-3-small` embeddings, and upserts one Qdrant point per chunk under the named
dense vector `"dense"` in the collection `learncycle_chunks`. Dense retrieval works
end-to-end via [rag_pipeline/retrieval.py](../../../rag_pipeline/retrieval.py).

This track extends that system with **sparse BM25-style vectors** (FastEmbed/Qdrant BM25)
and **hybrid (dense + sparse) retrieval with RRF fusion**, while preserving Supabase as
the canonical source of truth and Qdrant as a rebuildable retrieval index.

## Architectural rules
- Supabase remains the canonical source of truth.
- Qdrant remains a rebuildable retrieval index.
- Each Qdrant point supports:
  - dense vector named `"dense"` (existing)
  - sparse vector named `"sparse"` (new)
  - existing payload unchanged
- Dense search must continue to work.
- `user_id` filtering must always be applied on every retrieval path.
- `source_type` filtering remains optional on every retrieval path.
- Sparse vectors must not be silently faked. If sparse indexing is enabled and sparse
  embedding/upsert fails, the indexing job must **not** be marked `completed`.
- Inspect the installed `qdrant-client` and `fastembed` APIs before coding; use currently
  supported method names/types instead of guessing. Prefer model name `"Qdrant/bm25"`; if
  the exact name differs in the installed stack (e.g. `"qdrant/bm25"` lowercase), use the
  current supported BM25 sparse model and document the resolved name.
- The Qdrant sparse vector config for the BM25 vector must enable the `IDF` modifier when
  the installed `qdrant-client` exposes it (BM25 requires IDF for correct scoring). If
  the installed client does not expose the modifier, document the omission.
- Dense-only mode remains available for debugging and backward compatibility (toggled via
  `SPARSE_ENABLED=false`).

## Functional Requirements

### Configuration ([rag_pipeline/config.py](../../../rag_pipeline/config.py), [rag_pipeline/.env.example](../../../rag_pipeline/.env.example))
New defaults on `WorkerConfig`:
- `sparse_provider = "fastembed"`
- `sparse_model = "Qdrant/bm25"`
- `sparse_vector_name = "sparse"`
- `sparse_enabled = true`
- `hybrid_fusion = "rrf"`
- `hybrid_prefetch_limit = 30`
- `hybrid_top_k = 10`

`.env.example` adds:
```
SPARSE_PROVIDER=fastembed
SPARSE_MODEL=Qdrant/bm25
SPARSE_VECTOR_NAME=sparse
SPARSE_ENABLED=true
HYBRID_FUSION=rrf
HYBRID_PREFETCH_LIMIT=30
HYBRID_TOP_K=10
```

Existing dense config stays untouched. `SPARSE_ENABLED=false` keeps dense-only mode.

### Dependencies ([rag_pipeline/requirements.txt](../../../rag_pipeline/requirements.txt))
Add FastEmbed support **without duplicating** the existing `qdrant-client` entry. The
file currently lists a bare `qdrant-client`; replace that single line with the compatible
extras form (`qdrant-client[fastembed]`) so there is exactly one dependency line for
`qdrant-client`. If the installed `qdrant-client` version does not ship the
`[fastembed]` extra cleanly, add `fastembed` as a separate top-level dependency
**instead** — never both the extras form and a duplicate bare entry. Existing
`qdrant-client` usage must keep working unchanged.

### Supabase migration (`supabase/migrations/`)
New idempotent migration adds sparse metadata columns to `rag_chunks`:
- `sparse_embedding_status text default 'pending'`
- `sparse_embedding_model text`
- `sparse_embedded_at timestamptz`
- `sparse_embedding_error text`

Plus index `rag_chunks_sparse_embedding_status_idx on rag_chunks(sparse_embedding_status)`.

All `add column if not exists` / `create index if not exists`. RLS policies are unchanged,
old `rag_chunks` rows remain readable, dense metadata columns are unchanged.

### Sparse embeddings module (`rag_pipeline/sparse_embeddings.py`)
- `class SparseVectorData` with `indices: list[int]`, `values: list[float]`.
- `class SparseEmbedder`
  - `__init__(provider: str, model: str)`
  - `embed(texts: list[str]) -> list[SparseVectorData]`
- Behaviour:
  - `embed([])` returns `[]` and does not call the provider.
  - `provider="fastembed"` uses FastEmbed/Qdrant BM25 sparse embedding support.
  - Unsupported provider raises `RuntimeError`/`ValueError` with a clear message.
  - Output order matches input order; each result has both `indices` and `values`.
  - Missing FastEmbed dependency raises a clear, actionable error.
  - No fake/deterministic sparse vectors on the durable path.

### Qdrant store extensions ([rag_pipeline/qdrant_store.py](../../../rag_pipeline/qdrant_store.py))
- Extend `ensure_collection(dim: int, sparse_enabled: bool = False)`:
  - Missing + `sparse_enabled=False` → create dense-only collection (current behaviour).
  - Missing + `sparse_enabled=True` → create collection with dense `"dense"` and sparse
    `"sparse"`. The sparse vector params for `"sparse"` must enable the BM25 `IDF`
    modifier (`models.Modifier.IDF`) when the installed `qdrant-client` exposes it.
  - Exists with dense + sparse → no-op.
  - Exists with dense only + `sparse_enabled=True` → raise a clear `RuntimeError`
    explaining that the collection must be recreated/reindexed for hybrid search. Do
    **not** auto-delete during normal worker execution.
- Add `recreate_collection_for_hybrid(dim: int)` — only run when explicitly invoked by a
  reindex/maintenance script.
- Extend `upsert_chunks(...)`:
  - Dense-only mode: existing behaviour.
  - Hybrid mode: each point includes dense `"dense"`, sparse `"sparse"`, and existing
    payload (unchanged fields, `None` filtered, meaningful falsey values like `0`/`""`
    preserved). Point IDs remain `rag_chunks.id` UUIDs.
  - Dense vector always required.
  - Sparse vector required only when hybrid mode is enabled; otherwise raise a clear
    error if hybrid is requested but sparse is missing.

### Worker integration ([rag_pipeline/worker.py](../../../rag_pipeline/worker.py))
With `SPARSE_ENABLED=true`, the indexing flow becomes:
1. Source loaded.
2. Old source chunks / old Qdrant points cleaned up.
3. Source chunked.
4. `rag_chunks` upserted to Supabase.
5. Dense embeddings generated.
6. Sparse embeddings generated.
7. `QdrantStore.ensure_collection(dim, sparse_enabled=True)`.
8. Qdrant upserts dense + sparse vectors.
9. `rag_chunks` rows updated with:
   - `embedding_status='completed'`, `embedding_model`, `embedded_at`,
     `qdrant_collection`, `qdrant_point_id`
   - `sparse_embedding_status='completed'`, `sparse_embedding_model`,
     `sparse_embedded_at`
10. Job marked completed.

Failure handling:
- Dense embedding failure → existing behaviour preserved.
- Sparse embedding failure → mark affected rows `sparse_embedding_status='failed'`, set
  `sparse_embedding_error`, do **not** mark the job completed; existing failure/retry
  path applies.
- Qdrant hybrid upsert failure → mark affected rows `embedding_status` and/or
  `sparse_embedding_status` failed as appropriate, store a clear error, do **not** mark
  the job completed.
- Delete-before-upsert remains acceptable, but if delete succeeds and the subsequent
  sparse/hybrid upsert fails, the failure must surface — do not silently leave the index
  empty.

### Reindex / maintenance command (`rag_pipeline/reindex_qdrant.py`)
A standalone script for rebuilding Qdrant from existing `rag_chunks`:
```
python -m rag_pipeline.reindex_qdrant --user-id <id>
python -m rag_pipeline.reindex_qdrant --all
python -m rag_pipeline.reindex_qdrant --source-type pdf
python -m rag_pipeline.reindex_qdrant --recreate-collection
```
Behaviour:
- Read existing `rag_chunks` from Supabase, optionally filtered by `user_id`,
  `source_type`, or `source_id`.
- Generate dense embeddings via existing `Embedder` and sparse embeddings via
  `SparseEmbedder`.
- With `--recreate-collection`, explicitly recreate Qdrant with dense + sparse config.
- Upsert all selected chunks into Qdrant as hybrid points.
- Update dense + sparse metadata on `rag_chunks`.
- Never mix users; `user_id` filtering must be safe.

### Sparse-only retrieval ([rag_pipeline/retrieval.py](../../../rag_pipeline/retrieval.py))
```
search_sparse_chunks(
    query: str,
    user_id: str,
    source_types: list[str] | None = None,
    top_k: int = 10,
)
```
- Sparse-embed the query.
- Search Qdrant on named sparse vector `"sparse"`.
- Always filter by `user_id`, optionally by `source_types`.
- Return normalized result dicts with `chunk_id`, `text`, `score`, `source_type`,
  `source_id`, `page_index`, `title`, `heading`, `metadata`.

### Hybrid retrieval ([rag_pipeline/retrieval.py](../../../rag_pipeline/retrieval.py))
```
search_hybrid_chunks(
    query: str,
    user_id: str,
    source_types: list[str] | None = None,
    top_k: int = 10,
    prefetch_limit: int = 30,
)
```
- Dense-embed and sparse-embed the query.
- Use the Qdrant Query API / prefetch when supported by the installed `qdrant-client`,
  with dense prefetch over `"dense"` and sparse prefetch over `"sparse"`, fused by RRF.
- `user_id` filter applied to both retrieval paths.
- `source_type` filter applied when provided.
- Same normalized result shape as dense search.
- Local RRF fallback if server-side hybrid query is not cleanly supported:
  1. dense top `prefetch_limit`
  2. sparse top `prefetch_limit`
  3. fuse rankings locally with RRF
  4. return top `top_k`
- Whether server-side or local RRF is used must be documented in code and README.

### Evaluation helper (`rag_pipeline/evaluate_retrieval.py`)
Compare dense vs sparse vs hybrid on a small manually curated query set (JSON input).
Metrics: `Hit@1`, `Hit@3`, `Hit@5`; `MRR` optional. Output: per-mode results plus a
comparison table to console or JSON.

## Non-Functional Requirements
- **Source of truth:** Supabase remains authoritative. Sparse/hybrid failures keep
  affected rows in a clear `failed` state with `sparse_embedding_error`; the existing
  `rag_index_jobs` retry path applies.
- **Backward compatibility:** Dense-only retrieval continues to work unchanged. Existing
  dense tests and dense workers run with no behavioural change when `SPARSE_ENABLED=false`.
- **Idempotency:** Re-running an indexing job for the same source produces the same set
  of hybrid points (deterministic delete-before-insert) without stale duplicates.
- **Explicit maintenance:** Recreating a collection that exists with dense-only config
  must require explicit operator action (reindex script flag), never an auto-destruction
  inside the worker.
- **Test isolation:** All tests run without network access. `Embedder`, `SparseEmbedder`,
  and `QdrantStore` are injected/mocked. Use fake FastEmbed clients in unit tests.
- **No silent fallbacks:** Sparse vectors are never faked; missing dependencies, missing
  models, or upsert errors raise.

## Acceptance Criteria
- [ ] Dense retrieval still works.
- [ ] Sparse vectors are generated for chunks via FastEmbed BM25.
- [ ] Qdrant collection `learncycle_chunks` supports both dense vector `"dense"` and
      sparse vector `"sparse"`.
- [ ] Existing `rag_chunks` can be reindexed into hybrid Qdrant points via the new
      reindex command.
- [ ] New PDF/note/annotation jobs create dense + sparse Qdrant points when
      `SPARSE_ENABLED=true`.
- [ ] Sparse-only search works with `user_id` and optional `source_type` filters.
- [ ] Hybrid search works with `user_id` and optional `source_type` filters and uses
      server-side RRF when supported, otherwise the documented local RRF fallback.
- [ ] Updating a source does not leave stale hybrid Qdrant points.
- [ ] Deleting notes/annotations removes the corresponding hybrid Qdrant points.
- [ ] Sparse failures do not mark RAG jobs `completed`.
- [ ] `pytest rag_pipeline/tests` passes.
- [ ] `npx tsc --noEmit` passes.

## Out of Scope
- Pydantic AI / agentic retrieval.
- Reranking (cross-encoder or LLM-based).
- Knowledge graph retrieval.
- Web search.
- Chat UI changes.
- LLM answer generation.
- SPLADE / ColBERT / late interaction models.
