# Implementation Plan: Sparse Vectors & Hybrid Search for the RAG Pipeline

All tasks follow the TDD workflow defined in [conductor/workflow.md](../../workflow.md):
write failing tests first (Red), implement the smallest change to pass (Green), refactor,
run the quality gate, then commit with a conventional commit message and add a git note.

## Architectural rules (apply to every phase)
- Supabase remains the canonical source of truth.
- Qdrant remains a rebuildable retrieval index.
- Each Qdrant point supports dense vector `"dense"` and sparse vector `"sparse"` plus the
  existing payload (unchanged).
- Dense retrieval must continue to work; dense-only mode (`SPARSE_ENABLED=false`) stays
  available for debugging and backward compatibility.
- `user_id` filtering is always applied during retrieval; `source_type` filtering remains
  optional.
- Sparse vectors are never silently faked. If sparse indexing is enabled and sparse
  embedding/upsert fails, the indexing job must **not** be marked `completed`.
- Inspect installed `qdrant-client` and `fastembed` APIs before coding; use currently
  supported method names/types instead of guessing. Prefer `"Qdrant/bm25"` if supported;
  if the installed FastEmbed only exposes the lowercase `"qdrant/bm25"`, use that and
  document the resolved name in code + README.
- BM25 sparse vector config in Qdrant must enable the `IDF` modifier
  (`models.Modifier.IDF`) when the installed `qdrant-client` exposes it. If the modifier
  symbol is unavailable in the installed client, document the omission in code and
  README rather than silently dropping it.
- Avoid duplicate `qdrant-client` entries in `requirements.txt`. Use the compatible
  extras form (`qdrant-client[fastembed]`) as a single line, or fall back to a separate
  `fastembed` entry — never both the extras form and the bare entry.
- Recreating a Qdrant collection happens only via an explicit maintenance command, never
  inside normal worker execution.

## Phase 1: Dependencies & Configuration
- [x] Task: Update [rag_pipeline/requirements.txt](../../../rag_pipeline/requirements.txt)
    - [x] Replace the existing bare `qdrant-client` line with `qdrant-client[fastembed]`
          (single line — do **not** leave both the bare and the extras form)
    - [x] If the installed `qdrant-client` does not ship the `[fastembed]` extra cleanly,
          fall back to keeping bare `qdrant-client` plus a separate `fastembed` line
          (still no duplicates)
    - [x] Confirm existing `qdrant-client` usage still works after the dependency change
- [x] Task: Update [rag_pipeline/.env.example](../../../rag_pipeline/.env.example) with:
    - `SPARSE_PROVIDER=fastembed`
    - `SPARSE_MODEL=Qdrant/bm25`
    - `SPARSE_VECTOR_NAME=sparse`
    - `SPARSE_ENABLED=true`
    - `HYBRID_FUSION=rrf`
    - `HYBRID_PREFETCH_LIMIT=30`
    - `HYBRID_TOP_K=10`
- [x] Task: Update [rag_pipeline/config.py](../../../rag_pipeline/config.py)
    - [x] Add fields to `WorkerConfig` with defaults:
        - `sparse_provider = "fastembed"`
        - `sparse_model = "Qdrant/bm25"`
        - `sparse_vector_name = "sparse"`
        - `sparse_enabled = true`
        - `hybrid_fusion = "rrf"`
        - `hybrid_prefetch_limit = 30`
        - `hybrid_top_k = 10`
    - [x] Wire `from_env()` to read the matching env vars
    - [x] Parse `SPARSE_ENABLED` correctly as a boolean (e.g. `"false"` / `"0"` → `False`)
- [x] Task: Extend `rag_pipeline/tests/test_config.py` with failing tests first:
    - [x] Defaults are applied when env vars are missing
    - [x] Env vars override defaults
    - [x] Existing dense config tests still pass
    - [x] `SPARSE_ENABLED=false` keeps dense-only mode (`sparse_enabled is False`)
- [x] Task: Conductor - User Manual Verification 'Phase 1: Dependencies & Configuration'
      (Protocol in workflow.md)

## Phase 2: Supabase Sparse Metadata Migration
- [x] Task: Create new migration under `supabase/migrations/`
      (e.g. `20260506000002_rag_chunks_sparse_embedding_metadata.sql`)
    - [x] Add columns to `rag_chunks` (idempotent with `add column if not exists`):
        - `sparse_embedding_status text default 'pending'`
        - `sparse_embedding_model text`
        - `sparse_embedded_at timestamptz`
        - `sparse_embedding_error text`
    - [x] Add index (idempotent with `create index if not exists`):
        - `rag_chunks_sparse_embedding_status_idx on rag_chunks(sparse_embedding_status)`
    - [x] Confirm RLS policies are unchanged
    - [x] Confirm existing dense metadata columns are unchanged
    - [x] Confirm existing rows remain compatible (status defaults to `'pending'`)
- [x] Task: Conductor - User Manual Verification 'Phase 2: Sparse Metadata Migration'
      (Protocol in workflow.md)
    - [x] Confirm the migration applies cleanly
    - [x] Confirm old chunk rows remain readable
    - [x] Confirm new chunk rows can store sparse embedding metadata

## Phase 3: Sparse Embeddings Module
- [x] Task: Inspect installed `qdrant-client` / `fastembed` APIs before coding
    - [x] Document the actual BM25 sparse model name available in the installed stack
          (`"Qdrant/bm25"` vs `"qdrant/bm25"`); the chosen string must match what
          FastEmbed accepts, and the resolved name is what `SparseEmbedder` uses
    - [x] Document whether `qdrant_client.models.Modifier.IDF` is available — Phase 4
          will need it for the BM25 sparse vector config
- [x] Task: Create `rag_pipeline/sparse_embeddings.py`
    - [x] Write failing tests in `rag_pipeline/tests/test_sparse_embeddings.py`:
        - `test_sparse_embedder_returns_empty_list_for_empty_input` — `embed([])` → `[]`,
          provider not called
        - `test_sparse_embedder_uses_fastembed_bm25` — fakes FastEmbed, asserts
          `provider="fastembed"` + `model="Qdrant/bm25"` invokes BM25 sparse embedding
        - `test_sparse_embedder_preserves_order` — output order matches input order
        - `test_sparse_embedder_returns_indices_and_values` — each result has both
          `indices` and `values`
        - `test_sparse_embedder_raises_for_unknown_provider` — clear `ValueError` /
          `RuntimeError`
        - `test_sparse_embedder_raises_clear_error_when_dependency_missing` — actionable
          message naming the missing package
    - [x] Implement:
        - `class SparseVectorData` (`indices: list[int]`, `values: list[float]`)
        - `class SparseEmbedder`
            - `__init__(provider: str, model: str)`
            - `embed(texts: list[str]) -> list[SparseVectorData]`
        - [x] FastEmbed branch using the documented BM25 model
        - [x] No fake/deterministic sparse vectors on the durable path
- [x] Task: Conductor - User Manual Verification 'Phase 3: Sparse Embeddings Module'
      (Protocol in workflow.md)

## Phase 4: Qdrant Collection Supports Dense + Sparse
- [x] Task: Extend [rag_pipeline/qdrant_store.py](../../../rag_pipeline/qdrant_store.py)
    - [x] Write failing tests in `rag_pipeline/tests/test_qdrant_store.py`:
        - `test_ensure_collection_creates_dense_and_sparse_when_missing` — missing
          collection + `sparse_enabled=True` → creates dense `"dense"` + sparse `"sparse"`
        - `test_ensure_collection_detects_existing_hybrid_collection` — exists with both
          → no-op
        - `test_ensure_collection_raises_clear_error_when_sparse_missing` — exists with
          dense only + `sparse_enabled=True` → `RuntimeError` with actionable message
        - `test_recreate_collection_for_hybrid_is_explicit_only` — `ensure_collection`
          never auto-recreates; only `recreate_collection_for_hybrid(dim)` does
        - Existing dense-only tests still pass (regression coverage retained)
    - [x] Implement:
        - [x] Extend `ensure_collection(dim: int, sparse_enabled: bool = False)`
        - [x] When creating the sparse vector `"sparse"`, pass
          `models.SparseVectorParams(modifier=models.Modifier.IDF)` (BM25 needs IDF for
          correct scoring). If the installed `qdrant-client` does not expose
          `Modifier.IDF`, omit it and document the omission in code + README rather than
          silently substituting another modifier.
        - [x] Add `recreate_collection_for_hybrid(dim: int)` (explicit maintenance only) —
          must apply the same IDF modifier rule as `ensure_collection`.
- [x] Add a failing test asserting the BM25 sparse vector is created with the IDF
          modifier when the modifier symbol is available in the installed stack
- [x] Task: Conductor - User Manual Verification 'Phase 4: Qdrant Collection Hybrid Config'
      (Protocol in workflow.md)

## Phase 5: Qdrant Upsert with Dense + Sparse
- [x] Task: Extend `QdrantStore.upsert_chunks(...)` for hybrid mode
    - [x] Write failing tests in `rag_pipeline/tests/test_qdrant_store.py`:
        - `test_upsert_chunks_includes_dense_and_sparse_vectors` — point has both vectors
          under the correct named slots
        - `test_upsert_chunks_keeps_existing_payload_fields` — payload contract unchanged,
          `None` filtered, meaningful falsey values preserved
        - `test_upsert_chunks_raises_if_dense_missing` — clear error
        - `test_upsert_chunks_raises_if_sparse_missing_when_hybrid_enabled` — clear error
        - `test_upsert_chunks_still_supports_dense_only_mode` — backward compatible path
        - Existing upsert tests still pass
    - [x] Implement:
        - [x] Hybrid signature for `upsert_chunks(...)` accepting sparse vectors
        - [x] Validation rules above
        - [x] Point IDs remain `rag_chunks.id` UUIDs
- [x] Task: Conductor - User Manual Verification 'Phase 5: Qdrant Hybrid Upsert'
      (Protocol in workflow.md)

## Phase 6: Worker Integration for Sparse
- [x] Task: Wire `SparseEmbedder` into [rag_pipeline/worker.py](../../../rag_pipeline/worker.py)
    - [x] Extend [rag_pipeline/tests/test_worker.py](../../../rag_pipeline/tests/test_worker.py)
          using the existing `RecordingWorker` pattern; keep all current tests passing
    - [x] Add failing tests:
        - `test_worker_generates_sparse_embeddings_for_chunks`
        - `test_worker_upserts_dense_and_sparse_to_qdrant`
        - `test_worker_marks_sparse_embedding_completed` —
          `sparse_embedding_status='completed'`, `sparse_embedding_model`,
          `sparse_embedded_at` set
        - `test_worker_marks_sparse_embedding_failed_when_sparse_embedder_fails` —
          `sparse_embedding_status='failed'`, `sparse_embedding_error` set
        - `test_worker_does_not_mark_job_completed_when_sparse_fails`
        - `test_deleted_note_cleanup_still_deletes_qdrant_points`
        - `test_note_reindex_replaces_hybrid_points`
- [x] Implement worker flow when `SPARSE_ENABLED=true`:
        1. [x] Source loaded
        2. [x] Old source chunks / old Qdrant points cleaned up
        3. [x] Source chunked
        4. [x] `rag_chunks` upserted to Supabase
        5. [x] Dense embeddings generated
        6. [x] Sparse embeddings generated
        7. [x] `QdrantStore.ensure_collection(dim, sparse_enabled=True)`
        8. [x] Qdrant upsert (dense + sparse)
        9. [x] Update `rag_chunks` dense + sparse metadata
        10. [x] Job marked completed
    - [x] Failure handling:
        - [x] Sparse embedding failure → mark `sparse_embedding_status='failed'` with
          `sparse_embedding_error`, do not mark job completed
        - [x] Hybrid upsert failure → mark affected rows failed, do not mark job completed
        - [x] Delete-before-upsert remains acceptable, but failures must not be hidden
- [x] Task: Conductor - User Manual Verification 'Phase 6: Worker Hybrid Integration'
      (Protocol in workflow.md)

## Phase 7: Reindex / Maintenance Command
- [x] Task: Create `rag_pipeline/reindex_qdrant.py`
    - [x] Write failing tests in `rag_pipeline/tests/test_reindex_qdrant.py`:
        - `test_reindex_reads_existing_rag_chunks`
        - `test_reindex_generates_sparse_vectors`
        - `test_reindex_upserts_hybrid_points`
        - `test_reindex_filters_by_user_id`
        - `test_reindex_does_not_cross_users`
        - `test_reindex_recreate_collection_is_explicit`
- [x] Implement CLI:
        - [x] `python -m rag_pipeline.reindex_qdrant --user-id <id>`
        - [x] `python -m rag_pipeline.reindex_qdrant --all`
        - [x] `python -m rag_pipeline.reindex_qdrant --source-type pdf`
        - [x] `python -m rag_pipeline.reindex_qdrant --recreate-collection`
        - [x] Optional `--source-id` filter
    - [x] Behaviour:
        - [x] Read existing `rag_chunks` filtered by user_id / source_type / source_id
        - [x] Generate dense + sparse embeddings via existing `Embedder` and `SparseEmbedder`
        - [x] With `--recreate-collection`, explicitly call
          `QdrantStore.recreate_collection_for_hybrid(dim)`
        - [x] Upsert hybrid points
        - [x] Update dense + sparse metadata on `rag_chunks`
        - [x] Never mix users; `user_id` filtering is mandatory and safe
- [x] Task: Conductor - User Manual Verification 'Phase 7: Reindex Command'
      (Protocol in workflow.md)

## Phase 8: Sparse-only Retrieval
- [x] Task: Extend [rag_pipeline/retrieval.py](../../../rag_pipeline/retrieval.py) with
      `search_sparse_chunks(...)`
    - [x] Write failing tests in
          [rag_pipeline/tests/test_retrieval.py](../../../rag_pipeline/tests/test_retrieval.py):
        - `test_sparse_search_embeds_query_sparse` — uses `SparseEmbedder`
        - `test_sparse_search_uses_sparse_vector_name` — Qdrant call uses `"sparse"`
        - `test_sparse_search_filters_by_user_id` — always applied
        - `test_sparse_search_filters_by_source_types` — applied when provided
        - `test_sparse_search_returns_normalized_results` — same shape as dense
    - [x] Implement:
        ```
        search_sparse_chunks(
            query: str,
            user_id: str,
            source_types: list[str] | None = None,
            top_k: int = 10,
        )
        ```
- [x] Task: Conductor - User Manual Verification 'Phase 8: Sparse Retrieval'
      (Protocol in workflow.md)

## Phase 9: Hybrid Retrieval
- [x] Task: Inspect installed `qdrant-client` to determine whether server-side Query API
      / prefetch + RRF is available; document the choice (server-side vs local fallback)
- [x] Task: Extend [rag_pipeline/retrieval.py](../../../rag_pipeline/retrieval.py) with
      `search_hybrid_chunks(...)`
    - [x] Write failing tests in
          [rag_pipeline/tests/test_retrieval.py](../../../rag_pipeline/tests/test_retrieval.py):
        - `test_hybrid_search_embeds_dense_and_sparse_query`
        - `test_hybrid_search_uses_prefetch_for_dense_and_sparse_when_supported`
        - `test_hybrid_search_uses_rrf_fusion`
        - `test_hybrid_search_filters_by_user_id_in_both_paths`
        - `test_hybrid_search_filters_by_source_types`
        - `test_hybrid_search_returns_normalized_results`
        - `test_hybrid_search_can_use_local_rrf_fallback`
    - [x] Implement:
        ```
        search_hybrid_chunks(
            query: str,
            user_id: str,
            source_types: list[str] | None = None,
            top_k: int = 10,
            prefetch_limit: int = 30,
        )
        ```
        - Dense + sparse query embedding
        - Server-side hybrid via Query API / prefetch + RRF when supported
        - Local RRF fallback otherwise (dense top `prefetch_limit` + sparse top
          `prefetch_limit`, fused locally, top `top_k` returned)
        - `user_id` filter applied to both paths
        - `source_type` filter applied when provided
        - Same normalized result shape as dense search
- [x] Task: Conductor - User Manual Verification 'Phase 9: Hybrid Retrieval'
      (Protocol in workflow.md)

## Phase 10: Evaluation Helper
- [x] Task: Create `rag_pipeline/evaluate_retrieval.py`
    - [x] Write failing tests in `rag_pipeline/tests/test_evaluate_retrieval.py`:
        - `test_evaluation_computes_hit_at_k`
        - `test_evaluation_compares_dense_sparse_hybrid`
        - `test_evaluation_handles_missing_expected_source`
- [x] Implement:
        - [x] JSON input with `query` plus expected `source_id` / `page_index` /
          `source_type`
        - [x] Run dense, sparse, and hybrid searches per query
        - [x] Metrics: `Hit@1`, `Hit@3`, `Hit@5`; `MRR` optional
        - [x] Output: per-mode results + comparison table to console or JSON
- [x] Task: Conductor - User Manual Verification 'Phase 10: Evaluation Helper'
      (Protocol in workflow.md)

## Phase 11: Documentation & Quality Gate
- [x] Task: Update [rag_pipeline/README.md](../../../rag_pipeline/README.md)
    - [x] Explain dense vs sparse vs hybrid retrieval
    - [x] Document `SparseEmbedder`
    - [x] Document config variables: `SPARSE_PROVIDER`, `SPARSE_MODEL`,
          `SPARSE_VECTOR_NAME`, `SPARSE_ENABLED`, `HYBRID_FUSION`,
          `HYBRID_PREFETCH_LIMIT`, `HYBRID_TOP_K`
    - [x] Document the resolved BM25 model name (`"Qdrant/bm25"` vs `"qdrant/bm25"`)
          and whether the BM25 sparse vector was created with the `IDF` modifier in the
          installed `qdrant-client`
    - [x] Document the reindex command and how to recreate the Qdrant collection for
          hybrid
    - [x] Document whether server-side or local RRF fusion is used in this build
    - [x] Document that reranking, Pydantic AI, agentic retrieval, knowledge graph
          retrieval, and web search remain future phases
- [x] Task: Run full quality gate
    - [x] `pytest rag_pipeline/tests` — all green
    - [x] `npx tsc --noEmit` — still green
- [x] Task: Manual end-to-end smoke test (with real `QDRANT_URL`, `OPENAI_API_KEY`, and
      FastEmbed installed)
    1. [x] Install updated dependencies
    2. [x] Recreate the Qdrant collection for hybrid via the reindex script
    3. [x] Reindex existing `rag_chunks`
    4. [x] Verify the Qdrant collection has dense + sparse config
    5. [x] Verify Qdrant contains one point per selected `rag_chunks` row
    6. [x] Run dense search: "Was ist Process Mining?"
    7. [x] Run sparse search: "Process Mining", "HybridChunker", "PROCESS EQUALS"
    8. [x] Run hybrid search: "Was ist Process Mining?", "Power Automate Parent Child
           Flows", "QdrantStore"
    9. [x] Verify no search returns chunks from another user
    10. [x] Update a note, run the worker, verify old hybrid points are removed and new
            ones inserted
    11. [x] Delete a note and verify points are removed
- [x] Task: Conductor - User Manual Verification 'Phase 11: Documentation & Verification'
      (Protocol in workflow.md)

## Out of scope
- Pydantic AI / agentic retrieval
- Reranking (cross-encoder or LLM-based)
- Knowledge graph retrieval
- Web search
- Chat UI changes
- LLM answer generation
- SPLADE / ColBERT / late interaction models

## Acceptance criteria
- [x] Dense retrieval still works
- [x] Sparse vectors are generated for chunks via FastEmbed BM25
- [x] Qdrant supports dense vector `"dense"` and sparse vector `"sparse"` in collection
      `learncycle_chunks`
- [x] Existing `rag_chunks` can be reindexed into hybrid Qdrant points via the new
      reindex command
- [x] New PDF/note/annotation jobs create dense + sparse Qdrant points when
      `SPARSE_ENABLED=true`
- [x] Sparse-only search works with `user_id` and optional `source_type` filters
- [x] Hybrid search works with `user_id` and optional `source_type` filters and uses
      server-side RRF when supported, otherwise the documented local RRF fallback
- [x] Updating a source does not leave stale hybrid Qdrant points
- [x] Deleting notes/annotations removes corresponding hybrid Qdrant points
- [x] Sparse failures do not mark RAG jobs `completed`
- [x] `pytest rag_pipeline/tests` passes
- [x] `npx tsc --noEmit` passes
