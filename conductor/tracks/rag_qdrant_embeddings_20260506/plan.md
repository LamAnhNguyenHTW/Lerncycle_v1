# Implementation Plan: Durable Embeddings, Qdrant Upserts & First Dense Retrieval

All tasks follow the TDD workflow defined in [conductor/workflow.md](../../workflow.md):
write failing tests first (Red), implement the smallest change to pass (Green), refactor,
run the quality gate, then commit with a conventional commit message and add a git note.

## Architectural rules (apply to every phase)
- Supabase remains the canonical source of truth.
- Qdrant is only the retrieval index.
- Use `rag_chunks.id` (chunk UUIDs) as the Qdrant point IDs.
- Do not map returned chunk rows by `content_hash` alone — repeated chunks can share the
  same hash. Use a deterministic `client_chunk_key` generated before Supabase upsert,
  or fall back to `source_type + source_id + chunk_index + content_hash`.
- The `user_id` filter must always be applied during retrieval to prevent cross-user leakage.
- If embedding or Qdrant upsert fails, the RAG job must not be marked `completed`.

## Phase 1: Configuration & Dependencies
- [x] Task: Add `qdrant-client` to [rag_pipeline/requirements.txt](../../../rag_pipeline/requirements.txt)
- [x] Task: Update [rag_pipeline/.env.example](../../../rag_pipeline/.env.example) so the
      Qdrant block is documented as active (not "prepared for the next step") and includes:
    - `QDRANT_URL=`
    - `QDRANT_API_KEY=`
    - `QDRANT_COLLECTION=learncycle_chunks`
    - `EMBEDDING_PROVIDER=openai`
    - `OPENAI_API_KEY=`
    - `EMBEDDING_MODEL=text-embedding-3-small`
    - `EMBEDDING_BATCH_SIZE=100`
- [x] Task: Update local [rag_pipeline/.env](../../../rag_pipeline/.env)
    - [x] Set `QDRANT_COLLECTION=learncycle_chunks` (fix typo `lerncycle_chunks`)
- [x] Task: Update [rag_pipeline/config.py](../../../rag_pipeline/config.py)
    - [x] `qdrant_collection` defaults to `"learncycle_chunks"` when env var missing
    - [x] `embedding_provider` defaults to `"openai"`
    - [x] `embedding_model` defaults to `"text-embedding-3-small"`
    - [x] `embedding_batch_size` defaults to `100`
    - [x] `qdrant_api_key` remains optional
- [x] Task: Add or extend `rag_pipeline/tests/test_config.py`
    - [x] Verify defaults are applied correctly
    - [x] Verify env vars override defaults
- [x] Task: Conductor - User Manual Verification 'Phase 1: Configuration' (Protocol in workflow.md)

## Phase 1.5: Supabase Schema for Embedding Metadata
- [x] Task: Create a new migration under `supabase/migrations/`
      (e.g. `20260506000001_rag_chunks_embedding_metadata.sql`)
    - [x] Add columns to `rag_chunks` (idempotent with `add column if not exists`):
        - `embedding_status text default 'pending'`
        - `embedding_model text`
        - `embedded_at timestamptz`
        - `qdrant_collection text`
        - `qdrant_point_id text`
        - `embedding_error text`
    - [x] Add indexes (idempotent with `create index if not exists`):
        - `rag_chunks_embedding_status_idx on rag_chunks(embedding_status)`
        - `rag_chunks_qdrant_point_id_idx on rag_chunks(qdrant_point_id)`
    - [x] Confirm RLS policies and service-role worker behaviour are unchanged
    - [x] Confirm existing rows remain compatible (status defaults to `'pending'`)
- [x] Task: Conductor - User Manual Verification 'Phase 1.5: Migration' (Protocol in workflow.md)
    - [x] Confirm the migration applies cleanly
    - [x] Confirm old chunk rows remain readable
    - [x] Confirm new chunk rows can store embedding metadata

## Phase 2: Embeddings Module
- [x] Task: Create `rag_pipeline/embeddings.py` with a provider-switchable `Embedder`
    - [x] Write failing tests in `rag_pipeline/tests/test_embeddings.py`:
        - `test_openai_embedder_batches_inputs` — fakes the `OpenAI` client, asserts
          inputs are sent in batches of `batch_size` and returned vectors are flattened
          in original order
        - `test_embedder_raises_without_api_key` — provider=openai, no key →
          `RuntimeError` with a clear message
        - `test_gemini_branch_raises_not_implemented` — placeholder for future Gemini
        - `test_embedder_caches_dimension_after_first_call` — `dimension is None`
          before first call, set after first successful embed
        - `test_embedder_returns_empty_list_for_empty_input` — `embed([])` returns `[]`
          and does not call the provider
    - [x] Implement:
        - `class Embedder`
        - `__init__(provider, model, openai_api_key, gemini_api_key=None, batch_size=100)`
        - `embed(texts: list[str]) -> list[list[float]]`
        - `dimension: int | None`
        - `_embed_openai`, `_embed_gemini` (skeleton raising `NotImplementedError`)
    - [x] Validate that returned embeddings count matches input count
    - [x] Do not silently fall back to deterministic/fake embeddings in the durable path
- [x] Task: Conductor - User Manual Verification 'Phase 2: Embeddings' (Protocol in workflow.md)

## Phase 3: Qdrant Store Module
- [x] Task: Create `rag_pipeline/qdrant_store.py` with `QdrantStore`
    - [x] Write failing tests in `rag_pipeline/tests/test_qdrant_store.py` against a fake
          `QdrantClient`:
        - `test_ensure_collection_creates_when_missing` — creates with named vector
          `"dense"`, distance `cosine`
        - `test_ensure_collection_is_idempotent_when_present`
        - `test_upsert_builds_expected_payload` — `point.id` is chunk UUID, vector under
          named vector `"dense"`, payload contains documented fields, `None` keys filtered
        - `test_upsert_chunks_handles_empty_input_without_qdrant_call`
        - `test_delete_by_source_uses_filter` — `must` conditions on
          `user_id` + `source_type` + `source_id`
        - `test_search_chunks_passes_user_and_source_filters` — `user_id` always present,
          `source_type` filter present when `source_types` non-empty, named vector `"dense"`
        - `test_search_chunks_supports_empty_source_types_as_all_sources` — `[]` or `None`
          means all sources for the user; `user_id` filter still required
    - [x] Implement:
        - `class QdrantStore`
        - `__init__(url, api_key, collection_name)`
        - `ensure_collection(dim: int)`
        - `upsert_chunks(chunks)`
        - `delete_points_by_source(user_id, source_type, source_id)`
        - `search_chunks(query_vector, user_id, source_types, top_k)`
    - [x] Collection design:
        - Name: `learncycle_chunks`
        - Named vector: `dense` (size from runtime dim, distance cosine)
    - [x] Payload fields (filter `None`, keep meaningful falsey values):
          `chunk_id`, `user_id`, `source_type`, `source_id`,
          `pdf_id`, `note_id`, `annotation_id`, `document_id`,
          `page_index`, `title`, `heading`, `text`, `metadata`,
          `content_hash`, `chunk_index`, `created_at`, `updated_at`
- [x] Task: Conductor - User Manual Verification 'Phase 3: Qdrant Store' (Protocol in workflow.md)

## Phase 4: Worker Integration
- [x] Task: Wire `Embedder` and `QdrantStore` into `RagWorker`
    - [x] Extend tests in [rag_pipeline/tests/test_worker.py](../../../rag_pipeline/tests/test_worker.py)
          using the existing `RecordingWorker` pattern. Keep the existing tests passing.
    - [x] Add failing tests:
        - `test_worker_embeds_and_upserts_chunks_to_qdrant` — `Embedder.embed` called with
          chunk texts; `QdrantStore.ensure_collection` called with embedding dim;
          `QdrantStore.upsert_chunks` called with chunk UUIDs and documented payload
        - `test_note_reindex_deletes_old_qdrant_points_before_upsert`
        - `test_deleted_note_triggers_qdrant_cleanup` — note row missing →
          `QdrantStore.delete_points_by_source` called; old Supabase chunks removed;
          job completed with `skipped` metadata as today
        - `test_deleted_annotation_triggers_qdrant_cleanup`
        - `test_supabase_chunk_rows_marked_completed_after_qdrant_upsert` — sets
          `embedding_status='completed'`, `embedding_model`, `embedded_at`,
          `qdrant_collection`, `qdrant_point_id`
        - `test_worker_marks_chunks_failed_when_embedding_fails` — chunks updated with
          `embedding_status='failed'` and a clear `embedding_error`
        - `test_worker_does_not_mark_job_completed_when_embedding_fails`
        - `test_worker_marks_chunks_failed_when_qdrant_upsert_fails` — same as above
        - `test_worker_does_not_mark_job_completed_when_qdrant_upsert_fails`
    - [x] Implement `RagWorker.__init__`:
        - Construct `Embedder` from config
        - Construct `QdrantStore` from config
        - Allow dependency injection for tests
    - [x] Refactor `_upsert_chunks`:
        1. [x] Upsert into Supabase with `returning="representation"`
        2. [x] Map returned rows by `client_chunk_key` (preferred) or
           `(source_type, source_id, chunk_index, content_hash)` — never by `content_hash`
           alone
        3. [x] Call `Embedder.embed(texts)`
        4. [x] Call `QdrantStore.ensure_collection(dim)` (idempotent)
        5. [x] Call `QdrantStore.upsert_chunks(...)` with chunk UUIDs as point IDs and
           documented payload
        6. [x] Update matching `rag_chunks` rows with `embedding_status='completed'`,
           `embedding_model`, `embedded_at`, `qdrant_collection`, `qdrant_point_id`
    - [x] Extend `_replace_source_chunks`:
        - [x] Call `QdrantStore.delete_points_by_source(...)` before reinserting
    - [x] Extend `_delete_source_chunks`:
        - [x] Call `QdrantStore.delete_points_by_source(...)`
    - [x] Failure handling:
        - Embedding fails → mark affected chunks `embedding_status='failed'`, store
          `embedding_error`, do not mark job completed, let job retry
        - Qdrant upsert fails → same behaviour
        - Delete-before-upsert is acceptable, but if delete succeeds and the subsequent
          embedding/upsert fails, the job must fail/retry — never hide failures
- [x] Task: Conductor - User Manual Verification 'Phase 4: Worker Integration' (Protocol in workflow.md)

## Phase 5: Dense Retrieval Search
- [x] Task: Implement the first non-agentic dense retrieval search
    - [x] Decide location:
        - either new file `rag_pipeline/retrieval.py` with tests in
          `rag_pipeline/tests/test_retrieval.py`
        - or keep search inside `QdrantStore` and extend `test_qdrant_store.py`
    - [x] Add failing tests:
        - `test_search_chunks_embeds_query_and_searches_qdrant` — query text is embedded
          with the same `Embedder`; `QdrantStore.search_chunks` called with the vector
        - `test_search_chunks_filters_by_user_id` — `user_id` always passed
        - `test_search_chunks_filters_by_source_types` — `source_types` passed when given
        - `test_search_chunks_returns_normalized_results` — dicts with `chunk_id`, `text`,
          `score`, `source_type`, `source_id`, `page_index`, `title`/`heading`, `metadata`
    - [x] Implement:
        - `search_chunks(query: str, user_id: str, source_types: list[str] | None = None,
          top_k: int = 10)`
        - Embed query with the same `Embedder`/model
        - Search Qdrant on named vector `"dense"`
        - Always filter by `user_id`, optionally by `source_types`
        - Return normalized result dicts
    - [x] Out of scope (still): LLM answer generation, hybrid search, reranking
- [x] Task: Conductor - User Manual Verification 'Phase 5: Dense Retrieval' (Protocol in workflow.md)
    - [x] After running a real indexing job, manually test queries such as:
        - "Was steht im PDF über Hybrid Retrieval?"
        - "Welche Notizen habe ich zu RAG?"
        - "Welche Annotationen habe ich zu Chunking?"

## Phase 6: Documentation & Verification
- [x] Task: Update [rag_pipeline/README.md](../../../rag_pipeline/README.md)
    - [x] Document active Qdrant integration (remove "prepared" wording)
    - [x] Document `Embedder` and `QdrantStore` modules
    - [x] Document env vars: `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION`,
          `EMBEDDING_PROVIDER`, `OPENAI_API_KEY`, `EMBEDDING_MODEL`, `EMBEDDING_BATCH_SIZE`
    - [x] Document that only dense retrieval is implemented
    - [x] Document that hybrid search, sparse embeddings, reranking, Pydantic AI, and
          agentic retrieval are future phases
- [x] Task: Run full quality gate
    - [x] `pytest rag_pipeline/tests` — all green
    - [x] `npx tsc --noEmit` — still green (no TS changes expected)
- [x] Task: Manual end-to-end smoke test (with real `QDRANT_URL` and `OPENAI_API_KEY`)
    - [x] Run `python -m rag_pipeline.worker` against one PDF, one note, one
          annotation_comment job
    - [x] Verify in Supabase: `rag_chunks` rows with `embedding_status='completed'`,
          `embedding_model` set, `embedded_at` set,
          `qdrant_collection='learncycle_chunks'`, `qdrant_point_id` set
    - [x] Verify in Qdrant: collection `learncycle_chunks` exists with named vector
          `"dense"`; one point per `rag_chunks` row; payload contains `user_id`,
          `source_type`, `source_id`, `text`, `metadata`
    - [x] Verify update behaviour: updating a note removes old points and inserts
          refreshed ones
    - [x] Verify delete behaviour: deleting a note removes corresponding Qdrant points
    - [x] Verify dense search returns relevant chunks for PDF/note/annotation content and
          never returns chunks from another user
- [x] Task: Conductor - User Manual Verification 'Phase 6: Verification' (Protocol in workflow.md)

## Out of scope
- Pydantic AI / agentic retrieval
- Hybrid search (dense + sparse)
- Sparse embeddings
- Reranking
- Knowledge graph retrieval
- Web search
- Chat UI changes
- LLM answer generation

## Acceptance criteria
- [x] PDF, note, and annotation jobs still complete successfully
- [x] `rag_chunks` are still written as before
- [x] Supabase stores embedding/Qdrant metadata (`embedding_status`, `embedding_model`,
      `embedded_at`, `qdrant_collection`, `qdrant_point_id`, `embedding_error`)
- [x] Qdrant collection is created automatically if missing
- [x] Qdrant collection is named `learncycle_chunks`
- [x] Qdrant uses named vector `dense`
- [x] Qdrant contains one point per `rag_chunks` row
- [x] Updating a source does not leave stale Qdrant points
- [x] Deleting a note or annotation removes corresponding Qdrant points
- [x] Dense retrieval search works with `user_id` and optional `source_type` filters
- [x] Embedding/Qdrant failures do not mark jobs as completed
- [x] `pytest rag_pipeline/tests` passes
- [x] `npx tsc --noEmit` passes
