# Track: Durable Embeddings, Qdrant Upserts & First Dense Retrieval

## Overview
The existing RAG worker ([rag_pipeline/worker.py](../../../rag_pipeline/worker.py)) processes
`rag_index_jobs` for `pdf`, `note`, and `annotation_comment` sources and writes the resulting
chunks into the Supabase table `rag_chunks`. Qdrant is prepared in schema and config but no
vectors are written yet, and no retrieval search exists.

This track adds **durable dense embeddings** for every final chunk, **Qdrant upserts** with
one point per `rag_chunks` row, and a **first non-agentic dense retrieval search** over
Qdrant. Supabase remains the source of truth — Qdrant only mirrors `rag_chunks` for retrieval.

## Architectural rules
- Supabase remains the canonical database.
- Qdrant is only the retrieval index.
- Use `rag_chunks.id` (chunk UUIDs) as the Qdrant point IDs.
- Do not map returned rows by `content_hash` alone (repeated chunks may share the same
  hash). Prefer a deterministic `client_chunk_key` generated before Supabase upsert, or fall
  back to `source_type + source_id + chunk_index + content_hash`.
- The `user_id` filter must always be applied during retrieval to prevent cross-user leakage.
- If embedding or Qdrant upsert fails, the RAG job must not be marked `completed`.

## Functional Requirements

### Embeddings module (`rag_pipeline/embeddings.py`)
- Supports OpenAI `text-embedding-3-small` via `EMBEDDING_PROVIDER=openai`.
- Batches inputs (default `EMBEDDING_BATCH_SIZE=100`) and flattens results in input order.
- Returns dense vectors (`list[list[float]]`).
- Caches the vector dimension after the first successful call.
- Returns `[]` for empty input without calling the provider.
- Raises a clear `RuntimeError` when the provider's API key is missing.
- Provider switch is structured for later Gemini support (skeleton branch raising
  `NotImplementedError`).
- Does not silently fall back to fake/deterministic embeddings on the durable path.

### Qdrant module (`rag_pipeline/qdrant_store.py`)
- Collection name: `learncycle_chunks`.
- Named vector `dense` (distance `cosine`, size determined at runtime from the embedder).
  This leaves room for a later `sparse` named vector for hybrid search without a schema
  break.
- `ensure_collection(dim)` creates the collection on demand and is idempotent if it exists.
- `upsert_chunks(chunks)` writes one point per chunk under the named vector `dense`. Points
  use `rag_chunks.id` as the Qdrant point ID. Empty input is a no-op.
- `delete_points_by_source(user_id, source_type, source_id)` removes all points for a given
  source (used on reindex and on source deletion).
- `search_chunks(query_vector, user_id, source_types, top_k)` always filters by `user_id`,
  optionally by `source_types`, and queries the `dense` named vector.

### Qdrant point payload
For each chunk, the payload contains (only if available; `None` filtered out, but
meaningful falsey values like `0` or `""` are preserved):

`chunk_id`, `user_id`, `source_type`, `source_id`,
`pdf_id`, `note_id`, `annotation_id`, `document_id`,
`page_index`, `title`, `heading`, `text`, `metadata`,
`content_hash`, `chunk_index`, `created_at`, `updated_at`.

### Worker integration (`rag_pipeline/worker.py`)
- After Supabase writes the final chunks (with returned UUIDs), the worker embeds the chunk
  texts and upserts them into Qdrant.
- Reindex of a note/annotation/PDF source: old Qdrant points for that source are deleted
  before the refreshed points are inserted.
- Deletion of a source (note/annotation row no longer present): Qdrant points for that
  source are deleted.
- Existing chunking behaviour is unchanged.
- After successful Qdrant upsert, `rag_chunks` rows are updated with
  `embedding_status='completed'`, `embedding_model`, `embedded_at`, `qdrant_collection`,
  `qdrant_point_id`.
- On embedding or Qdrant failure, affected chunks are updated with
  `embedding_status='failed'` and a clear `embedding_error` message; the job is **not**
  marked `completed` so the existing retry path applies.

### Dense retrieval search (Phase 5)
- Public helper / function:
  `search_chunks(query: str, user_id: str, source_types: list[str] | None = None, top_k: int = 10)`.
- Embeds the query with the same `Embedder`/model used for indexing.
- Searches Qdrant on named vector `dense`.
- Always filters by `user_id`; optionally by `source_types`.
- Returns normalized result dicts containing `chunk_id`, `text`, `score`, `source_type`,
  `source_id`, `page_index`, `title`/`heading`, `metadata`.

### Supabase schema additions (`supabase/migrations/`)
A new idempotent migration adds (using `add column if not exists` /
`create index if not exists`):

- columns on `rag_chunks`:
  `embedding_status text default 'pending'`, `embedding_model text`,
  `embedded_at timestamptz`, `qdrant_collection text`, `qdrant_point_id text`,
  `embedding_error text`
- indexes:
  `rag_chunks_embedding_status_idx on rag_chunks(embedding_status)`,
  `rag_chunks_qdrant_point_id_idx on rag_chunks(qdrant_point_id)`

The migration must not change RLS or service-role behaviour, and must be safe to run on a
DB where some of these columns/indexes already exist.

## Non-Functional Requirements
- **Source of truth:** Supabase remains authoritative. A failed Qdrant write must not corrupt
  Supabase state — chunks remain in a clear `failed` state with `embedding_error`, and the
  job retries via the existing `rag_index_jobs` failure path.
- **Idempotency:** Re-running a job for the same source produces the same set of points
  (deterministic delete-before-insert) with no stale duplicates in Qdrant.
- **Provider switchability:** The embedding module is structured so a Gemini branch can be
  added without changing the worker.
- **Hybrid-readiness:** The `dense` named vector leaves room for a later `sparse` named
  vector without breaking the schema or migration path.
- **Test isolation:** All tests run without network access — `Embedder` and `QdrantStore`
  are injected/mocked.

## Acceptance Criteria
- [ ] PDF, note, and annotation jobs still complete successfully end-to-end.
- [ ] `rag_chunks` rows are still written exactly as before.
- [ ] Supabase stores embedding/Qdrant metadata (`embedding_status`, `embedding_model`,
      `embedded_at`, `qdrant_collection`, `qdrant_point_id`, `embedding_error`).
- [ ] Qdrant collection `learncycle_chunks` is created automatically if missing.
- [ ] Qdrant collection uses named vector `dense`.
- [ ] Qdrant contains exactly one point per `rag_chunks` row with the documented payload.
- [ ] Updating a note or annotation does not leave stale Qdrant points.
- [ ] Deleting a note or annotation removes the corresponding Qdrant points.
- [ ] Dense retrieval search works with `user_id` and optional `source_type` filters and
      never returns chunks from another user.
- [ ] Embedding or Qdrant failures do not mark RAG jobs as `completed`.
- [ ] `pytest rag_pipeline/tests` passes.
- [ ] `npx tsc --noEmit` still passes.

## Out of Scope
- Pydantic AI / agentic retrieval.
- Hybrid search (dense + sparse).
- Sparse embeddings.
- Reranking (cross-encoder or LLM-based).
- Knowledge graph retrieval.
- Web search.
- Chat UI changes.
- LLM answer generation.
- Frontend changes outside of any internal helper this track may expose.
