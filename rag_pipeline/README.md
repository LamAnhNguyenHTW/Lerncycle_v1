# RAG Pipeline

Python worker for PDF, note, and annotation chunk generation with durable
dense and sparse embeddings, plus Qdrant hybrid indexing.

## RAG Chat Service

The deployable chat path is non-agentic single-turn RAG:

```text
Browser -> Next.js /api/chat -> FastAPI /rag/answer -> Qdrant hybrid retrieval -> OpenAI
```

Next.js handles Supabase auth and extracts `user_id` from the server-side
session. The browser never calls the RAG service directly. FastAPI handles
hybrid retrieval, context building, and answer generation. Requests from Next.js
to FastAPI must include `Authorization: Bearer RAG_INTERNAL_API_KEY`; keep the
RAG service internal/private where possible.

Page numbers returned to the UI and included in LLM context are user-facing:
`page = page_index + 1`.

Required server-side environment variables:

- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`

Embedding and retrieval configuration:

- `EMBEDDING_PROVIDER` (`openai` or `gemini`)
- `EMBEDDING_MODEL` (default: `text-embedding-3-small`)
- `EMBEDDING_BATCH_SIZE` (default: `100`)
- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `GEMINI_OUTPUT_DIMENSIONALITY`
- `QDRANT_URL`
- `QDRANT_API_KEY`
- `QDRANT_COLLECTION` (default: `learncycle_chunks`)
- `SPARSE_PROVIDER` (default: `fastembed`)
- `SPARSE_MODEL` (default: `Qdrant/bm25`)
- `SPARSE_VECTOR_NAME` (default: `sparse`)
- `SPARSE_ENABLED` (default: `true`; set `false` for dense-only debugging)
- `HYBRID_FUSION` (default: `rrf`)
- `HYBRID_PREFETCH_LIMIT` (default: `30`)
- `HYBRID_TOP_K` (default: `10`)
- `RERANKING_ENABLED` (default: `false`)
- `RERANKING_PROVIDER` (`noop`, `fastembed`, or `llm`; default: `fastembed`)
- `RERANKING_MODEL`
- `RERANKING_CANDIDATE_K`
- `RERANKING_TOP_K`
- `CHAT_MEMORY_ENABLED` (default: `false`)
- `CHAT_MEMORY_SUMMARY_THRESHOLD` (default: `8`)
- `CHAT_MEMORY_SUMMARY_INTERVAL` (default: `4`)
- `CHAT_MEMORY_KEEP_RECENT` (default: `4`)
- `CHAT_MEMORY_MAX_SUMMARY_CHARS` (default: `2500`)
- `CHAT_MEMORY_RETRIEVAL_ENABLED` (default: `false`)
- `CHAT_MEMORY_DEFAULT_INCLUDED` (default: `false`)
- `CHAT_MEMORY_TOP_K` (default: `2`)
- `CHAT_MEMORY_SOURCE_TYPE` (must be `chat_memory`)
- `RAG_INTERNAL_API_KEY`

Next.js server-only chat variables:

- `RAG_API_URL` (local default: `http://localhost:8001`)
- `RAG_INTERNAL_API_KEY`
- `CHAT_MEMORY_ENABLED`
- `CHAT_MEMORY_SUMMARY_THRESHOLD`
- `CHAT_MEMORY_SUMMARY_INTERVAL`
- `CHAT_MEMORY_KEEP_RECENT`
- `CHAT_MEMORY_MAX_SUMMARY_CHARS`

Durable indexing currently supports OpenAI embeddings. If the durable embedding
provider has no API key, the job fails and retries instead of storing fake
embeddings.

OpenAI refinement example:

```env
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=sk-...
```

Gemini refinement example:

```env
EMBEDDING_PROVIDER=gemini
EMBEDDING_MODEL=gemini-embedding-001
GEMINI_API_KEY=...
GEMINI_OUTPUT_DIMENSIONALITY=1536
```

Run one job locally:

```bash
python -m rag_pipeline.worker
```

Run the RAG API locally:

```bash
uvicorn rag_pipeline.api:app --host 0.0.0.0 --port 8001
npm run dev
```

Docker service:

```bash
docker compose up rag-api
```

The worker uses Docling Hybrid Chunking first and then semantic refinement.
Final chunks are stored in Supabase, embedded with `Embedder` and
`SparseEmbedder`, and mirrored into Qdrant through `QdrantStore`. Supabase
remains the source of truth; Qdrant is only the retrieval index.

Retrieval supports dense, sparse, and hybrid modes:

```python
from rag_pipeline.retrieval import search_chunks, search_sparse_chunks
from rag_pipeline.retrieval import search_hybrid_chunks

dense = search_chunks("Was ist Process Mining?", user_id="...")
sparse = search_sparse_chunks("Process Mining", user_id="...")
hybrid = search_hybrid_chunks("Was ist Process Mining?", user_id="...")
```

Qdrant uses collection `learncycle_chunks` with named vectors `dense` and
`sparse`. The installed FastEmbed stack exposes BM25 as `Qdrant/bm25`, and the
installed `qdrant-client` exposes `models.Modifier.IDF`; new hybrid collections
therefore create BM25 sparse vector config with the IDF modifier.

Hybrid search uses Qdrant server-side Query API with dense and sparse
`Prefetch` plus RRF fusion when available. The code keeps a local RRF fallback
that runs dense and sparse searches separately and fuses the ranked lists.

Optional reranking runs only after user-scoped retrieval. The available
providers are `noop`, `fastembed`, and `llm`. FastEmbed reranking uses a local
cross-encoder. LLM reranking sends compact candidate descriptions to the
configured OpenAI model, costs tokens, and falls back to the original hybrid
ranking on malformed output or errors. Reranking prompts, rewritten queries,
scores, and reasons are internal and are not returned to the browser.

FastEmbed reranking example:

```env
RERANKING_ENABLED=true
RERANKING_PROVIDER=fastembed
RERANKING_MODEL=jinaai/jina-reranker-v2-base-multilingual
RERANKING_CANDIDATE_K=30
RERANKING_TOP_K=8
```

LLM reranking example:

```env
RERANKING_ENABLED=true
RERANKING_PROVIDER=llm
RERANKING_MODEL=gpt-4o-mini
RERANKING_CANDIDATE_K=20
RERANKING_TOP_K=8
```

For LLM reranking, `RERANKING_CANDIDATE_K <= 20` is recommended for token cost;
the hard maximum is `30`. Check the FastEmbed/Jina model license before
commercial or beta use, and monitor LLM model cost before enabling LLM reranking
for beta users.

Chat memory is optional and disabled by default. Next.js verifies the chat
session server-side, stores normal chat messages, then best-effort calls
`/rag/compress` after an assistant answer once threshold and interval guards are
met. The compressed rolling summary is stored in `chat_memory_summaries`, and a
normal `rag_index_jobs` row with `source_type=chat_memory` lets the Python
worker embed one summary chunk into the existing Qdrant collection. Normal
factual questions keep using PDF/note/annotation sources only. Memory-intent
questions such as "Was hatten wir besprochen?" can retrieve the current
session's `chat_memory` chunk when `CHAT_MEMORY_RETRIEVAL_ENABLED=true`.

Raw chat messages are never embedded. Browser requests cannot directly request
`chat_memory` as a source type; Next.js strips it before forwarding and FastAPI
rejects it on `/rag/answer`. Memory source cards expose only a short snippet, not
the full summary.

Existing dense-only collections are not recreated automatically during normal
worker execution. Recreate/rebuild hybrid points only via the explicit
maintenance command:

```bash
python -m rag_pipeline.reindex_qdrant --user-id <user-id>
python -m rag_pipeline.reindex_qdrant --all
python -m rag_pipeline.reindex_qdrant --user-id <user-id> --source-type pdf
python -m rag_pipeline.reindex_qdrant --user-id <user-id> --source-id <source-id>
python -m rag_pipeline.reindex_qdrant --user-id <user-id> --recreate-collection
```

Evaluation helper:

```bash
python -m rag_pipeline.evaluate_retrieval eval_queries.json --user-id <user-id>
```

Out of scope for the current RAG path remains GraphRAG, web search, agentic
RAG, raw chat message embeddings, cross-session global memory, and a UI memory
management page.
