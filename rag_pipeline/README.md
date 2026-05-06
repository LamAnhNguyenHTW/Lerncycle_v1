# RAG Pipeline

Python worker for PDF, note, and annotation chunk generation with durable
dense and sparse embeddings, plus Qdrant hybrid indexing.

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

Reranking, Pydantic AI, agentic retrieval, knowledge graph retrieval, web
search, and LLM answer generation remain future phases.
