# RAG Pipeline

Python worker for PDF, note, and annotation chunk generation with durable
dense embeddings and Qdrant indexing.

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
Final chunks are stored in Supabase, embedded with `Embedder`, and mirrored into
Qdrant through `QdrantStore`. Supabase remains the source of truth; Qdrant is
only the retrieval index.

The active retrieval path is dense search only:

```python
from rag_pipeline.retrieval import search_chunks

results = search_chunks("Welche Notizen habe ich zu RAG?", user_id="...")
```

Qdrant uses collection `learncycle_chunks` and named vector `dense`. Hybrid
search, sparse embeddings, reranking, Pydantic AI, agentic retrieval, and LLM
answer generation are future phases.
