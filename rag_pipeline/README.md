# RAG Pipeline

Python worker for PDF ingestion and chunk generation.

Required server-side environment variables:

- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`

Optional for semantic refinement:

- `EMBEDDING_PROVIDER` (`openai` or `gemini`)
- `EMBEDDING_MODEL`
- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `GEMINI_OUTPUT_DIMENSIONALITY`

If the configured provider has no API key, the worker falls back to
deterministic paragraph/sentence refinement. No temporary refinement embeddings
are stored.

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

Prepared for the next Qdrant step:

- `QDRANT_URL`
- `QDRANT_API_KEY`
- `QDRANT_COLLECTION`

Run one job locally:

```bash
python -m rag_pipeline.worker
```

The worker uses Docling Hybrid Chunking first and then semantic refinement.
Temporary OpenAI embeddings are used only for boundary detection and are not
stored in Supabase or Qdrant.
