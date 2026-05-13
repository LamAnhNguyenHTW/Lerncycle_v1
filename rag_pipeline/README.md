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
- `GRAPH_ENABLED` (default: `false`)
- `GRAPH_EXTRACTION_ENABLED` (default: `false`)
- `GRAPH_RETRIEVAL_ENABLED` (default: `false`)
- `GRAPH_STORE_PROVIDER` (must be `neo4j`)
- `NEO4J_URI`
- `NEO4J_USER`
- `NEO4J_PASSWORD`
- `NEO4J_DATABASE` (default: `neo4j`)
- `GRAPH_MAX_NODES_PER_CHUNK` (default: `12`)
- `GRAPH_MAX_EDGES_PER_CHUNK` (default: `20`)
- `GRAPH_RETRIEVAL_TOP_K` (default: `8`)
- `GRAPH_CONTEXT_MAX_CHARS` (default: `6000`)
- `GRAPH_SOURCE_TYPE` (must be `knowledge_graph`)
- `WEB_SEARCH_ENABLED` (default: `false`)
- `WEB_SEARCH_PROVIDER` (must be `tavily`)
- `WEB_SEARCH_TOP_K` (default: `5`)
- `WEB_SEARCH_TIMEOUT_SECONDS` (default: `15`)
- `WEB_SEARCH_MAX_QUERY_CHARS` (default: `300`)
- `WEB_SEARCH_MAX_CONTEXT_SOURCES` (default: `5`)
- `WEB_SEARCH_MAX_CHARS_PER_SOURCE` (default: `1000`)
- `WEB_SEARCH_MAX_TOTAL_CONTEXT_CHARS` (default: `4000`)
- `WEB_SEARCH_SOURCE_TYPE` (must be `web`)
- `TAVILY_API_KEY`
- `INTENT_CLASSIFIER_ENABLED` (default: `false`)
- `INTENT_CLASSIFIER_PROVIDER` (must be `openai`)
- `INTENT_CLASSIFIER_MODEL` (default: `gpt-4.1-mini`)
- `INTENT_CLASSIFIER_TIMEOUT_SECONDS` (default: `10`)
- `INTENT_CLASSIFIER_MAX_RECENT_MESSAGES` (default: `4`)
- `INTENT_CLASSIFIER_MAX_MESSAGE_CHARS` (default: `1000`)
- `INTENT_CLASSIFIER_FALLBACK_ENABLED` (default: `true`)
- `RAG_INTERNAL_API_KEY`

Next.js server-only chat variables:

- `RAG_API_URL` (local default: `http://localhost:8001`)
- `RAG_INTERNAL_API_KEY`
- `CHAT_MEMORY_ENABLED`
- `CHAT_MEMORY_SUMMARY_THRESHOLD`
- `CHAT_MEMORY_SUMMARY_INTERVAL`
- `CHAT_MEMORY_KEEP_RECENT`
- `CHAT_MEMORY_MAX_SUMMARY_CHARS`
- `GRAPH_RETRIEVAL_ENABLED`
- `WEB_SEARCH_ENABLED`
- `INTENT_CLASSIFIER_ENABLED`

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

Neo4j GraphRAG is optional and disabled by default. When
`GRAPH_EXTRACTION_ENABLED=true`, successful source indexing can enqueue a
`knowledge_graph` job that reads existing `rag_chunks`, extracts compact nodes
and relationships with an injected or OpenAI LLM client, validates strict JSON,
and writes user-scoped `Concept` and `Chunk` nodes to Neo4j. Neo4j is only a
rebuildable graph retrieval index; Supabase remains canonical and Qdrant remains
the vector index.

When `GRAPH_RETRIEVAL_ENABLED=true`, relationship and concept-map questions such
as "Wie hängt Process Mining mit Event Logs zusammen?" add a separate
Knowledge Graph Context section to the normal text-chunk prompt. Text chunks
remain the primary factual grounding, and graph retrieval failures fall back to
normal RAG. Browser requests cannot directly force graph retrieval; Next.js sets
`graph_mode` server-side from environment config. Graph source cards expose only
short snippets and backing chunk ids, never raw Cypher or full graph dumps.

Web Search is optional and disabled by default. When `WEB_SEARCH_ENABLED=true`,
Next.js can forward `web_mode="on"` only when the browser sends the
`enableWebSearch` hint. The Python RAG service calls Tavily through
`rag_pipeline.web_search.search_web`, normalizes results as ephemeral
`source_type="web"` chunks, and merges them with local RAG context for that one
answer. Web results are never written to Supabase, Qdrant, `rag_chunks`, or
`rag_index_jobs`. Missing Tavily credentials, provider errors, timeouts, and
empty results return safe web metadata and fall back to local retrieval.

Example direct RAG request:

```json
{
  "query": "Was ist aktuell neu bei OpenAI Agents SDK?",
  "user_id": "server-verified-user-id",
  "web_mode": "on"
}
```

Known limitations: web search does not add agentic browsing, web result caching,
web embeddings, or Supabase persistence for web results.

Intent classification is optional and disabled by default. When enabled, the RAG
service classifies the current query plus a small truncated recent-message
window into a validated JSON intent with retrieval flags. The classifier is only
a routing aid: chat memory still requires a verified `session_id`, web search
still requires `WEB_SEARCH_ENABLED=true`, and graph intent is returned as
metadata with `graph_available=false` in this track. LLM classifier failures
fall back to deterministic German/English keyword routing and never fail the
chat response.

Example classifier output:

```json
{
  "question_type": "concept_relationship",
  "needs_pdf": true,
  "needs_notes": false,
  "needs_annotations": false,
  "needs_chat_memory": false,
  "needs_graph": true,
  "needs_web": false,
  "confidence": 0.91,
  "reasoning_summary": "The user asks about the relationship between two concepts."
}
```

Retrieval planning is optional and disabled by default. When
`RETRIEVAL_PLANNER_ENABLED=true` and a validated intent is available, the RAG
service builds a deterministic `RetrievalPlan` with ordered steps such as
`search_pdf_chunks`, `search_notes`, `search_annotations`, `search_chat_memory`,
and `web_search`. The planner does not classify intent itself and does not run
autonomous loops; it only converts intent flags into safe executable steps.

Execution still enforces the same controls as the existing RAG path:
`user_id` is always server-side, selected PDF scope is preserved, chat memory
requires a verified `session_id`, and web search still requires
`WEB_SEARCH_ENABLED=true`. Graph questions may add a disabled
`query_knowledge_graph` step with `graph_available=false`; Graph RAG execution
is not part of this planner track.

Example retrieval plan:

```json
{
  "question_type": "concept_relationship",
  "planner_version": "v1",
  "fallback_used": false,
  "steps": [
    {
      "tool": "search_pdf_chunks",
      "query": "BPMN Process Mining Zusammenhang",
      "top_k": 6,
      "status": "enabled",
      "source_types": ["pdf"]
    },
    {
      "tool": "query_knowledge_graph",
      "query": "BPMN Process Mining Zusammenhang",
      "top_k": 5,
      "status": "disabled",
      "reason": "Graph RAG is not implemented yet."
    }
  ]
}
```

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

Out of scope for the current RAG path remains web search, agentic RAG, graph
embeddings, Neo4j GDS algorithms, visual mindmap rendering, raw chat message
embeddings, cross-session global memory, and a UI memory management page.
