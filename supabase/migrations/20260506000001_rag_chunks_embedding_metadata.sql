alter table public.rag_chunks
  add column if not exists embedding_status text default 'pending',
  add column if not exists embedding_model text,
  add column if not exists embedded_at timestamptz,
  add column if not exists qdrant_collection text,
  add column if not exists qdrant_point_id text,
  add column if not exists embedding_error text;

create index if not exists rag_chunks_embedding_status_idx
  on public.rag_chunks(embedding_status);

create index if not exists rag_chunks_qdrant_point_id_idx
  on public.rag_chunks(qdrant_point_id);
