alter table public.rag_chunks
  add column if not exists sparse_embedding_status text default 'pending',
  add column if not exists sparse_embedding_model text,
  add column if not exists sparse_embedded_at timestamptz,
  add column if not exists sparse_embedding_error text;

create index if not exists rag_chunks_sparse_embedding_status_idx
  on public.rag_chunks(sparse_embedding_status);
