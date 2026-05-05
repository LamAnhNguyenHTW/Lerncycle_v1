-- Keep only one active RAG job per source.

delete from public.rag_index_jobs keep
using public.rag_index_jobs drop_row
where keep.user_id = drop_row.user_id
  and keep.source_type = drop_row.source_type
  and keep.source_id = drop_row.source_id
  and keep.status in ('pending', 'processing')
  and drop_row.status in ('pending', 'processing')
  and keep.created_at > drop_row.created_at;

create unique index if not exists rag_index_jobs_one_active_source_idx
  on public.rag_index_jobs(user_id, source_type, source_id)
  where status in ('pending', 'processing');
