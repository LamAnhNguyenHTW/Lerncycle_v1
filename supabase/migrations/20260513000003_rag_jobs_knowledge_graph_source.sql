-- Allow rebuildable Neo4j GraphRAG jobs.
--
-- Job metadata shape:
--   source_type = 'knowledge_graph'
--   source_id = original_source_id
--   metadata.original_source_type = original source type ('pdf', 'note',
--     'annotation_comment', or 'chat_memory')
--   metadata.original_source_id = original source id
--   metadata.pdf_ids = optional list of scoped backing PDFs

alter table public.rag_index_jobs
  add column if not exists metadata jsonb not null default '{}';

do $$
begin
  if exists (
    select 1
    from pg_constraint
    where conname = 'rag_index_jobs_source_type_check'
      and conrelid = 'public.rag_index_jobs'::regclass
  ) then
    alter table public.rag_index_jobs
      drop constraint rag_index_jobs_source_type_check;
  end if;

  alter table public.rag_index_jobs
    add constraint rag_index_jobs_source_type_check
    check (source_type in (
      'pdf',
      'note',
      'annotation_comment',
      'chat_memory',
      'knowledge_graph'
    ));
end $$;

create unique index if not exists rag_index_jobs_one_active_source_idx
  on public.rag_index_jobs(user_id, source_type, source_id)
  where status in ('pending', 'processing');
