-- Add explicit job kinds while preserving Variant A scheduling:
-- one active job per (user_id, source_type, source_id), regardless of kind.
-- Learning-graph extraction is enqueued only after index_source succeeds, so the
-- existing rag_index_jobs_one_active_source_idx remains the intended guard.

alter table public.rag_index_jobs
  add column if not exists job_kind text not null default 'index_source';

do $$
begin
  if not exists (
    select 1
    from pg_constraint
    where conname = 'rag_index_jobs_job_kind_check'
      and conrelid = 'public.rag_index_jobs'::regclass
  ) then
    alter table public.rag_index_jobs
      add constraint rag_index_jobs_job_kind_check
      check (job_kind in ('index_source', 'extract_learning_graph'));
  end if;
end $$;
