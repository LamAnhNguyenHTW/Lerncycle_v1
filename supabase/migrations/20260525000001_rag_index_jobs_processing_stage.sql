-- Fine-grained processing status for user-facing RAG readiness UI.

alter table public.rag_index_jobs
  add column if not exists processing_stage text,
  add column if not exists stage_updated_at timestamptz,
  add column if not exists stage_error text;

do $$
begin
  if not exists (
    select 1
    from pg_constraint
    where conname = 'rag_index_jobs_processing_stage_check'
      and conrelid = 'public.rag_index_jobs'::regclass
  ) then
    alter table public.rag_index_jobs
      add constraint rag_index_jobs_processing_stage_check
      check (
        processing_stage is null
        or processing_stage in (
          'queued',
          'parsing',
          'indexing_dense',
          'indexing_sparse',
          'graph_extracting',
          'completed',
          'failed'
        )
      );
  end if;
end $$;

create index if not exists rag_index_jobs_user_source_idx
  on public.rag_index_jobs(user_id, source_type, source_id, created_at desc);

create or replace view public.v_source_processing_status_raw
with (security_invoker = true)
as
with latest_index_jobs as (
  select distinct on (user_id, source_type, source_id)
    id,
    user_id,
    source_type,
    source_id,
    pdf_id,
    note_id,
    annotation_id,
    status,
    processing_stage,
    stage_error,
    stage_updated_at,
    updated_at,
    created_at
  from public.rag_index_jobs
  where job_kind = 'index_source'
    and source_type in ('pdf', 'note', 'annotation_comment')
  order by user_id, source_type, source_id, created_at desc, id desc
),
latest_graph_jobs as (
  select distinct on (user_id, source_type, source_id)
    id,
    user_id,
    source_type,
    source_id,
    pdf_id,
    note_id,
    annotation_id,
    status,
    processing_stage,
    stage_error,
    stage_updated_at,
    updated_at,
    created_at
  from public.rag_index_jobs
  where job_kind = 'extract_learning_graph'
    and source_type in ('pdf', 'note', 'annotation_comment')
  order by user_id, source_type, source_id, created_at desc, id desc
),
source_keys as (
  select user_id, source_type, source_id from latest_index_jobs
  union
  select user_id, source_type, source_id from latest_graph_jobs
)
select
  source_keys.user_id,
  source_keys.source_type,
  source_keys.source_id,
  case
    when source_keys.source_type = 'pdf' then pdfs.id
    when source_keys.source_type = 'note' then notes.pdf_id
    when source_keys.source_type = 'annotation_comment' then pdf_annotations.pdf_id
  end as pdf_id,
  latest_index_jobs.processing_stage as rag_stage,
  latest_index_jobs.status as rag_status,
  latest_index_jobs.stage_error as rag_stage_error,
  latest_graph_jobs.processing_stage as graph_stage,
  latest_graph_jobs.status as graph_status,
  latest_graph_jobs.stage_error as graph_stage_error,
  exists (
    select 1
    from public.rag_chunks
    where rag_chunks.user_id = source_keys.user_id
      and rag_chunks.source_type = source_keys.source_type
      and rag_chunks.source_id = source_keys.source_id
  ) as has_chunks,
  greatest(
    coalesce(
      latest_index_jobs.stage_updated_at,
      latest_index_jobs.updated_at,
      latest_index_jobs.created_at,
      latest_graph_jobs.stage_updated_at,
      latest_graph_jobs.updated_at,
      latest_graph_jobs.created_at
    ),
    coalesce(
      latest_graph_jobs.stage_updated_at,
      latest_graph_jobs.updated_at,
      latest_graph_jobs.created_at,
      latest_index_jobs.stage_updated_at,
      latest_index_jobs.updated_at,
      latest_index_jobs.created_at
    )
  ) as updated_at
from source_keys
left join latest_index_jobs
  on latest_index_jobs.user_id = source_keys.user_id
  and latest_index_jobs.source_type = source_keys.source_type
  and latest_index_jobs.source_id = source_keys.source_id
left join latest_graph_jobs
  on latest_graph_jobs.user_id = source_keys.user_id
  and latest_graph_jobs.source_type = source_keys.source_type
  and latest_graph_jobs.source_id = source_keys.source_id
left join public.pdfs
  on source_keys.source_type = 'pdf'
  and pdfs.user_id = source_keys.user_id
  and pdfs.id = source_keys.source_id
left join public.notes
  on source_keys.source_type = 'note'
  and notes.user_id = source_keys.user_id
  and notes.id = source_keys.source_id
left join public.pdf_annotations
  on source_keys.source_type = 'annotation_comment'
  and pdf_annotations.user_id = source_keys.user_id
  and pdf_annotations.id = source_keys.source_id
where
  (source_keys.source_type = 'pdf' and pdfs.id is not null)
  or (source_keys.source_type = 'note' and notes.id is not null)
  or (
    source_keys.source_type = 'annotation_comment'
    and pdf_annotations.id is not null
  );
