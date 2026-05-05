-- Extend RAG jobs from PDF-only jobs to generic source jobs.

alter table public.rag_index_jobs
  alter column pdf_id drop not null;

alter table public.rag_index_jobs
  add column if not exists source_type text,
  add column if not exists source_id uuid,
  add column if not exists note_id uuid,
  add column if not exists annotation_id uuid;

update public.rag_index_jobs
  set source_type = 'pdf',
      source_id = pdf_id
  where source_type is null
    and pdf_id is not null;

alter table public.rag_index_jobs
  alter column source_type set not null,
  alter column source_id set not null;

alter table public.rag_index_jobs
  alter column source_type set default 'pdf';

do $$
begin
  if not exists (
    select 1
    from pg_constraint
    where conname = 'rag_index_jobs_source_type_check'
      and conrelid = 'public.rag_index_jobs'::regclass
  ) then
    alter table public.rag_index_jobs
      add constraint rag_index_jobs_source_type_check
      check (source_type in ('pdf', 'note', 'annotation_comment'));
  end if;

  if not exists (
    select 1
    from pg_constraint
    where conname = 'rag_index_jobs_note_id_fkey'
      and conrelid = 'public.rag_index_jobs'::regclass
  ) then
    alter table public.rag_index_jobs
      add constraint rag_index_jobs_note_id_fkey
      foreign key (note_id) references public.notes(id) on delete cascade;
  end if;

  if not exists (
    select 1
    from pg_constraint
    where conname = 'rag_index_jobs_annotation_id_fkey'
      and conrelid = 'public.rag_index_jobs'::regclass
  ) then
    alter table public.rag_index_jobs
      add constraint rag_index_jobs_annotation_id_fkey
      foreign key (annotation_id)
      references public.pdf_annotations(id)
      on delete cascade;
  end if;
end $$;

create index if not exists rag_index_jobs_user_source_idx
  on public.rag_index_jobs(user_id, source_type, source_id, created_at desc);

create index if not exists rag_index_jobs_note_id_idx
  on public.rag_index_jobs(note_id)
  where note_id is not null;

create index if not exists rag_index_jobs_annotation_id_idx
  on public.rag_index_jobs(annotation_id)
  where annotation_id is not null;
