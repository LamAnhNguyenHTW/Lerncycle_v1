-- RAG ingestion, chunking, and future Qdrant references.

-- Notes currently use (user_id, pdf_id) as primary key. Add a stable note id
-- so RAG chunks can use the generic source_id pattern for notes too.
alter table public.notes
  add column if not exists id uuid default gen_random_uuid();

update public.notes
  set id = gen_random_uuid()
  where id is null;

alter table public.notes
  alter column id set not null;

create unique index if not exists notes_id_key on public.notes(id);

-- Job queue for asynchronous indexing work. PDF uploads insert pending jobs;
-- workers claim jobs server-side with the claim_rag_index_job function below.
create table if not exists public.rag_index_jobs (
  id            uuid primary key default gen_random_uuid(),
  user_id       uuid not null references auth.users(id) on delete cascade,
  source_type   text not null default 'pdf'
    check (source_type in ('pdf', 'note', 'annotation_comment')),
  source_id     uuid not null,
  pdf_id        uuid references public.pdfs(id) on delete cascade,
  note_id       uuid references public.notes(id) on delete cascade,
  annotation_id uuid references public.pdf_annotations(id) on delete cascade,
  status        text not null default 'pending'
    check (status in ('pending', 'processing', 'completed', 'failed')),
  attempts      integer not null default 0,
  locked_at     timestamptz,
  started_at    timestamptz,
  completed_at  timestamptz,
  error_message text,
  created_at    timestamptz not null default now(),
  updated_at    timestamptz not null default now()
);

create index if not exists rag_index_jobs_worker_idx
  on public.rag_index_jobs(status, locked_at, created_at)
  where status in ('pending', 'processing');

create index if not exists rag_index_jobs_user_pdf_idx
  on public.rag_index_jobs(user_id, pdf_id, created_at desc);

create index if not exists rag_index_jobs_user_source_idx
  on public.rag_index_jobs(user_id, source_type, source_id, created_at desc);

-- Status and debugging record for each source processed by the RAG pipeline.
create table if not exists public.rag_documents (
  id                 uuid primary key default gen_random_uuid(),
  user_id            uuid not null references auth.users(id) on delete cascade,
  source_type        text not null
    check (source_type in ('pdf', 'note', 'annotation_comment')),
  source_id          uuid not null,
  pdf_id             uuid references public.pdfs(id) on delete cascade,
  note_id            uuid references public.notes(id) on delete cascade,
  annotation_id      uuid references public.pdf_annotations(id) on delete cascade,
  status             text not null default 'pending'
    check (status in ('pending', 'processing', 'completed', 'failed')),
  error_message      text,
  started_at         timestamptz,
  completed_at       timestamptz,
  docling_version    text,
  chunking_strategy  text not null default 'docling_hybrid_semantic_refinement',
  chunking_version   text not null default 'v1',
  metadata           jsonb not null default '{}',
  created_at         timestamptz not null default now(),
  updated_at         timestamptz not null default now(),
  unique (user_id, source_type, source_id, chunking_strategy, chunking_version)
);

create index if not exists rag_documents_user_source_idx
  on public.rag_documents(user_id, source_type, source_id);

create index if not exists rag_documents_pdf_idx
  on public.rag_documents(pdf_id);

-- Final chunks are stored in Supabase. Embeddings are intentionally not stored
-- here; Qdrant reference fields are prepared for the next pipeline step.
create table if not exists public.rag_chunks (
  id                 uuid primary key default gen_random_uuid(),
  user_id            uuid not null references auth.users(id) on delete cascade,
  rag_document_id    uuid references public.rag_documents(id) on delete cascade,
  source_type        text not null
    check (source_type in ('pdf', 'note', 'annotation_comment')),
  source_id          uuid not null,
  pdf_id             uuid references public.pdfs(id) on delete cascade,
  note_id            uuid references public.notes(id) on delete cascade,
  annotation_id      uuid references public.pdf_annotations(id) on delete cascade,
  page_index         integer,
  heading_path       text[] not null default '{}',
  chunk_kind         text not null default 'text',
  content            text not null,
  metadata           jsonb not null default '{}',
  content_hash       text not null,
  chunking_strategy  text not null default 'docling_hybrid_semantic_refinement',
  chunking_version   text not null default 'v1',
  embedding_status   text not null default 'pending'
    check (embedding_status in ('pending', 'processing', 'completed', 'failed')),
  embedding_model    text,
  embedded_at        timestamptz,
  qdrant_collection  text,
  qdrant_point_id    text,
  created_at         timestamptz not null default now(),
  updated_at         timestamptz not null default now(),
  unique (user_id, source_type, source_id, content_hash)
);

create index if not exists rag_chunks_user_source_idx
  on public.rag_chunks(user_id, source_type, source_id);

create index if not exists rag_chunks_pdf_idx
  on public.rag_chunks(pdf_id);

create index if not exists rag_chunks_qdrant_idx
  on public.rag_chunks(qdrant_collection, qdrant_point_id)
  where qdrant_point_id is not null;

alter table public.rag_index_jobs enable row level security;
alter table public.rag_documents enable row level security;
alter table public.rag_chunks enable row level security;

create policy "rag_index_jobs: user owns row"
  on public.rag_index_jobs
  for all
  using (user_id = auth.uid())
  with check (user_id = auth.uid());

create policy "rag_documents: user owns row"
  on public.rag_documents
  for all
  using (user_id = auth.uid())
  with check (user_id = auth.uid());

create policy "rag_chunks: user owns row"
  on public.rag_chunks
  for all
  using (user_id = auth.uid())
  with check (user_id = auth.uid());

create or replace function public.claim_rag_index_job(
  lock_timeout interval default interval '15 minutes',
  max_attempts integer default 3
)
returns setof public.rag_index_jobs
language sql
security definer
set search_path = public
as $$
  update public.rag_index_jobs job
    set status = 'processing',
        attempts = job.attempts + 1,
        locked_at = now(),
        started_at = coalesce(job.started_at, now()),
        updated_at = now(),
        error_message = null
    where job.id = (
      select id
      from public.rag_index_jobs
      where attempts < max_attempts
        and (
          status = 'pending'
          or (
            status = 'processing'
            and locked_at < now() - lock_timeout
          )
        )
      order by created_at asc
      for update skip locked
      limit 1
    )
    returning job.*;
$$;

revoke all on function public.claim_rag_index_job(interval, integer)
  from public, anon, authenticated;

grant execute on function public.claim_rag_index_job(interval, integer)
  to service_role;
