-- Chat memory rolling summaries for session-scoped RAG retrieval.

create table if not exists public.chat_memory_summaries (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  session_id uuid not null references public.chat_sessions(id) on delete cascade,
  summary text not null default '',
  message_range_start timestamptz,
  message_range_end timestamptz,
  represented_message_count integer not null default 0 check (represented_message_count >= 0),
  source_type text not null default 'chat_memory' check (source_type = 'chat_memory'),
  rag_job_id uuid references public.rag_index_jobs(id) on delete set null,
  rag_chunk_id uuid references public.rag_chunks(id) on delete set null,
  qdrant_point_id text,
  embedding_status text not null default 'pending'
    check (embedding_status in ('pending', 'processing', 'completed', 'failed', 'skipped')),
  embedded_at timestamptz,
  indexing_error text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique(user_id, session_id)
);

create index if not exists chat_memory_summaries_user_id_idx
  on public.chat_memory_summaries(user_id);
create index if not exists chat_memory_summaries_session_id_idx
  on public.chat_memory_summaries(session_id);
create index if not exists chat_memory_summaries_user_session_idx
  on public.chat_memory_summaries(user_id, session_id);
create index if not exists chat_memory_summaries_embedding_status_idx
  on public.chat_memory_summaries(embedding_status);

alter table public.chat_memory_summaries enable row level security;

drop policy if exists "chat_memory_summaries: user select own rows"
  on public.chat_memory_summaries;
create policy "chat_memory_summaries: user select own rows"
  on public.chat_memory_summaries
  for select
  using (auth.uid() = user_id);

drop policy if exists "chat_memory_summaries: user insert own rows"
  on public.chat_memory_summaries;
create policy "chat_memory_summaries: user insert own rows"
  on public.chat_memory_summaries
  for insert
  with check (auth.uid() = user_id);

drop policy if exists "chat_memory_summaries: user update own rows"
  on public.chat_memory_summaries;
create policy "chat_memory_summaries: user update own rows"
  on public.chat_memory_summaries
  for update
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

drop policy if exists "chat_memory_summaries: user delete own rows"
  on public.chat_memory_summaries;
create policy "chat_memory_summaries: user delete own rows"
  on public.chat_memory_summaries
  for delete
  using (auth.uid() = user_id);

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
    alter table public.rag_index_jobs
      add constraint rag_index_jobs_source_type_check
      check (source_type in ('pdf', 'note', 'annotation_comment', 'chat_memory'));
  end if;
end $$;
