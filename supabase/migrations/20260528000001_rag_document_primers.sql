create table if not exists public.rag_document_primers (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  source_type text not null
    check (source_type in ('pdf')),
  source_id text not null,
  pdf_id uuid references public.pdfs(id) on delete cascade,
  title text,
  summary text,
  main_topics jsonb not null default '[]'::jsonb,
  key_terms jsonb not null default '[]'::jsonb,
  learning_objectives jsonb not null default '[]'::jsonb,
  page_ranges jsonb not null default '[]'::jsonb,
  content_hash text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (user_id, source_type, source_id)
);

alter table public.rag_document_primers enable row level security;

create index if not exists rag_document_primers_user_source_idx
  on public.rag_document_primers(user_id, source_type, source_id);

create index if not exists rag_document_primers_pdf_idx
  on public.rag_document_primers(pdf_id);

drop policy if exists "rag_document_primers: user owns row" on public.rag_document_primers;
create policy "rag_document_primers: user owns row"
  on public.rag_document_primers
  for all
  to authenticated
  using (user_id = auth.uid())
  with check (user_id = auth.uid());
