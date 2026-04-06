-- Annotations and notes tables for the pdf_notetaking track.

-- ─────────────────────────────────────────────
-- pdf_annotations
-- ─────────────────────────────────────────────

create table if not exists public.pdf_annotations (
  id               uuid primary key default gen_random_uuid(),
  user_id          uuid not null references auth.users(id) on delete cascade,
  pdf_id           uuid not null references public.pdfs(id) on delete cascade,
  page_index       integer not null,
  highlight_areas  jsonb not null default '[]',
  quote            text,
  comment          text,
  color            text not null default 'yellow',
  created_at       timestamptz not null default now()
);

create index if not exists pdf_annotations_pdf_id_idx  on public.pdf_annotations(pdf_id);
create index if not exists pdf_annotations_user_id_idx on public.pdf_annotations(user_id);

alter table public.pdf_annotations enable row level security;

create policy "pdf_annotations: user owns row"
  on public.pdf_annotations
  for all
  using  (user_id = auth.uid())
  with check (user_id = auth.uid());

-- ─────────────────────────────────────────────
-- notes
-- ─────────────────────────────────────────────

create table if not exists public.notes (
  user_id    uuid not null references auth.users(id) on delete cascade,
  pdf_id     uuid not null references public.pdfs(id) on delete cascade,
  content    jsonb not null default '{}',
  updated_at timestamptz not null default now(),
  primary key (user_id, pdf_id)
);

create index if not exists notes_pdf_id_idx on public.notes(pdf_id);

alter table public.notes enable row level security;

create policy "notes: user owns row"
  on public.notes
  for all
  using  (user_id = auth.uid())
  with check (user_id = auth.uid());
