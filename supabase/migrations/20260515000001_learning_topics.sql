create table if not exists public.learning_topics (
  id              uuid primary key default gen_random_uuid(),
  user_id         uuid not null references auth.users(id) on delete cascade,
  course_id       uuid not null references public.courses(id) on delete cascade,
  pdf_id          uuid references public.pdfs(id) on delete cascade,
  name            text not null,
  normalized_name text not null,
  evidence        jsonb not null default '{}',
  score           numeric not null default 0,
  mastery_state   text not null default 'unassessed'
    check (mastery_state in ('unassessed', 'learning', 'understood', 'needs_review')),
  created_at      timestamptz not null default now(),
  updated_at      timestamptz not null default now(),
  unique (user_id, course_id, pdf_id, normalized_name)
);

create index if not exists learning_topics_user_course_pdf_idx
  on public.learning_topics(user_id, course_id, pdf_id, score desc);

alter table public.learning_topics enable row level security;

create policy "learning_topics: user owns row"
  on public.learning_topics
  for all
  using (user_id = auth.uid())
  with check (user_id = auth.uid());
