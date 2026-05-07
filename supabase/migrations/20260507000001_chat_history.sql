create table if not exists public.chat_sessions (
  id         uuid primary key default gen_random_uuid(),
  user_id    uuid not null references auth.users(id) on delete cascade,
  course_id  uuid references public.courses(id) on delete cascade,
  title      text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.chat_messages (
  id          uuid primary key default gen_random_uuid(),
  session_id  uuid not null references public.chat_sessions(id) on delete cascade,
  user_id     uuid not null references auth.users(id) on delete cascade,
  role        text not null check (role in ('user', 'assistant')),
  content     text not null,
  sources     jsonb not null default '[]',
  pdf_ids     uuid[] not null default '{}',
  created_at  timestamptz not null default now()
);

create index if not exists chat_sessions_user_updated_idx
  on public.chat_sessions(user_id, updated_at desc);

create index if not exists chat_messages_session_created_idx
  on public.chat_messages(session_id, created_at asc);

alter table public.chat_sessions enable row level security;
alter table public.chat_messages enable row level security;

create policy "chat_sessions: user owns row"
  on public.chat_sessions
  for all
  using (user_id = auth.uid())
  with check (user_id = auth.uid());

create policy "chat_messages: user owns row"
  on public.chat_messages
  for all
  using (user_id = auth.uid())
  with check (user_id = auth.uid());
