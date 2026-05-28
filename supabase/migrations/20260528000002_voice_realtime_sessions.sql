create table if not exists public.voice_realtime_sessions (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  session_id uuid not null references public.chat_sessions(id) on delete cascade,
  mode text not null default 'feynman' check (mode in ('feynman')),
  allowed_source_ids uuid[] not null default '{}',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  expires_at timestamptz
);

create unique index if not exists voice_realtime_sessions_user_session_idx
  on public.voice_realtime_sessions(user_id, session_id);

create index if not exists voice_realtime_sessions_user_updated_idx
  on public.voice_realtime_sessions(user_id, updated_at desc);

alter table public.voice_realtime_sessions enable row level security;

drop policy if exists "voice_realtime_sessions: user owns row" on public.voice_realtime_sessions;
create policy "voice_realtime_sessions: user owns row"
  on public.voice_realtime_sessions
  for all
  using (user_id = auth.uid())
  with check (user_id = auth.uid());
