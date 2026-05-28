create table if not exists public.realtime_tool_events (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  session_id uuid references public.chat_sessions(id) on delete set null,
  query text not null,
  result_count integer not null default 0 check (result_count >= 0),
  latency_ms integer not null default 0 check (latency_ms >= 0),
  created_at timestamptz not null default now()
);

create index if not exists realtime_tool_events_user_created_idx
  on public.realtime_tool_events(user_id, created_at desc);

create index if not exists realtime_tool_events_session_created_idx
  on public.realtime_tool_events(session_id, created_at desc);

alter table public.realtime_tool_events enable row level security;

drop policy if exists "realtime_tool_events: user owns row" on public.realtime_tool_events;
create policy "realtime_tool_events: user owns row"
  on public.realtime_tool_events
  for all
  using (user_id = auth.uid())
  with check (user_id = auth.uid());
