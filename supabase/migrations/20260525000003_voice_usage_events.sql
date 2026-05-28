create table if not exists public.voice_usage_events (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  session_id uuid references public.chat_sessions(id) on delete set null,
  input_seconds integer not null default 0,
  output_chars integer not null default 0,
  provider text not null,
  stt_model text,
  tts_model text,
  created_at timestamptz not null default now(),
  constraint voice_usage_events_provider_check
    check (provider in ('openai', 'gemini')),
  constraint voice_usage_events_non_negative_usage_check
    check (input_seconds >= 0 and output_chars >= 0)
);

alter table public.voice_usage_events enable row level security;
alter table public.voice_usage_events force row level security;

create index if not exists voice_usage_events_user_created_at_idx
  on public.voice_usage_events(user_id, created_at desc);

drop policy if exists "Users can read their own voice usage" on public.voice_usage_events;
create policy "Users can read their own voice usage"
  on public.voice_usage_events
  for select
  to authenticated
  using (auth.uid() = user_id);

revoke all on public.voice_usage_events from anon, authenticated;
grant select on public.voice_usage_events to authenticated;
