create table if not exists public.usage_events (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  feature text not null,
  quantity numeric not null,
  unit text not null,
  model text,
  estimated_cost_usd numeric not null default 0,
  session_id uuid references public.chat_sessions(id) on delete set null,
  created_at timestamptz not null default now(),
  constraint usage_events_feature_check
    check (feature in (
      'realtime_voice',
      'voice_stt',
      'voice_tts',
      'chat',
      'active_learning',
      'revision_generation',
      'embeddings_upload'
    )),
  constraint usage_events_unit_check
    check (unit in ('minutes', 'responses', 'messages', 'generations', 'documents')),
  constraint usage_events_non_negative_quantity_check
    check (quantity >= 0 and estimated_cost_usd >= 0)
);

alter table public.usage_events enable row level security;
alter table public.usage_events force row level security;

create index if not exists usage_events_user_feature_created_at_idx
  on public.usage_events(user_id, feature, created_at desc);

create index if not exists usage_events_feature_created_at_idx
  on public.usage_events(feature, created_at desc);

drop policy if exists "Users can read their own usage events" on public.usage_events;
create policy "Users can read their own usage events"
  on public.usage_events
  for select
  to authenticated
  using (auth.uid() = user_id);

revoke all on public.usage_events from anon, authenticated;
grant select on public.usage_events to authenticated;
