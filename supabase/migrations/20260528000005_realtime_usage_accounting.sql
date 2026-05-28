alter table public.voice_realtime_sessions
  add column if not exists duration_seconds integer not null default 0
    check (duration_seconds >= 0),
  add column if not exists usage_reported_at timestamptz;

create index if not exists voice_realtime_sessions_usage_reported_idx
  on public.voice_realtime_sessions(user_id, usage_reported_at)
  where usage_reported_at is null;
