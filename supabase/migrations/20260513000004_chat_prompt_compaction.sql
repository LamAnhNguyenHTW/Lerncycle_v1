alter table public.chat_sessions
  add column if not exists context_summary text,
  add column if not exists context_summary_cursor integer not null default 0;

do $$
begin
  if not exists (
    select 1
    from pg_constraint
    where conname = 'chat_sessions_context_summary_cursor_nonnegative'
      and conrelid = 'public.chat_sessions'::regclass
  ) then
    alter table public.chat_sessions
      add constraint chat_sessions_context_summary_cursor_nonnegative
      check (context_summary_cursor >= 0);
  end if;
end $$;
