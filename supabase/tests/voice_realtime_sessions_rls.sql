-- Catalog-level audit for voice_realtime_sessions storage and RLS.
do $$
declare
  policy_count integer;
  unique_columns text[];
begin
  if not exists (
    select 1
    from pg_class c
    join pg_namespace n on n.oid = c.relnamespace
    where n.nspname = 'public'
      and c.relname = 'voice_realtime_sessions'
      and c.relrowsecurity
  ) then
    raise exception 'voice_realtime_sessions must have RLS enabled';
  end if;

  select count(*)
  into policy_count
  from pg_policies
  where schemaname = 'public'
    and tablename = 'voice_realtime_sessions'
    and policyname = 'voice_realtime_sessions: user owns row'
    and roles = array['public']
    and cmd = 'ALL'
    and qual = '(user_id = auth.uid())'
    and with_check = '(user_id = auth.uid())';

  if policy_count != 1 then
    raise exception 'voice_realtime_sessions must expose exactly one authenticated own-row CRUD policy, got %',
      policy_count;
  end if;

  select array_agg(a.attname order by x.ord)
  into unique_columns
  from pg_class table_row
  join pg_index i on i.indrelid = table_row.oid
  join unnest(i.indkey) with ordinality as x(attnum, ord) on true
  join pg_attribute a on a.attrelid = table_row.oid and a.attnum = x.attnum
  where table_row.relname = 'voice_realtime_sessions'
    and i.indisunique
  group by i.indexrelid
  having array_agg(a.attname order by x.ord) = array['user_id', 'session_id'];

  if unique_columns is null then
    raise exception 'voice_realtime_sessions must be unique on (user_id, session_id)';
  end if;
end $$;
