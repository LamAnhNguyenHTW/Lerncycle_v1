-- Catalog-level audit for usage_events RLS and client privileges.
-- Run against a local Supabase database after migrations.

do $$
declare
  rls_enabled boolean;
  rls_forced boolean;
  select_policy_count integer;
  write_policy_count integer;
  client_write_grants integer;
  user_feature_index_count integer;
  feature_index_count integer;
begin
  select c.relrowsecurity, c.relforcerowsecurity
    into rls_enabled, rls_forced
  from pg_class c
  join pg_namespace n on n.oid = c.relnamespace
  where n.nspname = 'public'
    and c.relname = 'usage_events';

  if rls_enabled is not true then
    raise exception 'usage_events must have RLS enabled';
  end if;

  if rls_forced is not true then
    raise exception 'usage_events must force RLS for non-bypass table owners';
  end if;

  select count(*)
    into select_policy_count
  from pg_policies
  where schemaname = 'public'
    and tablename = 'usage_events'
    and cmd = 'SELECT'
    and roles = array['authenticated']
    and qual = '(auth.uid() = user_id)';

  if select_policy_count <> 1 then
    raise exception 'usage_events must expose exactly one authenticated own-user SELECT policy, got %',
      select_policy_count;
  end if;

  select count(*)
    into write_policy_count
  from pg_policies
  where schemaname = 'public'
    and tablename = 'usage_events'
    and cmd in ('INSERT', 'UPDATE', 'DELETE', 'ALL');

  if write_policy_count <> 0 then
    raise exception 'usage_events must not define client write policies, got %',
      write_policy_count;
  end if;

  select count(*)
    into client_write_grants
  from information_schema.role_table_grants
  where table_schema = 'public'
    and table_name = 'usage_events'
    and grantee in ('anon', 'authenticated')
    and privilege_type in ('INSERT', 'UPDATE', 'DELETE');

  if client_write_grants <> 0 then
    raise exception 'usage_events must not grant client write privileges, got %',
      client_write_grants;
  end if;

  select count(*)
    into user_feature_index_count
  from pg_indexes
  where schemaname = 'public'
    and tablename = 'usage_events'
    and indexname = 'usage_events_user_feature_created_at_idx';

  if user_feature_index_count <> 1 then
    raise exception 'usage_events must index (user_id, feature, created_at desc)';
  end if;

  select count(*)
    into feature_index_count
  from pg_indexes
  where schemaname = 'public'
    and tablename = 'usage_events'
    and indexname = 'usage_events_feature_created_at_idx';

  if feature_index_count <> 1 then
    raise exception 'usage_events must index (feature, created_at desc)';
  end if;
end $$;
