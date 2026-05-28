-- Catalog-level audit for rag_document_primers storage and RLS.
-- Run against a local Supabase database after migrations.

do $$
declare
  rls_enabled boolean;
  own_row_policy_count integer;
  unique_constraint_count integer;
  json_default_count integer;
begin
  select c.relrowsecurity
    into rls_enabled
  from pg_class c
  join pg_namespace n on n.oid = c.relnamespace
  where n.nspname = 'public'
    and c.relname = 'rag_document_primers';

  if rls_enabled is not true then
    raise exception 'rag_document_primers must have RLS enabled';
  end if;

  select count(*)
    into own_row_policy_count
  from pg_policies
  where schemaname = 'public'
    and tablename = 'rag_document_primers'
    and cmd = 'ALL'
    and roles = array['authenticated']
    and qual = '(user_id = auth.uid())'
    and with_check = '(user_id = auth.uid())';

  if own_row_policy_count <> 1 then
    raise exception 'rag_document_primers must expose exactly one authenticated own-row CRUD policy, got %',
      own_row_policy_count;
  end if;

  select count(*)
    into unique_constraint_count
  from pg_constraint constraint_row
  join pg_class table_row on table_row.oid = constraint_row.conrelid
  join pg_namespace namespace_row on namespace_row.oid = table_row.relnamespace
  where namespace_row.nspname = 'public'
    and table_row.relname = 'rag_document_primers'
    and constraint_row.contype = 'u'
    and constraint_row.conkey = array[
      (
        select attnum from pg_attribute
        where attrelid = table_row.oid and attname = 'user_id'
      ),
      (
        select attnum from pg_attribute
        where attrelid = table_row.oid and attname = 'source_type'
      ),
      (
        select attnum from pg_attribute
        where attrelid = table_row.oid and attname = 'source_id'
      )
    ]::smallint[];

  if unique_constraint_count <> 1 then
    raise exception 'rag_document_primers must be unique on (user_id, source_type, source_id), got %',
      unique_constraint_count;
  end if;

  select count(*)
    into json_default_count
  from information_schema.columns
  where table_schema = 'public'
    and table_name = 'rag_document_primers'
    and column_name in ('main_topics', 'key_terms', 'learning_objectives', 'page_ranges')
    and is_nullable = 'NO'
    and column_default like '%[]%';

  if json_default_count <> 4 then
    raise exception 'rag_document_primers json array columns must be non-null with [] defaults, got %',
      json_default_count;
  end if;
end $$;
