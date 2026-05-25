-- Seed-and-assert checks for v_source_processing_status_raw.
-- Run against a local Supabase database after migrations, then roll back.

begin;

insert into auth.users (
  id,
  aud,
  role,
  email,
  encrypted_password,
  email_confirmed_at,
  created_at,
  updated_at
)
values (
  '00000000-0000-0000-0000-00000000a001',
  'authenticated',
  'authenticated',
  'processing-status@example.test',
  'not-used',
  now(),
  now(),
  now()
)
on conflict (id) do nothing;

insert into public.courses (id, user_id, name)
values (
  '00000000-0000-0000-0000-00000000c001',
  '00000000-0000-0000-0000-00000000a001',
  'Processing Status Test'
)
on conflict (id) do nothing;

insert into public.pdfs (id, user_id, course_id, folder_id, name, storage_path, size_bytes)
values (
  '00000000-0000-0000-0000-00000000f001',
  '00000000-0000-0000-0000-00000000a001',
  '00000000-0000-0000-0000-00000000c001',
  null,
  'status.pdf',
  '00000000-0000-0000-0000-00000000a001/status.pdf',
  1
)
on conflict (id) do nothing;

insert into public.rag_index_jobs (
  id,
  user_id,
  source_type,
  source_id,
  pdf_id,
  job_kind,
  status,
  processing_stage,
  created_at,
  updated_at,
  stage_updated_at
)
values
  (
    '00000000-0000-0000-0000-00000000j001',
    '00000000-0000-0000-0000-00000000a001',
    'pdf',
    '00000000-0000-0000-0000-00000000f001',
    '00000000-0000-0000-0000-00000000f001',
    'index_source',
    'failed',
    'parsing',
    '2026-05-25 08:00:00+00',
    '2026-05-25 08:00:00+00',
    null
  ),
  (
    '00000000-0000-0000-0000-00000000j002',
    '00000000-0000-0000-0000-00000000a001',
    'pdf',
    '00000000-0000-0000-0000-00000000f001',
    '00000000-0000-0000-0000-00000000f001',
    'index_source',
    'completed',
    'completed',
    '2026-05-25 09:00:00+00',
    '2026-05-25 09:05:00+00',
    null
  ),
  (
    '00000000-0000-0000-0000-00000000j003',
    '00000000-0000-0000-0000-00000000a001',
    'pdf',
    '00000000-0000-0000-0000-00000000f001',
    '00000000-0000-0000-0000-00000000f001',
    'extract_learning_graph',
    'processing',
    'graph_extracting',
    '2026-05-25 10:00:00+00',
    '2026-05-25 10:00:00+00',
    '2026-05-25 10:15:00+00'
  ),
  (
    '00000000-0000-0000-0000-00000000j004',
    '00000000-0000-0000-0000-00000000a001',
    'pdf',
    '00000000-0000-0000-0000-00000000dead',
    '00000000-0000-0000-0000-00000000dead',
    'index_source',
    'completed',
    'completed',
    '2026-05-25 11:00:00+00',
    '2026-05-25 11:00:00+00',
    '2026-05-25 11:00:00+00'
  );

insert into public.rag_chunks (
  id,
  user_id,
  source_type,
  source_id,
  pdf_id,
  content,
  content_hash,
  embedding_status
)
values (
  '00000000-0000-0000-0000-00000000b001',
  '00000000-0000-0000-0000-00000000a001',
  'pdf',
  '00000000-0000-0000-0000-00000000f001',
  '00000000-0000-0000-0000-00000000f001',
  'Chunk text',
  'processing-status-hash',
  'completed'
)
on conflict (user_id, source_type, source_id, content_hash) do nothing;

do $$
declare
  row_count integer;
  status_row public.v_source_processing_status_raw%rowtype;
  has_business_logic_column boolean;
begin
  select count(*)
    into row_count
  from public.v_source_processing_status_raw
  where user_id = '00000000-0000-0000-0000-00000000a001';

  if row_count <> 1 then
    raise exception 'Expected deleted source to be omitted and one row to remain, got %', row_count;
  end if;

  select *
    into status_row
  from public.v_source_processing_status_raw
  where source_id = '00000000-0000-0000-0000-00000000f001';

  if status_row.rag_stage <> 'completed' or status_row.rag_status <> 'completed' then
    raise exception 'Expected most recent index_source job to win, got %/%',
      status_row.rag_stage,
      status_row.rag_status;
  end if;

  if status_row.graph_stage <> 'graph_extracting' or status_row.graph_status <> 'processing' then
    raise exception 'Expected graph job fields to be raw, got %/%',
      status_row.graph_stage,
      status_row.graph_status;
  end if;

  if status_row.has_chunks is not true then
    raise exception 'Expected has_chunks=true for seeded rag_chunks row';
  end if;

  if status_row.updated_at <> '2026-05-25 10:15:00+00'::timestamptz then
    raise exception 'Expected updated_at to prefer stage_updated_at fallback chain, got %',
      status_row.updated_at;
  end if;

  select exists (
    select 1
    from information_schema.columns
    where table_schema = 'public'
      and table_name = 'v_source_processing_status_raw'
      and column_name in ('overall_stage', 'rag_ready', 'graph_ready', 'user_safe_error')
  )
    into has_business_logic_column;

  if has_business_logic_column then
    raise exception 'Raw view must not expose business-logic columns';
  end if;
end $$;

rollback;
