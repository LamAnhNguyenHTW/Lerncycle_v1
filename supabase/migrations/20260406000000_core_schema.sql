-- Core schema for Lerncycle
-- Creates the structure: courses > folders > pdfs
-- All tables have RLS enabled with user-scoped policies.

-- ─────────────────────────────────────────────
-- TABLES
-- ─────────────────────────────────────────────

create table if not exists public.courses (
  id         uuid primary key default gen_random_uuid(),
  user_id    uuid not null references auth.users(id) on delete cascade,
  name       text not null,
  created_at timestamptz not null default now()
);

create table if not exists public.folders (
  id         uuid primary key default gen_random_uuid(),
  user_id    uuid not null references auth.users(id) on delete cascade,
  course_id  uuid not null references public.courses(id) on delete cascade,
  name       text not null,
  created_at timestamptz not null default now()
);

create table if not exists public.pdfs (
  id           uuid primary key default gen_random_uuid(),
  user_id      uuid not null references auth.users(id) on delete cascade,
  folder_id    uuid not null references public.folders(id) on delete cascade,
  name         text not null,
  storage_path text not null unique,
  size_bytes   bigint not null default 0,
  created_at   timestamptz not null default now()
);

-- ─────────────────────────────────────────────
-- INDEXES
-- ─────────────────────────────────────────────

create index if not exists courses_user_id_idx on public.courses(user_id);
create index if not exists folders_course_id_idx on public.folders(course_id);
create index if not exists pdfs_folder_id_idx on public.pdfs(folder_id);

-- ─────────────────────────────────────────────
-- ROW LEVEL SECURITY
-- ─────────────────────────────────────────────

alter table public.courses enable row level security;
alter table public.folders  enable row level security;
alter table public.pdfs     enable row level security;

create policy "courses: user owns row"
  on public.courses
  for all
  using  (user_id = auth.uid())
  with check (user_id = auth.uid());

create policy "folders: user owns row"
  on public.folders
  for all
  using  (user_id = auth.uid())
  with check (user_id = auth.uid());

create policy "pdfs: user owns row"
  on public.pdfs
  for all
  using  (user_id = auth.uid())
  with check (user_id = auth.uid());

-- ─────────────────────────────────────────────
-- STORAGE
-- ─────────────────────────────────────────────
-- Run these statements in the Supabase dashboard SQL editor
-- AFTER creating the 'pdfs' bucket via Storage > New Bucket (private).

-- Allow users to upload only to their own prefix: {user_id}/{folder_id}/...
create policy "pdfs storage: user can upload"
  on storage.objects
  for insert
  to authenticated
  with check (
    bucket_id = 'pdfs'
    and (storage.foldername(name))[1] = auth.uid()::text
  );

-- Allow users to read only their own objects.
create policy "pdfs storage: user can read"
  on storage.objects
  for select
  to authenticated
  using (
    bucket_id = 'pdfs'
    and (storage.foldername(name))[1] = auth.uid()::text
  );

-- Allow users to delete only their own objects.
create policy "pdfs storage: user can delete"
  on storage.objects
  for delete
  to authenticated
  using (
    bucket_id = 'pdfs'
    and (storage.foldername(name))[1] = auth.uid()::text
  );
