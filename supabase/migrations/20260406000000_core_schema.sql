-- Core schema for Lerncycle
-- Creates the hierarchical structure: semesters > subjects > weeks > pdfs
-- All tables have RLS enabled with user-scoped policies.

-- ─────────────────────────────────────────────
-- TABLES
-- ─────────────────────────────────────────────

create table if not exists public.semesters (
  id         uuid primary key default gen_random_uuid(),
  user_id    uuid not null references auth.users(id) on delete cascade,
  name       text not null,
  created_at timestamptz not null default now()
);

create table if not exists public.subjects (
  id          uuid primary key default gen_random_uuid(),
  user_id     uuid not null references auth.users(id) on delete cascade,
  semester_id uuid not null references public.semesters(id) on delete cascade,
  name        text not null,
  created_at  timestamptz not null default now()
);

create table if not exists public.weeks (
  id         uuid primary key default gen_random_uuid(),
  user_id    uuid not null references auth.users(id) on delete cascade,
  subject_id uuid not null references public.subjects(id) on delete cascade,
  name       text not null,
  created_at timestamptz not null default now()
);

create table if not exists public.pdfs (
  id           uuid primary key default gen_random_uuid(),
  user_id      uuid not null references auth.users(id) on delete cascade,
  week_id      uuid not null references public.weeks(id) on delete cascade,
  name         text not null,
  storage_path text not null unique,
  size_bytes   bigint not null default 0,
  created_at   timestamptz not null default now()
);

-- ─────────────────────────────────────────────
-- INDEXES
-- ─────────────────────────────────────────────

create index if not exists semesters_user_id_idx on public.semesters(user_id);
create index if not exists subjects_semester_id_idx on public.subjects(semester_id);
create index if not exists weeks_subject_id_idx on public.weeks(subject_id);
create index if not exists pdfs_week_id_idx on public.pdfs(week_id);

-- ─────────────────────────────────────────────
-- ROW LEVEL SECURITY
-- ─────────────────────────────────────────────

alter table public.semesters enable row level security;
alter table public.subjects  enable row level security;
alter table public.weeks     enable row level security;
alter table public.pdfs      enable row level security;

-- Semesters: users can only see and modify their own rows.
create policy "semesters: user owns row"
  on public.semesters
  for all
  using  (user_id = auth.uid())
  with check (user_id = auth.uid());

-- Subjects: users can only see and modify their own rows.
create policy "subjects: user owns row"
  on public.subjects
  for all
  using  (user_id = auth.uid())
  with check (user_id = auth.uid());

-- Weeks: users can only see and modify their own rows.
create policy "weeks: user owns row"
  on public.weeks
  for all
  using  (user_id = auth.uid())
  with check (user_id = auth.uid());

-- PDFs: users can only see and modify their own rows.
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

-- Allow users to upload only to their own prefix: {user_id}/{week_id}/...
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
