-- Allow PDFs to be uploaded directly to a course without a folder.
-- Adds course_id (NOT NULL) to pdfs, makes folder_id nullable,
-- and changes the folder FK to ON DELETE SET NULL so that deleting
-- a folder can optionally orphan its PDFs rather than cascade-delete them.

-- Step 1: add course_id if it doesn't already exist
alter table public.pdfs
  add column if not exists course_id uuid references public.courses(id) on delete cascade;

-- Step 2: backfill course_id from the parent folder for all existing rows
update public.pdfs p
  set course_id = f.course_id
  from public.folders f
  where p.folder_id = f.id;

-- Step 3: make course_id NOT NULL (all existing rows are now backfilled)
alter table public.pdfs
  alter column course_id set not null;

-- Step 4: make folder_id nullable
alter table public.pdfs
  alter column folder_id drop not null;

-- Step 5: replace folder FK with ON DELETE SET NULL
--         (so deleting a folder can optionally keep its PDFs as loose files)
alter table public.pdfs
  drop constraint pdfs_folder_id_fkey;

alter table public.pdfs
  add constraint pdfs_folder_id_fkey
    foreign key (folder_id)
    references public.folders(id)
    on delete set null;

-- Index for querying all PDFs by course and for loose-PDF queries
create index if not exists pdfs_course_id_idx on public.pdfs(course_id);
