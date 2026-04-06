-- Tightens and fixes RLS policies for uploads to the 'avatars' bucket.
-- Ensures users can only access files under their own prefix: {user_id}/...

drop policy if exists "avatars: authenticated can upload" on storage.objects;
drop policy if exists "avatars: authenticated can update" on storage.objects;
drop policy if exists "avatars: authenticated can delete" on storage.objects;
drop policy if exists "avatars: authenticated can read own" on storage.objects;

create policy "avatars: authenticated can upload"
  on storage.objects
  for insert
  to authenticated
  with check (
    bucket_id = 'avatars'
    and (storage.foldername(name))[1] = auth.uid()::text
  );

create policy "avatars: authenticated can update"
  on storage.objects
  for update
  to authenticated
  using (
    bucket_id = 'avatars'
    and (storage.foldername(name))[1] = auth.uid()::text
  )
  with check (
    bucket_id = 'avatars'
    and (storage.foldername(name))[1] = auth.uid()::text
  );

create policy "avatars: authenticated can delete"
  on storage.objects
  for delete
  to authenticated
  using (
    bucket_id = 'avatars'
    and (storage.foldername(name))[1] = auth.uid()::text
  );

-- Needed for upsert flows that may read the existing object row.
create policy "avatars: authenticated can read own"
  on storage.objects
  for select
  to authenticated
  using (
    bucket_id = 'avatars'
    and (storage.foldername(name))[1] = auth.uid()::text
  );
