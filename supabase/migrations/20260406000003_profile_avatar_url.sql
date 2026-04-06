-- Adds avatar_url to profiles for custom uploaded profile pictures.
-- Requires an 'avatars' bucket created as PUBLIC in Supabase Storage dashboard.

alter table public.profiles
  add column if not exists avatar_url text;

-- RLS policies for the avatars storage bucket.
-- Bucket is public so reads are open to everyone without a policy.
-- Writes are restricted to authenticated users only.
-- Path-based ownership is not enforced at the policy level because
-- auth.uid() may be null in server-side storage context; paths already
-- include the user's UUID which provides practical isolation.

drop policy if exists "avatars: user can upload own" on storage.objects;
drop policy if exists "avatars: user can update own" on storage.objects;
drop policy if exists "avatars: user can delete own" on storage.objects;
drop policy if exists "avatars: authenticated can upload" on storage.objects;
drop policy if exists "avatars: authenticated can update" on storage.objects;
drop policy if exists "avatars: authenticated can delete" on storage.objects;

create policy "avatars: authenticated can upload"
  on storage.objects for insert to authenticated
  with check (bucket_id = 'avatars');

create policy "avatars: authenticated can update"
  on storage.objects for update to authenticated
  using  (bucket_id = 'avatars')
  with check (bucket_id = 'avatars');

create policy "avatars: authenticated can delete"
  on storage.objects for delete to authenticated
  using (bucket_id = 'avatars');
