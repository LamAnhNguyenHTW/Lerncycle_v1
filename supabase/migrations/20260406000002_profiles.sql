-- Creates the profiles table and a trigger to auto-insert a row
-- whenever a new user signs up via Supabase Auth.

create table if not exists public.profiles (
  id           uuid primary key references auth.users(id) on delete cascade,
  display_name text,
  avatar_name  text,
  updated_at   timestamptz not null default now()
);

alter table public.profiles enable row level security;

create policy "profiles: user owns row"
  on public.profiles
  for all
  using  (id = auth.uid())
  with check (id = auth.uid());

-- Auto-create a profile row whenever a new auth user is created.
create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer set search_path = public
as $$
begin
  insert into public.profiles (id)
  values (new.id)
  on conflict (id) do nothing;
  return new;
end;
$$;

create or replace trigger on_auth_user_created
  after insert on auth.users
  for each row execute procedure public.handle_new_user();
