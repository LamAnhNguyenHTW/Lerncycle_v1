-- Beta signups captured from the public marketing landing page.
-- RLS: insert allowed for anon + authenticated; no select/update/delete
-- policies — the list is readable only via the service role in Supabase Studio.

create table if not exists public.beta_signups (
  id          uuid primary key default gen_random_uuid(),
  email       text not null,
  locale      text not null check (locale in ('de', 'en')),
  source      text,
  user_agent  text,
  created_at  timestamptz not null default now()
);

-- Case-insensitive uniqueness — used by the server action to detect
-- duplicate signups and return `alreadySubscribed: true`.
create unique index if not exists beta_signups_email_lower_idx
  on public.beta_signups (lower(email));

alter table public.beta_signups enable row level security;

-- Insert is the only allowed operation for non-service-role clients.
-- `WITH CHECK (true)` because the landing page must work for fully
-- unauthenticated visitors (`anon` role).
drop policy if exists "beta_signups_insert_public" on public.beta_signups;
create policy "beta_signups_insert_public"
  on public.beta_signups
  for insert
  to anon, authenticated
  with check (true);

-- RLS policies don't override base table privileges — anon/authenticated
-- also need the raw INSERT grant, otherwise Postgres returns 42501
-- (permission denied) before RLS is even evaluated.
grant insert on public.beta_signups to anon, authenticated;
