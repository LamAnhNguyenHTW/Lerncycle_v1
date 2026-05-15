-- Revision feature: flashcards (SM-2), mindmap (view-only over learning_structure),
-- and multiple-choice mock tests. All tables are RLS-protected with user_id = auth.uid().

create table if not exists public.flashcard_decks (
  id                uuid primary key default gen_random_uuid(),
  user_id           uuid not null references auth.users(id) on delete cascade,
  course_id         uuid references public.courses(id) on delete set null,
  title             text not null,
  source_pdf_ids    uuid[] not null default '{}',
  status            text not null default 'pending'
                    check (status in ('pending', 'ready', 'failed')),
  generation_error  text,
  created_at        timestamptz not null default now(),
  updated_at        timestamptz not null default now()
);

create table if not exists public.flashcards (
  id                 uuid primary key default gen_random_uuid(),
  deck_id            uuid not null references public.flashcard_decks(id) on delete cascade,
  user_id            uuid not null references auth.users(id) on delete cascade,
  front              text not null,
  back               text not null,
  source_chunk_ids   uuid[] not null default '{}',
  ease_factor        real not null default 2.5,
  interval_days      integer not null default 0,
  repetitions        integer not null default 0,
  due_at             timestamptz not null default now(),
  last_reviewed_at   timestamptz,
  created_at         timestamptz not null default now()
);

create index if not exists flashcards_due_idx
  on public.flashcards (deck_id, due_at);

create index if not exists flashcard_decks_user_updated_idx
  on public.flashcard_decks (user_id, updated_at desc);

create table if not exists public.mock_tests (
  id                uuid primary key default gen_random_uuid(),
  user_id           uuid not null references auth.users(id) on delete cascade,
  course_id         uuid references public.courses(id) on delete set null,
  title             text not null,
  source_pdf_ids    uuid[] not null default '{}',
  question_count    integer not null default 0,
  status            text not null default 'pending'
                    check (status in ('pending', 'ready', 'failed')),
  generation_error  text,
  created_at        timestamptz not null default now()
);

create index if not exists mock_tests_user_created_idx
  on public.mock_tests (user_id, created_at desc);

create table if not exists public.mock_test_questions (
  id                uuid primary key default gen_random_uuid(),
  mock_test_id      uuid not null references public.mock_tests(id) on delete cascade,
  user_id           uuid not null references auth.users(id) on delete cascade,
  prompt            text not null,
  choices           jsonb not null,
  correct_index     integer not null check (correct_index between 0 and 3),
  explanation       text,
  source_chunk_ids  uuid[] not null default '{}',
  position          integer not null
);

create index if not exists mock_test_questions_test_position_idx
  on public.mock_test_questions (mock_test_id, position);

create table if not exists public.mock_test_attempts (
  id              uuid primary key default gen_random_uuid(),
  mock_test_id    uuid not null references public.mock_tests(id) on delete cascade,
  user_id         uuid not null references auth.users(id) on delete cascade,
  score_percent   real not null,
  answers         jsonb not null,
  started_at      timestamptz not null default now(),
  completed_at    timestamptz not null default now()
);

create index if not exists mock_test_attempts_test_completed_idx
  on public.mock_test_attempts (mock_test_id, completed_at desc);

alter table public.flashcard_decks      enable row level security;
alter table public.flashcards           enable row level security;
alter table public.mock_tests           enable row level security;
alter table public.mock_test_questions  enable row level security;
alter table public.mock_test_attempts   enable row level security;

do $$ begin
  if not exists (select 1 from pg_policies
                 where schemaname = 'public' and tablename = 'flashcard_decks'
                   and policyname = 'flashcard_decks: user owns row') then
    create policy "flashcard_decks: user owns row"
      on public.flashcard_decks for all
      using (user_id = auth.uid())
      with check (user_id = auth.uid());
  end if;

  if not exists (select 1 from pg_policies
                 where schemaname = 'public' and tablename = 'flashcards'
                   and policyname = 'flashcards: user owns row') then
    create policy "flashcards: user owns row"
      on public.flashcards for all
      using (user_id = auth.uid())
      with check (user_id = auth.uid());
  end if;

  if not exists (select 1 from pg_policies
                 where schemaname = 'public' and tablename = 'mock_tests'
                   and policyname = 'mock_tests: user owns row') then
    create policy "mock_tests: user owns row"
      on public.mock_tests for all
      using (user_id = auth.uid())
      with check (user_id = auth.uid());
  end if;

  if not exists (select 1 from pg_policies
                 where schemaname = 'public' and tablename = 'mock_test_questions'
                   and policyname = 'mock_test_questions: user owns row') then
    create policy "mock_test_questions: user owns row"
      on public.mock_test_questions for all
      using (user_id = auth.uid())
      with check (user_id = auth.uid());
  end if;

  if not exists (select 1 from pg_policies
                 where schemaname = 'public' and tablename = 'mock_test_attempts'
                   and policyname = 'mock_test_attempts: user owns row') then
    create policy "mock_test_attempts: user owns row"
      on public.mock_test_attempts for all
      using (user_id = auth.uid())
      with check (user_id = auth.uid());
  end if;
end $$;
