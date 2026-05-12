-- Store the learning context scope used to share chat memory across sessions.

alter table public.chat_memory_summaries
  add column if not exists course_id uuid references public.courses(id) on delete cascade,
  add column if not exists pdf_id uuid references public.pdfs(id) on delete set null;

create index if not exists chat_memory_summaries_user_course_idx
  on public.chat_memory_summaries(user_id, course_id)
  where course_id is not null;

create index if not exists chat_memory_summaries_user_pdf_idx
  on public.chat_memory_summaries(user_id, pdf_id)
  where pdf_id is not null;
