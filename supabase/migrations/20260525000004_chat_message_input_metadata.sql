alter table public.chat_messages
  add column if not exists input_metadata jsonb;
