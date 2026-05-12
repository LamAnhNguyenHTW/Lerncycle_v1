-- Allow chat memory rows in RAG document and chunk tracking tables.

do $$
begin
  if exists (
    select 1
    from pg_constraint
    where conname = 'rag_documents_source_type_check'
      and conrelid = 'public.rag_documents'::regclass
  ) then
    alter table public.rag_documents
      drop constraint rag_documents_source_type_check;
    alter table public.rag_documents
      add constraint rag_documents_source_type_check
      check (source_type in ('pdf', 'note', 'annotation_comment', 'chat_memory'));
  end if;

  if exists (
    select 1
    from pg_constraint
    where conname = 'rag_chunks_source_type_check'
      and conrelid = 'public.rag_chunks'::regclass
  ) then
    alter table public.rag_chunks
      drop constraint rag_chunks_source_type_check;
    alter table public.rag_chunks
      add constraint rag_chunks_source_type_check
      check (source_type in ('pdf', 'note', 'annotation_comment', 'chat_memory'));
  end if;
end $$;
