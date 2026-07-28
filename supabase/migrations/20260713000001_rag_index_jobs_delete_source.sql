-- Allow explicit source-deletion cleanup jobs (track rag_data_lifecycle_20260613).
-- delete_source jobs carry the target in source_type/source_id (+ metadata.deleted_pdf_id)
-- and are enqueued with pdf_id/note_id/annotation_id = NULL so they survive the
-- ON DELETE CASCADE of the source row they clean up after.

alter table public.rag_index_jobs
  drop constraint if exists rag_index_jobs_job_kind_check;

alter table public.rag_index_jobs
  add constraint rag_index_jobs_job_kind_check
  check (job_kind in ('index_source', 'extract_learning_graph', 'delete_source'));
