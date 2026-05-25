-- Atomically reset user-facing processing stage when a worker claims a job.

create or replace function public.claim_rag_index_job(
  lock_timeout interval default interval '15 minutes',
  max_attempts integer default 3
)
returns setof public.rag_index_jobs
language sql
security definer
set search_path = public
as $$
  update public.rag_index_jobs job
    set status = 'processing',
        attempts = job.attempts + 1,
        locked_at = now(),
        started_at = coalesce(job.started_at, now()),
        updated_at = now(),
        error_message = null,
        processing_stage = 'parsing',
        stage_error = null,
        stage_updated_at = now()
    where job.id = (
      select id
      from public.rag_index_jobs
      where attempts < max_attempts
        and (
          status = 'pending'
          or (
            status = 'processing'
            and locked_at < now() - lock_timeout
          )
        )
      order by created_at asc
      for update skip locked
      limit 1
    )
    returning job.*;
$$;

revoke all on function public.claim_rag_index_job(interval, integer)
  from public, anon, authenticated;

grant execute on function public.claim_rag_index_job(interval, integer)
  to service_role;
