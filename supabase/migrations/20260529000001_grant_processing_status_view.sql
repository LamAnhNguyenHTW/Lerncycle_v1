-- Grant read access on the processing-status view to authenticated users.
-- The view uses security_invoker = true, so per-user RLS on the underlying
-- tables still applies; this only allows the authenticated role to select
-- the view at all (missing grant caused "permission denied for view").

grant select on public.v_source_processing_status_raw to authenticated;
