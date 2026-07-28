---
name: prod-supabase-and-grants
description: Production Supabase project differs from local; RAG/voice migrations lack explicit GRANTs and 403 on authenticated
metadata:
  type: project
---

Production (Vercel) and local use **different Supabase projects**:
- Local (`.env.local`): project ref `pyvrgjigxlxzhsitoreo`
- Production (`learncycle-beta.vercel.app`): project ref `tsibqjevhzmavibhzvhz`

So a feature working locally can fail in prod if migrations / grants were not applied to `tsibqjevhzmavibhzvhz`.

Several RAG/voice migrations enable RLS + policies but contain **no explicit `GRANT`** — they relied on Supabase default privileges, which did NOT apply for some tables in the prod project. Result: PostgREST `403 permission denied for table` (PG code 42501) for the `authenticated` role, which surfaced as HTTP 500 in `/api/voice/realtime-token` (and a non-fatal console.error for `realtime_tool_events`).

Fixed in prod by running:
`grant select, insert, update, delete on public.<table> to authenticated;`
for `rag_document_primers`, `voice_realtime_sessions`, `realtime_tool_events`.

Tables touched only via the **service client** (`createServiceClient`, `service_role`) — `usage_events`, `voice_usage_events` — do NOT need authenticated DML grants. `beta_signups` is insert-only by design.

**Why:** migrations omitted explicit grants; default privileges are unreliable across projects.
**How to apply:** when adding a table the app reads/writes via the user (authenticated) client, add an explicit `grant ... to authenticated;` to the migration, not just RLS + policy. Audit query: list public tables where `authenticated` is missing any of SELECT/INSERT/UPDATE/DELETE via `information_schema.role_table_grants`.
