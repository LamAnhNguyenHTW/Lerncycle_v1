@AGENTS.md

# Lerncycle — Project Guide for AI Agents

## Project Overview

Lerncycle is an AI-powered learning companion (web app). It lets students organize study materials in a Semester → Subject → Week hierarchy, upload PDFs, and use AI features (chatbot, Feynman technique, flashcards).

Full product definition: [conductor/product.md](conductor/product.md)
Design guidelines: [conductor/product-guidelines.md](conductor/product-guidelines.md)

## Tech Stack

- **Framework:** Next.js 16 (App Router, TypeScript, `src/` layout) — read `node_modules/next/dist/docs/` before writing any Next.js code
- **UI:** shadcn/ui components + Tailwind CSS v4 — Notion-style minimalist aesthetic
- **Backend:** Supabase (Postgres + Auth + Storage via `@supabase/ssr`)
- **AI:** OpenAI GPT-4 (planned)
- **Deployment:** Vercel

Full tech stack: [conductor/tech-stack.md](conductor/tech-stack.md)

## Architecture Patterns

- **Supabase browser client:** `src/lib/supabase/client.ts` → `createClient()` — use in Client Components
- **Supabase server client:** `src/lib/supabase/server.ts` → `async createClient()` — use in Server Components, Server Actions, Route Handlers
- **Auth:** Magic-link via `supabase.auth.signInWithOtp`; session refreshed in `src/middleware.ts`; callback at `src/app/auth/callback/route.ts`
- **Server Actions:** All in `src/actions/` — `'use server'`, return `{error?: string}`
- **Route protection:** Middleware redirects unauthenticated users to `/login`; redirects logged-in users away from `/login`

## Code Style

Follow [conductor/code_styleguides/typescript.md](conductor/code_styleguides/typescript.md):
- Named exports only (no default exports for non-page files)
- `const`/`let` only, no `var`
- Single quotes, explicit semicolons
- No `any` — use `unknown` or specific types
- No `#private` fields — use TypeScript `private`
- JSDoc `/** */` for public functions

## Development Commands

```bash
npm install          # install dependencies
npm run dev          # start dev server (http://localhost:3000)
npm run build        # production build
npm run lint         # run ESLint
```

## Workflow & Planning

All work follows the TDD workflow defined in [conductor/workflow.md](conductor/workflow.md):
1. Pick next `[ ]` task from the plan, mark `[~]`
2. Write failing tests (Red), implement (Green), refactor
3. Commit with conventional commit message, attach git note summary
4. Mark task `[x]` with short SHA in plan, commit plan update
5. On phase completion: run Phase Completion Verification Protocol (see workflow.md)

Active track plan: [conductor/tracks/core_mgmt_20260406/plan.md](conductor/tracks/core_mgmt_20260406/plan.md)
All tracks: [conductor/tracks.md](conductor/tracks.md)

## Current Implementation Status

### Phase 0: Project Scaffold & Auth — COMPLETE
All scaffold and auth files are implemented:
- `src/lib/supabase/client.ts` — browser Supabase client
- `src/lib/supabase/server.ts` — server Supabase client
- `src/middleware.ts` — session refresh + route protection
- `src/app/login/page.tsx` — magic-link login UI
- `src/app/auth/callback/route.ts` — OTP exchange handler
- `src/actions/auth.ts` — `signInWithEmail`, `signOut`

### Phase 1: Foundation & Data Modeling — PENDING
Supabase DB migration (tables + RLS + storage bucket) not yet applied.

### Phase 2: Core API & Folder Management — COMPLETE (ahead of schedule)
- `src/actions/semesters.ts` — `createSemester`, `deleteSemester`
- `src/actions/subjects.ts` — `createSubject`, `deleteSubject`
- `src/actions/weeks.ts` — `createWeek`, `deleteWeek`

### Phase 3: PDF Upload — PARTIALLY COMPLETE
- `src/actions/pdfs.ts` — `uploadPdf`, `getPdfSignedUrl`, `deletePdf` ✅
- `PdfDropzone` client component — NOT YET BUILT

### Phase 4: UI & Navigation — NOT STARTED
Dashboard, sidebar, and content area still to build.

## Environment Variables

See `.env.local` (not committed). Required keys:
- `NEXT_PUBLIC_SUPABASE_URL`
- `NEXT_PUBLIC_SUPABASE_ANON_KEY`
- `NEXT_PUBLIC_SITE_URL` (optional, defaults to `http://localhost:3000`)
