# Implementation Plan: Core Management & PDF Upload

## Phase 0: Project Scaffold & Auth

- [x] Task: Scaffold Next.js App (bf8156e)
    - [x] Run create-next-app with TypeScript, Tailwind, App Router, src/ dir.
    - [x] Install Shadcn UI and configure theme.
    - [x] Install @supabase/supabase-js, @supabase/ssr, react-dropzone.
    - [x] Create .env.local with Supabase project URL and anon key.
    - [x] Create Supabase browser client (src/lib/supabase/client.ts).
    - [x] Create Supabase server client (src/lib/supabase/server.ts).
    - [x] Create auth middleware (middleware.ts) for route protection.
    - [x] Build magic-link login page (src/app/login/page.tsx).
    - [x] Build auth callback route (src/app/auth/callback/route.ts).
- [ ] Task: Conductor - User Manual Verification 'Phase 0: Project Scaffold & Auth' (Protocol in workflow.md)

## Phase 1: Foundation & Data Modeling

- [x] Task: Supabase Setup & Database Schema (17967d4)
    - [x] Apply migration: create semesters, subjects, weeks, pdfs tables with user_id FK to auth.users.
    - [x] Enable RLS on all tables with user-scoped policies.
    - [x] Create private Supabase Storage bucket for PDFs with user-scoped storage policy.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Foundation & Data Modeling' (Protocol in workflow.md)

## Phase 2: Core API & Folder Management

- [x] Task: Folder Management Logic (e7414ee)
    - [x] Implement Server Actions for creating, listing, and deleting Semesters.
    - [x] Implement Server Actions for creating, listing, and deleting Subjects.
    - [x] Implement Server Actions for creating, listing, and deleting Weeks.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Core API & Folder Management' (Protocol in workflow.md)

## Phase 3: PDF Upload & Integration

- [x] Task: Secure PDF Upload Functionality (417acd5)
    - [x] Develop the server-side uploadPdf action to upload to Supabase Storage (private bucket, signed URLs).
    - [x] Associate uploaded PDF metadata (name, size, storage_path) with the corresponding Week in the DB.
    - [x] Implement deletePdf action (removes Storage object + metadata row).
    - [x] Implement the client-side PdfDropzone component using react-dropzone.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: PDF Upload & Integration' (Protocol in workflow.md)

## Phase 4: UI & Navigation

- [x] Task: Minimalist Dashboard & Navigation (417acd5)
    - [x] Build collapsible sidebar with Semester → Subject → Week → PDF tree.
    - [x] Inline create/delete controls at each tree level.
    - [x] Main content area: selected Week's PDF list + PdfDropzone.
    - [x] Empty states and loading skeletons.
    - [ ] Custom icon component wired to Figma SVG assets.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: UI & Navigation' (Protocol in workflow.md)
