# Implementation Plan: Core Management & PDF Upload

## Phase 0: Project Scaffold & Auth

- [~] Task: Scaffold Next.js App
    - [~] Run create-next-app with TypeScript, Tailwind, App Router, src/ dir.
    - [ ] Install Shadcn UI and configure theme.
    - [ ] Install @supabase/supabase-js, @supabase/ssr, react-dropzone.
    - [ ] Create .env.local with Supabase project URL and anon key.
    - [ ] Create Supabase browser client (src/lib/supabase/client.ts).
    - [ ] Create Supabase server client (src/lib/supabase/server.ts).
    - [ ] Create auth middleware (middleware.ts) for route protection.
    - [ ] Build magic-link login page (src/app/login/page.tsx).
    - [ ] Build auth callback route (src/app/auth/callback/route.ts).
- [ ] Task: Conductor - User Manual Verification 'Phase 0: Project Scaffold & Auth' (Protocol in workflow.md)

## Phase 1: Foundation & Data Modeling

- [ ] Task: Supabase Setup & Database Schema
    - [ ] Apply migration: create semesters, subjects, weeks, pdfs tables with user_id FK to auth.users.
    - [ ] Enable RLS on all tables with user-scoped policies.
    - [ ] Create private Supabase Storage bucket for PDFs with user-scoped storage policy.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Foundation & Data Modeling' (Protocol in workflow.md)

## Phase 2: Core API & Folder Management

- [ ] Task: Folder Management Logic
    - [ ] Implement Server Actions for creating, listing, and deleting Semesters.
    - [ ] Implement Server Actions for creating, listing, and deleting Subjects.
    - [ ] Implement Server Actions for creating, listing, and deleting Weeks.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Core API & Folder Management' (Protocol in workflow.md)

## Phase 3: PDF Upload & Integration

- [ ] Task: Secure PDF Upload Functionality
    - [ ] Implement the client-side PdfDropzone component using react-dropzone.
    - [ ] Develop the server-side uploadPdf action to upload to Supabase Storage (private bucket, signed URLs).
    - [ ] Associate uploaded PDF metadata (name, size, storage_path) with the corresponding Week in the DB.
    - [ ] Implement deletePdf action (removes Storage object + metadata row).
- [ ] Task: Conductor - User Manual Verification 'Phase 3: PDF Upload & Integration' (Protocol in workflow.md)

## Phase 4: UI & Navigation

- [ ] Task: Minimalist Dashboard & Navigation
    - [ ] Build collapsible sidebar with Semester → Subject → Week → PDF tree.
    - [ ] Inline create/delete controls at each tree level.
    - [ ] Main content area: selected Week's PDF list + PdfDropzone.
    - [ ] Empty states and loading skeletons.
    - [ ] Custom icon component wired to Figma SVG assets.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: UI & Navigation' (Protocol in workflow.md)
