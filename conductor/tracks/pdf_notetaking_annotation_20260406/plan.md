# Implementation Plan: PDF Side-by-Side Note-Taking & Annotation

## Phase 1: Environment & Dependency Setup
- [ ] Task: Install required libraries for PDF, Editor, and Split-Pane
    - [ ] Install `react-pdf-viewer` and `@react-pdf-viewer/default-layout`
    - [ ] Install `@react-pdf-viewer/highlight` for annotation support
    - [ ] Install `@tiptap/react`, `@tiptap/starter-kit`, and related TipTap dependencies
    - [ ] Install `react-resizable-panels`
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Setup' (Protocol in workflow.md)

## Phase 2: Database Schema & API
- [ ] Task: Create `notes` table in Supabase
    - [ ] Define columns: `id`, `file_id` (foreign key), `user_id` (foreign key), `content` (json/text), `created_at`, `updated_at`
- [ ] Task: Create `pdf_annotations` table in Supabase
    - [ ] Define columns: `id`, `file_id` (foreign key), `user_id` (foreign key), `page_index` (int), `highlight_area` (json), `comment` (text), `created_at`, `updated_at`
- [ ] Task: Implement Supabase Edge Functions or API routes for fetching/saving notes and annotations
    - [ ] Write tests for fetching/saving data
    - [ ] Implement the API logic
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Database' (Protocol in workflow.md)

## Phase 3: UI Foundation - Resizable Layout
- [ ] Task: Create the `StudyInterface` container component
    - [ ] Write unit tests for the layout structure and responsiveness
    - [ ] Implement the `react-resizable-panels` layout with two panes
    - [ ] Add the draggable slider (handle) between panes
- [ ] Task: Conductor - User Manual Verification 'Phase 3: UI Foundation' (Protocol in workflow.md)

## Phase 4: PDF Viewer Integration
- [ ] Task: Implement the `PDFViewer` component
    - [ ] Write unit tests for loading and rendering the PDF
    - [ ] Integrate `react-pdf-viewer` to load the PDF from storage
- [ ] Task: Conductor - User Manual Verification 'Phase 4: PDF Viewer Integration' (Protocol in workflow.md)

## Phase 5: PDF Annotation System (Highlights & Comments)
- [ ] Task: Implement active highlighting on the PDF
    - [ ] Write tests for annotation logic (creating, loading)
    - [ ] Integrate the `react-pdf-viewer` highlight plugin
    - [ ] Store highlights in the `pdf_annotations` table
- [ ] Task: Implement commenting on highlights
    - [ ] Write tests for adding/editing comments on a highlight
    - [ ] Create a UI for the user to input comments associated with a highlight
    - [ ] Store and retrieve comments from the `pdf_annotations` table
- [ ] Task: Conductor - User Manual Verification 'Phase 5: Annotation System' (Protocol in workflow.md)

## Phase 6: Rich-Text Editor (TipTap) Integration
- [ ] Task: Implement the `NoteEditor` component using TipTap
    - [ ] Write unit tests for basic editor functionality and persistence triggers
    - [ ] Implement the TipTap editor with a basic starter kit
- [ ] Task: Implement Auto-save logic for `NoteEditor`
    - [ ] Add debouncing (e.g., 500ms-1s) to the editor's update cycle
    - [ ] Connect the auto-save to the `notes` table via the API
- [ ] Task: Conductor - User Manual Verification 'Phase 6: Editor Integration' (Protocol in workflow.md)

## Phase 7: Final Integration & State Sync
- [ ] Task: Connect components for a unified state
    - [ ] Write integration tests for the full user flow (Open PDF -> Annotate -> Take Notes -> Refresh)
    - [ ] Ensure the UI updates correctly when annotations or notes are modified
- [ ] Task: Conductor - User Manual Verification 'Phase 7: Final Integration' (Protocol in workflow.md)
