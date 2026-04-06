# Track: Implement PDF-Side-by-Side Note-Taking & Annotation

## Overview
This track focuses on creating a dual-pane study interface where users can view an uploaded PDF on the left and take notes using a rich-text editor on the right. The PDF pane will support active highlighting and commenting directly on its content, while the note pane will handle general summarization and active recall exercises.

## Functional Requirements
- **PDF Viewer:** Integrate `react-pdf-viewer` to display the PDF associated with a selected file.
- **PDF Annotation (Highlight & Comment):** Implement the ability to highlight specific sentences or sections in the PDF and add associated comments directly on the PDF pane.
- **Rich-Text Editor:** Implement a `TipTap` (ProseMirror-based) editor for advanced note-taking on the right side.
- **Splitscreen Layout:** Use `react-resizable-panels` to create a horizontal split layout.
- **Resizable Slider:** Provide a draggable divider to adjust the width of the PDF and note sections.
- **Auto-save:** Notes and PDF annotations (highlights/comments) should automatically save to the database as the user interacts (with debouncing).
- **Context Linking:** Notes and annotations must be uniquely linked to the specific PDF/File ID, page number (for annotations), and the current user.

## Non-Functional Requirements
- **RAG Readiness:** Store notes and PDF annotations in a relational database format (Supabase tables) to facilitate future vectorization and retrieval-augmented generation (RAG).
- **Responsive Design:** Ensure the split-pane layout adapts reasonably to different screen widths.
- **Performance:** Handle PDF rendering and real-time annotation overlay efficiently.

## Acceptance Criteria
- [ ] User can open a specific PDF from their library.
- [ ] The PDF displays correctly on the left side of the screen.
- [ ] User can highlight text and add comments directly onto the PDF.
- [ ] The TipTap editor is functional and editable on the right side.
- [ ] Dragging the divider resizes both panes in real-time.
- [ ] Notes are persisted to the `notes` table in Supabase.
- [ ] PDF annotations (highlights/comments) are persisted to a `pdf_annotations` table in Supabase.
- [ ] Reloading the page restores both general notes and all PDF-side highlights/comments.

## Out of Scope
- Initial PDF upload functionality (assumed already exists).
- Implementation of the AI RAG system or vector database embeddings (this will be a future track).
- Exporting notes or the annotated PDF to external formats (PDF, Markdown export).
