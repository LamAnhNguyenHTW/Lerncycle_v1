# Track Specification: Core Management & PDF Upload

## Overview
This track focuses on building the foundational file management system and PDF upload capabilities for the Lerncycle AI Learning App. It establishes the hierarchical folder structure (Semester > Subject > Week) and the ability for users to securely upload and manage their study materials.

## Scope
*   **Hierarchical Folder Management:**
    *   Create, list, and delete Semester folders.
    *   Create, list, and delete Subject folders within a Semester.
    *   Create, list, and delete Week folders within a Subject.
*   **PDF Upload & Storage:**
    *   Securely upload PDF files to Supabase Storage.
    *   Store PDF metadata (name, size, path) in the PostgreSQL database.
    *   Associate uploaded PDFs with a specific Week folder.
*   **Basic UI:**
    *   A clean, minimalist dashboard for navigating the folder hierarchy.
    *   Intuitive upload interface for adding PDFs to folders.

## Tech Stack (Subset)
*   **Next.js (React):** Frontend framework and routing.
*   **Shadcn UI:** Minimalist UI components.
*   **Supabase:** PostgreSQL database, Storage, and Authentication.

## Key Constraints
*   **Minimalist Aesthetic:** Adhere to the Notion-style design guidelines.
*   **Intuitive Navigation:** Ensure the folder hierarchy is easy to traverse and manage.
