'use client';

import { useState } from 'react';
import { Course, Folder, PdfFile } from '@/lib/data';
import { NotionIcon } from './NotionIcon';
import { createFolder, deleteFolder } from '@/actions/folders';
import { deletePdf } from '@/actions/pdfs';
import { PdfDropzone } from '@/components/PdfDropzone';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';

export function FolderList({ course }: { course: Course }) {
  const [newFolderName, setNewFolderName] = useState('');
  const [isCreating, setIsCreating] = useState(false);

  const handleCreateFolder = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newFolderName.trim()) return;
    await createFolder(course.id, newFolderName);
    setNewFolderName('');
    setIsCreating(false);
  };

  return (
    <div className="w-full mt-12 mb-20">
      <div className="flex items-center justify-between mb-8">
        <div className="flex items-center gap-3">
          <div className="w-1 h-5 bg-primary rounded-full"></div>
          <h2 className="text-xl font-semibold tracking-tight">Materials & Folders</h2>
        </div>
        <button
          onClick={() => setIsCreating(true)}
          className="flex items-center gap-2 rounded-lg bg-black px-4 py-2 text-base font-medium text-white hover:bg-black/80"
        >
          <NotionIcon name="ni-plus" className="w-[22px] h-[22px]" />
          New Folder
        </button>
      </div>

      {/* Direct uploads (no folder) */}
      <LoosePdfsSection courseId={course.id} pdfs={course.loose_pdfs} />

      {/* Folder creation form */}
      {isCreating && (
        <form onSubmit={handleCreateFolder} className="mb-8 p-5 rounded-xl border border-border bg-white shadow-sm flex items-center gap-4">
          <input
            autoFocus
            type="text"
            placeholder="Folder name (e.g. Week 1)"
            className="flex-1 rounded-md border border-border px-4 py-2.5 text-base outline-none focus:border-primary"
            value={newFolderName}
            onChange={(e) => setNewFolderName(e.target.value)}
          />
          <button type="submit" className="text-base font-medium bg-black text-white px-5 py-2.5 rounded-md">
            Save
          </button>
          <button type="button" onClick={() => setIsCreating(false)} className="text-base font-medium px-5 py-2.5 hover:bg-black/5 rounded-md">
            Cancel
          </button>
        </form>
      )}

      {/* Folders */}
      {course.folders.length === 0 ? (
        <div className="rounded-xl border border-dashed border-border p-16 text-center text-muted-foreground flex flex-col items-center justify-center">
          <NotionIcon name="ni-folders" className="w-[48px] h-[48px] mb-4 opacity-20" />
          <p className="text-lg">No folders yet.</p>
          <p className="text-base mt-2">Create a folder to organize your PDFs.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-6">
          {course.folders.map((folder) => (
            <FolderView key={folder.id} folder={folder} />
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Direct uploads section ───────────────────────────────────────────────────

function LoosePdfsSection({ courseId, pdfs }: { courseId: string; pdfs: PdfFile[] }) {
  const [isUploading, setIsUploading] = useState(false);

  return (
    <div className="mb-8 rounded-xl border border-border bg-white shadow-sm overflow-hidden">
      <div className="flex items-center gap-3 border-b border-border bg-gray-50/50 px-6 py-4 font-medium text-base">
        <NotionIcon name="ni-file-text" className="w-[24px] h-[24px] text-muted-foreground" />
        <span>Direct Uploads</span>
        <span className="ml-auto text-sm text-muted-foreground font-normal">
          Unfiled documents
        </span>
      </div>

      <div className="p-5">
        {pdfs.length > 0 && (
          <div className="flex flex-col gap-2 mb-4">
            {pdfs.map((pdf) => (
              <PdfRow key={pdf.id} pdf={pdf} />
            ))}
          </div>
        )}

        {isUploading ? (
          <div className="mt-2 relative">
            <button
              onClick={() => setIsUploading(false)}
              className="absolute -top-2 right-2 text-xs text-muted-foreground hover:text-foreground z-10 p-2"
            >
              ✕ Close
            </button>
            <PdfDropzone
              targetId={courseId}
              targetType="course"
              onUploaded={() => setIsUploading(false)}
            />
          </div>
        ) : (
          <button
            onClick={() => setIsUploading(true)}
            className="text-base font-medium text-muted-foreground hover:text-foreground hover:underline flex items-center gap-2 w-full justify-center py-5 rounded-lg border border-dashed border-border bg-gray-50/50 hover:bg-black/5 transition-colors"
          >
            Upload PDF directly
          </button>
        )}
      </div>
    </div>
  );
}

// ─── Folder view ──────────────────────────────────────────────────────────────

function FolderView({ folder }: { folder: Folder }) {
  const [isUploading, setIsUploading] = useState(false);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const hasPdfs = folder.pdfs.length > 0;

  const handleDelete = async (keepFiles: boolean) => {
    setDeleting(true);
    await deleteFolder(folder.id, keepFiles);
    setDeleting(false);
    setDeleteOpen(false);
  };

  return (
    <div className="rounded-xl border border-border bg-white shadow-sm overflow-hidden flex flex-col">
      <div className="flex items-center justify-between border-b border-border bg-gray-50/50 px-6 py-4">
        <div className="flex items-center gap-3 font-medium text-base">
          <div className="text-foreground"><NotionIcon name="ni-folder" className="w-[24px] h-[24px]" /></div>
          {folder.name}
        </div>

        <AlertDialog open={deleteOpen} onOpenChange={setDeleteOpen}>
          <button
            onClick={() => setDeleteOpen(true)}
            className="text-muted-foreground hover:text-red-500 transition-colors"
            title="Delete folder"
          >
            <NotionIcon name="ni-x" className="w-[22px] h-[22px]" />
          </button>
          <AlertDialogContent size="default">
            <AlertDialogHeader>
              <AlertDialogTitle>Delete folder "{folder.name}"?</AlertDialogTitle>
              <AlertDialogDescription>
                {hasPdfs
                  ? `This folder contains ${folder.pdfs.length} PDF${folder.pdfs.length === 1 ? '' : 's'}. What should happen to the files?`
                  : 'This empty folder will be permanently deleted.'}
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel disabled={deleting}>Cancel</AlertDialogCancel>
              {hasPdfs && (
                <AlertDialogAction
                  variant="outline"
                  disabled={deleting}
                  onClick={() => handleDelete(true)}
                >
                  Keep files
                </AlertDialogAction>
              )}
              <AlertDialogAction
                variant="destructive"
                disabled={deleting}
                onClick={() => handleDelete(false)}
              >
                {deleting ? 'Deleting...' : hasPdfs ? 'Delete everything' : 'Delete'}
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </div>

      <div className="p-5">
        {folder.pdfs.length > 0 && (
          <div className="flex flex-col gap-2 mb-4">
            {folder.pdfs.map((pdf) => (
              <PdfRow key={pdf.id} pdf={pdf} />
            ))}
          </div>
        )}

        {isUploading ? (
          <div className="mt-2 relative">
            <button
              onClick={() => setIsUploading(false)}
              className="absolute -top-2 right-2 text-xs text-muted-foreground hover:text-foreground z-10 p-2"
            >
              ✕ Close
            </button>
            <PdfDropzone
              targetId={folder.id}
              targetType="folder"
              onUploaded={() => setIsUploading(false)}
            />
          </div>
        ) : (
          <button
            onClick={() => setIsUploading(true)}
            className="text-base font-medium text-muted-foreground hover:text-foreground hover:underline flex items-center gap-2 w-full justify-center py-5 rounded-lg border border-dashed border-border bg-gray-50/50 hover:bg-black/5 transition-colors"
          >
            Upload PDF
          </button>
        )}
      </div>
    </div>
  );
}

// ─── Shared PDF row ───────────────────────────────────────────────────────────

function PdfRow({ pdf }: { pdf: PdfFile }) {
  return (
    <div className="flex items-center justify-between rounded-lg border border-border px-5 py-4 text-base hover:border-black/20 transition-colors group cursor-pointer">
      <div className="flex items-center gap-3">
        <NotionIcon name="ni-file-text" className="w-[24px] h-[24px] text-muted-foreground" />
        <span>{pdf.name}</span>
      </div>
      <div className="flex items-center gap-5 text-muted-foreground">
        <span className="text-sm">{(pdf.size_bytes / 1024 / 1024).toFixed(2)} MB</span>
        <button
          onClick={(e) => {
            e.stopPropagation();
            deletePdf(pdf.id, pdf.storage_path);
          }}
          className="opacity-0 group-hover:opacity-100 hover:text-red-500 transition-all"
        >
          <NotionIcon name="ni-x" className="w-[20px] h-[20px]" />
        </button>
      </div>
    </div>
  );
}
