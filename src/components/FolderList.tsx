'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
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
import {useLanguage} from '@/lib/i18n';

export function FolderList({ course }: { course: Course }) {
  const {t} = useLanguage();
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
    <div className="w-full mt-8 mb-16 max-w-4xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-2">
          <div className="w-1 h-5 bg-primary rounded-full"></div>
          <h2 className="text-lg font-semibold tracking-tight">{t('materials.title')}</h2>
        </div>
        <button
          onClick={() => setIsCreating(true)}
          className="flex items-center gap-1.5 rounded-lg bg-black px-3 py-2 text-sm font-medium text-white hover:bg-black/80 transition-colors"
        >
          <NotionIcon name="ni-plus" className="w-[18px] h-[18px]" />
          {t('materials.newFolder')}
        </button>
      </div>

      {/* Direct uploads (no folder) */}
      <LoosePdfsSection courseId={course.id} pdfs={course.loose_pdfs} />

      {isCreating && (
        <form onSubmit={handleCreateFolder} className="mb-6 p-4 rounded-xl border border-border bg-white shadow-sm flex items-center gap-3">
          <input
            autoFocus
            type="text"
            placeholder={t('materials.folderPlaceholder')}
            className="flex-1 rounded-md border border-border px-3 py-2 text-sm outline-none focus:border-primary"
            value={newFolderName}
            onChange={(e) => setNewFolderName(e.target.value)}
          />
          <button type="submit" className="text-sm font-medium bg-black text-white px-4 py-2 rounded-md">
            {t('materials.save')}
          </button>
          <button type="button" onClick={() => setIsCreating(false)} className="text-sm font-medium px-4 py-2 hover:bg-black/5 rounded-md">
            {t('materials.cancel')}
          </button>
        </form>
      )}

      {/* Folders */}
      {course.folders.length === 0 ? (
        <div className="rounded-xl border border-dashed border-border p-16 text-center text-muted-foreground flex flex-col items-center justify-center">
          <NotionIcon name="ni-folders" className="w-[48px] h-[48px] mb-4 opacity-20" />
          <p className="text-lg">{t('materials.noFolders')}</p>
          <p className="text-base mt-2">{t('materials.noFoldersHint')}</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-6">
          {course.folders.map((folder) => (
            <FolderView key={folder.id} folder={folder} courseId={course.id} />
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Direct uploads section ───────────────────────────────────────────────────

function LoosePdfsSection({ courseId, pdfs }: { courseId: string; pdfs: PdfFile[] }) {
  const {t} = useLanguage();
  const [isUploading, setIsUploading] = useState(false);

  return (
    <div className="mb-6 rounded-xl border border-border bg-white shadow-sm overflow-hidden">
      <div className="flex items-center gap-2.5 border-b border-border bg-gray-50/50 px-5 py-3 font-medium text-sm">
        <NotionIcon name="ni-file-text" className="w-[20px] h-[20px] text-muted-foreground" />
        <span>{t('materials.directUploads')}</span>
        <span className="ml-auto text-xs text-muted-foreground font-normal">
          {t('materials.unfiledDocuments')}
        </span>
      </div>

      <div className="p-4">
        {pdfs.length > 0 && (
          <div className="flex flex-col gap-2 mb-3">
            {pdfs.map((pdf) => (
              <PdfRow key={pdf.id} pdf={pdf} courseId={courseId} />
            ))}
          </div>
        )}

        {isUploading ? (
          <div className="mt-2 relative">
            <button
              onClick={() => setIsUploading(false)}
              className="absolute -top-2 right-2 text-xs text-muted-foreground hover:text-foreground z-10 p-2"
            >
              × {t('materials.close')}
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
            className="text-sm font-medium text-muted-foreground hover:text-foreground flex items-center gap-2 w-full justify-center py-4 rounded-lg border border-dashed border-border bg-gray-50/50 hover:bg-black/5 transition-colors"
          >
            {t('materials.uploadDirect')}
          </button>
        )}
      </div>
    </div>
  );
}

// ─── Folder view ──────────────────────────────────────────────────────────────

function FolderView({ folder, courseId }: { folder: Folder; courseId: string }) {
  const {t} = useLanguage();
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
      <div className="flex items-center justify-between border-b border-border bg-gray-50/50 px-5 py-3">
        <div className="flex items-center gap-2.5 font-medium text-sm">
          <div className="text-foreground"><NotionIcon name="ni-folder" className="w-[20px] h-[20px]" /></div>
          {folder.name}
        </div>

        <AlertDialog open={deleteOpen} onOpenChange={setDeleteOpen}>
          <button
            onClick={() => setDeleteOpen(true)}
            className="text-muted-foreground hover:text-red-500 transition-colors"
            title={t('materials.deleteFolder')}
          >
            <NotionIcon name="ni-x" className="w-[18px] h-[18px]" />
          </button>
          <AlertDialogContent size="default">
            <AlertDialogHeader>
              <AlertDialogTitle>{t('materials.deleteFolderTitle', {name: folder.name})}</AlertDialogTitle>
              <AlertDialogDescription>
                {hasPdfs
                  ? t('materials.deleteFolderWithFiles', {count: String(folder.pdfs.length), plural: folder.pdfs.length === 1 ? '' : 's'})
                  : t('materials.deleteEmptyFolder')}
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel disabled={deleting}>{t('materials.cancel')}</AlertDialogCancel>
              {hasPdfs && (
                <AlertDialogAction
                  variant="outline"
                  disabled={deleting}
                  onClick={() => handleDelete(true)}
                >
                  {t('materials.keepFiles')}
                </AlertDialogAction>
              )}
              <AlertDialogAction
                variant="destructive"
                disabled={deleting}
                onClick={() => handleDelete(false)}
              >
                {deleting ? t('materials.deleting') : hasPdfs ? t('materials.deleteEverything') : t('materials.delete')}
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </div>

      <div className="p-4">
        {folder.pdfs.length > 0 && (
          <div className="flex flex-col gap-2 mb-3">
            {folder.pdfs.map((pdf) => (
              <PdfRow key={pdf.id} pdf={pdf} courseId={courseId} />
            ))}
          </div>
        )}

        {isUploading ? (
          <div className="mt-2 relative">
            <button
              onClick={() => setIsUploading(false)}
              className="absolute -top-2 right-2 text-xs text-muted-foreground hover:text-foreground z-10 p-2"
            >
              × {t('materials.close')}
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
            className="text-sm font-medium text-muted-foreground hover:text-foreground flex items-center gap-2 w-full justify-center py-4 rounded-lg border border-dashed border-border bg-gray-50/50 hover:bg-black/5 transition-colors"
          >
            {t('materials.uploadPdf')}
          </button>
        )}
      </div>
    </div>
  );
}

// ─── Shared PDF row ───────────────────────────────────────────────────────────

function PdfRow({ pdf, courseId }: { pdf: PdfFile; courseId: string }) {
  const router = useRouter();
  const {t} = useLanguage();

  return (
    <div
      className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-3 rounded-lg border border-border px-4 py-3 text-sm hover:border-black/20 transition-colors group cursor-pointer"
      onClick={() => {
        router.push(`/?courseId=${courseId}&tab=notetaking&pdfId=${pdf.id}`);
      }}
      title={pdf.name}
    >
      <div className="flex min-w-0 items-center gap-2.5">
        <NotionIcon name="ni-file-text" className="h-[18px] w-[18px] shrink-0 text-muted-foreground" />
        <span className="truncate">{pdf.name}</span>
      </div>
      <div className="flex shrink-0 items-center gap-4 text-muted-foreground">
        <span className="hidden text-sm sm:inline">{(pdf.size_bytes / 1024 / 1024).toFixed(2)} MB</span>
        <button
          onClick={(e) => {
            e.stopPropagation();
            router.push(`/?courseId=${courseId}&tab=learn&pdfId=${pdf.id}`);
          }}
          className="opacity-0 group-hover:opacity-100 hover:text-foreground transition-all text-sm"
        >
          {t('materials.chat')}
        </button>
        <button
          onClick={(e) => {
            e.stopPropagation();
            deletePdf(pdf.id, pdf.storage_path);
          }}
          className="opacity-0 group-hover:opacity-100 hover:text-red-500 transition-all"
        >
          <NotionIcon name="ni-x" className="w-[18px] h-[18px]" />
        </button>
      </div>
    </div>
  );
}
