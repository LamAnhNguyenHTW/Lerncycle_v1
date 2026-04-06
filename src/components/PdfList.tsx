'use client';

import {useTransition} from 'react';
import {useRouter} from 'next/navigation';
import {deletePdf} from '@/actions/pdfs';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '@/components/ui/alert-dialog';
import type {Pdf} from '@/types';

interface Props {
  pdfs: Pdf[];
  weekId: string;
}

/** Client Component: renders a PDF list with delete controls. */
export function PdfList({pdfs}: Props) {
  if (pdfs.length === 0) return null;

  return (
    <ul className="flex flex-col gap-1" role="list">
      {pdfs.map((pdf) => (
        <PdfRow key={pdf.id} pdf={pdf} />
      ))}
    </ul>
  );
}

function PdfRow({pdf}: {pdf: Pdf}) {
  const router = useRouter();
  const [pending, startTransition] = useTransition();

  return (
    <li className="group flex items-center gap-3 rounded-lg border border-border bg-card px-3 py-2">
      <FileIcon className="size-4 shrink-0 text-muted-foreground" />

      <div className="flex min-w-0 flex-1 flex-col">
        <span className="truncate text-sm font-medium">{pdf.name}</span>
        <span className="text-xs text-muted-foreground">{formatBytes(pdf.size_bytes)}</span>
      </div>

      <AlertDialog>
        <AlertDialogTrigger
          className="rounded p-1 text-muted-foreground opacity-0 transition-opacity hover:bg-muted hover:text-destructive group-hover:opacity-100"
          title="Delete PDF"
          disabled={pending}
        >
          <TrashIcon className="size-3.5" />
        </AlertDialogTrigger>
        <AlertDialogContent size="sm">
          <AlertDialogHeader>
            <AlertDialogTitle>Delete &quot;{pdf.name}&quot;?</AlertDialogTitle>
            <AlertDialogDescription>
              The file will be permanently removed. This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              disabled={pending}
              onClick={() =>
                startTransition(async () => {
                  await deletePdf(pdf.id, pdf.storage_path);
                  router.refresh();
                })
              }
            >
              {pending ? 'Deleting…' : 'Delete'}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </li>
  );
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function FileIcon({className}: {className?: string}) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden="true"
    >
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14 2 14 8 20 8" />
    </svg>
  );
}

function TrashIcon({className}: {className?: string}) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden="true"
    >
      <polyline points="3 6 5 6 21 6" />
      <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
      <path d="M10 11v6M14 11v6" />
      <path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2" />
    </svg>
  );
}
