'use client';

import {useCallback, useState} from 'react';
import {useDropzone} from 'react-dropzone';
import {uploadPdf} from '@/actions/pdfs';
import {cn} from '@/lib/utils';

interface Props {
  targetId: string;
  targetType: 'course' | 'folder';
  onUploaded?: () => void;
}

type UploadState = 'idle' | 'uploading' | 'error';

/** Drag-and-drop PDF upload zone for a specific folder. */
export function PdfDropzone({targetId, targetType, onUploaded}: Props) {
  const [state, setState] = useState<UploadState>('idle');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const onDrop = useCallback(
    async (accepted: File[]) => {
      if (accepted.length === 0) return;

      setState('uploading');
      setErrorMsg(null);

      const formData = new FormData();
      formData.set('file', accepted[0]);

      try {
        const result = await uploadPdf(targetId, targetType, formData);

        if (result.error) {
          setState('error');
          setErrorMsg(result.error);
        } else {
          setState('idle');
          onUploaded?.();
        }
      } catch (error) {
        setState('error');
        setErrorMsg(
          error instanceof Error ? error.message : 'Upload failed unexpectedly.',
        );
      }
    },
    [targetId, targetType, onUploaded],
  );

  const {getRootProps, getInputProps, isDragActive, isDragReject} = useDropzone({
    onDrop,
    accept: {'application/pdf': ['.pdf']},
    maxFiles: 1,
    disabled: state === 'uploading',
  });

  return (
    <div
      {...getRootProps()}
      className={cn(
        'flex flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed px-6 py-10 text-center transition-colors cursor-pointer',
        isDragActive && !isDragReject && 'border-primary bg-primary/5',
        isDragReject && 'border-destructive bg-destructive/5',
        !isDragActive && 'border-border hover:border-muted-foreground/40 hover:bg-muted/30',
        state === 'uploading' && 'pointer-events-none opacity-60',
      )}
    >
      <input {...getInputProps()} />

      {state === 'uploading' ? (
        <>
          <UploadIcon className="size-8 text-muted-foreground animate-pulse" />
          <p className="text-sm text-muted-foreground">Uploading…</p>
        </>
      ) : (
        <>
          <UploadIcon className={cn('size-8', isDragReject ? 'text-destructive' : 'text-muted-foreground')} />
          <div>
            <p className="text-sm font-medium">
              {isDragReject
                ? 'Only PDF files are accepted'
                : isDragActive
                  ? 'Drop to upload'
                  : 'Drop a PDF here, or click to select'}
            </p>
            <p className="mt-0.5 text-xs text-muted-foreground">PDF only · max 50 MB</p>
          </div>
        </>
      )}

      {state === 'error' && errorMsg && (
        <p className="text-xs text-destructive" role="alert">
          {errorMsg}
        </p>
      )}
    </div>
  );
}

function UploadIcon({className}: {className?: string}) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden="true"
    >
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" x2="12" y1="3" y2="15" />
    </svg>
  );
}
