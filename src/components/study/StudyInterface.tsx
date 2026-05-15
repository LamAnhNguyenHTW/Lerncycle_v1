'use client';

import {useState, useEffect, useMemo} from 'react';
import {Group, Panel, Separator} from 'react-resizable-panels';
import dynamic from 'next/dynamic';
import type {JSONContent} from '@tiptap/core';
import {type Course, type PdfFile} from '@/lib/data';
import {type Annotation} from '@/actions/annotations';
import {getAnnotations} from '@/actions/annotations';
import {getNote} from '@/actions/notes';
import {NotionIcon} from '@/components/NotionIcon';
import {PdfPicker} from './PdfPicker';
import {useLanguage} from '@/lib/i18n';
import type {TranslationKey} from '@/lib/i18n';

function StudyLoadingFallback({labelKey, spinner = false}: {labelKey: TranslationKey; spinner?: boolean}) {
  const {t} = useLanguage();
  return (
    <div className="pdf-loading">
      {spinner && <div className="pdf-loading-spinner" />}
      <p>{t(labelKey)}</p>
    </div>
  );
}

// Lazy-load heavy components (PDF worker & TipTap) only on the client
const PDFViewer = dynamic(
  () => import('./PDFViewer').then((m) => m.PDFViewer),
  {
    ssr: false,
    loading: () => <StudyLoadingFallback labelKey="study.loadingViewer" spinner />,
  },
);

const NoteEditor = dynamic(
  () => import('./NoteEditor').then((m) => m.NoteEditor),
  {
    ssr: false,
    loading: () => <StudyLoadingFallback labelKey="study.loadingEditor" />,
  },
);

interface StudyInterfaceProps {
  course: Course;
  initialPdfId?: string;
}

export function StudyInterface({course, initialPdfId}: StudyInterfaceProps) {
  const {t} = useLanguage();
  const initialPdf = useMemo(() => {
    if (!initialPdfId) return null;
    const allPdfs = [
      ...course.loose_pdfs,
      ...course.folders.flatMap((folder) => folder.pdfs),
    ];
    return allPdfs.find((pdf) => pdf.id === initialPdfId) ?? null;
  }, [initialPdfId, course]);

  const [activePdf, setActivePdf] = useState<PdfFile | null>(initialPdf);
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [noteContent, setNoteContent] = useState<JSONContent | null>(null);
  const [loading, setLoading] = useState(Boolean(initialPdf));
  const activePdfId = activePdf?.id;

  // When the active PDF changes, load its annotations and note
  useEffect(() => {
    if (!activePdfId) {
      return;
    }

    let active = true;

    Promise.all([getAnnotations(activePdfId), getNote(activePdfId)]).then(
      ([annResult, noteResult]) => {
        if (!active) return;
        setAnnotations(annResult.annotations ?? []);
        setNoteContent((noteResult.content ?? null) as JSONContent | null);
        setLoading(false);
      },
    );

    return () => {
      active = false;
    };
  }, [activePdfId]);

  const handleSelectPdf = (pdf: PdfFile) => {
    setActivePdf(pdf);
    setAnnotations([]);
    setNoteContent(null);
    setLoading(true);
  };

  const handleBackToAllPdfs = () => {
    setActivePdf(null);
    setAnnotations([]);
    setNoteContent(null);
    setLoading(false);
  };

  const handleAnnotationCreated = (annotation: Annotation) => {
    setAnnotations((prev) => [...prev, annotation]);
  };

  const handleAnnotationDeleted = (id: string) => {
    setAnnotations((prev) => prev.filter((a) => a.id !== id));
  };

  const handleAnnotationUpdated = (annotation: Annotation) => {
    setAnnotations((prev) =>
      prev.map((item) => (item.id === annotation.id ? annotation : item)),
    );
  };

  // ── No PDF selected ───────────────────────────────────────────────────────
  if (!activePdf) {
    return (
      <div className="study-interface-wrap">
        <PdfPicker course={course} onSelect={handleSelectPdf} />
      </div>
    );
  }

  // ── PDF selected: split-pane layout ──────────────────────────────────────
  return (
    <div className="study-interface-wrap">
      {/* Top bar */}
      <div className="study-topbar">
        <button
          className="study-back-btn"
          onClick={handleBackToAllPdfs}
          title={t('study.chooseDifferentPdf')}
        >
          <NotionIcon name="ni-arrow-left" className="w-[18px] h-[18px]" />
          <span>{t('study.allPdfs')}</span>
        </button>
        <span className="study-pdf-name">{activePdf.name}</span>
        {loading && <span className="study-loading-badge">{t('study.loading')}</span>}
      </div>

      {/* Resizable panels */}
      <div className="study-panels-wrap">
        <Group orientation="horizontal" className="study-panel-group">
          {/* Left: PDF Viewer */}
          <Panel id="pdf-panel" defaultSize="72%" minSize="40%" className="study-panel">
            <PDFViewer
              storagePath={activePdf.storage_path}
              pdfId={activePdf.id}
              pdfName={activePdf.name}
              annotations={annotations}
              onAnnotationCreated={handleAnnotationCreated}
              onAnnotationDeleted={handleAnnotationDeleted}
              onAnnotationUpdated={handleAnnotationUpdated}
            />
          </Panel>

          {/* Drag handle */}
          <Separator className="study-separator" />

          {/* Right: Note Editor */}
          <Panel id="note-panel" defaultSize="28%" minSize="20%" className="study-panel">
            {!loading && (
              <NoteEditor pdfId={activePdf.id} initialContent={noteContent} />
            )}
          </Panel>
        </Group>
      </div>
    </div>
  );
}
