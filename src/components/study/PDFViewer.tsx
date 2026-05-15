'use client';

import {useEffect, useState} from 'react';
import {Worker, Viewer, type SpecialZoomLevel} from '@react-pdf-viewer/core';
import {defaultLayoutPlugin} from '@react-pdf-viewer/default-layout';
import {
  highlightPlugin,
  Trigger,
  type HighlightArea,
  type RenderHighlightTargetProps,
  type RenderHighlightsProps,
} from '@react-pdf-viewer/highlight';
import {getPdfSignedUrl} from '@/actions/pdfs';
import {
  createAnnotation,
  deleteAnnotation,
  updateAnnotation,
  type Annotation,
} from '@/actions/annotations';
import {NotionIcon} from '@/components/NotionIcon';
import {useLanguage} from '@/lib/i18n';
import type {TranslationKey} from '@/lib/i18n';

import '@react-pdf-viewer/core/lib/styles/index.css';
import '@react-pdf-viewer/default-layout/lib/styles/index.css';
import '@react-pdf-viewer/highlight/lib/styles/index.css';

interface PDFViewerProps {
  storagePath: string;
  pdfId: string;
  pdfName: string;
  annotations: Annotation[];
  onAnnotationCreated: (annotation: Annotation) => void;
  onAnnotationDeleted: (annotationId: string) => void;
  onAnnotationUpdated: (annotation: Annotation) => void;
}

type AnnotationColor = 'yellow' | 'green' | 'blue' | 'pink';

const COLOR_STYLES: Record<AnnotationColor, {bg: string; border: string}> = {
  yellow: {bg: 'rgba(255, 222, 0, 0.3)', border: '#e6c800'},
  green: {bg: 'rgba(74, 222, 128, 0.3)', border: '#22c55e'},
  blue: {bg: 'rgba(96, 165, 250, 0.3)', border: '#3b82f6'},
  pink: {bg: 'rgba(244, 114, 182, 0.3)', border: '#ec4899'},
};

const COLOR_LABEL_KEYS: Record<AnnotationColor, TranslationKey> = {
  yellow: 'pdf.color.yellow',
  green: 'pdf.color.green',
  blue: 'pdf.color.blue',
  pink: 'pdf.color.pink',
};

export function PDFViewer({
  storagePath,
  pdfId,
  pdfName,
  annotations,
  onAnnotationCreated,
  onAnnotationDeleted,
  onAnnotationUpdated,
}: PDFViewerProps) {
  const {t} = useLanguage();
  const [fileUrl, setFileUrl] = useState<string | null>(null);
  const [urlError, setUrlError] = useState<string | null>(null);
  const [message, setMessage] = useState('');
  const [selectedColor, setSelectedColor] = useState<AnnotationColor>('yellow');
  const [isHighlightMode, setIsHighlightMode] = useState(true);
  const [hoveredAnnotationId, setHoveredAnnotationId] = useState<string | null>(null);
  const [expandedCommentDraft, setExpandedCommentDraft] = useState('');
  const [savingAnnotationId, setSavingAnnotationId] = useState<string | null>(null);
  const [deletingAnnotationId, setDeletingAnnotationId] = useState<string | null>(null);
  const [isHighlightsOpen, setIsHighlightsOpen] = useState(false);
  const [expandedAnnotationId, setExpandedAnnotationId] = useState<string | null>(null);

  useEffect(() => {
    let active = true;

    getPdfSignedUrl(storagePath).then(({url, error}) => {
      if (!active) return;
      if (error || !url) {
        setUrlError(error ?? t('pdf.couldNotLoad'));
      } else {
        setUrlError(null);
        setFileUrl(url);
      }
    });

    return () => {
      active = false;
    };
  }, [storagePath, t]);

  const renderHighlightTarget = (props: RenderHighlightTargetProps) => (
    <div
      style={{
        position: 'absolute',
        left: `${props.selectionRegion.left + props.selectionRegion.width}%`,
        top: `${props.selectionRegion.top + props.selectionRegion.height}%`,
        transform: 'translate(-100%, 8px)',
        zIndex: 10,
      }}
    >
      <div className="annotation-popup">
        <div className="annotation-color-row">
          {(Object.keys(COLOR_STYLES) as AnnotationColor[]).map((colorKey) => (
            <button
              key={colorKey}
              className={`annotation-color-btn ${colorKey === selectedColor ? 'selected' : ''}`}
              style={{background: COLOR_STYLES[colorKey].border}}
              onClick={() => setSelectedColor(colorKey)}
              title={t(COLOR_LABEL_KEYS[colorKey])}
            />
          ))}
        </div>
        <textarea
          className="annotation-comment-input"
          placeholder={t('pdf.addComment')}
          value={message}
          onChange={(event) => setMessage(event.target.value)}
          rows={2}
        />
        <div className="annotation-popup-actions">
          <button
            className="annotation-btn-save"
            onClick={async () => {
              if (!props.selectionData) return;
              const result = await createAnnotation(pdfId, {
                page_index: props.highlightAreas[0]?.pageIndex ?? 0,
                highlight_areas: props.highlightAreas,
                quote: props.selectionData.selectedText,
                comment: message,
                color: selectedColor,
              });
              if (result.annotation) onAnnotationCreated(result.annotation);
              setMessage('');
              props.cancel();
            }}
          >
            {t('pdf.saveHighlight')}
          </button>
          <button className="annotation-btn-cancel" onClick={props.cancel}>
            {t('materials.cancel')}
          </button>
        </div>
      </div>
    </div>
  );

  const renderHighlightContent = () => <></>;

  const renderHighlights = (props: RenderHighlightsProps) => (
    <div>
      {annotations
        .filter((annotation) => annotation.page_index === props.pageIndex)
        .map((annotation) => {
          const areas = (annotation.highlight_areas as HighlightArea[]).filter(
            (area) => area.pageIndex === props.pageIndex,
          );
          const color = COLOR_STYLES[(annotation.color as AnnotationColor) ?? 'yellow'];

          return (
            <div key={annotation.id} className="annotation-highlight-group">
              {areas.map((area, idx) => (
                <div
                  key={idx}
                  className="annotation-highlight-area"
                  style={{
                    ...props.getCssProperties(area, props.rotation),
                    background: color.bg,
                    borderBottom: `2px solid ${color.border}`,
                  }}
                />
              ))}
            </div>
          );
        })}
    </div>
  );

  const highlightPluginInstance = highlightPlugin({
    renderHighlightTarget,
    renderHighlightContent,
    renderHighlights,
  });

  useEffect(() => {
    highlightPluginInstance.switchTrigger(
      isHighlightMode ? Trigger.TextSelection : Trigger.None,
    );
  }, [highlightPluginInstance, isHighlightMode]);

  const defaultLayoutPluginInstance = defaultLayoutPlugin({
    sidebarTabs: () => [],
  });
  const sortedAnnotations = [...annotations].sort((a, b) =>
    b.created_at.localeCompare(a.created_at),
  );

  const jumpToAnnotation = (annotation: Annotation) => {
    const firstArea = (annotation.highlight_areas as HighlightArea[])[0];
    if (!firstArea) return;
    highlightPluginInstance.jumpToHighlightArea(firstArea);
  };

  const saveExpandedAnnotation = async (annotationId: string) => {
    setSavingAnnotationId(annotationId);
    const {annotation, error} = await updateAnnotation(annotationId, {
      comment: expandedCommentDraft.trim(),
    });
    setSavingAnnotationId(null);
    if (!error && annotation) {
      onAnnotationUpdated(annotation);
    }
  };

  const removeAnnotation = async (annotationId: string) => {
    setDeletingAnnotationId(annotationId);
    const {error} = await deleteAnnotation(annotationId);
    setDeletingAnnotationId(null);
    if (!error) {
      onAnnotationDeleted(annotationId);
      if (expandedAnnotationId === annotationId) setExpandedAnnotationId(null);
      if (hoveredAnnotationId === annotationId) setHoveredAnnotationId(null);
    }
  };

  if (urlError) {
    return (
      <div className="pdf-error">
        <p>{t('pdf.couldNotLoad')}</p>
        <p className="pdf-error-detail">{urlError}</p>
      </div>
    );
  }

  if (!fileUrl) {
    return (
      <div className="pdf-loading">
        <div className="pdf-loading-spinner" />
        <p>{t('pdf.loading', {name: pdfName})}</p>
      </div>
    );
  }

  return (
    <div className="pdf-viewer-container">
      <div className="pdf-viewer-layout">
        <aside className={`pdf-annotations-sidebar ${isHighlightsOpen ? 'is-open' : 'is-collapsed'}`}>
          <div className="pdf-annotations-header">
            <button
              className="pdf-annotations-toggle"
              onClick={() => setIsHighlightsOpen((prev) => !prev)}
              title={isHighlightsOpen ? t('pdf.collapseHighlights') : t('pdf.expandHighlights')}
            >
              <NotionIcon
                name="ni-sidebar-text"
                className="pdf-annotations-toggle-icon w-[16px] h-[16px]"
              />
              {isHighlightsOpen && <span className="pdf-annotations-label">{t('pdf.highlights')}</span>}
              <NotionIcon
                name={isHighlightsOpen ? 'ni-chevron-left' : 'ni-chevron-right'}
                className="pdf-annotations-toggle-chevron w-[14px] h-[14px]"
              />
            </button>
          </div>

          {isHighlightsOpen && (
            <div className="pdf-annotations-list">
            <div className="pdf-annotations-meta">
              <span className="pdf-annotations-meta-count">
                {annotations.length} {annotations.length === 1 ? t('pdf.highlight') : t('pdf.highlights')}
              </span>
            </div>
            {sortedAnnotations.length === 0 && (
              <p className="pdf-annotations-empty">{t('pdf.noHighlights')}</p>
            )}

            {sortedAnnotations.map((annotation) => (
              <div
                key={annotation.id}
                className={`pdf-annotation-item ${hoveredAnnotationId === annotation.id ? 'is-hovered' : ''}`}
                onMouseEnter={() => setHoveredAnnotationId(annotation.id)}
                onMouseLeave={() => setHoveredAnnotationId(null)}
                onClick={() => jumpToAnnotation(annotation)}
              >
                <button
                  className="pdf-annotation-jump"
                  onClick={() => jumpToAnnotation(annotation)}
                  title={t('pdf.jumpToHighlight')}
                >
                  <span className="pdf-annotation-page">{t('pdf.page')} {annotation.page_index + 1}</span>
                  <span className="pdf-annotation-quote">
                    {annotation.quote?.trim() || t('pdf.noText')}
                  </span>
                </button>

                {expandedAnnotationId === annotation.id ? (
                  <div className="pdf-annotation-expanded">
                    <textarea
                      className="pdf-annotation-comment-editor"
                      rows={4}
                      value={expandedCommentDraft}
                      onChange={(event) => setExpandedCommentDraft(event.target.value)}
                      onClick={(event) => event.stopPropagation()}
                    />
                    <div className="pdf-annotation-actions">
                      <button
                        className="pdf-annotation-inline-save"
                        onClick={(event) => {
                          event.stopPropagation();
                          saveExpandedAnnotation(annotation.id);
                        }}
                        disabled={savingAnnotationId === annotation.id}
                      >
                        {savingAnnotationId === annotation.id ? t('note.saving') : t('materials.save')}
                      </button>
                      <button
                        className="pdf-annotation-inline-delete"
                        onClick={(event) => {
                          event.stopPropagation();
                          removeAnnotation(annotation.id);
                        }}
                        disabled={deletingAnnotationId === annotation.id}
                      >
                        {deletingAnnotationId === annotation.id ? t('materials.deleting') : t('materials.delete')}
                      </button>
                      <button
                        className="pdf-annotation-inline-edit"
                        onClick={(event) => {
                          event.stopPropagation();
                          setExpandedAnnotationId(null);
                        }}
                      >
                        {t('pdf.collapse')}
                      </button>
                    </div>
                  </div>
                ) : (
                  <button
                    className="pdf-annotation-expand-btn"
                    onClick={(event) => {
                      event.stopPropagation();
                      setExpandedAnnotationId(annotation.id);
                      setExpandedCommentDraft(annotation.comment ?? '');
                    }}
                  >
                    {t('pdf.expand')}
                  </button>
                )}
              </div>
            ))}
            </div>
          )}
        </aside>

        <div className="pdf-viewer-main">
          <div className="pdf-viewer-toolbar">
            <button
              className={`pdf-highlight-toggle ${isHighlightMode ? 'active' : ''}`}
              onClick={() => setIsHighlightMode((prev) => !prev)}
              title={
                isHighlightMode
                  ? t('pdf.highlightOnTitle')
                  : t('pdf.highlightOffTitle')
              }
            >
              <NotionIcon name="ni-pen-line" className="w-[14px] h-[14px]" />
              <span>{isHighlightMode ? t('pdf.highlightOn') : t('pdf.highlightOff')}</span>
            </button>
          </div>
          <Worker workerUrl="/pdf.worker.min.js">
            <Viewer
              fileUrl={fileUrl}
              plugins={[defaultLayoutPluginInstance, highlightPluginInstance]}
              defaultScale={'PageWidth' as SpecialZoomLevel}
            />
          </Worker>
        </div>
      </div>
    </div>
  );
}
