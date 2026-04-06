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

export function PDFViewer({
  storagePath,
  pdfId,
  pdfName,
  annotations,
  onAnnotationCreated,
  onAnnotationDeleted,
  onAnnotationUpdated,
}: PDFViewerProps) {
  const [fileUrl, setFileUrl] = useState<string | null>(null);
  const [urlError, setUrlError] = useState<string | null>(null);
  const [message, setMessage] = useState('');
  const [selectedColor, setSelectedColor] = useState<AnnotationColor>('yellow');
  const [isHighlightMode, setIsHighlightMode] = useState(true);
  const [hoveredAnnotationId, setHoveredAnnotationId] = useState<string | null>(null);
  const [expandedCommentDraft, setExpandedCommentDraft] = useState('');
  const [savingAnnotationId, setSavingAnnotationId] = useState<string | null>(null);
  const [deletingAnnotationId, setDeletingAnnotationId] = useState<string | null>(null);
  const [isHighlightsOpen, setIsHighlightsOpen] = useState(true);
  const [expandedAnnotationId, setExpandedAnnotationId] = useState<string | null>(null);

  useEffect(() => {
    let active = true;

    getPdfSignedUrl(storagePath).then(({url, error}) => {
      if (!active) return;
      if (error || !url) {
        setUrlError(error ?? 'Failed to load PDF');
      } else {
        setUrlError(null);
        setFileUrl(url);
      }
    });

    return () => {
      active = false;
    };
  }, [storagePath]);

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
              title={colorKey}
            />
          ))}
        </div>
        <textarea
          className="annotation-comment-input"
          placeholder="Add a comment (optional)..."
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
            Save highlight
          </button>
          <button className="annotation-btn-cancel" onClick={props.cancel}>
            Cancel
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
        <p>Could not load PDF</p>
        <p className="pdf-error-detail">{urlError}</p>
      </div>
    );
  }

  if (!fileUrl) {
    return (
      <div className="pdf-loading">
        <div className="pdf-loading-spinner" />
        <p>Loading {pdfName}...</p>
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
              title={isHighlightsOpen ? 'Collapse highlights' : 'Expand highlights'}
            >
              <NotionIcon
                name="ni-sidebar-text"
                className="pdf-annotations-toggle-icon w-[16px] h-[16px]"
              />
              {isHighlightsOpen && <span className="pdf-annotations-label">Highlights</span>}
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
                {annotations.length} {annotations.length === 1 ? 'highlight' : 'highlights'}
              </span>
            </div>
            {sortedAnnotations.length === 0 && (
              <p className="pdf-annotations-empty">No highlights yet.</p>
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
                  title="Jump to highlight"
                >
                  <span className="pdf-annotation-page">Page {annotation.page_index + 1}</span>
                  <span className="pdf-annotation-quote">
                    {annotation.quote?.trim() || '(No text)'}
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
                        {savingAnnotationId === annotation.id ? 'Saving...' : 'Save'}
                      </button>
                      <button
                        className="pdf-annotation-inline-delete"
                        onClick={(event) => {
                          event.stopPropagation();
                          removeAnnotation(annotation.id);
                        }}
                        disabled={deletingAnnotationId === annotation.id}
                      >
                        {deletingAnnotationId === annotation.id ? 'Deleting...' : 'Delete'}
                      </button>
                      <button
                        className="pdf-annotation-inline-edit"
                        onClick={(event) => {
                          event.stopPropagation();
                          setExpandedAnnotationId(null);
                        }}
                      >
                        Collapse
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
                    Expand
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
                  ? 'Highlight mode on: select text to create highlight'
                  : 'Highlight mode off'
              }
            >
              <NotionIcon name="ni-pen-line" className="w-[14px] h-[14px]" />
              <span>{isHighlightMode ? 'Highlight: On' : 'Highlight: Off'}</span>
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
