'use client';

import {useEffect, useState} from 'react';
import {useLanguage} from '@/lib/i18n';
import type {PdfOption} from '../pdf-utils';

type LearningTreeNode = {
  id: string;
  label: string;
  type: 'document' | 'topic' | 'subtopic' | 'concept' | 'objective';
  summary?: string | null;
  pageStart?: number | null;
  pageEnd?: number | null;
  confidence?: number | null;
  chunkIds: string[];
  children: LearningTreeNode[];
};

const TYPE_COLORS: Record<LearningTreeNode['type'], string> = {
  document: 'border-primary/40 bg-primary/5 text-foreground',
  topic: 'border-blue-500/30 bg-blue-500/5',
  subtopic: 'border-emerald-500/30 bg-emerald-500/5',
  concept: 'border-amber-500/30 bg-amber-500/5',
  objective: 'border-pink-500/30 bg-pink-500/5',
};

const TYPE_LABELS: Record<LearningTreeNode['type'], string> = {
  document: 'Document',
  topic: 'Topic',
  subtopic: 'Subtopic',
  concept: 'Concept',
  objective: 'Objective',
};

function TreeNode({node, depth = 0}: {node: LearningTreeNode; depth?: number}) {
  return (
    <li className="relative group">
      <div
        className={
          'inline-flex items-center rounded-lg border px-3 py-2 text-sm shadow-sm transition-all hover:shadow-md ' +
          TYPE_COLORS[node.type]
        }
      >
        <span className="text-[10px] uppercase tracking-wider font-semibold opacity-70 mr-2.5">
          {TYPE_LABELS[node.type]}
        </span>
        <span className="font-medium">{node.label}</span>
        {(node.pageStart !== null && node.pageStart !== undefined) && (
          <span className="ml-3 px-1.5 py-0.5 rounded text-[10px] font-medium bg-background/50 text-muted-foreground">
            p.{node.pageStart}
            {node.pageEnd && node.pageEnd !== node.pageStart ? `–${node.pageEnd}` : ''}
          </span>
        )}
      </div>
      {node.children.length > 0 && (
        <ul className="mt-3 ml-6 space-y-3 border-l-2 border-border/50 pl-5 py-1">
          {node.children.map((child) => (
            <TreeNode key={child.id} node={child} depth={depth + 1} />
          ))}
        </ul>
      )}
    </li>
  );
}

interface Props {
  pdfOptions: PdfOption[];
}

export function MindmapTab({pdfOptions}: Props) {
  const {t} = useLanguage();
  const [selectedPdfId, setSelectedPdfId] = useState<string | null>(pdfOptions[0]?.id ?? null);
  const [tree, setTree] = useState<LearningTreeNode | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!selectedPdfId) {
      setTree(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    setTree(null);
    (async () => {
      try {
        const response = await fetch(`/api/learning-graph/${selectedPdfId}/tree`);
        if (cancelled) return;
        if (response.status === 404) {
          setError('empty');
          setLoading(false);
          return;
        }
        if (!response.ok) {
          setError('Failed to load mindmap.');
          setLoading(false);
          return;
        }
        const data = (await response.json()) as LearningTreeNode;
        if (cancelled) return;
        setTree(data);
        setLoading(false);
      } catch {
        if (cancelled) return;
        setError('Failed to load mindmap.');
        setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [selectedPdfId]);

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center gap-3">
        <label className="text-sm font-medium text-muted-foreground shrink-0">{t('revision.mindmap.selectPdf')}</label>
        <select
          value={selectedPdfId ?? ''}
          onChange={(e) => setSelectedPdfId(e.target.value || null)}
          className="w-full max-w-md h-10 rounded-md border border-input bg-background px-3 text-sm focus:ring-1 focus:ring-primary focus:border-primary transition-colors shadow-sm cursor-pointer"
        >
          <option value="">—</option>
          {pdfOptions.map((opt) => (
            <option key={opt.id} value={opt.id}>
              {opt.folderName ? `${opt.folderName} / ` : ''}{opt.name}
            </option>
          ))}
        </select>
      </div>

      <div className="pt-2">
        {loading && (
          <div className="flex items-center gap-3 text-sm text-muted-foreground p-4">
             <div className="h-4 w-4 animate-spin rounded-full border-2 border-primary border-t-transparent"></div>
             {t('revision.mindmap.loading')}
          </div>
        )}

        {error === 'empty' && (
          <div className="rounded-xl border border-dashed border-border bg-muted/20 p-8 text-center text-sm text-muted-foreground space-y-3">
            <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-muted/50">
              <svg className="h-6 w-6 text-muted-foreground" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 13h6m-3-3v6m5 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg>
            </div>
            <p className="font-medium text-foreground text-base">{t('revision.mindmap.empty')}</p>
            <p className="max-w-md mx-auto">{t('revision.mindmap.generatingHint')}</p>
          </div>
        )}
        
        {error && error !== 'empty' && (
          <div className="p-4 rounded-lg bg-destructive/10 text-destructive text-sm border border-destructive/20">
            {error}
          </div>
        )}

        {tree && (
          <div className="overflow-x-auto rounded-xl border border-border bg-card p-6 shadow-sm">
            <ul className="space-y-3">
              <TreeNode node={tree} />
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}
