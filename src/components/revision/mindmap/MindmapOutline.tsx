'use client';

import { useState } from 'react';
import { ChevronRight, ChevronDown, FileText, Folder, FolderOpen } from 'lucide-react';
import { TYPE_COLORS, TYPE_LABELS } from './CustomNode';

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

function OutlineNode({ node, depth = 0 }: { node: LearningTreeNode; depth?: number }) {
  const [expanded, setExpanded] = useState(depth === 0); // Auto-expand only root node

  const visibleChildren = node.children;
  
  const hasChildren = visibleChildren.length > 0;

  return (
    <div className="flex flex-col">
      <div 
        className={`group flex items-center gap-2 py-1.5 px-2 -ml-2 rounded-md hover:bg-muted/50 cursor-pointer transition-colors ${depth === 0 ? 'mt-2' : ''}`}
        onClick={() => setExpanded(!expanded)}
      >
        {/* Toggle Icon */}
        <div className="w-5 h-5 flex items-center justify-center shrink-0 text-muted-foreground hover:text-foreground transition-colors">
          {hasChildren ? (
            expanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />
          ) : (
            <div className="w-4 h-4" /> // Spacer
          )}
        </div>

        {/* Node Icon based on type */}
        <div className="shrink-0 flex items-center justify-center w-5 h-5 text-muted-foreground opacity-70">
          {node.type === 'document' ? (
            <FileText className="w-4 h-4" />
          ) : expanded && hasChildren ? (
            <FolderOpen className="w-4 h-4" />
          ) : (
            <Folder className="w-4 h-4" />
          )}
        </div>

        {/* Content */}
        <div className="flex-1 flex items-center flex-wrap gap-2 min-w-0">
          <span className="font-medium text-sm text-foreground truncate">{node.label}</span>
          
          <span className="text-[10px] uppercase tracking-wider font-semibold opacity-50 px-1.5 py-0.5 rounded bg-muted">
            {TYPE_LABELS[node.type] || node.type}
          </span>

          {(node.pageStart !== null && node.pageStart !== undefined) && (
            <span className="text-[10px] font-medium text-muted-foreground">
              p.{node.pageStart}{node.pageEnd && node.pageEnd !== node.pageStart ? `–${node.pageEnd}` : ''}
            </span>
          )}
        </div>
      </div>

      {/* Children */}
      {expanded && hasChildren && (
        <div className="ml-5 pl-2 mt-1 border-l border-border/50 flex flex-col gap-1">
          {visibleChildren.map((child) => (
            <OutlineNode key={child.id} node={child} depth={depth + 1} />
          ))}
        </div>
      )}
    </div>
  );
}

export function MindmapOutline({ tree }: { tree: LearningTreeNode }) {
  // If the root is concept/objective, don't show (unlikely)
  if (tree.type === 'concept' || tree.type === 'objective') return null;

  return (
    <div className="w-full border border-border rounded-xl bg-card p-6 overflow-x-auto min-h-[600px] shadow-sm">
      <div className="min-w-[400px]">
        <OutlineNode node={tree} />
      </div>
    </div>
  );
}
