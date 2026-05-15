import {Handle, Position} from '@xyflow/react';

type LearningTreeNode = {
  id: string;
  label: string;
  type: 'document' | 'topic' | 'subtopic' | 'concept' | 'objective';
  summary?: string | null;
  pageStart?: number | null;
  pageEnd?: number | null;
  confidence?: number | null;
  chunkIds: string[];
};

export const TYPE_COLORS: Record<LearningTreeNode['type'], string> = {
  document: 'border-primary/40 bg-primary/5 text-foreground',
  topic: 'border-blue-500/30 bg-blue-500/5',
  subtopic: 'border-emerald-500/30 bg-emerald-500/5',
  concept: 'border-amber-500/30 bg-amber-500/5',
  objective: 'border-pink-500/30 bg-pink-500/5',
};

export const TYPE_LABELS: Record<LearningTreeNode['type'], string> = {
  document: 'Document',
  topic: 'Topic',
  subtopic: 'Subtopic',
  concept: 'Concept',
  objective: 'Objective',
};

export function CustomNode({data}: {data: LearningTreeNode}) {
  return (
    <div
      className={
        'relative flex items-center rounded-lg border px-4 py-3 shadow-sm bg-card transition-all min-w-[200px] max-w-[300px] ' +
        TYPE_COLORS[data.type]
      }
    >
      <Handle
        type="target"
        position={Position.Left}
        className="w-2 h-2 !bg-muted-foreground/50 border-none"
      />
      
      <div className="flex-1">
        <div className="flex items-center gap-2 mb-1.5">
          <span className="text-[10px] uppercase tracking-wider font-semibold opacity-70">
            {TYPE_LABELS[data.type]}
          </span>
          {data.pageStart !== null && data.pageStart !== undefined && (
            <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-background/50 text-muted-foreground">
              p.{data.pageStart}
              {data.pageEnd && data.pageEnd !== data.pageStart ? `–${data.pageEnd}` : ''}
            </span>
          )}
        </div>
        <div className="font-medium text-sm leading-tight">{data.label}</div>
        {data.summary && (
          <p className="text-xs text-muted-foreground mt-2 line-clamp-2 leading-relaxed">
            {data.summary}
          </p>
        )}
      </div>

      <Handle
        type="source"
        position={Position.Right}
        className="w-2 h-2 !bg-muted-foreground/50 border-none"
      />
    </div>
  );
}
