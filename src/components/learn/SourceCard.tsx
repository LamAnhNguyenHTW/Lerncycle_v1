import type {ChatSource} from '@/types/chat';
import {Globe} from 'lucide-react';

const SOURCE_LABELS = {
  pdf: 'PDF',
  note: 'Note',
  annotation_comment: 'Annotation',
  chat_memory: 'Memory',
  knowledge_graph: 'Graph',
  web: 'Web',
};

function getSourceTitle(source: ChatSource) {
  return source.title
    ?? source.metadata.filename
    ?? (source.source_type === 'pdf'
      ? 'PDF'
      : source.source_type === 'chat_memory'
        ? 'Chat Memory'
      : source.source_type === 'knowledge_graph'
        ? 'Knowledge Graph'
      : source.source_type === 'web'
        ? 'Web Source'
      : source.source_type === 'annotation_comment'
        ? 'Annotation'
        : source.source_type === 'note'
          ? 'Note'
          : 'Source');
}

export function SourceCard({source}: {source: ChatSource}) {
  const webUrl = source.source_type === 'web' ? source.metadata.url : undefined;
  const title = getSourceTitle(source);
  return (
    <article className="rounded-lg border border-border bg-white p-3 shadow-sm">
      <div className="mb-2 flex items-center gap-2">
        <span className="rounded-md border border-border bg-muted px-2 py-0.5 text-xs font-medium text-muted-foreground">
          {SOURCE_LABELS[source.source_type]}
        </span>
        {source.source_type === 'web' && <Globe className="h-3.5 w-3.5 text-muted-foreground" />}
        {source.page !== null && source.source_type !== 'web' && source.source_type !== 'knowledge_graph' && (
          <span className="text-xs text-muted-foreground">Page {source.page}</span>
        )}
      </div>
      <h3 className="text-sm font-medium text-foreground">
        {webUrl ? (
          <a href={webUrl} target="_blank" rel="noopener noreferrer" className="underline-offset-2 hover:underline">
            {title}
          </a>
        ) : title}
      </h3>
      {webUrl && <p className="mt-1 truncate text-xs text-muted-foreground">{webUrl}</p>}
      {source.source_type === 'web' && source.metadata.provider && (
        <p className="mt-1 text-xs text-muted-foreground">Provider: {source.metadata.provider}</p>
      )}
      {source.source_type === 'knowledge_graph' && source.metadata.provider && (
        <p className="mt-1 text-xs text-muted-foreground">Provider: {source.metadata.provider}</p>
      )}
      {source.source_type === 'knowledge_graph' && source.metadata.node_names && source.metadata.node_names.length > 0 && (
        <p className="mt-1 text-xs text-muted-foreground">
          Entities: {source.metadata.node_names.slice(0, 5).join(', ')}
        </p>
      )}
      {source.heading && <p className="mt-1 text-xs text-muted-foreground">{source.heading}</p>}
      <p className="mt-2 text-sm leading-6 text-muted-foreground">{source.snippet}</p>
    </article>
  );
}
