import type {ChatSource} from '@/types/chat';

const SOURCE_LABELS = {
  pdf: 'PDF',
  note: 'Note',
  annotation_comment: 'Annotation',
  chat_memory: 'Memory',
};

function getSourceTitle(source: ChatSource) {
  return source.title
    ?? source.metadata.filename
    ?? (source.source_type === 'pdf'
      ? 'PDF'
      : source.source_type === 'chat_memory'
        ? 'Chat Memory'
      : source.source_type === 'annotation_comment'
        ? 'Annotation'
        : source.source_type === 'note'
          ? 'Note'
          : 'Source');
}

export function SourceCard({source}: {source: ChatSource}) {
  return (
    <article className="rounded-lg border border-border bg-white p-3 shadow-sm">
      <div className="mb-2 flex items-center gap-2">
        <span className="rounded-md border border-border bg-muted px-2 py-0.5 text-xs font-medium text-muted-foreground">
          {SOURCE_LABELS[source.source_type]}
        </span>
        {source.page !== null && (
          <span className="text-xs text-muted-foreground">Page {source.page}</span>
        )}
      </div>
      <h3 className="text-sm font-medium text-foreground">
        {getSourceTitle(source)}
      </h3>
      {source.heading && <p className="mt-1 text-xs text-muted-foreground">{source.heading}</p>}
      <p className="mt-2 text-sm leading-6 text-muted-foreground">{source.snippet}</p>
    </article>
  );
}
