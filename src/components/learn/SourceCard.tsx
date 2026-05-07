import type {ChatSource} from '@/types/chat';

const SOURCE_LABELS = {
  pdf: 'PDF',
  note: 'Note',
  annotation_comment: 'Annotation',
};

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
        {source.title ?? source.metadata.filename ?? 'Untitled source'}
      </h3>
      {source.heading && <p className="mt-1 text-xs text-muted-foreground">{source.heading}</p>}
      <p className="mt-2 text-sm leading-6 text-muted-foreground">{source.snippet}</p>
    </article>
  );
}
