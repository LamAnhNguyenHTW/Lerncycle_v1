'use client';

import {useLanguage} from '@/lib/i18n';
import type {PdfOption} from './pdf-utils';

export function PdfMultiSelect({
  options,
  selectedIds,
  onChange,
}: {
  options: PdfOption[];
  selectedIds: string[];
  onChange: (ids: string[]) => void;
}) {
  const {t} = useLanguage();
  const selected = new Set(selectedIds);

  const toggle = (id: string) => {
    const next = new Set(selected);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    onChange(Array.from(next));
  };

  const allSelected = options.length > 0 && options.every((o) => selected.has(o.id));

  return (
    <div className="space-y-2">
      <div className="flex justify-between items-center text-xs text-muted-foreground">
        <span>{options.length === 0 ? '—' : `${selectedIds.length} / ${options.length}`}</span>
        <div className="flex gap-2">
          <button
            type="button"
            className="hover:text-foreground underline-offset-2 hover:underline disabled:opacity-50"
            disabled={options.length === 0 || allSelected}
            onClick={() => onChange(options.map((o) => o.id))}
          >
            {t('revision.common.pdfSelectAll')}
          </button>
          <button
            type="button"
            className="hover:text-foreground underline-offset-2 hover:underline disabled:opacity-50"
            disabled={selectedIds.length === 0}
            onClick={() => onChange([])}
          >
            {t('revision.common.pdfSelectNone')}
          </button>
        </div>
      </div>
      <div className="max-h-48 overflow-y-auto rounded-md border border-border bg-background">
        {options.length === 0 ? (
          <div className="p-3 text-xs text-muted-foreground">No PDFs available in this course.</div>
        ) : (
          options.map((option) => (
            <label
              key={option.id}
              className="flex items-start gap-2 px-3 py-2 text-sm hover:bg-muted/50 cursor-pointer border-b border-border/50 last:border-b-0"
            >
              <input
                type="checkbox"
                className="mt-0.5"
                checked={selected.has(option.id)}
                onChange={() => toggle(option.id)}
              />
              <span className="flex-1 min-w-0">
                <span className="block truncate">{option.name}</span>
                {option.folderName && (
                  <span className="block text-xs text-muted-foreground truncate">{option.folderName}</span>
                )}
              </span>
            </label>
          ))
        )}
      </div>
    </div>
  );
}
