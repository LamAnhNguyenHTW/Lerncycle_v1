'use client';

import {useState} from 'react';
import {useLanguage} from '@/lib/i18n';
import {createMockTest} from '@/actions/revision';
import {Button} from '@/components/ui/button';
import {Input} from '@/components/ui/input';
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import {PdfMultiSelect} from '../PdfMultiSelect';
import type {PdfOption} from '../pdf-utils';

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  pdfOptions: PdfOption[];
  courseId: string;
  onCreated: () => void;
}

export function MockTestGenerationDialog({open, onOpenChange, pdfOptions, courseId, onCreated}: Props) {
  const {t, language} = useLanguage();
  const [title, setTitle] = useState('');
  const [count, setCount] = useState(10);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const reset = () => {
    setTitle('');
    setCount(10);
    setSelectedIds([]);
    setError(null);
  };

  const handleClose = (next: boolean) => {
    if (submitting) return;
    onOpenChange(next);
    if (!next) reset();
  };

  const handleSubmit = async () => {
    setError(null);
    if (!title.trim()) {
      setError(t('revision.mocktest.title'));
      return;
    }
    if (selectedIds.length === 0) {
      setError(t('revision.flashcards.sources'));
      return;
    }
    setSubmitting(true);
    const result = await createMockTest({
      title: title.trim(),
      pdfIds: selectedIds,
      count,
      language,
      courseId,
    });
    setSubmitting(false);
    if (result.error) {
      setError(result.error);
      return;
    }
    onCreated();
    handleClose(false);
  };

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>{t('revision.mocktest.newTest')}</DialogTitle>
        </DialogHeader>
        <div className="space-y-4 py-2">
          <div className="space-y-1">
            <label className="text-sm font-medium">{t('revision.mocktest.title')}</label>
            <Input
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              disabled={submitting}
            />
          </div>
          <div className="space-y-1">
            <label className="text-sm font-medium flex justify-between">
              <span>{t('revision.mocktest.count')}</span>
              <span className="text-muted-foreground">{count}</span>
            </label>
            <input
              type="range"
              min={3}
              max={20}
              value={count}
              onChange={(e) => setCount(Number(e.target.value))}
              disabled={submitting}
              className="w-full"
            />
          </div>
          <div className="space-y-1">
            <label className="text-sm font-medium">{t('revision.flashcards.sources')}</label>
            <PdfMultiSelect options={pdfOptions} selectedIds={selectedIds} onChange={setSelectedIds} />
          </div>
          {error && <p className="text-sm text-destructive">{error}</p>}
          {submitting && (
            <p className="text-sm text-muted-foreground">{t('revision.flashcards.generating')}</p>
          )}
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => handleClose(false)} disabled={submitting}>
            {t('revision.common.cancel')}
          </Button>
          <Button onClick={handleSubmit} disabled={submitting}>
            {submitting ? t('revision.flashcards.generating') : t('revision.common.create')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
