'use client';

import {useState} from 'react';
import {useLanguage} from '@/lib/i18n';
import {createFlashcardDeck} from '@/actions/revision';
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

export function DeckGenerationDialog({
  open,
  onOpenChange,
  pdfOptions,
  courseId,
  onCreated,
}: Props) {
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
      setError(t('revision.flashcards.title'));
      return;
    }
    
    setSubmitting(true);
    const result = await createFlashcardDeck({
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
        <DialogHeader className="pb-2">
          <DialogTitle>{t('revision.flashcards.newDeck')}</DialogTitle>
        </DialogHeader>
        <div className="space-y-6 py-4">
          <div className="space-y-2">
            <label className="text-sm font-semibold">{t('revision.flashcards.title')}</label>
            <Input
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder={t('revision.flashcards.titlePlaceholder')}
              disabled={submitting}
              className="h-10"
            />
          </div>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <label className="text-sm font-semibold">{t('revision.flashcards.count')}</label>
              <span className="text-sm font-medium bg-muted px-2.5 py-0.5 rounded-md text-muted-foreground">{count}</span>
            </div>
            <input
              type="range"
              min={3}
              max={30}
              value={count}
              onChange={(e) => setCount(Number(e.target.value))}
              disabled={submitting}
              className="w-full accent-primary cursor-pointer"
            />
          </div>
          <div className="space-y-2">
            <label className="text-sm font-semibold block">{t('revision.flashcards.sources')}</label>
            <PdfMultiSelect options={pdfOptions} selectedIds={selectedIds} onChange={setSelectedIds} />
          </div>
          {error && <p className="text-sm text-destructive">{error}</p>}
          {submitting && (
            <p className="text-sm text-muted-foreground">{t('revision.flashcards.generating')}</p>
          )}
        </div>
        <DialogFooter className="pt-2">
          <Button variant="outline" onClick={() => handleClose(false)} disabled={submitting}>
            {t('revision.common.cancel')}
          </Button>
          <Button onClick={handleSubmit} disabled={submitting}>
            {submitting 
              ? (selectedIds.length > 0 ? t('revision.flashcards.generating') : t('materials.save'))
              : (selectedIds.length > 0 ? t('revision.flashcards.generate') : t('revision.common.create'))}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
