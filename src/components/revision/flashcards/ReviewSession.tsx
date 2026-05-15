'use client';

import {useCallback, useEffect, useState} from 'react';
import {useLanguage} from '@/lib/i18n';
import {Button} from '@/components/ui/button';
import {getDueCards, reviewFlashcard} from '@/actions/revision';
import type {Flashcard, ReviewQuality} from '@/types/revision';

interface Props {
  deckId: string;
  onBack: () => void;
}

const QUALITY_BUTTONS: {quality: ReviewQuality; labelKey: 'revision.flashcards.again' | 'revision.flashcards.hard' | 'revision.flashcards.good' | 'revision.flashcards.easy'; variant: 'destructive' | 'outline' | 'secondary' | 'default'}[] = [
  {quality: 0, labelKey: 'revision.flashcards.again', variant: 'destructive'},
  {quality: 3, labelKey: 'revision.flashcards.hard', variant: 'outline'},
  {quality: 4, labelKey: 'revision.flashcards.good', variant: 'secondary'},
  {quality: 5, labelKey: 'revision.flashcards.easy', variant: 'default'},
];

export function ReviewSession({deckId, onBack}: Props) {
  const {t} = useLanguage();
  const [cards, setCards] = useState<Flashcard[]>([]);
  const [loading, setLoading] = useState(true);
  const [index, setIndex] = useState(0);
  const [flipped, setFlipped] = useState(false);
  const [working, setWorking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    const result = await getDueCards(deckId, 50);
    setLoading(false);
    if (result.error) {
      setError(result.error);
      return;
    }
    setCards(result.data ?? []);
    setIndex(0);
    setFlipped(false);
  }, [deckId]);

  useEffect(() => {
    void load();
  }, [load]);

  const current = cards[index];

  const handleReview = async (quality: ReviewQuality) => {
    if (!current) return;
    setWorking(true);
    const result = await reviewFlashcard({cardId: current.id, quality});
    setWorking(false);
    if (result.error) {
      setError(result.error);
      return;
    }
    if (index + 1 >= cards.length) {
      setCards([]);
      setIndex(0);
    } else {
      setIndex((i) => i + 1);
    }
    setFlipped(false);
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <Button variant="ghost" size="sm" onClick={onBack}>
          ← {t('revision.common.back')}
        </Button>
        <div className="text-sm text-muted-foreground">
          {cards.length > 0 && `${index + 1} / ${cards.length}`}
        </div>
      </div>

      {error && <p className="text-sm text-destructive">{error}</p>}

      {loading ? (
        <div className="text-center text-muted-foreground py-12">...</div>
      ) : !current ? (
        <div className="text-center py-12">
          <p className="text-muted-foreground">{t('revision.flashcards.reviewEmpty')}</p>
        </div>
      ) : (
        <div className="space-y-4">
          <div 
            className="min-h-[280px] rounded-xl border border-border bg-card p-6 shadow-sm flex items-center justify-center text-center cursor-pointer hover:border-primary/50 transition-colors"
            onClick={() => setFlipped(!flipped)}
          >
            <div className="max-w-prose">
              <p className="text-xs uppercase tracking-wide text-muted-foreground mb-3">
                {flipped ? t('revision.flashcards.answer') : t('revision.flashcards.question')}
              </p>
              <p className="text-lg whitespace-pre-wrap">
                {flipped ? current.back : current.front}
              </p>
            </div>
          </div>

          {!flipped ? (
            <div className="flex justify-center">
              <Button onClick={() => setFlipped(true)} disabled={working}>
                {t('revision.flashcards.flip')}
              </Button>
            </div>
          ) : (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
              {QUALITY_BUTTONS.map(({quality, labelKey, variant}) => (
                <Button
                  key={quality}
                  variant={variant}
                  disabled={working}
                  onClick={() => handleReview(quality)}
                >
                  {t(labelKey)}
                </Button>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
