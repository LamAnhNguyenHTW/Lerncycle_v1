'use client';

import {useCallback, useEffect, useState} from 'react';
import {useLanguage} from '@/lib/i18n';
import {Button} from '@/components/ui/button';
import {
  deleteDeck,
  getDeckCards,
  listFlashcardDecks,
} from '@/actions/revision';
import type {FlashcardDeck} from '@/types/revision';
import {DeckGenerationDialog} from './DeckGenerationDialog';
import {ReviewSession} from './ReviewSession';
import {DeckCardsView} from './DeckCardsView';
import {Trash2, Play, Plus, BookOpen} from 'lucide-react';
import type {PdfOption} from '../pdf-utils';

interface Props {
  courseId: string;
  pdfOptions: PdfOption[];
}

export function FlashcardsTab({courseId, pdfOptions}: Props) {
  const {t, language} = useLanguage();
  const [decks, setDecks] = useState<FlashcardDeck[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [reviewingDeckId, setReviewingDeckId] = useState<string | null>(null);
  const [viewingDeckId, setViewingDeckId] = useState<string | null>(null);
  const [cardCounts, setCardCounts] = useState<Record<string, number>>({});

  const loadDecks = useCallback(async () => {
    setLoading(true);
    setError(null);
    const result = await listFlashcardDecks(courseId);
    setLoading(false);
    if (result.error) {
      setError(result.error);
      return;
    }
    setDecks(result.data ?? []);
  }, [courseId]);

  useEffect(() => {
    void loadDecks();
  }, [loadDecks]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const next: Record<string, number> = {};
      for (const deck of decks) {
        if (deck.status !== 'ready') continue;
        const res = await getDeckCards(deck.id);
        if (cancelled) return;
        if (res.data) next[deck.id] = res.data.length;
      }
      if (!cancelled) setCardCounts((prev) => ({...prev, ...next}));
    })();
    return () => {
      cancelled = true;
    };
  }, [decks]);

  if (reviewingDeckId) {
    return (
      <ReviewSession
        deckId={reviewingDeckId}
        onBack={() => {
          setReviewingDeckId(null);
          void loadDecks();
        }}
      />
    );
  }

  if (viewingDeckId) {
    return (
      <DeckCardsView
        deckId={viewingDeckId}
        onBack={() => setViewingDeckId(null)}
        onReview={() => {
          const id = viewingDeckId;
          setViewingDeckId(null);
          setReviewingDeckId(id);
        }}
      />
    );
  }

  const handleDelete = async (deckId: string) => {
    if (!window.confirm(t('revision.flashcards.deleteConfirm'))) return;
    const result = await deleteDeck(deckId);
    if (result.error) {
      setError(result.error);
      return;
    }
    setDecks((prev) => prev.filter((d) => d.id !== deckId));
  };

  return (
    <div className="space-y-4">
      <div className="flex justify-end mb-2">
        <Button onClick={() => setDialogOpen(true)} size="sm">
          <Plus className="w-4 h-4 mr-1.5" />
          {t('revision.flashcards.newDeck')}
        </Button>
      </div>

      {error && <p className="text-sm text-destructive">{error}</p>}

      {loading ? (
        <div className="text-sm text-muted-foreground py-8">...</div>
      ) : decks.length === 0 ? (
        <div className="text-center py-12 text-muted-foreground border border-dashed border-border rounded-lg">
          {t('revision.flashcards.empty')}
        </div>
      ) : (
        <ul className="space-y-2">
          {decks.map((deck) => {
            const count = cardCounts[deck.id];
            return (
              <li
                key={deck.id}
                onClick={() => {
                  if (deck.status === 'ready') setViewingDeckId(deck.id);
                }}
                className={`flex items-center justify-between gap-3 p-4 rounded-lg border border-border bg-card transition ${deck.status === 'ready' ? 'cursor-pointer hover:shadow-md hover:border-primary/50' : 'opacity-70'}`}
              >
                <div className="min-w-0 flex-1">
                  <p className="font-medium truncate flex items-center gap-2">
                    <BookOpen className="w-4 h-4 text-muted-foreground" />
                    {deck.title}
                  </p>
                  <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs text-muted-foreground mt-1 pl-6">
                    {deck.status === 'ready' && (
                      <span>
                        {t('revision.flashcards.cardCount', {
                          count: String(count ?? '—'),
                          plural: (count ?? 0) === 1 ? '' : (language === 'de' ? 'n' : 's'),
                        })}
                      </span>
                    )}
                    <span
                      className={
                        deck.status === 'ready'
                          ? 'text-emerald-600 dark:text-emerald-400'
                          : deck.status === 'failed'
                            ? 'text-destructive'
                            : ''
                      }
                    >
                      Status:{' '}
                      {deck.status === 'ready'
                        ? t('revision.flashcards.statusReady')
                        : deck.status === 'failed'
                          ? `${t('revision.flashcards.statusFailed')}${deck.generationError ? `: ${deck.generationError}` : ''}`
                          : t('revision.flashcards.statusPending')}
                    </span>
                  </div>
                </div>
                <div className="flex gap-2 shrink-0">
                  {deck.status === 'ready' && (
                    <Button 
                      size="sm" 
                      onClick={(e) => {
                        e.stopPropagation();
                        setReviewingDeckId(deck.id);
                      }}
                    >
                      <Play className="w-4 h-4 mr-1.5" />
                      {t('revision.flashcards.review')}
                    </Button>
                  )}
                  <Button
                    size="sm"
                    variant="ghost"
                    className="text-muted-foreground hover:text-destructive"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDelete(deck.id);
                    }}
                    title={t('revision.flashcards.delete')}
                  >
                    <Trash2 className="w-4 h-4" />
                  </Button>
                </div>
              </li>
            );
          })}
        </ul>
      )}

      <DeckGenerationDialog
        open={dialogOpen}
        onOpenChange={setDialogOpen}
        pdfOptions={pdfOptions}
        courseId={courseId}
        onCreated={() => void loadDecks()}
      />
    </div>
  );
}
