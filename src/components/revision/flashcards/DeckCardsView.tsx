'use client';

import {useCallback, useEffect, useState} from 'react';
import {useLanguage} from '@/lib/i18n';
import {Button} from '@/components/ui/button';
import {getDeckCards, getDeck, createFlashcard, updateFlashcard, deleteFlashcard} from '@/actions/revision';
import type {Flashcard, FlashcardDeck} from '@/types/revision';
import {ArrowLeft, RotateCcw, Zap, Calendar, AlertCircle, Plus, Pencil, Trash2} from 'lucide-react';

interface Props {
  deckId: string;
  onBack: () => void;
  onReview: () => void;
}

export function DeckCardsView({deckId, onBack, onReview}: Props) {
  const {t} = useLanguage();
  const [deck, setDeck] = useState<FlashcardDeck | null>(null);
  const [cards, setCards] = useState<Flashcard[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Manual card creation state
  const [adding, setAdding] = useState(false);
  const [newFront, setNewFront] = useState('');
  const [newBack, setNewBack] = useState('');
  const [savingCard, setSavingCard] = useState(false);

  // Edit state
  const [editingCardId, setEditingCardId] = useState<string | null>(null);
  const [editFront, setEditFront] = useState('');
  const [editBack, setEditBack] = useState('');
  const [updatingCard, setUpdatingCard] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    const [deckRes, cardsRes] = await Promise.all([
      getDeck(deckId),
      getDeckCards(deckId),
    ]);
    setLoading(false);
    if (deckRes.error) setError(deckRes.error);
    else if (cardsRes.error) setError(cardsRes.error);
    else {
      setDeck(deckRes.data ?? null);
      setCards(cardsRes.data ?? []);
    }
  }, [deckId]);

  useEffect(() => {
    void load();
  }, [load]);

  const handleAddCard = async () => {
    if (!newFront.trim() || !newBack.trim()) return;
    setSavingCard(true);
    const result = await createFlashcard({deckId, front: newFront, back: newBack});
    setSavingCard(false);
    if (result.error) {
      setError(result.error);
    } else {
      setAdding(false);
      setNewFront('');
      setNewBack('');
      void load();
    }
  };

  const handleDeleteCard = async (cardId: string) => {
    if (!window.confirm(t('revision.flashcards.deleteConfirm'))) return;
    const result = await deleteFlashcard(cardId);
    if (result.error) {
      setError(result.error);
    } else {
      void load();
    }
  };

  const startEditing = (card: Flashcard) => {
    setEditingCardId(card.id);
    setEditFront(card.front);
    setEditBack(card.back);
  };

  const handleUpdateCard = async () => {
    if (!editingCardId || !editFront.trim() || !editBack.trim()) return;
    setUpdatingCard(true);
    const result = await updateFlashcard({cardId: editingCardId, front: editFront, back: editBack});
    setUpdatingCard(false);
    if (result.error) {
      setError(result.error);
    } else {
      setEditingCardId(null);
      void load();
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between pb-3 border-b border-border/50">
        <div className="flex items-center gap-3">
          <Button variant="outline" size="sm" onClick={onBack} className="h-9 px-3">
            <ArrowLeft className="w-4 h-4 mr-1.5" />
            {t('revision.common.back')}
          </Button>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="secondary" size="sm" onClick={() => setAdding(true)} className="h-9">
            <Plus className="w-4 h-4 mr-1.5" />
            {t('revision.flashcards.newCard')}
          </Button>
          <Button size="sm" onClick={onReview} className="h-9" disabled={cards.length === 0}>
            {t('revision.flashcards.review')}
          </Button>
        </div>
      </div>

      {error && <p className="text-sm text-destructive">{error}</p>}

      {adding && (
        <div className="p-5 border border-border rounded-xl bg-card shadow-sm space-y-4">
          <h4 className="font-semibold">{t('revision.flashcards.newCard')}</h4>
          <div className="space-y-3">
            <textarea 
              placeholder={t('revision.flashcards.question')} 
              value={newFront} 
              onChange={e => setNewFront(e.target.value)}
              className="flex min-h-[80px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
              disabled={savingCard}
            />
            <textarea 
              placeholder={t('revision.flashcards.answer')} 
              value={newBack} 
              onChange={e => setNewBack(e.target.value)}
              className="flex min-h-[80px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
              disabled={savingCard}
            />
            <div className="flex gap-2 justify-end pt-2">
              <Button variant="ghost" size="sm" onClick={() => setAdding(false)} disabled={savingCard}>
                {t('revision.common.cancel')}
              </Button>
              <Button size="sm" onClick={handleAddCard} disabled={savingCard || !newFront.trim() || !newBack.trim()}>
                {t('materials.save')}
              </Button>
            </div>
          </div>
        </div>
      )}

      {loading ? (
        <div className="text-center text-muted-foreground py-12">
          <div className="mx-auto h-6 w-6 animate-spin rounded-full border-2 border-primary border-t-transparent mb-4"></div>
          {t('revision.mindmap.loading')}
        </div>
      ) : cards.length === 0 ? (
        <div className="text-center py-12 text-muted-foreground border border-dashed border-border rounded-lg">
          {t('revision.flashcards.empty')}
        </div>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {cards.map((card, index) => {
            if (editingCardId === card.id) {
              return (
                <div key={card.id} className="p-4 border border-border rounded-xl bg-card shadow-sm space-y-4 flex flex-col h-full">
                  <h4 className="font-semibold text-sm">Edit Card {index + 1}</h4>
                  <div className="space-y-3 flex-1 flex flex-col">
                    <textarea 
                      placeholder={t('revision.flashcards.question')} 
                      value={editFront} 
                      onChange={e => setEditFront(e.target.value)}
                      className="flex min-h-[80px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                      disabled={updatingCard}
                    />
                    <textarea 
                      placeholder={t('revision.flashcards.answer')} 
                      value={editBack} 
                      onChange={e => setEditBack(e.target.value)}
                      className="flex min-h-[80px] flex-1 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                      disabled={updatingCard}
                    />
                    <div className="flex gap-2 justify-end pt-2">
                      <Button variant="ghost" size="sm" onClick={() => setEditingCardId(null)} disabled={updatingCard}>
                        {t('revision.common.cancel')}
                      </Button>
                      <Button size="sm" onClick={handleUpdateCard} disabled={updatingCard || !editFront.trim() || !editBack.trim()}>
                        {t('materials.save')}
                      </Button>
                    </div>
                  </div>
                </div>
              );
            }

            const isDue = new Date(card.dueAt) <= new Date();
            return (
              <div key={card.id} className="flex flex-col rounded-xl border border-border bg-card shadow-sm h-full overflow-hidden hover:shadow-md transition-shadow">
                <div className="bg-muted/30 px-4 py-2.5 border-b border-border flex justify-between items-center">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-semibold text-muted-foreground">
                      {t('revision.flashcards.cardLabel', {n: String(index + 1)})}
                    </span>
                    <button 
                      onClick={() => startEditing(card)} 
                      className="text-muted-foreground hover:text-foreground opacity-50 hover:opacity-100 transition-opacity"
                      title="Edit card"
                    >
                      <Pencil className="w-3 h-3" />
                    </button>
                    <button 
                      onClick={() => handleDeleteCard(card.id)} 
                      className="text-destructive/50 hover:text-destructive opacity-50 hover:opacity-100 transition-opacity"
                      title="Delete card"
                    >
                      <Trash2 className="w-3 h-3" />
                    </button>
                  </div>
                  <div className="flex items-center gap-1.5 text-[10px] font-medium text-muted-foreground">
                    <span title={t('revision.flashcards.repetitions')} className="flex items-center gap-1 bg-background px-1.5 py-0.5 rounded shadow-sm border border-border/50">
                      <RotateCcw className="w-3 h-3 opacity-70" />
                      {card.repetitions}
                    </span>
                    <span title={t('revision.flashcards.easeFactor')} className="flex items-center gap-1 bg-background px-1.5 py-0.5 rounded shadow-sm border border-border/50">
                      <Zap className="w-3 h-3 opacity-70" />
                      {card.easeFactor.toFixed(1)}
                    </span>
                    {isDue ? (
                      <span title={t('revision.flashcards.due')} className="flex items-center gap-1 bg-destructive/10 text-destructive px-1.5 py-0.5 rounded shadow-sm border border-destructive/20">
                        <AlertCircle className="w-3 h-3" />
                        {t('revision.flashcards.due')}
                      </span>
                    ) : (
                      <span title={t('revision.flashcards.nextDue')} className="flex items-center gap-1 bg-background px-1.5 py-0.5 rounded shadow-sm border border-border/50">
                        <Calendar className="w-3 h-3 opacity-70" />
                        {new Date(card.dueAt).toLocaleDateString()}
                      </span>
                    )}
                  </div>
                </div>
                <div className="p-4 flex-1">
                  <span className="text-[10px] uppercase tracking-wider font-bold text-primary/70 block mb-1.5">
                    {t('revision.flashcards.question')}
                  </span>
                  <p className="text-sm font-medium leading-relaxed">{card.front}</p>
                </div>
                <div className="p-4 bg-muted/10 border-t border-border/50 mt-auto">
                  <span className="text-[10px] uppercase tracking-wider font-bold text-emerald-600/70 dark:text-emerald-400/70 block mb-1.5">
                    {t('revision.flashcards.answer')}
                  </span>
                  <p className="text-sm text-muted-foreground whitespace-pre-wrap">{card.back}</p>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
