'use client';

import {useCallback, useEffect, useState} from 'react';
import {useLanguage} from '@/lib/i18n';
import {Button} from '@/components/ui/button';
import {deleteMockTest, listMockTests, listAllUserMockTestAttempts} from '@/actions/revision';
import type {MockTest, MockTestAttempt} from '@/types/revision';
import {MockTestGenerationDialog} from './MockTestGenerationDialog';
import {TestRunner} from './TestRunner';
import {TestPreviewView} from './TestPreviewView';
import {Trash2, Play, Plus, FileQuestion} from 'lucide-react';
import type {PdfOption} from '../pdf-utils';

interface Props {
  courseId: string;
  pdfOptions: PdfOption[];
}

export function MockTestTab({courseId, pdfOptions}: Props) {
  const {t, language} = useLanguage();
  const [tests, setTests] = useState<MockTest[]>([]);
  const [attempts, setAttempts] = useState<Record<string, MockTestAttempt[]>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [activeTestId, setActiveTestId] = useState<string | null>(null);
  const [viewingTestId, setViewingTestId] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    const [testResult, attemptResult] = await Promise.all([
      listMockTests(courseId),
      listAllUserMockTestAttempts(),
    ]);
    setLoading(false);
    if (testResult.error) {
      setError(testResult.error);
      return;
    }
    setTests(testResult.data ?? []);
    
    if (!attemptResult.error) {
      const grouped: Record<string, MockTestAttempt[]> = {};
      for (const a of attemptResult.data ?? []) {
        if (!grouped[a.mockTestId]) grouped[a.mockTestId] = [];
        grouped[a.mockTestId].push(a);
      }
      setAttempts(grouped);
    }
  }, [courseId]);

  useEffect(() => {
    void load();
  }, [load]);

  if (activeTestId) {
    return (
      <TestRunner
        testId={activeTestId}
        onBack={() => {
          setActiveTestId(null);
          void load();
        }}
      />
    );
  }

  if (viewingTestId) {
    return (
      <TestPreviewView
        testId={viewingTestId}
        onBack={() => setViewingTestId(null)}
        onStart={() => {
          const id = viewingTestId;
          setViewingTestId(null);
          setActiveTestId(id);
        }}
      />
    );
  }

  const handleDelete = async (testId: string) => {
    if (!window.confirm(t('revision.flashcards.deleteConfirm'))) return;
    const result = await deleteMockTest(testId);
    if (result.error) {
      setError(result.error);
      return;
    }
    setTests((prev) => prev.filter((tst) => tst.id !== testId));
  };

  return (
    <div className="space-y-4">
      <div className="flex justify-end mb-2">
        <Button onClick={() => setDialogOpen(true)} size="sm">
          <Plus className="w-4 h-4 mr-1.5" />
          {t('revision.mocktest.newTest')}
        </Button>
      </div>

      {error && <p className="text-sm text-destructive">{error}</p>}

      {loading ? (
        <div className="text-sm text-muted-foreground py-8">...</div>
      ) : tests.length === 0 ? (
        <div className="text-center py-12 text-muted-foreground border border-dashed border-border rounded-lg">
          {t('revision.mocktest.empty')}
        </div>
      ) : (
        <ul className="space-y-2">
          {tests.map((mt) => {
            const testAttempts = attempts[mt.id] || [];
            const bestScore = testAttempts.length > 0 
              ? Math.max(...testAttempts.map(a => a.scorePercent))
              : null;

            return (
              <li
                key={mt.id}
                onClick={() => {
                  if (mt.status === 'ready') setViewingTestId(mt.id);
                }}
                className={`flex items-center justify-between gap-3 p-4 rounded-lg border border-border bg-card transition ${mt.status === 'ready' ? 'cursor-pointer hover:shadow-md hover:border-primary/50' : 'opacity-70'}`}
              >
                <div className="min-w-0 flex-1">
                  <p className="font-medium truncate flex items-center gap-2">
                    <FileQuestion className="w-4 h-4 text-muted-foreground" />
                    {mt.title}
                  </p>
                  <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs text-muted-foreground mt-1 pl-6">
                    <span>
                      {t('revision.mocktest.questionCount', {
                        count: String(mt.questionCount),
                        plural: mt.questionCount === 1 ? '' : (language === 'de' ? 'n' : 's'),
                      })}
                    </span>
                    <span
                      className={
                        mt.status === 'ready'
                          ? 'text-emerald-600 dark:text-emerald-400'
                          : mt.status === 'failed'
                            ? 'text-destructive'
                            : ''
                      }
                    >
                      Status:{' '}
                      {mt.status === 'ready'
                        ? t('revision.flashcards.statusReady')
                        : mt.status === 'failed'
                          ? `${t('revision.flashcards.statusFailed')}${mt.generationError ? `: ${mt.generationError}` : ''}`
                          : t('revision.flashcards.statusPending')}
                    </span>
                    {bestScore !== null && (
                      <span className="text-primary font-medium border-l border-border pl-3">
                        {t('revision.mocktest.completed', {
                          score: String(Math.round(bestScore)),
                        })}
                      </span>
                    )}
                  </div>
                </div>
                <div className="flex gap-2 shrink-0">
                  {mt.status === 'ready' && (
                    <Button 
                      size="sm" 
                      onClick={(e) => {
                        e.stopPropagation();
                        setActiveTestId(mt.id);
                      }}
                    >
                      <Play className="w-4 h-4 mr-1.5" />
                      {t('revision.mocktest.start')}
                    </Button>
                  )}
                  <Button
                    size="sm"
                    variant="ghost"
                    className="text-muted-foreground hover:text-destructive"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDelete(mt.id);
                    }}
                    title={t('revision.mocktest.delete')}
                  >
                    <Trash2 className="w-4 h-4" />
                  </Button>
                </div>
              </li>
            );
          })}
        </ul>
      )}

      <MockTestGenerationDialog
        open={dialogOpen}
        onOpenChange={setDialogOpen}
        pdfOptions={pdfOptions}
        courseId={courseId}
        onCreated={() => void load()}
      />
    </div>
  );
}
