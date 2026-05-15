'use client';

import {useEffect, useState} from 'react';
import {useLanguage} from '@/lib/i18n';
import {Button} from '@/components/ui/button';
import {getMockTestWithQuestions} from '@/actions/revision';
import type {MockTest, MockTestQuestionPublic} from '@/types/revision';
import {ArrowLeft} from 'lucide-react';

interface Props {
  testId: string;
  onBack: () => void;
  onStart: () => void;
}

export function TestPreviewView({testId, onBack, onStart}: Props) {
  const {t} = useLanguage();
  const [test, setTest] = useState<MockTest | null>(null);
  const [questions, setQuestions] = useState<MockTestQuestionPublic[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setLoading(true);
      const result = await getMockTestWithQuestions(testId);
      if (cancelled) return;
      setLoading(false);
      if (result.error || !result.data) {
        setError(result.error ?? 'Test not found.');
        return;
      }
      setTest(result.data.test);
      setQuestions(result.data.questions);
    })();
    return () => {
      cancelled = true;
    };
  }, [testId]);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between pb-3 border-b border-border/50">
        <div className="flex items-center gap-3">
          <Button variant="outline" size="sm" onClick={onBack} className="h-9 px-3">
            <ArrowLeft className="w-4 h-4 mr-1.5" />
            {t('revision.common.back')}
          </Button>
        </div>
        <Button size="sm" onClick={onStart} className="h-9">
          {t('revision.mocktest.start')}
        </Button>
      </div>

      {error && <p className="text-sm text-destructive">{error}</p>}

      {loading ? (
        <div className="text-center text-muted-foreground py-12">
          <div className="mx-auto h-6 w-6 animate-spin rounded-full border-2 border-primary border-t-transparent mb-4"></div>
          {t('revision.mindmap.loading')}
        </div>
      ) : (
        <div className="space-y-4">
          <p className="text-sm text-muted-foreground font-medium">
            {t('revision.mocktest.count')}: {questions.length}
          </p>
          <ul className="space-y-3">
            {questions.map((q, index) => (
              <li key={q.id} className="p-4 rounded-xl border border-border bg-card shadow-sm">
                <p className="font-medium text-sm mb-3">
                  <span className="text-muted-foreground mr-2">{index + 1}.</span>
                  {q.prompt}
                </p>
                <div className="grid gap-2 pl-6">
                  {q.choices.map((choice, cIndex) => (
                    <div key={cIndex} className="text-sm text-muted-foreground flex gap-2">
                      <span className="opacity-50">{String.fromCharCode(97 + cIndex)})</span>
                      <span>{choice}</span>
                    </div>
                  ))}
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
