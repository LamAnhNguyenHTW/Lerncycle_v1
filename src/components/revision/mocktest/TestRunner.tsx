'use client';

import {useEffect, useState} from 'react';
import {useLanguage} from '@/lib/i18n';
import {Button} from '@/components/ui/button';
import {
  getMockTestWithQuestions,
  submitMockTestAttempt,
} from '@/actions/revision';
import type {
  MockTest,
  MockTestAttempt,
  MockTestQuestionPublic,
} from '@/types/revision';

interface Props {
  testId: string;
  onBack: () => void;
}

export function TestRunner({testId, onBack}: Props) {
  const {t} = useLanguage();
  const [test, setTest] = useState<MockTest | null>(null);
  const [questions, setQuestions] = useState<MockTestQuestionPublic[]>([]);
  const [answers, setAnswers] = useState<Record<string, number>>({});
  const [current, setCurrent] = useState(0);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [attempt, setAttempt] = useState<MockTestAttempt | null>(null);
  const [startedAt] = useState(() => new Date().toISOString());
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
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

  if (loading) {
    return <div className="py-12 text-center text-muted-foreground">...</div>;
  }
  if (error || !test) {
    return (
      <div className="space-y-3">
        <Button variant="ghost" size="sm" onClick={onBack}>
          ← {t('revision.mocktest.back')}
        </Button>
        <p className="text-sm text-destructive">{error}</p>
      </div>
    );
  }

  if (attempt) {
    const scoredById = new Map(attempt.answers.map((a) => [a.questionId, a]));
    return (
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <Button variant="ghost" size="sm" onClick={onBack}>
            ← {t('revision.mocktest.back')}
          </Button>
          <div className="text-right">
            <p className="text-xs text-muted-foreground">{t('revision.mocktest.score')}</p>
            <p className="text-2xl font-semibold">{Math.round(attempt.scorePercent)}%</p>
          </div>
        </div>
        <ol className="space-y-4">
          {questions.map((q, i) => {
            const scored = scoredById.get(q.id);
            const correctIdx = scored?.correctIndex ?? -1;
            const chosenIdx = scored?.chosenIndex ?? -1;
            return (
              <li key={q.id} className="rounded-lg border border-border bg-card p-4">
                <p className="font-medium mb-2">
                  {i + 1}. {q.prompt}
                </p>
                <ul className="space-y-1 mb-2">
                  {q.choices.map((choice, ci) => {
                    const isCorrect = ci === correctIdx;
                    const isChosen = ci === chosenIdx;
                    return (
                      <li
                        key={ci}
                        className={
                          'rounded-md px-3 py-1.5 text-sm border ' +
                          (isCorrect
                            ? 'border-emerald-500/40 bg-emerald-500/10'
                            : isChosen
                              ? 'border-destructive/40 bg-destructive/10'
                              : 'border-transparent')
                        }
                      >
                        {choice} {isCorrect && '✓'} {!isCorrect && isChosen && '✗'}
                      </li>
                    );
                  })}
                </ul>
                {scored?.explanation && (
                  <p className="text-xs text-muted-foreground">
                    <span className="font-medium">{t('revision.mocktest.explanation')}: </span>
                    {scored.explanation}
                  </p>
                )}
              </li>
            );
          })}
        </ol>
      </div>
    );
  }

  const q = questions[current];
  const totalAnswered = Object.keys(answers).length;
  const allAnswered = totalAnswered === questions.length;

  const handleSubmit = async () => {
    setSubmitting(true);
    const payload = Object.entries(answers).map(([questionId, chosenIndex]) => ({
      questionId,
      chosenIndex,
    }));
    const result = await submitMockTestAttempt({
      testId,
      answers: payload,
      startedAt,
    });
    setSubmitting(false);
    if (result.error || !result.data) {
      setError(result.error ?? 'Submission failed.');
      return;
    }
    setAttempt(result.data);
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <Button variant="ghost" size="sm" onClick={onBack}>
          ← {t('revision.mocktest.back')}
        </Button>
        <p className="text-sm text-muted-foreground">
          {t('revision.mocktest.questionOf', {
            n: String(current + 1),
            total: String(questions.length),
          })}
        </p>
      </div>

      {q ? (
        <div className="rounded-xl border border-border bg-card p-6 space-y-4">
          <p className="text-lg font-medium">{q.prompt}</p>
          <div className="space-y-2">
            {q.choices.map((choice, ci) => {
              const selected = answers[q.id] === ci;
              return (
                <button
                  key={ci}
                  type="button"
                  onClick={() => setAnswers((prev) => ({...prev, [q.id]: ci}))}
                  className={
                    'block w-full text-left rounded-md border px-3 py-2 text-sm transition ' +
                    (selected
                      ? 'border-primary bg-primary/10'
                      : 'border-border hover:bg-muted/50')
                  }
                >
                  {choice}
                </button>
              );
            })}
          </div>
        </div>
      ) : null}

      <div className="flex items-center justify-between gap-2">
        <Button
          variant="outline"
          onClick={() => setCurrent((c) => Math.max(0, c - 1))}
          disabled={current === 0}
        >
          {t('revision.mocktest.previous')}
        </Button>
        {current < questions.length - 1 ? (
          <Button onClick={() => setCurrent((c) => Math.min(questions.length - 1, c + 1))}>
            {t('revision.mocktest.next')}
          </Button>
        ) : (
          <Button onClick={handleSubmit} disabled={!allAnswered || submitting}>
            {submitting ? '...' : t('revision.mocktest.submit')}
          </Button>
        )}
      </div>
    </div>
  );
}
