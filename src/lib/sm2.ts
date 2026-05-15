import type { ReviewQuality } from '@/types/revision';

export interface Sm2State {
  easeFactor: number;
  intervalDays: number;
  repetitions: number;
}

export interface Sm2Result extends Sm2State {
  dueAt: Date;
}

const MIN_EASE_FACTOR = 1.3;
const DEFAULT_EASE_FACTOR = 2.5;

/**
 * Standard SM-2 spaced repetition scheduler.
 * Quality scale: 0-2 = fail (reset reps), 3 = hard pass, 4 = good, 5 = easy.
 */
export function applySm2(
  prev: Sm2State,
  quality: ReviewQuality,
  now: Date = new Date()
): Sm2Result {
  const prevEase = Number.isFinite(prev.easeFactor) && prev.easeFactor > 0
    ? prev.easeFactor
    : DEFAULT_EASE_FACTOR;
  const prevInterval = Math.max(0, prev.intervalDays | 0);
  const prevReps = Math.max(0, prev.repetitions | 0);

  const nextEase = Math.max(
    MIN_EASE_FACTOR,
    prevEase + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
  );

  let nextReps: number;
  let nextInterval: number;

  if (quality < 3) {
    nextReps = 0;
    nextInterval = 1;
  } else if (prevReps === 0) {
    nextReps = 1;
    nextInterval = 1;
  } else if (prevReps === 1) {
    nextReps = 2;
    nextInterval = 6;
  } else {
    nextReps = prevReps + 1;
    nextInterval = Math.round(prevInterval * nextEase);
  }

  const dueAt = new Date(now.getTime() + nextInterval * 24 * 60 * 60 * 1000);

  return {
    easeFactor: nextEase,
    intervalDays: nextInterval,
    repetitions: nextReps,
    dueAt,
  };
}
