export type RevisionStatus = 'pending' | 'ready' | 'failed';

export type ReviewQuality = 0 | 1 | 2 | 3 | 4 | 5;

export interface FlashcardDeck {
  id: string;
  userId: string;
  courseId: string | null;
  title: string;
  sourcePdfIds: string[];
  status: RevisionStatus;
  generationError: string | null;
  createdAt: string;
  updatedAt: string;
}

export interface Flashcard {
  id: string;
  deckId: string;
  userId: string;
  front: string;
  back: string;
  sourceChunkIds: string[];
  easeFactor: number;
  intervalDays: number;
  repetitions: number;
  dueAt: string;
  lastReviewedAt: string | null;
  createdAt: string;
}

export interface MockTest {
  id: string;
  userId: string;
  courseId: string | null;
  title: string;
  sourcePdfIds: string[];
  questionCount: number;
  status: RevisionStatus;
  generationError: string | null;
  createdAt: string;
}

export interface MockTestQuestion {
  id: string;
  mockTestId: string;
  prompt: string;
  choices: string[];
  correctIndex: number;
  explanation: string | null;
  sourceChunkIds: string[];
  position: number;
}

export interface MockTestQuestionPublic {
  id: string;
  mockTestId: string;
  prompt: string;
  choices: string[];
  position: number;
}

export interface MockTestAttemptAnswer {
  questionId: string;
  chosenIndex: number;
}

export interface MockTestAttemptScoredAnswer extends MockTestAttemptAnswer {
  correct: boolean;
  correctIndex: number;
  explanation: string | null;
}

export interface MockTestAttempt {
  id: string;
  mockTestId: string;
  userId: string;
  scorePercent: number;
  answers: MockTestAttemptScoredAnswer[];
  startedAt: string;
  completedAt: string;
}

export type RevisionLanguage = 'de' | 'en';
