'use server';

import {createClient} from '@/lib/supabase/server';
import {applySm2} from '@/lib/sm2';
import type {
  Flashcard,
  FlashcardDeck,
  MockTest,
  MockTestAttempt,
  MockTestAttemptAnswer,
  MockTestAttemptScoredAnswer,
  MockTestQuestion,
  MockTestQuestionPublic,
  ReviewQuality,
  RevisionLanguage,
} from '@/types/revision';

const FLASHCARD_GENERATION_TIMEOUT_MS = 90_000;

type DeckRow = {
  id: string;
  user_id: string;
  course_id: string | null;
  title: string;
  source_pdf_ids: string[];
  status: 'pending' | 'ready' | 'failed';
  generation_error: string | null;
  created_at: string;
  updated_at: string;
};

type CardRow = {
  id: string;
  deck_id: string;
  user_id: string;
  front: string;
  back: string;
  source_chunk_ids: string[];
  ease_factor: number;
  interval_days: number;
  repetitions: number;
  due_at: string;
  last_reviewed_at: string | null;
  created_at: string;
};

type MockTestRow = {
  id: string;
  user_id: string;
  course_id: string | null;
  title: string;
  source_pdf_ids: string[];
  question_count: number;
  status: 'pending' | 'ready' | 'failed';
  generation_error: string | null;
  created_at: string;
};

type QuestionRow = {
  id: string;
  mock_test_id: string;
  prompt: string;
  choices: unknown;
  correct_index: number;
  explanation: string | null;
  source_chunk_ids: string[];
  position: number;
};

type AttemptRow = {
  id: string;
  mock_test_id: string;
  user_id: string;
  score_percent: number;
  answers: unknown;
  started_at: string;
  completed_at: string;
};

function deckFromRow(row: DeckRow): FlashcardDeck {
  return {
    id: row.id,
    userId: row.user_id,
    courseId: row.course_id,
    title: row.title,
    sourcePdfIds: row.source_pdf_ids ?? [],
    status: row.status,
    generationError: row.generation_error,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
  };
}

function cardFromRow(row: CardRow): Flashcard {
  return {
    id: row.id,
    deckId: row.deck_id,
    userId: row.user_id,
    front: row.front,
    back: row.back,
    sourceChunkIds: row.source_chunk_ids ?? [],
    easeFactor: row.ease_factor,
    intervalDays: row.interval_days,
    repetitions: row.repetitions,
    dueAt: row.due_at,
    lastReviewedAt: row.last_reviewed_at,
    createdAt: row.created_at,
  };
}

function mockTestFromRow(row: MockTestRow): MockTest {
  return {
    id: row.id,
    userId: row.user_id,
    courseId: row.course_id,
    title: row.title,
    sourcePdfIds: row.source_pdf_ids ?? [],
    questionCount: row.question_count,
    status: row.status,
    generationError: row.generation_error,
    createdAt: row.created_at,
  };
}

function questionFromRow(row: QuestionRow): MockTestQuestion {
  const choices = Array.isArray(row.choices) ? row.choices.map(String) : [];
  return {
    id: row.id,
    mockTestId: row.mock_test_id,
    prompt: row.prompt,
    choices,
    correctIndex: row.correct_index,
    explanation: row.explanation,
    sourceChunkIds: row.source_chunk_ids ?? [],
    position: row.position,
  };
}

function attemptFromRow(row: AttemptRow): MockTestAttempt {
  const answers = Array.isArray(row.answers)
    ? (row.answers as MockTestAttemptScoredAnswer[])
    : [];
  return {
    id: row.id,
    mockTestId: row.mock_test_id,
    userId: row.user_id,
    scorePercent: row.score_percent,
    answers,
    startedAt: row.started_at,
    completedAt: row.completed_at,
  };
}

function ragEnv(): {ragApiUrl: string; internalApiKey: string} | {error: string} {
  const ragApiUrl = process.env.RAG_API_URL;
  const internalApiKey = process.env.RAG_INTERNAL_API_KEY;
  if (!ragApiUrl || !internalApiKey) {
    return {error: 'Revision is not configured (missing RAG_API_URL or RAG_INTERNAL_API_KEY).'};
  }
  return {ragApiUrl, internalApiKey};
}

type GeneratedFlashcardPayload = {
  front: string;
  back: string;
  source_chunk_ids?: string[];
};

type GeneratedMockQuestionPayload = {
  prompt: string;
  choices: string[];
  correct_index: number;
  explanation?: string;
  source_chunk_ids?: string[];
};

async function callRagPost<T>(
  endpoint: string,
  body: unknown,
  timeoutMs = FLASHCARD_GENERATION_TIMEOUT_MS,
): Promise<{data?: T; error?: string}> {
  const env = ragEnv();
  if ('error' in env) return {error: env.error};
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${env.ragApiUrl.replace(/\/$/, '')}${endpoint}`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${env.internalApiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    if (!response.ok) {
      const detail = await response.text().catch(() => '');
      return {error: `Generation failed: HTTP ${response.status} ${detail.slice(0, 200)}`};
    }
    const json = (await response.json()) as T;
    return {data: json};
  } catch (err) {
    if (err instanceof Error && err.name === 'AbortError') {
      return {error: 'Generation timed out.'};
    }
    return {error: err instanceof Error ? err.message : 'Generation failed.'};
  } finally {
    clearTimeout(timeout);
  }
}

// ---------------------------------------------------------------------------
// Flashcard decks
// ---------------------------------------------------------------------------

export async function createFlashcardDeck(input: {
  title: string;
  pdfIds: string[];
  count: number;
  language?: RevisionLanguage;
  courseId?: string | null;
}): Promise<{data?: FlashcardDeck; error?: string}> {
  const supabase = await createClient();
  const {data: {user}} = await supabase.auth.getUser();
  if (!user) return {error: 'Not authenticated.'};

  const title = input.title.trim();
  if (!title) return {error: 'Title is required.'};
  if (!Array.isArray(input.pdfIds)) {
    return {error: 'Invalid PDFs selection.'};
  }
  const isManualDeck = input.pdfIds.length === 0;
  const count = Math.max(1, Math.min(Math.floor(input.count || 0), 30));
  const language: RevisionLanguage = input.language === 'en' ? 'en' : 'de';

  const {data: deckRow, error: insertError} = await supabase
    .from('flashcard_decks')
    .insert({
      user_id: user.id,
      course_id: input.courseId ?? null,
      title,
      source_pdf_ids: input.pdfIds,
      status: isManualDeck ? 'ready' : 'pending',
    })
    .select('*')
    .single<DeckRow>();

  if (insertError || !deckRow) {
    return {error: insertError?.message ?? 'Could not create deck.'};
  }

  if (isManualDeck) {
    return {data: deckFromRow(deckRow)};
  }

  const result = await callRagPost<{cards: GeneratedFlashcardPayload[]}>(
    '/revision/flashcards',
    {
      user_id: user.id,
      pdf_ids: input.pdfIds,
      count,
      language,
    },
  );

  if (result.error || !result.data) {
    await supabase
      .from('flashcard_decks')
      .update({status: 'failed', generation_error: result.error ?? 'Unknown error', updated_at: new Date().toISOString()})
      .eq('id', deckRow.id)
      .eq('user_id', user.id);
    return {error: result.error ?? 'Generation failed.'};
  }

  const cards = result.data.cards ?? [];
  if (cards.length === 0) {
    await supabase
      .from('flashcard_decks')
      .update({status: 'failed', generation_error: 'No cards generated.', updated_at: new Date().toISOString()})
      .eq('id', deckRow.id)
      .eq('user_id', user.id);
    return {error: 'No cards generated.'};
  }

  const nowIso = new Date().toISOString();
  const insertRows = cards.map((card) => ({
    deck_id: deckRow.id,
    user_id: user.id,
    front: card.front,
    back: card.back,
    source_chunk_ids: card.source_chunk_ids ?? [],
    ease_factor: 2.5,
    interval_days: 0,
    repetitions: 0,
    due_at: nowIso,
  }));

  const {error: cardError} = await supabase.from('flashcards').insert(insertRows);
  if (cardError) {
    await supabase
      .from('flashcard_decks')
      .update({status: 'failed', generation_error: cardError.message, updated_at: nowIso})
      .eq('id', deckRow.id)
      .eq('user_id', user.id);
    return {error: cardError.message};
  }

  const {data: updated, error: updateError} = await supabase
    .from('flashcard_decks')
    .update({status: 'ready', generation_error: null, updated_at: nowIso})
    .eq('id', deckRow.id)
    .eq('user_id', user.id)
    .select('*')
    .single<DeckRow>();

  if (updateError || !updated) {
    return {error: updateError?.message ?? 'Could not finalise deck.'};
  }
  return {data: deckFromRow(updated)};
}

export async function listFlashcardDecks(courseId?: string | null): Promise<{
  data?: FlashcardDeck[];
  error?: string;
}> {
  const supabase = await createClient();
  const {data: {user}} = await supabase.auth.getUser();
  if (!user) return {error: 'Not authenticated.'};

  let query = supabase
    .from('flashcard_decks')
    .select('*')
    .eq('user_id', user.id)
    .order('updated_at', {ascending: false});
  if (courseId) query = query.eq('course_id', courseId);

  const {data, error} = await query.returns<DeckRow[]>();
  if (error) return {error: error.message};
  return {data: (data ?? []).map(deckFromRow)};
}

export async function getDeck(deckId: string): Promise<{
  data?: FlashcardDeck;
  error?: string;
}> {
  const supabase = await createClient();
  const {data: {user}} = await supabase.auth.getUser();
  if (!user) return {error: 'Not authenticated.'};

  const {data, error} = await supabase
    .from('flashcard_decks')
    .select('*')
    .eq('id', deckId)
    .eq('user_id', user.id)
    .maybeSingle<DeckRow>();
  if (error) return {error: error.message};
  if (!data) return {error: 'Deck not found.'};
  return {data: deckFromRow(data)};
}

export async function getDeckCards(deckId: string): Promise<{
  data?: Flashcard[];
  error?: string;
}> {
  const supabase = await createClient();
  const {data: {user}} = await supabase.auth.getUser();
  if (!user) return {error: 'Not authenticated.'};

  const {data, error} = await supabase
    .from('flashcards')
    .select('*')
    .eq('deck_id', deckId)
    .eq('user_id', user.id)
    .order('created_at', {ascending: true})
    .returns<CardRow[]>();
  if (error) return {error: error.message};
  return {data: (data ?? []).map(cardFromRow)};
}

export async function deleteDeck(deckId: string): Promise<{error?: string}> {
  const supabase = await createClient();
  const {data: {user}} = await supabase.auth.getUser();
  if (!user) return {error: 'Not authenticated.'};

  const {error} = await supabase
    .from('flashcard_decks')
    .delete()
    .eq('id', deckId)
    .eq('user_id', user.id);
  if (error) return {error: error.message};
  return {};
}

export async function getDueCards(
  deckId: string,
  limit = 20,
): Promise<{data?: Flashcard[]; error?: string}> {
  const supabase = await createClient();
  const {data: {user}} = await supabase.auth.getUser();
  if (!user) return {error: 'Not authenticated.'};

  const cappedLimit = Math.max(1, Math.min(limit, 100));
  const {data, error} = await supabase
    .from('flashcards')
    .select('*')
    .eq('deck_id', deckId)
    .eq('user_id', user.id)
    .lte('due_at', new Date().toISOString())
    .order('due_at', {ascending: true})
    .limit(cappedLimit)
    .returns<CardRow[]>();
  if (error) return {error: error.message};
  return {data: (data ?? []).map(cardFromRow)};
}

export async function reviewFlashcard(input: {
  cardId: string;
  quality: ReviewQuality;
}): Promise<{data?: Flashcard; error?: string}> {
  const supabase = await createClient();
  const {data: {user}} = await supabase.auth.getUser();
  if (!user) return {error: 'Not authenticated.'};

  if (![0, 1, 2, 3, 4, 5].includes(input.quality)) {
    return {error: 'Invalid quality.'};
  }

  const {data: existing, error: fetchError} = await supabase
    .from('flashcards')
    .select('*')
    .eq('id', input.cardId)
    .eq('user_id', user.id)
    .maybeSingle<CardRow>();
  if (fetchError) return {error: fetchError.message};
  if (!existing) return {error: 'Card not found.'};

  const now = new Date();
  const next = applySm2(
    {
      easeFactor: existing.ease_factor,
      intervalDays: existing.interval_days,
      repetitions: existing.repetitions,
    },
    input.quality,
    now,
  );

  const {data: updated, error: updateError} = await supabase
    .from('flashcards')
    .update({
      ease_factor: next.easeFactor,
      interval_days: next.intervalDays,
      repetitions: next.repetitions,
      due_at: next.dueAt.toISOString(),
      last_reviewed_at: now.toISOString(),
    })
    .eq('id', input.cardId)
    .eq('user_id', user.id)
    .select('*')
    .single<CardRow>();
  if (updateError || !updated) {
    return {error: updateError?.message ?? 'Could not update card.'};
  }
  return {data: cardFromRow(updated)};
}

// ---------------------------------------------------------------------------
// Mock tests
// ---------------------------------------------------------------------------

export async function createMockTest(input: {
  title: string;
  pdfIds: string[];
  count: number;
  language?: RevisionLanguage;
  courseId?: string | null;
}): Promise<{data?: MockTest; error?: string}> {
  const supabase = await createClient();
  const {data: {user}} = await supabase.auth.getUser();
  if (!user) return {error: 'Not authenticated.'};

  const title = input.title.trim();
  if (!title) return {error: 'Title is required.'};
  if (!Array.isArray(input.pdfIds) || input.pdfIds.length === 0) {
    return {error: 'At least one PDF must be selected.'};
  }
  const count = Math.max(1, Math.min(Math.floor(input.count || 0), 20));
  const language: RevisionLanguage = input.language === 'en' ? 'en' : 'de';

  const {data: testRow, error: insertError} = await supabase
    .from('mock_tests')
    .insert({
      user_id: user.id,
      course_id: input.courseId ?? null,
      title,
      source_pdf_ids: input.pdfIds,
      question_count: 0,
      status: 'pending',
    })
    .select('*')
    .single<MockTestRow>();
  if (insertError || !testRow) {
    return {error: insertError?.message ?? 'Could not create test.'};
  }

  const result = await callRagPost<{questions: GeneratedMockQuestionPayload[]}>(
    '/revision/mocktest',
    {
      user_id: user.id,
      pdf_ids: input.pdfIds,
      count,
      language,
    },
  );

  if (result.error || !result.data) {
    await supabase
      .from('mock_tests')
      .update({status: 'failed', generation_error: result.error ?? 'Unknown error'})
      .eq('id', testRow.id)
      .eq('user_id', user.id);
    return {error: result.error ?? 'Generation failed.'};
  }

  const questions = result.data.questions ?? [];
  if (questions.length === 0) {
    await supabase
      .from('mock_tests')
      .update({status: 'failed', generation_error: 'No questions generated.'})
      .eq('id', testRow.id)
      .eq('user_id', user.id);
    return {error: 'No questions generated.'};
  }

  const insertRows = questions.map((q, index) => ({
    mock_test_id: testRow.id,
    user_id: user.id,
    prompt: q.prompt,
    choices: q.choices,
    correct_index: q.correct_index,
    explanation: q.explanation ?? '',
    source_chunk_ids: q.source_chunk_ids ?? [],
    position: index,
  }));
  const {error: qError} = await supabase.from('mock_test_questions').insert(insertRows);
  if (qError) {
    await supabase
      .from('mock_tests')
      .update({status: 'failed', generation_error: qError.message})
      .eq('id', testRow.id)
      .eq('user_id', user.id);
    return {error: qError.message};
  }

  const {data: updated, error: updateError} = await supabase
    .from('mock_tests')
    .update({status: 'ready', question_count: insertRows.length, generation_error: null})
    .eq('id', testRow.id)
    .eq('user_id', user.id)
    .select('*')
    .single<MockTestRow>();
  if (updateError || !updated) {
    return {error: updateError?.message ?? 'Could not finalise test.'};
  }
  return {data: mockTestFromRow(updated)};
}

export async function listMockTests(courseId?: string | null): Promise<{
  data?: MockTest[];
  error?: string;
}> {
  const supabase = await createClient();
  const {data: {user}} = await supabase.auth.getUser();
  if (!user) return {error: 'Not authenticated.'};

  let query = supabase
    .from('mock_tests')
    .select('*')
    .eq('user_id', user.id)
    .order('created_at', {ascending: false});
  if (courseId) query = query.eq('course_id', courseId);

  const {data, error} = await query.returns<MockTestRow[]>();
  if (error) return {error: error.message};
  return {data: (data ?? []).map(mockTestFromRow)};
}

export async function deleteMockTest(testId: string): Promise<{error?: string}> {
  const supabase = await createClient();
  const {data: {user}} = await supabase.auth.getUser();
  if (!user) return {error: 'Not authenticated.'};

  const {error} = await supabase
    .from('mock_tests')
    .delete()
    .eq('id', testId)
    .eq('user_id', user.id);
  if (error) return {error: error.message};
  return {};
}

export async function getMockTestWithQuestions(testId: string): Promise<{
  data?: {test: MockTest; questions: MockTestQuestionPublic[]};
  error?: string;
}> {
  const supabase = await createClient();
  const {data: {user}} = await supabase.auth.getUser();
  if (!user) return {error: 'Not authenticated.'};

  const {data: test, error: testError} = await supabase
    .from('mock_tests')
    .select('*')
    .eq('id', testId)
    .eq('user_id', user.id)
    .maybeSingle<MockTestRow>();
  if (testError) return {error: testError.message};
  if (!test) return {error: 'Test not found.'};

  const {data: questions, error: qError} = await supabase
    .from('mock_test_questions')
    .select('id, mock_test_id, prompt, choices, position')
    .eq('mock_test_id', testId)
    .eq('user_id', user.id)
    .order('position', {ascending: true})
    .returns<Pick<QuestionRow, 'id' | 'mock_test_id' | 'prompt' | 'choices' | 'position'>[]>();
  if (qError) return {error: qError.message};

  const publicQuestions: MockTestQuestionPublic[] = (questions ?? []).map((row) => ({
    id: row.id,
    mockTestId: row.mock_test_id,
    prompt: row.prompt,
    choices: Array.isArray(row.choices) ? (row.choices as string[]).map(String) : [],
    position: row.position,
  }));

  return {data: {test: mockTestFromRow(test), questions: publicQuestions}};
}

export async function submitMockTestAttempt(input: {
  testId: string;
  answers: MockTestAttemptAnswer[];
  startedAt?: string;
}): Promise<{data?: MockTestAttempt; error?: string}> {
  const supabase = await createClient();
  const {data: {user}} = await supabase.auth.getUser();
  if (!user) return {error: 'Not authenticated.'};

  if (!Array.isArray(input.answers) || input.answers.length === 0) {
    return {error: 'No answers submitted.'};
  }

  const {data: questions, error: qError} = await supabase
    .from('mock_test_questions')
    .select('*')
    .eq('mock_test_id', input.testId)
    .eq('user_id', user.id)
    .returns<QuestionRow[]>();
  if (qError) return {error: qError.message};
  if (!questions || questions.length === 0) {
    return {error: 'Test has no questions.'};
  }

  const byId = new Map(questions.map((q) => [q.id, q]));
  const scored: MockTestAttemptScoredAnswer[] = [];
  let correctCount = 0;
  for (const answer of input.answers) {
    const q = byId.get(answer.questionId);
    if (!q) continue;
    const isCorrect = Number(answer.chosenIndex) === q.correct_index;
    if (isCorrect) correctCount += 1;
    scored.push({
      questionId: q.id,
      chosenIndex: Number(answer.chosenIndex),
      correct: isCorrect,
      correctIndex: q.correct_index,
      explanation: q.explanation,
    });
  }

  if (scored.length === 0) return {error: 'No answers matched the test questions.'};

  const scorePercent = (correctCount / questions.length) * 100;
  const startedAt = input.startedAt ?? new Date().toISOString();

  const {data: attempt, error: attemptError} = await supabase
    .from('mock_test_attempts')
    .insert({
      mock_test_id: input.testId,
      user_id: user.id,
      score_percent: scorePercent,
      answers: scored,
      started_at: startedAt,
      completed_at: new Date().toISOString(),
    })
    .select('*')
    .single<AttemptRow>();
  if (attemptError || !attempt) {
    return {error: attemptError?.message ?? 'Could not save attempt.'};
  }
  return {data: attemptFromRow(attempt)};
}

export async function listMockTestAttempts(testId: string): Promise<{
  data?: MockTestAttempt[];
  error?: string;
}> {
  const supabase = await createClient();
  const {data: {user}} = await supabase.auth.getUser();
  if (!user) return {error: 'Not authenticated.'};

  const {data, error} = await supabase
    .from('mock_test_attempts')
    .select('*')
    .eq('mock_test_id', testId)
    .eq('user_id', user.id)
    .order('completed_at', {ascending: false})
    .returns<AttemptRow[]>();
  if (error) return {error: error.message};
  return {data: (data ?? []).map(attemptFromRow)};
}

export async function listAllUserMockTestAttempts(): Promise<{
  data?: MockTestAttempt[];
  error?: string;
}> {
  const supabase = await createClient();
  const {data: {user}} = await supabase.auth.getUser();
  if (!user) return {error: 'Not authenticated.'};

  const {data, error} = await supabase
    .from('mock_test_attempts')
    .select('*')
    .eq('user_id', user.id)
    .order('completed_at', {ascending: false})
    .returns<AttemptRow[]>();
  if (error) return {error: error.message};
  return {data: (data ?? []).map(attemptFromRow)};
}

export async function createFlashcard(input: {
  deckId: string;
  front: string;
  back: string;
}): Promise<{data?: Flashcard; error?: string}> {
  const supabase = await createClient();
  const {data: {user}} = await supabase.auth.getUser();
  if (!user) return {error: 'Not authenticated.'};

  if (!input.front.trim() || !input.back.trim()) {
    return {error: 'Front and back content required.'};
  }

  const nowIso = new Date().toISOString();
  const insertRow = {
    deck_id: input.deckId,
    user_id: user.id,
    front: input.front.trim(),
    back: input.back.trim(),
    source_chunk_ids: [],
    ease_factor: 2.5,
    interval_days: 0,
    repetitions: 0,
    due_at: nowIso,
  };

  const {data: cardRow, error} = await supabase
    .from('flashcards')
    .insert(insertRow)
    .select('*')
    .single<CardRow>();

  if (error || !cardRow) return {error: error?.message ?? 'Could not create card.'};
  return {data: cardFromRow(cardRow)};
}

export async function deleteFlashcard(cardId: string): Promise<{error?: string}> {
  const supabase = await createClient();
  const {data: {user}} = await supabase.auth.getUser();
  if (!user) return {error: 'Not authenticated.'};

  const {error} = await supabase
    .from('flashcards')
    .delete()
    .eq('id', cardId)
    .eq('user_id', user.id);

  if (error) return {error: error.message};
  return {};
}

export async function updateFlashcard(input: {
  cardId: string;
  front: string;
  back: string;
}): Promise<{data?: Flashcard; error?: string}> {
  const supabase = await createClient();
  const {data: {user}} = await supabase.auth.getUser();
  if (!user) return {error: 'Not authenticated.'};

  if (!input.front.trim() || !input.back.trim()) {
    return {error: 'Front and back content required.'};
  }

  const {data: cardRow, error} = await supabase
    .from('flashcards')
    .update({
      front: input.front.trim(),
      back: input.back.trim(),
      updated_at: new Date().toISOString(),
    })
    .eq('id', input.cardId)
    .eq('user_id', user.id)
    .select('*')
    .single<CardRow>();

  if (error || !cardRow) return {error: error?.message ?? 'Could not update card.'};
  return {data: cardFromRow(cardRow)};
}

