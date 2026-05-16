import {NextResponse} from 'next/server';
import {createClient} from '@/lib/supabase/server';
import type {SupabaseClient} from '@supabase/supabase-js';
import type {ActiveLearningState, ChatMode, ChatRequest, ChatResponse, ChatRole, ChatSourceType, RecentChatMessage} from '@/types/chat';

const MATERIAL_SOURCE_TYPES: ChatSourceType[] = ['pdf', 'note', 'annotation_comment'];
const SOURCE_TYPES: ChatSourceType[] = [...MATERIAL_SOURCE_TYPES, 'chat_memory', 'web'];
const CHAT_MODES: ChatMode[] = ['normal', 'guided_learning', 'feynman'];
const RECENT_MESSAGE_LIMIT = 10;
const RECENT_MESSAGE_CONTENT_LIMIT = 2000;
const CHAT_MEMORY_DEFAULT_THRESHOLD = 8;
const CHAT_MEMORY_DEFAULT_INTERVAL = 4;
const CHAT_MEMORY_DEFAULT_KEEP_RECENT = 4;
const CHAT_MEMORY_DEFAULT_MAX_CHARS = 2500;
const PROMPT_COMPACTION_DEFAULT_THRESHOLD = 12;
const PROMPT_COMPACTION_DEFAULT_INTERVAL = 4;
const PROMPT_COMPACTION_DEFAULT_KEEP_RECENT = 6;
const PROMPT_COMPACTION_DEFAULT_MAX_CHARS = 1500;
const FORBIDDEN_RAG_TOOL_FIELDS = [
  'tools',
  'tool',
  'tool_args',
  'tool_registry',
  'allowed_tools',
  'cypher',
  'neo4j_query',
  'retrieval_plan',
  'intent',
  'agentic_decision',
  'refinement_action',
  'agentic_tool',
  'agentic_tool_args',
  'max_tool_calls',
  'max_refinement_rounds',
  'raw_tool_calls',
] as const;

class SessionNotFoundError extends Error {}
class MissingActiveLearningMigrationError extends Error {}

type ChatSessionContext = {
  id: string;
  mode: ChatMode;
  activeLearningState: ActiveLearningState;
};

function errorResponse(message: string, status: number) {
  return NextResponse.json({error: message}, {status});
}

function validateBody(body: Partial<ChatRequest>) {
  if (typeof body.message !== 'string' || body.message.trim().length === 0) {
    return 'Message must be a non-empty string.';
  }
  if (body.message.length > 2000) {
    return 'Message must be at most 2000 characters.';
  }
  if (body.top_k !== undefined && (!Number.isInteger(body.top_k) || body.top_k < 1 || body.top_k > 20)) {
    return 'top_k must be an integer between 1 and 20.';
  }
  if (
    body.source_types !== undefined &&
    (!Array.isArray(body.source_types) || body.source_types.some((sourceType) => !SOURCE_TYPES.includes(sourceType)))
  ) {
    return 'source_types contains an unsupported value.';
  }
  if (body.pdf_ids !== undefined && (!Array.isArray(body.pdf_ids) || body.pdf_ids.some((pdfId) => typeof pdfId !== 'string'))) {
    return 'pdf_ids must be an array of strings.';
  }
  if (body.mode !== undefined && !CHAT_MODES.includes(body.mode)) {
    return 'mode must be normal, guided_learning, or feynman.';
  }
  if (body.topic !== undefined && typeof body.topic !== 'string') {
    return 'topic must be a string.';
  }
  if (
    body.difficulty !== undefined &&
    !['beginner', 'intermediate', 'advanced'].includes(body.difficulty)
  ) {
    return 'difficulty must be beginner, intermediate, or advanced.';
  }
  if (body.language !== undefined && !['de', 'en'].includes(body.language)) {
    return 'language must be de or en.';
  }
  if (body.learner_name !== undefined && typeof body.learner_name !== 'string') {
    return 'learner_name must be a string.';
  }
  return null;
}

function normalizeChatMode(value: unknown): ChatMode {
  return CHAT_MODES.includes(value as ChatMode) ? (value as ChatMode) : 'normal';
}

function normalizeActiveLearningState(value: unknown): ActiveLearningState {
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    return value as ActiveLearningState;
  }
  return {};
}

function isMissingActiveLearningColumnError(error: unknown) {
  if (!error || typeof error !== 'object') {
    return false;
  }
  const maybeError = error as {code?: string; message?: string; details?: string};
  const text = `${maybeError.message ?? ''} ${maybeError.details ?? ''}`.toLowerCase();
  return (
    maybeError.code === 'PGRST204' ||
    maybeError.code === '42703' ||
    text.includes('active_learning_state') ||
    text.includes('chat_sessions.mode') ||
    text.includes("column 'mode'") ||
    text.includes('column mode') ||
    text.includes('schema cache')
  );
}

async function getOrCreateSession(
  supabase: Awaited<ReturnType<typeof createClient>>,
  userId: string,
  body: Partial<ChatRequest>,
) {
  if (body.session_id) {
    const {data, error} = await supabase
      .from('chat_sessions')
      .select('id, mode, active_learning_state')
      .eq('id', body.session_id)
      .eq('user_id', userId)
      .maybeSingle();
    if (error && isMissingActiveLearningColumnError(error)) {
      const fallback = await supabase
        .from('chat_sessions')
        .select('id')
        .eq('id', body.session_id)
        .eq('user_id', userId)
        .maybeSingle();
      if (fallback.data?.id) {
        return {
          id: fallback.data.id as string,
          mode: 'normal',
          activeLearningState: {},
        };
      }
    }
    if (data?.id) {
      return {
        id: data.id as string,
        mode: normalizeChatMode(data.mode),
        activeLearningState: normalizeActiveLearningState(data.active_learning_state),
      };
    }
    throw new SessionNotFoundError('Session not found.');
  }

  const initialActiveLearningState = body.mode && body.mode !== 'normal'
    ? {
      mode: body.mode,
      ...(body.topic ? {topic: body.topic.trim(), target_concept: body.topic.trim()} : {}),
      ...(body.difficulty ? {difficulty: body.difficulty} : {}),
      ...(body.language ? {language: body.language} : {}),
      ...(body.learner_name ? {learner_name: body.learner_name.trim().slice(0, 80)} : {}),
    }
    : {};
  const insertPayload = {
    user_id: userId,
    course_id: body.course_id ?? null,
    title: body.message?.trim().slice(0, 80) ?? null,
    mode: body.mode ?? 'normal',
    active_learning_state: initialActiveLearningState,
  };
  const {data, error} = await supabase
    .from('chat_sessions')
    .insert(insertPayload)
    .select('id, mode, active_learning_state')
    .single();
  if (error && isMissingActiveLearningColumnError(error)) {
    if (body.mode && body.mode !== 'normal') {
      throw new MissingActiveLearningMigrationError('Active learning database migration has not been applied.');
    }
    const fallback = await supabase
      .from('chat_sessions')
      .insert({
        user_id: userId,
        course_id: body.course_id ?? null,
        title: body.message?.trim().slice(0, 80) ?? null,
      })
      .select('id')
      .single();
    if (fallback.error || !fallback.data?.id) {
      console.error('Failed to create chat session', fallback.error ?? error);
      throw new Error('Failed to create chat session.');
    }
    return {
      id: fallback.data.id as string,
      mode: 'normal',
      activeLearningState: {},
    };
  }
  if (error || !data?.id) {
    console.error('Failed to create chat session', error);
    throw new Error('Failed to create chat session.');
  }
  return {
    id: data.id as string,
    mode: normalizeChatMode(data.mode),
    activeLearningState: normalizeActiveLearningState(data.active_learning_state),
  };
}

async function loadSessionPromptContext(
  supabase: SupabaseClient,
  sessionId: string,
  userId: string,
) {
  const {data, error} = await supabase
    .from('chat_sessions')
    .select('course_id, context_summary, context_summary_cursor, mode, active_learning_state')
    .eq('id', sessionId)
    .eq('user_id', userId)
    .maybeSingle();
  if (error && isMissingActiveLearningColumnError(error)) {
    const fallback = await supabase
      .from('chat_sessions')
      .select('course_id, context_summary, context_summary_cursor')
      .eq('id', sessionId)
      .eq('user_id', userId)
      .maybeSingle();
    const fallbackData = fallback.data;
    return {
      courseId: typeof fallbackData?.course_id === 'string' ? fallbackData.course_id : null,
      contextSummary: typeof fallbackData?.context_summary === 'string' && fallbackData.context_summary.trim().length > 0
        ? fallbackData.context_summary.trim()
        : null,
      mode: 'normal' as ChatMode,
      activeLearningState: {},
    };
  }
  return {
    courseId: typeof data?.course_id === 'string' ? data.course_id : null,
    contextSummary: typeof data?.context_summary === 'string' && data.context_summary.trim().length > 0
      ? data.context_summary.trim()
      : null,
    mode: normalizeChatMode(data?.mode),
    activeLearningState: normalizeActiveLearningState(data?.active_learning_state),
  };
}

async function loadRecentMessages(
  supabase: SupabaseClient,
  sessionId: string,
  userId: string,
  limit: number = RECENT_MESSAGE_LIMIT,
): Promise<RecentChatMessage[]> {
  const {data, error} = await supabase
    .from('chat_messages')
    .select('role, content')
    .eq('session_id', sessionId)
    .eq('user_id', userId)
    .order('created_at', {ascending: false})
    .limit(limit);
  if (error) {
    console.error('Failed to load recent chat messages', error);
    return [];
  }

  return (data ?? [])
    .reverse()
    .filter((message): message is {role: ChatRole; content: string} =>
      (message.role === 'user' || message.role === 'assistant') && typeof message.content === 'string',
    )
    .map((message) => ({
      role: message.role,
      content: message.content.slice(0, RECENT_MESSAGE_CONTENT_LIMIT),
    }));
}

function chatMemoryConfig() {
  return {
    enabled: parseBool(process.env.CHAT_MEMORY_ENABLED, false),
    threshold: parseIntEnv(process.env.CHAT_MEMORY_SUMMARY_THRESHOLD, CHAT_MEMORY_DEFAULT_THRESHOLD),
    interval: parseIntEnv(process.env.CHAT_MEMORY_SUMMARY_INTERVAL, CHAT_MEMORY_DEFAULT_INTERVAL),
    keepRecent: parseIntEnv(process.env.CHAT_MEMORY_KEEP_RECENT, CHAT_MEMORY_DEFAULT_KEEP_RECENT),
    maxSummaryChars: parseIntEnv(process.env.CHAT_MEMORY_MAX_SUMMARY_CHARS, CHAT_MEMORY_DEFAULT_MAX_CHARS),
  };
}

function promptCompactionConfig() {
  const keepRecent = Math.max(
    2,
    parseIntEnv(process.env.PROMPT_COMPACTION_KEEP_RECENT, PROMPT_COMPACTION_DEFAULT_KEEP_RECENT),
  );
  const threshold = Math.max(
    keepRecent + 1,
    parseIntEnv(process.env.PROMPT_COMPACTION_THRESHOLD, PROMPT_COMPACTION_DEFAULT_THRESHOLD),
  );
  const interval = Math.max(
    1,
    parseIntEnv(process.env.PROMPT_COMPACTION_INTERVAL, PROMPT_COMPACTION_DEFAULT_INTERVAL),
  );
  const maxSummaryChars = clampInt(
    parseIntEnv(process.env.PROMPT_COMPACTION_MAX_SUMMARY_CHARS, PROMPT_COMPACTION_DEFAULT_MAX_CHARS),
    300,
    4000,
  );
  return {
    enabled: parseBool(process.env.PROMPT_COMPACTION_ENABLED, false),
    threshold,
    interval,
    keepRecent,
    maxSummaryChars,
  };
}

function parseBool(value: string | undefined, fallback: boolean) {
  if (value === undefined || value === '') {
    return fallback;
  }
  return !['0', 'false', 'no', 'off'].includes(value.toLowerCase());
}

function parseIntEnv(value: string | undefined, fallback: number) {
  if (!value) {
    return fallback;
  }
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function clampInt(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function stripNonMaterialSourceTypes(sourceTypes: ChatSourceType[] | undefined) {
  const stripped = (sourceTypes ?? MATERIAL_SOURCE_TYPES).filter((sourceType) =>
    sourceType !== 'chat_memory' && sourceType !== 'web'
  );
  return stripped.length > 0 ? stripped : MATERIAL_SOURCE_TYPES;
}

async function triggerChatMemorySummary({
  supabase,
  sessionId,
  userId,
  courseId,
  pdfIds,
  ragApiUrl,
  internalApiKey,
}: {
  supabase: SupabaseClient;
  sessionId: string;
  userId: string;
  courseId: string | null;
  pdfIds: string[];
  ragApiUrl: string;
  internalApiKey: string;
}): Promise<void> {
  try {
    const config = chatMemoryConfig();
    if (!config.enabled) {
      return;
    }

    const {data: existingSummary} = await supabase
      .from('chat_memory_summaries')
      .select('summary, represented_message_count')
      .eq('user_id', userId)
      .eq('session_id', sessionId)
      .maybeSingle();
    const representedMessageCount = Number(existingSummary?.represented_message_count ?? 0);

    const {count, error: countError} = await supabase
      .from('chat_messages')
      .select('id', {count: 'exact', head: true})
      .eq('user_id', userId)
      .eq('session_id', sessionId);
    if (countError || count === null || count < config.threshold) {
      return;
    }
    if (count - representedMessageCount < config.interval) {
      return;
    }

    const compressLimit = Math.max(0, count - representedMessageCount - config.keepRecent);
    if (compressLimit <= 0) {
      return;
    }
    const {data: messages, error: messagesError} = await supabase
      .from('chat_messages')
      .select('role, content, created_at')
      .eq('user_id', userId)
      .eq('session_id', sessionId)
      .order('created_at', {ascending: true})
      .range(representedMessageCount, representedMessageCount + compressLimit - 1);
    if (messagesError || !messages || messages.length === 0) {
      return;
    }

    const response = await fetch(`${ragApiUrl.replace(/\/$/, '')}/rag/compress`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${internalApiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        messages: messages
          .filter((message) => message.role === 'user' || message.role === 'assistant')
          .map((message) => ({
            role: message.role,
            content: String(message.content ?? '').slice(0, RECENT_MESSAGE_CONTENT_LIMIT),
          })),
        existing_summary: existingSummary?.summary ?? null,
        max_chars: config.maxSummaryChars,
      }),
    });
    if (!response.ok) {
      console.error('RAG compression failed', {status: response.status});
      return;
    }
    const {summary} = await response.json();
    if (typeof summary !== 'string' || summary.trim().length === 0) {
      return;
    }

    const lastCompressed = messages[messages.length - 1];
    const newRepresentedCount = representedMessageCount + messages.length;
    const {data: pendingJob} = await supabase
      .from('rag_index_jobs')
      .select('id')
      .eq('user_id', userId)
      .eq('source_type', 'chat_memory')
      .eq('source_id', sessionId)
      .in('status', ['pending', 'processing'])
      .maybeSingle();

    const {error: upsertError} = await supabase.from('chat_memory_summaries').upsert({
      user_id: userId,
      session_id: sessionId,
      summary: summary.trim(),
      represented_message_count: newRepresentedCount,
      message_range_start: messages[0]?.created_at ?? null,
      message_range_end: lastCompressed?.created_at ?? null,
      source_type: 'chat_memory',
      course_id: courseId,
      pdf_id: pdfIds[0] ?? null,
      embedding_status: 'pending',
      updated_at: new Date().toISOString(),
    }, {onConflict: 'user_id,session_id'});
    if (upsertError || pendingJob?.id) {
      return;
    }

    const {data: job} = await supabase
      .from('rag_index_jobs')
      .insert({
        user_id: userId,
        source_type: 'chat_memory',
        source_id: sessionId,
        job_kind: 'index_source',
        status: 'pending',
      })
      .select('id')
      .single();
    if (job?.id) {
      await supabase
        .from('chat_memory_summaries')
        .update({rag_job_id: job.id, updated_at: new Date().toISOString()})
        .eq('user_id', userId)
        .eq('session_id', sessionId);
    }
  } catch (error) {
    console.error('Chat memory summary trigger failed', error);
  }
}

async function triggerPromptCompaction({
  supabase,
  sessionId,
  userId,
  ragApiUrl,
  internalApiKey,
}: {
  supabase: SupabaseClient;
  sessionId: string;
  userId: string;
  ragApiUrl: string;
  internalApiKey: string;
}): Promise<void> {
  try {
    const config = promptCompactionConfig();
    if (!config.enabled) {
      return;
    }

    const {count, error: countError} = await supabase
      .from('chat_messages')
      .select('id', {count: 'exact', head: true})
      .eq('user_id', userId)
      .eq('session_id', sessionId);
    if (countError || count === null || count < config.threshold) {
      return;
    }

    const {data: session, error: sessionError} = await supabase
      .from('chat_sessions')
      .select('context_summary, context_summary_cursor')
      .eq('id', sessionId)
      .eq('user_id', userId)
      .maybeSingle();
    if (sessionError || !session) {
      return;
    }

    const cursor = Math.max(0, Number(session.context_summary_cursor ?? 0));
    const compactUntil = count - config.keepRecent;
    const newCompressibleCount = compactUntil - cursor;
    if (compactUntil <= cursor) {
      return;
    }
    if (cursor > 0 && newCompressibleCount < config.interval) {
      return;
    }

    const {data: messages, error: messagesError} = await supabase
      .from('chat_messages')
      .select('role, content')
      .eq('user_id', userId)
      .eq('session_id', sessionId)
      .order('created_at', {ascending: true})
      .order('id', {ascending: true})
      .range(cursor, compactUntil - 1);
    if (messagesError || !messages || messages.length === 0) {
      return;
    }

    const response = await fetch(`${ragApiUrl.replace(/\/$/, '')}/rag/compress`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${internalApiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        messages: messages
          .filter((message) => message.role === 'user' || message.role === 'assistant')
          .map((message) => ({
            role: message.role,
            content: String(message.content ?? '').slice(0, RECENT_MESSAGE_CONTENT_LIMIT),
          })),
        existing_summary: typeof session.context_summary === 'string' ? session.context_summary : null,
        max_chars: config.maxSummaryChars,
      }),
    });
    if (!response.ok) {
      console.error('RAG prompt compaction failed', {status: response.status});
      return;
    }

    const {summary} = await response.json();
    if (typeof summary !== 'string' || summary.trim().length === 0) {
      return;
    }

    const {data: updatedSession, error: updateError} = await supabase
      .from('chat_sessions')
      .update({
        context_summary: summary.trim().slice(0, config.maxSummaryChars),
        context_summary_cursor: compactUntil,
        updated_at: new Date().toISOString(),
      })
      .eq('id', sessionId)
      .eq('user_id', userId)
      .eq('context_summary_cursor', cursor)
      .select('id')
      .maybeSingle();
    if (updateError) {
      console.error('Failed to update prompt compaction summary', updateError);
    } else if (!updatedSession) {
      console.info('Skipped prompt compaction update because another compaction advanced the cursor.');
    }
  } catch (error) {
    console.error('Prompt compaction trigger failed', error);
  }
}

async function loadRelatedMemorySourceIds(
  supabase: SupabaseClient,
  userId: string,
  sessionId: string,
  courseId: string | null,
  pdfIds: string[],
) {
  const ids = new Set<string>([sessionId]);
  try {
    if (courseId) {
      const {data} = await supabase
        .from('chat_memory_summaries')
        .select('session_id')
        .eq('user_id', userId)
        .eq('course_id', courseId)
        .eq('embedding_status', 'completed')
        .limit(10);
      for (const row of data ?? []) {
        if (typeof row.session_id === 'string') {
          ids.add(row.session_id);
        }
      }
    }
    if (pdfIds.length > 0) {
      const {data} = await supabase
        .from('chat_memory_summaries')
        .select('session_id')
        .eq('user_id', userId)
        .in('pdf_id', pdfIds)
        .eq('embedding_status', 'completed')
        .limit(10);
      for (const row of data ?? []) {
        if (typeof row.session_id === 'string') {
          ids.add(row.session_id);
        }
      }
    }
  } catch (error) {
    console.error('Failed to load related chat memory ids', error);
  }
  return Array.from(ids);
}

export async function POST(request: Request) {
  const supabase = await createClient();
  const {data: {user}} = await supabase.auth.getUser();
  if (!user) {
    return errorResponse('Unauthorized', 401);
  }

  let rawBody: Record<string, unknown>;
  try {
    rawBody = await request.json();
  } catch {
    return errorResponse('Invalid JSON body.', 400);
  }
  // Explicitly strip fields that must never be forwarded to the Python RAG service.
  // Browser-controlled tool execution is not allowed; these are server-side-only concerns.
  const safeRawBody = {...rawBody};
  for (const field of FORBIDDEN_RAG_TOOL_FIELDS) {
    delete safeRawBody[field];
  }
  const body: Partial<ChatRequest> = safeRawBody as Partial<ChatRequest>;

  const validationError = validateBody(body);
  if (validationError) {
    return errorResponse(validationError, 400);
  }
  if (body.use_rag !== true) {
    return errorResponse('Non-RAG chat is not implemented.', 400);
  }

  const ragApiUrl = process.env.RAG_API_URL;
  const internalApiKey = process.env.RAG_INTERNAL_API_KEY;
  if (!ragApiUrl || !internalApiKey) {
    console.error('RAG chat env vars missing.');
    return errorResponse('RAG chat is not configured.', 500);
  }

  const topK = body.top_k ?? 8;
  const sourceTypes = stripNonMaterialSourceTypes(body.source_types);
  const pdfIds = body.pdf_ids?.filter(Boolean) ?? [];
  const webMode = body.enableWebSearch === true ? 'on' : 'off';
  const useIntentClassifier = parseBool(process.env.INTENT_CLASSIFIER_ENABLED, false);
  const useRetrievalPlanner = parseBool(process.env.RETRIEVAL_PLANNER_ENABLED, false);
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 60000);

  try {
    const session = await getOrCreateSession(supabase, user.id, body);
    const sessionId = session.id;
    const promptContext = await loadSessionPromptContext(supabase, sessionId, user.id);
    const sessionMode = session.mode ?? promptContext.mode;
    const activeLearningState = Object.keys(session.activeLearningState).length > 0
      ? session.activeLearningState
      : promptContext.activeLearningState;
    const requestActiveLearningState = sessionMode === 'normal'
      ? activeLearningState
      : {
        ...activeLearningState,
        ...(body.topic ? {topic: body.topic.trim(), target_concept: body.topic.trim()} : {}),
        ...(body.difficulty ? {difficulty: body.difficulty} : {}),
        ...(body.language ? {language: body.language} : {}),
        ...(body.learner_name ? {learner_name: body.learner_name.trim().slice(0, 80)} : {}),
      };
    const courseId = body.course_id ?? promptContext.courseId;
    const compactionConfig = promptCompactionConfig();
    const recentMessageLimit = compactionConfig.enabled && promptContext.contextSummary
      ? compactionConfig.keepRecent
      : RECENT_MESSAGE_LIMIT;
    const recentMessages = await loadRecentMessages(supabase, sessionId, user.id, recentMessageLimit);
    const memorySourceIds = await loadRelatedMemorySourceIds(
      supabase,
      user.id,
      sessionId,
      courseId,
      pdfIds,
    );
    await supabase.from('chat_messages').insert({
      session_id: sessionId,
      user_id: user.id,
      role: 'user',
      content: body.message!.trim(),
      pdf_ids: pdfIds,
    });

    const response = await fetch(`${ragApiUrl.replace(/\/$/, '')}/rag/answer`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${internalApiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query: body.message!.trim(),
        user_id: user.id,
        source_types: sourceTypes,
        top_k: topK,
        pdf_ids: pdfIds.length > 0 ? pdfIds : undefined,
        recent_messages: recentMessages,
        session_id: sessionId,
        memory_source_ids: memorySourceIds,
        memory_mode: 'auto',
        graph_mode: parseBool(process.env.GRAPH_RETRIEVAL_ENABLED, false) ? 'auto' : 'off',
        context_summary: promptContext.contextSummary ?? undefined,
        web_mode: webMode,
        use_intent_classifier: useIntentClassifier,
        use_retrieval_planner: useRetrievalPlanner,
        chat_mode: sessionMode,
        active_learning_state: requestActiveLearningState,
        chat_language: body.language,
      }),
      signal: controller.signal,
    });

    if (!response.ok) {
      console.error('RAG service error', {status: response.status, body: await response.text()});
      return errorResponse('RAG service failed to answer.', 500);
    }

    const ragResponse = await response.json();
    if (
      ragResponse.updated_active_learning_state &&
      typeof ragResponse.updated_active_learning_state === 'object' &&
      !Array.isArray(ragResponse.updated_active_learning_state)
    ) {
      const {error: stateError} = await supabase
        .from('chat_sessions')
        .update({active_learning_state: ragResponse.updated_active_learning_state})
        .eq('id', sessionId)
        .eq('user_id', user.id);
      if (stateError) {
        console.error('Failed to persist active learning state', stateError);
      }
    }
    await supabase.from('chat_messages').insert({
      session_id: sessionId,
      user_id: user.id,
      role: 'assistant',
      content: ragResponse.answer,
      sources: ragResponse.sources ?? [],
      pdf_ids: pdfIds,
    });
    await supabase
      .from('chat_sessions')
      .update({updated_at: new Date().toISOString()})
      .eq('id', sessionId)
      .eq('user_id', user.id);
    void triggerChatMemorySummary({
      supabase,
      sessionId,
      userId: user.id,
      courseId,
      pdfIds,
      ragApiUrl,
      internalApiKey,
    });
    void triggerPromptCompaction({
      supabase,
      sessionId,
      userId: user.id,
      ragApiUrl,
      internalApiKey,
    });

    const chatResponse: ChatResponse = {
      session_id: sessionId,
      answer: ragResponse.answer,
      sources: ragResponse.sources ?? [],
      retrieval: {mode: 'hybrid', top_k: topK},
      active_learning_state: normalizeActiveLearningState(ragResponse.updated_active_learning_state ?? activeLearningState),
    };
    return NextResponse.json(chatResponse);
  } catch (error) {
    if (error instanceof SessionNotFoundError) {
      return errorResponse('Session not found.', 404);
    }
    if (error instanceof MissingActiveLearningMigrationError) {
      return errorResponse(
        'Active Learning database migration is missing. Apply supabase/migrations/20260514000001_active_learning_chat_sessions.sql and restart the app.',
        500,
      );
    }
    console.error('RAG service request failed', error);
    return errorResponse('RAG service failed to answer.', 500);
  } finally {
    clearTimeout(timeout);
  }
}

export async function GET(request: Request) {
  const supabase = await createClient();
  const {data: {user}} = await supabase.auth.getUser();
  if (!user) {
    return errorResponse('Unauthorized', 401);
  }

  const {searchParams} = new URL(request.url);
  const courseId = searchParams.get('courseId');
  const requestedMode = searchParams.get('mode');
  if (requestedMode && !CHAT_MODES.includes(requestedMode as ChatMode)) {
    return errorResponse('mode must be normal, guided_learning, or feynman.', 400);
  }
  const modeFilter = requestedMode as ChatMode | null;
  let sessionQuery = supabase
    .from('chat_sessions')
    .select('id, title, course_id, updated_at, mode, active_learning_state')
    .eq('user_id', user.id)
    .order('updated_at', {ascending: false})
    .limit(20);
  if (courseId) {
    sessionQuery = sessionQuery.eq('course_id', courseId);
  }
  if (modeFilter) {
    sessionQuery = sessionQuery.eq('mode', modeFilter);
  }

  const {data: sessions, error: sessionsError} = await sessionQuery;
  let normalizedSessions = (sessions ?? []).map((session) => ({
    ...session,
    mode: normalizeChatMode(session.mode),
    active_learning_state: normalizeActiveLearningState(session.active_learning_state),
  }));
  if (sessionsError && isMissingActiveLearningColumnError(sessionsError)) {
    let fallbackSessionQuery = supabase
      .from('chat_sessions')
      .select('id, title, course_id, updated_at')
      .eq('user_id', user.id)
      .order('updated_at', {ascending: false})
      .limit(20);
    if (courseId) {
      fallbackSessionQuery = fallbackSessionQuery.eq('course_id', courseId);
    }
    if (modeFilter && modeFilter !== 'normal') {
      return NextResponse.json({sessions: []});
    }
    const fallback = await fallbackSessionQuery;
    if (fallback.error) {
      console.error('Failed to load chat sessions', fallback.error);
      return errorResponse('Failed to load chat sessions.', 500);
    }
    normalizedSessions = (fallback.data ?? []).map((session) => ({
      ...session,
      mode: 'normal' as ChatMode,
      active_learning_state: {},
    }));
  } else if (sessionsError) {
    console.error('Failed to load chat sessions', sessionsError);
    return errorResponse('Failed to load chat sessions.', 500);
  }

  const sessionIds = normalizedSessions.map((session) => session.id);
  if (sessionIds.length === 0) {
    return NextResponse.json({sessions: []});
  }

  const {data: messages, error: messagesError} = await supabase
    .from('chat_messages')
    .select('id, session_id, role, content, sources, pdf_ids, created_at')
    .eq('user_id', user.id)
    .in('session_id', sessionIds)
    .order('created_at', {ascending: true});
  if (messagesError) {
    console.error('Failed to load chat messages', messagesError);
    return errorResponse('Failed to load chat messages.', 500);
  }

  return NextResponse.json({
    sessions: normalizedSessions.map((session) => ({
      ...session,
      mode: normalizeChatMode(session.mode),
      active_learning_state: normalizeActiveLearningState(session.active_learning_state),
      messages: (messages ?? []).filter((message) => message.session_id === session.id),
    })),
  });
}
