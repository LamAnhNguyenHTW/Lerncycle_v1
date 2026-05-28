import {NextResponse} from 'next/server';
import {createClient} from '@/lib/supabase/server';
import {createServiceClient} from '@/lib/supabase/service';
import {assertUsageQuota, recordUsage, UsageQuotaError, type UsageClient} from '@/lib/limits/guard';

export const runtime = 'nodejs';

const RECENT_MESSAGE_LIMIT = 6;
const RECENT_MESSAGE_CONTENT_LIMIT = 1200;

type RealtimeToolBody = {
  sessionId: string;
  query: string;
  topK: number;
};

type RealtimeToolBodyValidation =
  | {ok: true; value: RealtimeToolBody}
  | {ok: false; error: string};

export function validateRealtimeToolBody(value: unknown): RealtimeToolBodyValidation {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return {ok: false, error: 'Invalid JSON body.'};
  }
  const body = value as Record<string, unknown>;
  const rawQuery = firstNonEmptyString(body.query, body.question, body.topic);
  const requestedTopK = typeof body.top_k === 'number' ? body.top_k : body.topK;
  const topK = Number.isInteger(requestedTopK) ? Number(requestedTopK) : 4;
  return {
    ok: true,
    value: {
      sessionId: typeof body.sessionId === 'string' ? body.sessionId.trim() : '',
      query: (rawQuery ?? 'Ueberblick Ist-Prozess Start Ablauf Einsatzplanung zentrale Schritte').slice(0, 500),
      topK: Math.min(5, Math.max(1, topK)),
    },
  };
}

function firstNonEmptyString(...values: unknown[]) {
  for (const value of values) {
    if (typeof value === 'string' && value.trim()) {
      return value.trim();
    }
  }
  return null;
}

function errorResponse(message: string, status: number) {
  return NextResponse.json({error: message}, {status});
}

function isActiveRealtimeSession(value: {expires_at?: string | null; usage_reported_at?: string | null}) {
  if (value.usage_reported_at) {
    return false;
  }
  if (!value.expires_at) {
    return true;
  }
  return new Date(value.expires_at).getTime() > Date.now();
}

export function compactRealtimeSources(value: unknown) {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.slice(0, 5).map((source) => {
    const row = source && typeof source === 'object' ? source as Record<string, unknown> : {};
    const text = firstNonEmptyString(row.snippet, row.content, row.text, row.excerpt) ?? '';
    return {
      source_id: firstNonEmptyString(row.source_id, row.pdf_id, row.id) ?? null,
      title: firstNonEmptyString(row.title, row.filename, row.pdf_name) ?? null,
      page: typeof row.page === 'number'
        ? row.page
        : typeof row.page_index === 'number'
          ? row.page_index + 1
          : null,
      snippet: text.replace(/\s+/g, ' ').trim().slice(0, 500),
    };
  }).filter((source) => source.snippet || source.title || source.source_id);
}

export function noRealtimeResultsResponse() {
  return {
    answer: 'Ich habe dazu in deinen ausgewaehlten Dokumenten nichts Passendes gefunden.',
    sources: [],
    no_results: true,
  };
}

export async function POST(request: Request) {
  const supabase = await createClient();
  const {data: {user}} = await supabase.auth.getUser();
  if (!user) {
    return errorResponse('Unauthorized', 401);
  }

  let rawBody: unknown;
  try {
    rawBody = await request.json();
  } catch {
    return errorResponse('Invalid JSON body.', 400);
  }

  const validation = validateRealtimeToolBody(rawBody);
  if (!validation.ok) {
    return errorResponse(validation.error, 400);
  }
  if (!validation.value.sessionId) {
    return errorResponse('sessionId is required.', 400);
  }

  const {data: session, error: sessionError} = await supabase
    .from('chat_sessions')
    .select('id, mode, active_learning_state')
    .eq('id', validation.value.sessionId)
    .eq('user_id', user.id)
    .maybeSingle();
  if (sessionError || !session?.id) {
    return errorResponse('Session not found.', 404);
  }

  const {data: realtimeSession, error: realtimeSessionError} = await supabase
    .from('voice_realtime_sessions')
    .select('allowed_source_ids, expires_at, usage_reported_at')
    .eq('session_id', validation.value.sessionId)
    .eq('user_id', user.id)
    .maybeSingle();
  if (realtimeSessionError || !realtimeSession) {
    return errorResponse('Realtime session source scope not found.', 404);
  }
  if (!isActiveRealtimeSession(realtimeSession)) {
    return errorResponse('Realtime voice session is no longer active.', 409);
  }
  const usageClient = createServiceClient() as unknown as UsageClient;
  try {
    await assertUsageQuota({
      supabase: usageClient,
      userId: user.id,
      feature: 'chat',
      requested: 1,
    });
  } catch (error) {
    if (error instanceof UsageQuotaError) {
      return errorResponse(error.message, error.status);
    }
    console.error('Failed to check realtime tool usage quota', error);
    return errorResponse('Realtime RAG tool failed.', 500);
  }
  const allowedSourceIds = Array.isArray(realtimeSession.allowed_source_ids)
    ? realtimeSession.allowed_source_ids.filter((sourceId): sourceId is string => typeof sourceId === 'string')
    : [];

  const {data: messages} = await supabase
    .from('chat_messages')
    .select('role, content')
    .eq('session_id', validation.value.sessionId)
    .eq('user_id', user.id)
    .order('created_at', {ascending: false})
    .limit(RECENT_MESSAGE_LIMIT);

  const ragApiUrl = process.env.RAG_API_URL;
  const internalApiKey = process.env.RAG_INTERNAL_API_KEY;
  if (!ragApiUrl || !internalApiKey) {
    return errorResponse('RAG chat is not configured.', 500);
  }

  const startedAt = Date.now();
  const response = await fetch(`${ragApiUrl.replace(/\/$/, '')}/rag/answer`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${internalApiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      query: validation.value.query,
      user_id: user.id,
      source_types: ['pdf', 'note', 'annotation_comment'],
      top_k: validation.value.topK,
      pdf_ids: allowedSourceIds,
      recent_messages: (messages ?? [])
        .reverse()
        .filter((message) => message.role === 'user' || message.role === 'assistant')
        .map((message) => ({
          role: message.role,
          content: String(message.content ?? '').slice(0, RECENT_MESSAGE_CONTENT_LIMIT),
        })),
      session_id: validation.value.sessionId,
      chat_mode: session.mode ?? 'feynman',
      active_learning_state: session.active_learning_state ?? {mode: 'feynman'},
    }),
  });

  if (!response.ok) {
    return errorResponse('Realtime RAG tool failed.', 502);
  }

  const result = await response.json();
  const sources = compactRealtimeSources(result.sources);
  try {
    await recordUsage({
      supabase: usageClient,
      userId: user.id,
      feature: 'chat',
      quantity: 1,
      model: 'realtime-rag-tool',
      sessionId: validation.value.sessionId,
    });
  } catch (error) {
    console.error('Failed to record realtime tool usage', error);
  }
  await recordRealtimeToolEvent({
    supabase,
    userId: user.id,
    sessionId: validation.value.sessionId,
    query: validation.value.query,
    resultCount: sources.length,
    latencyMs: Date.now() - startedAt,
  });
  if (sources.length === 0 && !String(result.answer ?? '').trim()) {
    return NextResponse.json(noRealtimeResultsResponse());
  }
  return NextResponse.json({
    answer: String(result.answer ?? ''),
    sources,
    no_results: sources.length === 0,
  });
}

async function recordRealtimeToolEvent(params: {
  supabase: Awaited<ReturnType<typeof createClient>>;
  userId: string;
  sessionId: string;
  query: string;
  resultCount: number;
  latencyMs: number;
}) {
  const {error} = await params.supabase
    .from('realtime_tool_events')
    .insert({
      user_id: params.userId,
      session_id: params.sessionId,
      query: params.query,
      result_count: params.resultCount,
      latency_ms: Math.max(0, Math.round(params.latencyMs)),
    });
  if (error) {
    console.error('Failed to record realtime tool event', error);
  }
}
