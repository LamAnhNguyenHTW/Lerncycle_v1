import {NextResponse} from 'next/server';
import {createClient} from '@/lib/supabase/server';
import {createServiceClient} from '@/lib/supabase/service';
import {getUsageLimitsConfig} from '@/lib/limits/config';
import {getVoiceServerConfig} from '@/lib/voice/config';
import {OpenAIVoiceProvider} from '@/lib/voice/openai';
import {UsageQuotaError, assertUsageQuota, type UsageClient} from '@/lib/limits/guard';
import {
  RealtimeContextError,
  buildRealtimeContext,
  buildRealtimeInstructions,
  persistRealtimeSessionSources,
} from '@/lib/voice/realtimeContext';
import type {RealtimeToolDefinition} from '@/lib/voice/provider';

export const runtime = 'nodejs';

type RealtimeTokenBody = {
  sessionId?: string;
  courseId?: string;
  pdfIds: string[];
  pdfNames: string[];
  mode: 'feynman';
};

type RealtimeTokenBodyValidation =
  | {ok: true; value: RealtimeTokenBody}
  | {ok: false; error: string};

type UsageQuantityRow = {
  quantity: number | string | null;
};

export function validateRealtimeTokenBody(value: unknown): RealtimeTokenBodyValidation {
  if (value === undefined || value === null) {
    return {ok: true, value: {mode: 'feynman', pdfIds: [], pdfNames: []}};
  }
  if (typeof value !== 'object' || Array.isArray(value)) {
    return {ok: false, error: 'Invalid JSON body.'};
  }
  const body = value as Record<string, unknown>;
  if (body.mode !== undefined && body.mode !== 'feynman') {
    return {ok: false, error: 'Realtime voice is only available for Feynman mode.'};
  }
  return {
    ok: true,
    value: {
      mode: 'feynman',
      ...(typeof body.sessionId === 'string' && body.sessionId.trim() ? {sessionId: body.sessionId.trim()} : {}),
      ...(typeof body.course_id === 'string' && body.course_id.trim() ? {courseId: body.course_id.trim()} : {}),
      pdfIds: parseStringArray(body.pdf_ids),
      pdfNames: parseStringArray(body.pdf_names).slice(0, 8),
    },
  };
}

export function realtimeInstructions(pdfNames: string[] = [], selectedContext = '') {
  return buildRealtimeInstructions({
    mode: 'feynman',
    pdfNames,
    documentPrimer: selectedContext,
  });
}

export function realtimeRagToolDefinition(): RealtimeToolDefinition {
  return {
    type: 'function',
    name: 'search_learncycle_context',
    description: 'Searches the authenticated user LearnCycle study materials for relevant course context.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        query: {
          type: 'string',
          description: 'The learner question or concept to search for.',
        },
        top_k: {
          type: 'integer',
          minimum: 1,
          maximum: 8,
          description: 'Maximum number of context snippets to return.',
        },
      },
      required: ['query'],
    },
  };
}

function parseStringArray(value: unknown) {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === 'string' && item.trim().length > 0).map((item) => item.trim())
    : [];
}

function errorResponse(message: string, status: number) {
  return NextResponse.json({error: message}, {status});
}

function startOfCurrentUtcMonth(now = new Date()) {
  return new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), 1, 0, 0, 0, 0));
}

function normalizeQuantity(value: unknown) {
  const parsed = Number(value ?? 0);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 0;
}

async function sumRealtimeUsageMinutes(params: {
  supabase: UsageClient;
  userId?: string;
}) {
  let query = params.supabase
    .from('usage_events')
    .select('quantity')
    .eq('feature', 'realtime_voice');
  if (params.userId) {
    query = query.eq('user_id', params.userId);
  }
  const {data, error} = await query.gte('created_at', startOfCurrentUtcMonth().toISOString());
  if (error) {
    throw new UsageQuotaError('realtime_voice', 'usage_read_error');
  }
  return (data as UsageQuantityRow[] | null ?? []).reduce(
    (sum, row) => sum + normalizeQuantity(row.quantity),
    0,
  );
}

async function remainingRealtimeSeconds(params: {
  supabase: UsageClient;
  userId: string;
}) {
  const limit = getUsageLimitsConfig().features.realtime_voice;
  const [userUsedMinutes, globalUsedMinutes] = await Promise.all([
    sumRealtimeUsageMinutes({supabase: params.supabase, userId: params.userId}),
    sumRealtimeUsageMinutes({supabase: params.supabase}),
  ]);
  const userRemainingMinutes = Math.max(0, limit.perUser.quantity - userUsedMinutes);
  const globalRemainingMinutes = Math.max(0, limit.globalMonthly.quantity - globalUsedMinutes);
  return Math.floor(Math.min(userRemainingMinutes, globalRemainingMinutes) * 60);
}

async function createRealtimeChatSession(params: {
  supabase: Awaited<ReturnType<typeof createClient>>;
  userId: string;
  body: RealtimeTokenBody;
}) {
  const {data, error} = await params.supabase
    .from('chat_sessions')
    .insert({
      user_id: params.userId,
      course_id: params.body.courseId ?? null,
      title: 'Live voice',
      mode: 'feynman',
      active_learning_state: {mode: 'feynman'},
    })
    .select('id')
    .single();
  if (error || !data?.id) {
    console.error('Failed to create realtime chat session', error);
    throw new RealtimeContextError('Failed to create realtime session.', 500);
  }
  return String(data.id);
}

export async function POST(request: Request) {
  const config = getVoiceServerConfig();
  if (!config.enabled) {
    return errorResponse('Voice mode is disabled.', 404);
  }

  const supabase = await createClient();
  const {data: {user}} = await supabase.auth.getUser();
  if (!user) {
    return errorResponse('Unauthorized', 401);
  }

  let rawBody: unknown = null;
  try {
    const text = await request.text();
    rawBody = text.trim() ? JSON.parse(text) : null;
  } catch {
    return errorResponse('Invalid JSON body.', 400);
  }

  const validation = validateRealtimeTokenBody(rawBody);
  if (!validation.ok) {
    return errorResponse(validation.error, 400);
  }

  try {
    const sessionId = validation.value.sessionId ?? await createRealtimeChatSession({
      supabase,
      userId: user.id,
      body: validation.value,
    });
    const realtimeContext = await buildRealtimeContext({
      supabase,
      userId: user.id,
      config,
      body: {
        sessionId,
        sourceIds: validation.value.pdfIds,
        mode: validation.value.mode,
      },
    });
    await persistRealtimeSessionSources({
      supabase,
      userId: user.id,
      sessionId,
      allowedSourceIds: realtimeContext.allowedSourceIds,
      mode: validation.value.mode,
    });
    const serviceSupabase = createServiceClient() as unknown as UsageClient;
    await assertUsageQuota({
      supabase: serviceSupabase,
      userId: user.id,
      feature: 'realtime_voice',
      requested: 1 / 60,
    });
    const expiresAfterSeconds = await remainingRealtimeSeconds({
      supabase: serviceSupabase,
      userId: user.id,
    });
    if (expiresAfterSeconds <= 0) {
      throw new UsageQuotaError('realtime_voice', 'per_user_limit');
    }
    const provider = new OpenAIVoiceProvider({
      apiKey: config.openaiApiKey,
      sttModel: config.sttModel,
      ttsModel: config.ttsModel,
      realtimeModel: config.realtimeModel,
    });
    const session = await provider.createRealtimeSession({
      userId: user.id,
      model: config.realtimeModel,
      voice: 'alloy',
      instructions: realtimeContext.realtimeInstructions,
      tools: [realtimeRagToolDefinition()],
      expiresAfterSeconds,
    });
    await supabase
      .from('voice_realtime_sessions')
      .update({expires_at: session.expiresAt, updated_at: new Date().toISOString()})
      .eq('session_id', sessionId)
      .eq('user_id', user.id);

    return NextResponse.json({
      clientSecret: session.clientSecret,
      expiresAt: session.expiresAt,
      model: session.model,
      tools: session.tools.map((tool) => tool.name),
      sessionId,
      allowedSourceIds: realtimeContext.allowedSourceIds,
    });
  } catch (error) {
    if (error instanceof UsageQuotaError) {
      return errorResponse(error.message, error.status);
    }
    if (error instanceof RealtimeContextError) {
      return errorResponse(error.message, error.status);
    }
    console.error('Realtime voice token creation failed', error);
    return errorResponse('Realtime voice is not configured.', 500);
  }
}
