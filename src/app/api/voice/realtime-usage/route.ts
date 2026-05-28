import {NextResponse} from 'next/server';
import {createClient} from '@/lib/supabase/server';
import {createServiceClient} from '@/lib/supabase/service';
import {recordUsage, type UsageClient} from '@/lib/limits/guard';

export const runtime = 'nodejs';

const MAX_REPORTED_DURATION_SECONDS = 10 * 60;

type RealtimeUsageBody = {
  sessionId: string;
  durationSeconds: number;
};

type RealtimeUsageValidation =
  | {ok: true; value: RealtimeUsageBody}
  | {ok: false; error: string; status: number};

type RealtimeUsageSessionRow = {
  session_id: string;
  expires_at: string | null;
  created_at: string;
  usage_reported_at: string | null;
};

export function validateRealtimeUsageBody(value: unknown): RealtimeUsageValidation {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return {ok: false, error: 'Invalid JSON body.', status: 400};
  }
  const body = value as Record<string, unknown>;
  if (typeof body.sessionId !== 'string' || !body.sessionId.trim()) {
    return {ok: false, error: 'sessionId is required.', status: 400};
  }
  const durationSeconds = Number(body.durationSeconds ?? body.duration_seconds);
  if (!Number.isFinite(durationSeconds) || durationSeconds <= 0) {
    return {ok: false, error: 'durationSeconds must be positive.', status: 400};
  }
  return {
    ok: true,
    value: {
      sessionId: body.sessionId.trim(),
      durationSeconds: Math.min(MAX_REPORTED_DURATION_SECONDS, Math.ceil(durationSeconds)),
    },
  };
}

export function clampRealtimeUsageSeconds(params: {
  reportedSeconds: number;
  createdAt: Date;
  expiresAt: Date | null;
  now?: Date;
}) {
  const now = params.now ?? new Date();
  const expiry = params.expiresAt && params.expiresAt.getTime() > 0 ? params.expiresAt : now;
  const allowedMs = Math.max(0, Math.min(now.getTime(), expiry.getTime()) - params.createdAt.getTime());
  const allowedSeconds = Math.ceil(allowedMs / 1000);
  return Math.max(0, Math.min(Math.ceil(params.reportedSeconds), allowedSeconds, MAX_REPORTED_DURATION_SECONDS));
}

function errorResponse(message: string, status: number) {
  return NextResponse.json({error: message}, {status});
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

  const validation = validateRealtimeUsageBody(rawBody);
  if (!validation.ok) {
    return errorResponse(validation.error, validation.status);
  }

  const {data: realtimeSession, error: sessionError} = await supabase
    .from('voice_realtime_sessions')
    .select('session_id, expires_at, created_at, usage_reported_at')
    .eq('session_id', validation.value.sessionId)
    .eq('user_id', user.id)
    .maybeSingle();
  if (sessionError || !realtimeSession) {
    return errorResponse('Realtime session not found.', 404);
  }

  const session = realtimeSession as RealtimeUsageSessionRow;
  if (session.usage_reported_at) {
    return NextResponse.json({ok: true, recorded: false});
  }

  const durationSeconds = clampRealtimeUsageSeconds({
    reportedSeconds: validation.value.durationSeconds,
    createdAt: new Date(session.created_at),
    expiresAt: session.expires_at ? new Date(session.expires_at) : null,
  });
  const reportedAt = new Date().toISOString();
  const {data: updatedRows, error: updateError} = await supabase
    .from('voice_realtime_sessions')
    .update({
      duration_seconds: durationSeconds,
      usage_reported_at: reportedAt,
      updated_at: reportedAt,
    })
    .eq('session_id', validation.value.sessionId)
    .eq('user_id', user.id)
    .is('usage_reported_at', null)
    .select('session_id');
  if (updateError) {
    console.error('Failed to mark realtime usage as reported', updateError);
    return errorResponse('Failed to record realtime voice usage.', 500);
  }
  if (!Array.isArray(updatedRows) || updatedRows.length === 0) {
    return NextResponse.json({ok: true, recorded: false});
  }
  if (durationSeconds > 0) {
    try {
      await recordUsage({
        supabase: createServiceClient() as unknown as UsageClient,
        userId: user.id,
        feature: 'realtime_voice',
        quantity: durationSeconds / 60,
        model: 'gpt-realtime-2',
        sessionId: validation.value.sessionId,
      });
    } catch (error) {
      console.error('Failed to write realtime usage event', error);
      return errorResponse('Failed to record realtime voice usage.', 500);
    }
  }

  return NextResponse.json({ok: true, recorded: durationSeconds > 0, durationSeconds});
}
