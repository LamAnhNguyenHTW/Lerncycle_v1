import {NextResponse} from 'next/server';
import {createClient} from '@/lib/supabase/server';

export const runtime = 'nodejs';

const MAX_TRANSCRIPT_TURNS = 80;
const MAX_TURN_CHARS = 1000;

type RealtimeSummaryTurn = {
  role: 'user' | 'assistant';
  text: string;
};

type RealtimeSummaryBody = {
  sessionId: string;
  turns: RealtimeSummaryTurn[];
};

type RealtimeSummaryValidation =
  | {ok: true; value: RealtimeSummaryBody}
  | {ok: false; error: string; status: number};

type RealtimeChatMessageRow = {
  session_id: string;
  user_id: string;
  role: 'user' | 'assistant';
  content: string;
  input_metadata: {
    input_type: 'voice';
    realtime_transcript: true;
    turn_index: number;
  };
};

export function validateRealtimeSummaryBody(value: unknown): RealtimeSummaryValidation {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return {ok: false, error: 'Invalid JSON body.', status: 400};
  }
  const body = value as Record<string, unknown>;
  if (typeof body.sessionId !== 'string' || !body.sessionId.trim()) {
    return {ok: false, error: 'sessionId is required.', status: 400};
  }
  if (!Array.isArray(body.turns)) {
    return {ok: false, error: 'turns must be an array.', status: 400};
  }

  const turns = body.turns
    .slice(0, MAX_TRANSCRIPT_TURNS)
    .map((turn): RealtimeSummaryTurn | null => {
      if (!turn || typeof turn !== 'object' || Array.isArray(turn)) {
        return null;
      }
      const record = turn as Record<string, unknown>;
      if (record.role !== 'user' && record.role !== 'assistant') {
        return null;
      }
      if (typeof record.text !== 'string' || !record.text.trim()) {
        return null;
      }
      return {
        role: record.role,
        text: normalizeTranscriptText(record.text).slice(0, MAX_TURN_CHARS),
      };
    })
    .filter((turn): turn is RealtimeSummaryTurn => turn !== null);

  if (turns.length === 0) {
    return {ok: false, error: 'No transcript text to persist.', status: 400};
  }

  return {
    ok: true,
    value: {
      sessionId: body.sessionId.trim(),
      turns,
    },
  };
}

export function buildRealtimeChatMessageRows(params: {
  sessionId: string;
  userId: string;
  turns: RealtimeSummaryTurn[];
}): RealtimeChatMessageRow[] {
  return params.turns.map((turn, index) => ({
    session_id: params.sessionId,
    user_id: params.userId,
    role: turn.role,
    content: turn.text,
    input_metadata: {
      input_type: 'voice',
      realtime_transcript: true,
      turn_index: index,
    },
  }));
}

function normalizeTranscriptText(value: string) {
  return value.replace(/\s+/g, ' ').trim();
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

  const validation = validateRealtimeSummaryBody(rawBody);
  if (!validation.ok) {
    return errorResponse(validation.error, validation.status);
  }

  const {data: session, error: sessionError} = await supabase
    .from('chat_sessions')
    .select('id, mode')
    .eq('id', validation.value.sessionId)
    .eq('user_id', user.id)
    .maybeSingle();
  if (sessionError || !session?.id) {
    return errorResponse('Session not found.', 404);
  }
  if (session.mode !== 'feynman') {
    return errorResponse('Realtime voice summaries are only available for Feynman sessions.', 400);
  }
  const {data: realtimeSession, error: realtimeSessionError} = await supabase
    .from('voice_realtime_sessions')
    .select('session_id')
    .eq('session_id', validation.value.sessionId)
    .eq('user_id', user.id)
    .maybeSingle();
  if (realtimeSessionError || !realtimeSession) {
    return errorResponse('Realtime session source scope not found.', 404);
  }

  const rows = buildRealtimeChatMessageRows({
    sessionId: validation.value.sessionId,
    userId: user.id,
    turns: validation.value.turns,
  });
  const {error: insertError} = await supabase.from('chat_messages').insert(rows);
  if (insertError) {
    console.error('Failed to persist realtime voice transcript', insertError);
    return errorResponse('Failed to persist realtime voice transcript.', 500);
  }

  await supabase
    .from('chat_sessions')
    .update({updated_at: new Date().toISOString()})
    .eq('id', validation.value.sessionId)
    .eq('user_id', user.id);

  return NextResponse.json({ok: true});
}
