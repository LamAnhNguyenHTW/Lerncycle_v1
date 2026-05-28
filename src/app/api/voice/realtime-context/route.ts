import {NextResponse} from 'next/server';
import {createClient} from '@/lib/supabase/server';
import {getVoiceServerConfig} from '@/lib/voice/config';
import {
  RealtimeContextError,
  buildRealtimeContext,
  persistRealtimeSessionSources,
  validateRealtimeContextBody,
} from '@/lib/voice/realtimeContext';

export const runtime = 'nodejs';

function errorResponse(message: string, status: number) {
  return NextResponse.json({error: message}, {status});
}

export async function POST(request: Request) {
  let rawBody: unknown;
  try {
    rawBody = await request.json();
  } catch {
    return errorResponse('Invalid JSON body.', 400);
  }

  const validation = validateRealtimeContextBody(rawBody);
  if (!validation.ok) {
    return errorResponse(validation.error, 400);
  }

  const supabase = await createClient();
  const {data: {user}} = await supabase.auth.getUser();
  const config = getVoiceServerConfig();

  try {
    const context = await buildRealtimeContext({
      supabase,
      userId: user?.id,
      config,
      body: validation.value,
    });
    if (!user?.id) {
      return errorResponse('Unauthorized', 401);
    }
    await persistRealtimeSessionSources({
      supabase,
      userId: user.id,
      sessionId: validation.value.sessionId,
      allowedSourceIds: context.allowedSourceIds,
      mode: validation.value.mode,
    });
    return NextResponse.json(context);
  } catch (error) {
    if (error instanceof RealtimeContextError) {
      return errorResponse(error.message, error.status);
    }
    console.error('Realtime context build failed', error);
    return errorResponse('Realtime context is not configured.', 500);
  }
}
