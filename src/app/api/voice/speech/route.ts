import {NextResponse} from 'next/server';
import {createClient} from '@/lib/supabase/server';
import {createServiceClient} from '@/lib/supabase/service';
import {getVoiceServerConfig} from '@/lib/voice/config';
import {OpenAIVoiceProvider} from '@/lib/voice/openai';
import {VoiceQuotaError, assertVoiceQuota, recordVoiceUsage, type VoiceUsageClient} from '@/lib/voice/quota';
import type {ChatMode} from '@/types/chat';

export const runtime = 'nodejs';

type SpeechBody =
  | {messageId: string; voice?: string}
  | {text: string; mode?: ChatMode; sessionId?: string; voice?: string};

type SpeechBodyValidation =
  | {ok: true; value: SpeechBody}
  | {ok: false; error: string};

const CHAT_MODES: ChatMode[] = ['normal', 'guided_learning', 'feynman'];
const DEFAULT_VOICE = 'alloy';
const SPOKEN_SUMMARY_PREFIX = 'Hier eine kurze gesprochene Zusammenfassung.';

export function validateSpeechBody(value: unknown): SpeechBodyValidation {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return {ok: false, error: 'Invalid JSON body.'};
  }
  const body = value as Record<string, unknown>;
  if (typeof body.messageId === 'string' && body.messageId.trim()) {
    return {
      ok: true,
      value: {
        messageId: body.messageId.trim(),
        ...(typeof body.voice === 'string' ? {voice: body.voice.trim()} : {}),
      },
    };
  }
  if (typeof body.text === 'string' && body.text.trim()) {
    const mode = CHAT_MODES.includes(body.mode as ChatMode) ? (body.mode as ChatMode) : 'normal';
    return {
      ok: true,
      value: {
        text: body.text.trim(),
        mode,
        ...(typeof body.sessionId === 'string' && body.sessionId.trim() ? {sessionId: body.sessionId.trim()} : {}),
        ...(typeof body.voice === 'string' ? {voice: body.voice.trim()} : {}),
      },
    };
  }
  return {ok: false, error: 'messageId or text is required.'};
}

export function truncateTextForSpeech(text: string, maxChars: number) {
  const normalized = text.replace(/\s+/g, ' ').trim();
  if (normalized.length <= maxChars) {
    return normalized;
  }
  const candidate = normalized.slice(0, Math.max(1, maxChars));
  const sentenceBoundary = Math.max(
    candidate.lastIndexOf('.'),
    candidate.lastIndexOf('!'),
    candidate.lastIndexOf('?'),
  );
  const truncated = sentenceBoundary > 20 ? candidate.slice(0, sentenceBoundary + 1) : candidate.trimEnd();
  return `${SPOKEN_SUMMARY_PREFIX} ${truncated}`;
}

export function resolveVoiceStyle(mode: ChatMode) {
  if (mode === 'feynman') {
    return {voice: DEFAULT_VOICE, speed: 0.92};
  }
  if (mode === 'guided_learning') {
    return {voice: DEFAULT_VOICE, speed: 0.96};
  }
  return {voice: DEFAULT_VOICE, speed: 1};
}

function errorResponse(message: string, status: number) {
  return NextResponse.json({error: message}, {status});
}

async function resolveSpeechText(
  supabase: Awaited<ReturnType<typeof createClient>>,
  userId: string,
  body: SpeechBody,
): Promise<{text: string; mode: ChatMode; sessionId: string | null}> {
  if ('messageId' in body) {
    const {data, error} = await supabase
      .from('chat_messages')
      .select('content, role, session_id, chat_sessions!inner(mode, user_id)')
      .eq('id', body.messageId)
      .eq('user_id', userId)
      .maybeSingle();
    if (error || !data || data.role !== 'assistant') {
      throw new Response('Assistant message not found.', {status: 404});
    }
    const joinedSession = Array.isArray(data.chat_sessions) ? data.chat_sessions[0] : data.chat_sessions;
    const mode = CHAT_MODES.includes(joinedSession?.mode as ChatMode) ? (joinedSession?.mode as ChatMode) : 'normal';
    return {text: String(data.content ?? ''), mode, sessionId: String(data.session_id)};
  }

  if (body.sessionId) {
    const {data, error} = await supabase
      .from('chat_sessions')
      .select('id')
      .eq('id', body.sessionId)
      .eq('user_id', userId)
      .maybeSingle();
    if (error || !data?.id) {
      throw new Response('Session not found.', {status: 404});
    }
  }
  return {text: body.text, mode: body.mode ?? 'normal', sessionId: body.sessionId ?? null};
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

  let rawBody: unknown;
  try {
    rawBody = await request.json();
  } catch {
    return errorResponse('Invalid JSON body.', 400);
  }

  const validation = validateSpeechBody(rawBody);
  if (!validation.ok) {
    return errorResponse(validation.error, 400);
  }

  let serviceSupabase;
  try {
    serviceSupabase = createServiceClient() as unknown as VoiceUsageClient;
    await assertVoiceQuota(serviceSupabase, user.id, 'tts', config);
  } catch (error) {
    if (error instanceof VoiceQuotaError) {
      return errorResponse(error.message, error.status);
    }
    console.error('Voice TTS quota check failed', error);
    return errorResponse('Voice mode is not configured.', 500);
  }

  try {
    const resolved = await resolveSpeechText(supabase, user.id, validation.value);
    const voiceStyle = resolveVoiceStyle(resolved.mode);
    const voice = validation.value.voice || voiceStyle.voice;
    const speechText = truncateTextForSpeech(resolved.text, config.ttsMaxChars);
    const provider = new OpenAIVoiceProvider({
      apiKey: config.openaiApiKey,
      sttModel: config.sttModel,
      ttsModel: config.ttsModel,
    });
    const result = await provider.synthesize(speechText, {voice, speed: voiceStyle.speed});
    await recordVoiceUsage(serviceSupabase, {
      userId: user.id,
      sessionId: resolved.sessionId,
      outputChars: result.characterCount,
      provider: config.provider,
      ttsModel: result.model,
    });

    return new Response(result.audio, {
      headers: {
        'Content-Type': result.audio.type || 'audio/mpeg',
        'Cache-Control': 'no-store',
        'X-Voice-Characters': String(result.characterCount),
        'X-Voice-Model': result.model,
      },
    });
  } catch (error) {
    if (error instanceof Response) {
      return errorResponse(await error.text(), error.status);
    }
    console.error('Voice speech synthesis failed', error);
    return errorResponse('Voice playback failed. Please read the text response.', 502);
  }
}
