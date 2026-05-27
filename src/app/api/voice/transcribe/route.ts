import {NextResponse} from 'next/server';
import {createClient} from '@/lib/supabase/server';
import {createServiceClient} from '@/lib/supabase/service';
import {getVoiceServerConfig} from '@/lib/voice/config';
import {EmptyTranscriptionError, OpenAIVoiceProvider} from '@/lib/voice/openai';
import {
  VoiceQuotaError,
  assertVoiceQuota,
  recordVoiceUsage,
  resolveInputSeconds,
  type VoiceUsageClient,
} from '@/lib/voice/quota';

export const runtime = 'nodejs';

export const AUDIO_MIME_TYPES = new Set([
  'audio/webm',
  'audio/ogg',
  'audio/mp4',
  'audio/aac',
  'audio/mpeg',
]);

type UploadValidationInput = {
  contentLength: number | null;
  maxUploadBytes: number;
  file: File | null;
};

type UploadValidationResult =
  | {ok: true}
  | {ok: false; status: number; error: string};

export function validateVoiceUpload(input: UploadValidationInput): UploadValidationResult {
  if (input.contentLength !== null && input.contentLength > input.maxUploadBytes) {
    return {ok: false, status: 413, error: 'Audio upload is too large.'};
  }
  if (!input.file) {
    return {ok: false, status: 400, error: 'Missing audio file.'};
  }
  if (input.file.size > input.maxUploadBytes) {
    return {ok: false, status: 413, error: 'Audio upload is too large.'};
  }
  const baseMimeType = input.file.type.split(';')[0].trim().toLowerCase();
  if (!AUDIO_MIME_TYPES.has(baseMimeType)) {
    return {ok: false, status: 415, error: 'Unsupported audio type.'};
  }
  return {ok: true};
}

export function parseRecordingSeconds(value: FormDataEntryValue | null): number | undefined {
  if (typeof value !== 'string') {
    return undefined;
  }
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : undefined;
}

function errorResponse(message: string, status: number) {
  return NextResponse.json({error: message}, {status});
}

function genericVoiceFailure() {
  return errorResponse('Voice transcription failed. Please try again or type your message.', 502);
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

  const contentLengthHeader = request.headers.get('content-length');
  const contentLength = contentLengthHeader ? Number.parseInt(contentLengthHeader, 10) : null;
  if (contentLength !== null && Number.isFinite(contentLength) && contentLength > config.maxUploadBytes) {
    return errorResponse('Audio upload is too large.', 413);
  }

  let formData: FormData;
  try {
    formData = await request.formData();
  } catch {
    return errorResponse('Invalid multipart form data.', 400);
  }

  const audio = formData.get('audio');
  const file = audio instanceof File ? audio : null;
  const validation = validateVoiceUpload({
    contentLength: Number.isFinite(contentLength) ? contentLength : null,
    maxUploadBytes: config.maxUploadBytes,
    file,
  });
  if (!validation.ok) {
    return errorResponse(validation.error, validation.status);
  }
  if (!file) {
    return errorResponse('Missing audio file.', 400);
  }

  let serviceSupabase;
  try {
    serviceSupabase = createServiceClient() as unknown as VoiceUsageClient;
    await assertVoiceQuota(serviceSupabase, user.id, 'stt', config);
  } catch (error) {
    if (error instanceof VoiceQuotaError) {
      return errorResponse(error.message, error.status);
    }
    console.error('Voice quota check failed', error);
    return errorResponse('Voice mode is not configured.', 500);
  }

  try {
    const provider = new OpenAIVoiceProvider({
      apiKey: config.openaiApiKey,
      sttModel: config.sttModel,
      ttsModel: config.ttsModel,
    });
    const languageHint = formData.get('language');
    const language = languageHint === 'de' || languageHint === 'en' ? languageHint : undefined;
    const result = await provider.transcribe(file, language ? {language} : undefined);
    const inputSeconds = resolveInputSeconds({
      providerDurationSeconds: result.durationSeconds,
      clientRecordingSeconds: parseRecordingSeconds(formData.get('recording_seconds')),
      audioBytes: file.size,
      maxRecordingSeconds: config.maxRecordingSeconds,
    });

    await recordVoiceUsage(serviceSupabase, {
      userId: user.id,
      inputSeconds,
      provider: config.provider,
      sttModel: result.model,
    });

    return NextResponse.json({
      text: result.text,
      durationSeconds: inputSeconds,
      ...(result.language ? {language: result.language} : {}),
      model: result.model,
    });
  } catch (error) {
    if (error instanceof EmptyTranscriptionError) {
      console.warn('Voice transcription empty', {bytes: file.size, mime: file.type});
      return errorResponse('No speech detected. Please try again.', 422);
    }
    console.error('Voice transcription failed', error);
    return genericVoiceFailure();
  }
}
