import 'server-only';

import type {SpeechResult, TranscriptionResult} from '@/types/voice';
import type {RealtimeSessionConfig, RealtimeSessionOptions, VoiceProvider} from './provider';

const OPENAI_AUDIO_TRANSCRIPTIONS_URL = 'https://api.openai.com/v1/audio/transcriptions';
const OPENAI_AUDIO_SPEECH_URL = 'https://api.openai.com/v1/audio/speech';
const OPENAI_REALTIME_CLIENT_SECRETS_URL = 'https://api.openai.com/v1/realtime/client_secrets';

type OpenAIVoiceProviderOptions = {
  apiKey: string | undefined;
  sttModel: string;
  ttsModel: string;
  realtimeModel?: string;
  fetchImpl?: typeof fetch;
};

export class OpenAIVoiceProvider implements VoiceProvider {
  private readonly apiKey: string;
  private readonly sttModel: string;
  private readonly ttsModel: string;
  private readonly realtimeModel: string;
  private readonly fetchImpl: typeof fetch;

  constructor(options: OpenAIVoiceProviderOptions) {
    if (!options.apiKey?.trim()) {
      throw new Error('OpenAI API key is required for voice mode.');
    }
    this.apiKey = options.apiKey;
    this.sttModel = options.sttModel;
    this.ttsModel = options.ttsModel;
    this.realtimeModel = options.realtimeModel ?? 'gpt-realtime-2';
    this.fetchImpl = options.fetchImpl ?? fetch;
  }

  async transcribe(audio: Blob, opts?: {language?: string}): Promise<TranscriptionResult> {
    const formData = new FormData();
    formData.set('file', new File([audio], 'voice-input.webm', {type: audio.type || 'audio/webm'}));
    formData.set('model', this.sttModel);
    // Only whisper-1 supports verbose_json (with duration); gpt-4o-*-transcribe only support json/text.
    const supportsVerboseJson = this.sttModel === 'whisper-1';
    formData.set('response_format', supportsVerboseJson ? 'verbose_json' : 'json');
    if (opts?.language) {
      formData.set('language', opts.language);
    }

    const response = await this.fetchImpl(OPENAI_AUDIO_TRANSCRIPTIONS_URL, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: formData,
    });

    if (!response.ok) {
      const body = await response.text().catch(() => '');
      throw new Error(`OpenAI transcription failed (${response.status}): ${body.slice(0, 500)}`);
    }

    return parseOpenAITranscriptionResponse(await response.json(), this.sttModel);
  }

  async synthesize(text: string, opts: {voice: string; speed?: number}): Promise<SpeechResult> {
    const response = await this.fetchImpl(OPENAI_AUDIO_SPEECH_URL, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: this.ttsModel,
        voice: opts.voice,
        input: text,
        ...(opts.speed ? {speed: opts.speed} : {}),
      }),
    });

    if (!response.ok) {
      throw new Error('OpenAI speech synthesis failed.');
    }

    const contentType = response.headers.get('content-type') ?? 'audio/mpeg';
    const audio = new Blob([await response.arrayBuffer()], {type: contentType});
    return {
      audio,
      characterCount: text.length,
      model: this.ttsModel,
    };
  }

  async createRealtimeSession(opts: RealtimeSessionOptions): Promise<RealtimeSessionConfig> {
    const model = opts.model ?? this.realtimeModel;
    const tools = opts.tools ?? [];
    const response = await this.fetchImpl(OPENAI_REALTIME_CLIENT_SECRETS_URL, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json',
        'OpenAI-Safety-Identifier': await hashedSafetyIdentifier(opts.userId),
      },
      body: JSON.stringify({
        expires_after: {
          anchor: 'created_at',
          seconds: clampRealtimeExpirySeconds(opts.expiresAfterSeconds),
        },
        session: {
          type: 'realtime',
          model,
          ...(opts.instructions ? {instructions: opts.instructions} : {}),
          audio: {
            input: {
              turn_detection: {
                type: 'semantic_vad',
                eagerness: 'medium',
                create_response: true,
                interrupt_response: true,
              },
              transcription: {
                model: this.sttModel,
              },
            },
            output: {
              voice: opts.voice ?? 'alloy',
            },
          },
          ...(tools.length > 0 ? {tools, tool_choice: 'auto'} : {}),
        },
      }),
    });

    if (!response.ok) {
      throw new Error('OpenAI realtime session creation failed.');
    }

    return parseOpenAIRealtimeSessionResponse(await response.json(), model, tools);
  }
}

export class EmptyTranscriptionError extends Error {
  constructor() {
    super('OpenAI transcription returned empty text.');
    this.name = 'EmptyTranscriptionError';
  }
}

export function parseOpenAITranscriptionResponse(value: unknown, model: string): TranscriptionResult {
  if (!value || typeof value !== 'object') {
    throw new Error('OpenAI transcription returned an invalid response.');
  }
  const response = value as {text?: unknown; duration?: unknown; language?: unknown};
  const text = typeof response.text === 'string' ? response.text.trim() : '';
  if (!text) {
    throw new EmptyTranscriptionError();
  }
  const duration = typeof response.duration === 'number' && Number.isFinite(response.duration)
    ? Math.max(0, Math.ceil(response.duration))
    : 0;

  return {
    text,
    durationSeconds: duration,
    ...(typeof response.language === 'string' && response.language ? {language: response.language} : {}),
    model,
  };
}

export function parseOpenAIRealtimeSessionResponse(
  value: unknown,
  model: string,
  tools: RealtimeSessionConfig['tools'] = [],
): RealtimeSessionConfig {
  if (!value || typeof value !== 'object') {
    throw new Error('OpenAI realtime session returned an invalid response.');
  }
  const response = value as {
    value?: unknown;
    expires_at?: unknown;
    client_secret?: {value?: unknown; expires_at?: unknown};
    session?: {client_secret?: {value?: unknown; expires_at?: unknown}; model?: unknown};
  };
  const clientSecret =
    typeof response.value === 'string'
      ? response.value
      : typeof response.client_secret?.value === 'string'
        ? response.client_secret.value
        : typeof response.session?.client_secret?.value === 'string'
          ? response.session.client_secret.value
          : '';
  const expiresAtSeconds =
    typeof response.expires_at === 'number'
      ? response.expires_at
      : typeof response.client_secret?.expires_at === 'number'
        ? response.client_secret.expires_at
        : typeof response.session?.client_secret?.expires_at === 'number'
          ? response.session.client_secret.expires_at
          : 0;

  if (!clientSecret) {
    throw new Error('OpenAI realtime session did not include a client secret.');
  }

  return {
    clientSecret,
    expiresAt: expiresAtSeconds > 0 ? new Date(expiresAtSeconds * 1000).toISOString() : new Date(Date.now() + 60_000).toISOString(),
    model: typeof response.session?.model === 'string' ? response.session.model : model,
    tools,
  };
}

export async function hashedSafetyIdentifier(userId: string): Promise<string> {
  const bytes = new TextEncoder().encode(userId);
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, '0'))
    .join('');
}

function clampRealtimeExpirySeconds(value: number | undefined) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return 600;
  }
  return Math.min(600, Math.max(1, Math.floor(value)));
}
