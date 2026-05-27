import 'server-only';

import type {SpeechResult, TranscriptionResult} from '@/types/voice';
import type {VoiceProvider} from './provider';

const OPENAI_AUDIO_TRANSCRIPTIONS_URL = 'https://api.openai.com/v1/audio/transcriptions';
const OPENAI_AUDIO_SPEECH_URL = 'https://api.openai.com/v1/audio/speech';

type OpenAIVoiceProviderOptions = {
  apiKey: string | undefined;
  sttModel: string;
  ttsModel: string;
  fetchImpl?: typeof fetch;
};

export class OpenAIVoiceProvider implements VoiceProvider {
  private readonly apiKey: string;
  private readonly sttModel: string;
  private readonly ttsModel: string;
  private readonly fetchImpl: typeof fetch;

  constructor(options: OpenAIVoiceProviderOptions) {
    if (!options.apiKey?.trim()) {
      throw new Error('OpenAI API key is required for voice mode.');
    }
    this.apiKey = options.apiKey;
    this.sttModel = options.sttModel;
    this.ttsModel = options.ttsModel;
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
