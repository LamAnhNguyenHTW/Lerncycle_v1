import 'server-only';

import type {SpeechResult, TranscriptionResult} from '@/types/voice';
import type {RealtimeSessionConfig, VoiceProvider} from './provider';

/**
 * Interface-compatible placeholder for a future Gemini Live implementation.
 *
 * This class intentionally does not call Gemini yet. It exists to keep the
 * provider seam honest while preserving the current OpenAI-only runtime path.
 */
export class GeminiVoiceProvider implements VoiceProvider {
  async transcribe(): Promise<TranscriptionResult> {
    throw new Error('GeminiVoiceProvider is not implemented.');
  }

  async synthesize(): Promise<SpeechResult> {
    throw new Error('GeminiVoiceProvider is not implemented.');
  }

  async createRealtimeSession(): Promise<RealtimeSessionConfig> {
    throw new Error('GeminiVoiceProvider is not implemented.');
  }
}
