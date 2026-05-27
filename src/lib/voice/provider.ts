import type {SpeechResult, TranscriptionResult} from '@/types/voice';

export type RealtimeSessionConfig = {
  clientSecret: string;
  expiresAt: string;
  model: string;
};

/**
 * Provider seam for voice I/O.
 *
 * Implementations may call external APIs, but callers can rely on normalized
 * transcription text, bounded speech audio blobs, and provider-reported model
 * metadata without knowing which vendor produced the result.
 */
export interface VoiceProvider {
  transcribe(audio: Blob, opts?: {language?: string}): Promise<TranscriptionResult>;
  synthesize(text: string, opts: {voice: string; speed?: number}): Promise<SpeechResult>;
  createRealtimeSession?(opts: {userId: string}): Promise<RealtimeSessionConfig>;
}
