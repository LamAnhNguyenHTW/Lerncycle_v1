import type {SpeechResult, TranscriptionResult} from '@/types/voice';

export type RealtimeToolDefinition = {
  type: 'function';
  name: string;
  description: string;
  parameters: Record<string, unknown>;
};

export type RealtimeSessionConfig = {
  clientSecret: string;
  expiresAt: string;
  model: string;
  tools: RealtimeToolDefinition[];
};

export type RealtimeSessionOptions = {
  userId: string;
  model?: string;
  voice?: string;
  instructions?: string;
  tools?: RealtimeToolDefinition[];
  expiresAfterSeconds?: number;
};

/**
 * Provider seam for all server-side voice I/O.
 *
 * Implementations may call external APIs, but route handlers and UI code can
 * rely on these normalized contracts:
 * - `transcribe()` accepts an in-memory browser audio blob and returns text plus
 *   provider/model metadata. It must not persist raw audio.
 * - `synthesize()` accepts already-authorized chat text and returns a playable
 *   audio blob plus provider/model metadata. Quota enforcement remains outside
 *   the provider in the route handler.
 * - `createRealtimeSession()` returns only short-lived browser credentials and
 *   declarative tool metadata. It must never return a long-lived provider API
 *   key or grant direct database access.
 *
 * Providers are intentionally stateless. User/session/PDF authorization,
 * feature flags, quota checks, and usage logging stay in Next.js route handlers
 * so OpenAI, Gemini, or another backend can be swapped without changing those
 * security boundaries.
 */
export interface VoiceProvider {
  transcribe(audio: Blob, opts?: {language?: string}): Promise<TranscriptionResult>;
  synthesize(text: string, opts: {voice: string; speed?: number}): Promise<SpeechResult>;
  createRealtimeSession?(opts: RealtimeSessionOptions): Promise<RealtimeSessionConfig>;
}
