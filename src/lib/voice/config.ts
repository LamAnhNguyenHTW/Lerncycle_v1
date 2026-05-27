import 'server-only';

import type {ChatMode} from '@/types/chat';
import type {PublicVoiceConfig, VoiceProviderName, VoiceServerConfig} from '@/types/voice';

const DEFAULT_PROVIDER: VoiceProviderName = 'openai';
const DEFAULT_STT_MODEL = 'gpt-4o-mini-transcribe';
const DEFAULT_TTS_MODEL = 'gpt-4o-mini-tts';
const DEFAULT_MAX_RECORDING_SECONDS = 60;
const DEFAULT_MAX_UPLOAD_BYTES = 5_242_880;
const DEFAULT_DAILY_MINUTES_PER_USER = 10;
const DEFAULT_DAILY_TTS_RESPONSES_PER_USER = 20;
const DEFAULT_TTS_MAX_CHARS = 1000;
const MVP_ENABLED_MODES: ChatMode[] = ['feynman'];

type VoiceEnv = Record<string, string | undefined>;

export function getVoiceConfig(env: VoiceEnv = process.env): PublicVoiceConfig {
  const serverConfig = getVoiceServerConfig(env);
  return {
    enabled: serverConfig.enabled,
    maxRecordingSeconds: serverConfig.maxRecordingSeconds,
    maxUploadBytes: serverConfig.maxUploadBytes,
    enabledModes: serverConfig.enabledModes,
    ttsMaxChars: serverConfig.ttsMaxChars,
  };
}

export function getVoiceServerConfig(env: VoiceEnv = process.env): VoiceServerConfig {
  return {
    enabled: parseBool(env.VOICE_MODE_ENABLED, false),
    provider: parseProvider(env.VOICE_PROVIDER),
    sttModel: nonEmpty(env.VOICE_STT_MODEL, DEFAULT_STT_MODEL),
    ttsModel: nonEmpty(env.VOICE_TTS_MODEL, DEFAULT_TTS_MODEL),
    maxRecordingSeconds: parsePositiveInt(env.VOICE_MAX_RECORDING_SECONDS, DEFAULT_MAX_RECORDING_SECONDS),
    maxUploadBytes: parsePositiveInt(env.VOICE_MAX_UPLOAD_BYTES, DEFAULT_MAX_UPLOAD_BYTES),
    dailyMinutesPerUser: parsePositiveInt(env.VOICE_DAILY_MINUTES_PER_USER, DEFAULT_DAILY_MINUTES_PER_USER),
    dailyTtsResponsesPerUser: parsePositiveInt(
      env.VOICE_DAILY_TTS_RESPONSES_PER_USER,
      DEFAULT_DAILY_TTS_RESPONSES_PER_USER,
    ),
    ttsMaxChars: parsePositiveInt(env.VOICE_TTS_MAX_CHARS, DEFAULT_TTS_MAX_CHARS),
    enabledModes: MVP_ENABLED_MODES,
    openaiApiKey: env.OPENAI_API_KEY,
  };
}

function parseBool(value: string | undefined, fallback: boolean) {
  if (value === undefined || value.trim() === '') {
    return fallback;
  }
  return !['0', 'false', 'no', 'off'].includes(value.trim().toLowerCase());
}

function parsePositiveInt(value: string | undefined, fallback: number) {
  if (value === undefined || value.trim() === '') {
    return fallback;
  }
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function parseProvider(value: string | undefined): VoiceProviderName {
  return value === 'gemini' ? 'gemini' : DEFAULT_PROVIDER;
}

function nonEmpty(value: string | undefined, fallback: string) {
  const trimmed = value?.trim();
  return trimmed ? trimmed : fallback;
}
