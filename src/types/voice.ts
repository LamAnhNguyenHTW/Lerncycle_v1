import type {ChatMode} from '@/types/chat';

export type VoiceProviderName = 'openai' | 'gemini';

export type VoiceSessionState =
  | 'idle'
  | 'recording'
  | 'transcribing'
  | 'thinking'
  | 'speaking'
  | 'error';

export type TranscriptionResult = {
  text: string;
  durationSeconds: number;
  language?: string;
  model: string;
};

export type SpeechResult = {
  audio: Blob;
  characterCount: number;
  model: string;
};

export type VoiceUsage = {
  userId: string;
  inputSeconds: number;
  outputCharacters: number;
  provider: VoiceProviderName;
  sttModel?: string;
  ttsModel?: string;
};

export type PublicVoiceConfig = {
  enabled: boolean;
  maxRecordingSeconds: number;
  maxUploadBytes: number;
  enabledModes: ChatMode[];
  ttsMaxChars: number;
};

export type VoiceServerConfig = PublicVoiceConfig & {
  provider: VoiceProviderName;
  sttModel: string;
  ttsModel: string;
  dailyMinutesPerUser: number;
  dailyTtsResponsesPerUser: number;
  openaiApiKey?: string;
};
