import 'server-only';

import type {VoiceProviderName} from '@/types/voice';
import type {VoiceServerConfig} from '@/types/voice';

export type VoiceQuotaKind = 'stt' | 'tts';

type VoiceUsageRow = {
  input_seconds: number | null;
  output_chars: number | null;
};

type VoiceUsageQueryResult = PromiseLike<{
  data: VoiceUsageRow[] | null;
  error: {message: string} | null;
}>;

type VoiceUsageInsertResult = PromiseLike<{
  error: {message: string} | null;
}>;

type VoiceUsageSelectQuery = {
  eq: (column: string, value: string) => {
    gte: (column: string, value: string) => VoiceUsageQueryResult;
  };
};

type VoiceUsageTable = {
  select: (columns: string) => VoiceUsageSelectQuery;
  insert: (payload: Record<string, unknown>) => VoiceUsageInsertResult;
};

export type VoiceUsageClient = {
  from: (table: string) => VoiceUsageTable;
};

type VoiceUsageInsert = {
  userId: string;
  sessionId?: string | null;
  inputSeconds?: number;
  outputChars?: number;
  provider: VoiceProviderName;
  sttModel?: string | null;
  ttsModel?: string | null;
};

const VOICE_LIMIT_ERROR =
  "Du hast dein heutiges Voice-Limit erreicht. Textchat funktioniert weiterhin. / You have reached today's voice limit. Text chat still works.";
const NOMINAL_AUDIO_BYTES_PER_SECOND = 16_000;

export class VoiceQuotaError extends Error {
  readonly status = 429;

  constructor(readonly kind: VoiceQuotaKind, message = VOICE_LIMIT_ERROR) {
    super(message);
    this.name = 'VoiceQuotaError';
  }
}

export async function assertVoiceQuota(
  supabase: VoiceUsageClient,
  userId: string,
  kind: VoiceQuotaKind,
  config: VoiceServerConfig,
): Promise<void> {
  const {data, error} = await supabase
    .from('voice_usage_events')
    .select('input_seconds, output_chars')
    .eq('user_id', userId)
    .gte('created_at', sinceLast24Hours());

  if (error) {
    throw new Error('Failed to load voice usage.');
  }

  const totals = calculateVoiceUsageTotals(data ?? []);
  if (kind === 'stt' && totals.inputSeconds >= config.dailyMinutesPerUser * 60) {
    throw new VoiceQuotaError(kind);
  }
  if (kind === 'tts' && totals.ttsResponses >= config.dailyTtsResponsesPerUser) {
    throw new VoiceQuotaError(kind);
  }
}

export async function recordVoiceUsage(
  supabase: VoiceUsageClient,
  usage: VoiceUsageInsert,
): Promise<void> {
  const {error} = await supabase.from('voice_usage_events').insert({
    user_id: usage.userId,
    session_id: usage.sessionId ?? null,
    input_seconds: Math.max(0, Math.ceil(usage.inputSeconds ?? 0)),
    output_chars: Math.max(0, Math.ceil(usage.outputChars ?? 0)),
    provider: usage.provider,
    stt_model: usage.sttModel ?? null,
    tts_model: usage.ttsModel ?? null,
  });
  if (error) {
    throw new Error('Failed to write voice usage.');
  }
}

export function calculateVoiceUsageTotals(rows: VoiceUsageRow[]) {
  return rows.reduce(
    (totals, row) => ({
      inputSeconds: totals.inputSeconds + Math.max(0, Number(row.input_seconds ?? 0)),
      ttsResponses: totals.ttsResponses + (Number(row.output_chars ?? 0) > 0 ? 1 : 0),
    }),
    {inputSeconds: 0, ttsResponses: 0},
  );
}

export function resolveInputSeconds({
  providerDurationSeconds,
  clientRecordingSeconds,
  audioBytes,
  maxRecordingSeconds,
}: {
  providerDurationSeconds?: number;
  clientRecordingSeconds?: number;
  audioBytes: number;
  maxRecordingSeconds: number;
}) {
  if (providerDurationSeconds !== undefined && Number.isFinite(providerDurationSeconds) && providerDurationSeconds > 0) {
    return Math.min(Math.ceil(providerDurationSeconds), maxRecordingSeconds);
  }
  const clampedClientValue = truncateVoiceInputSeconds(clientRecordingSeconds, maxRecordingSeconds);
  if (clampedClientValue > 0) {
    return clampedClientValue;
  }
  return estimateInputSecondsFromAudio(audioBytes, maxRecordingSeconds);
}

export function truncateVoiceInputSeconds(value: unknown, maxRecordingSeconds: number) {
  const parsed = typeof value === 'number'
    ? value
    : typeof value === 'string'
      ? Number.parseFloat(value)
      : 0;
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return 0;
  }
  return Math.min(Math.ceil(parsed), maxRecordingSeconds);
}

export function estimateInputSecondsFromAudio(audioBytes: number, maxRecordingSeconds: number) {
  if (!Number.isFinite(audioBytes) || audioBytes <= 0) {
    return 0;
  }
  return Math.min(Math.max(1, Math.ceil(audioBytes / NOMINAL_AUDIO_BYTES_PER_SECOND)), maxRecordingSeconds);
}

function sinceLast24Hours(): string {
  return new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString();
}
