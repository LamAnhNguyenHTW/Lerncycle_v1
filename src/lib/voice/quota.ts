import 'server-only';

import {assertUsageQuota, recordUsage, type UsageClient} from '@/lib/limits/guard';
import type {VoiceProviderName} from '@/types/voice';
import type {VoiceServerConfig} from '@/types/voice';

export type VoiceQuotaKind = 'stt' | 'tts';

export type VoiceUsageRow = {
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

export type VoiceUsageClient = UsageClient & {
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
  try {
    await assertUsageQuota({
      supabase,
      userId,
      feature: kind === 'stt' ? 'voice_stt' : 'voice_tts',
      requested: kind === 'stt' ? 1 / 60 : 1,
      config: {
        lockdown: false,
        features: {
          voice_stt: {
            unit: 'minutes',
            estimatedUnitCostUsd: 0.003,
            perUser: {quantity: config.dailyMinutesPerUser, window: 'day'},
            globalMonthly: {quantity: 600},
          },
          voice_tts: {
            unit: 'responses',
            estimatedUnitCostUsd: 0.015,
            perUser: {quantity: config.dailyTtsResponsesPerUser, window: 'day'},
            globalMonthly: {quantity: 400},
          },
        },
      },
    });
  } catch {
    throw new VoiceQuotaError(kind);
  }
}

export async function recordVoiceUsage(
  supabase: VoiceUsageClient,
  usage: VoiceUsageInsert,
): Promise<void> {
  const inputSeconds = Math.max(0, Math.ceil(usage.inputSeconds ?? 0));
  const outputChars = Math.max(0, Math.ceil(usage.outputChars ?? 0));
  if (inputSeconds > 0) {
    await recordUsage({
      supabase,
      userId: usage.userId,
      feature: 'voice_stt',
      quantity: inputSeconds / 60,
      model: usage.sttModel ?? null,
      sessionId: usage.sessionId ?? null,
    });
  }
  if (outputChars > 0) {
    await recordUsage({
      supabase,
      userId: usage.userId,
      feature: 'voice_tts',
      quantity: 1,
      model: usage.ttsModel ?? null,
      sessionId: usage.sessionId ?? null,
    });
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
