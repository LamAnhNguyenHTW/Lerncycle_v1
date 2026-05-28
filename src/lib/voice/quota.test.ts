import {
  VoiceQuotaError,
  calculateVoiceUsageTotals,
  estimateInputSecondsFromAudio,
  recordVoiceUsage,
  truncateVoiceInputSeconds,
} from './quota';

export function assertVoiceUsageTotalsSeparateSttAndTts() {
  const totals = calculateVoiceUsageTotals([
    {input_seconds: 10, output_chars: 0},
    {input_seconds: 5, output_chars: 120},
    {input_seconds: 0, output_chars: 60},
  ]);

  if (totals.inputSeconds !== 15 || totals.ttsResponses !== 2) {
    throw new Error(`Unexpected usage totals: ${JSON.stringify(totals)}`);
  }
}

export function assertVoiceQuotaErrorCarriesKindAndStatus() {
  const error = new VoiceQuotaError('stt', 'limit reached');

  if (error.kind !== 'stt' || error.status !== 429 || error.message !== 'limit reached') {
    throw new Error('VoiceQuotaError must expose kind, status, and message.');
  }
}

export function assertVoiceDurationFallbacksClampClientInput() {
  const clamped = truncateVoiceInputSeconds(999, 60);
  if (clamped !== 60) {
    throw new Error(`Expected client advisory duration to clamp at 60, got ${clamped}.`);
  }

  const estimated = estimateInputSecondsFromAudio(32_000, 60);
  if (estimated !== 2) {
    throw new Error(`Expected nominal bitrate estimate to be 2 seconds, got ${estimated}.`);
  }
}

export async function assertVoiceUsageRecordsUnifiedUsageEvents() {
  const inserts: unknown[] = [];
  const query = {
    eq: () => query,
    gte: () => Promise.resolve({data: [], error: null}),
  };
  const client = {
    from: (table: string) => ({
      select: () => query,
      insert: (payload: unknown) => {
        inserts.push({table, payload});
        return Promise.resolve({error: null});
      },
    }),
  };

  await recordVoiceUsage(client, {
    userId: 'user-1',
    sessionId: '00000000-0000-0000-0000-000000000001',
    inputSeconds: 60,
    outputChars: 120,
    provider: 'openai',
    sttModel: 'stt',
    ttsModel: 'tts',
  });

  if (
    inserts.length !== 2 ||
    !inserts.every((insert) => (insert as {table: string}).table === 'usage_events') ||
    !JSON.stringify(inserts).includes('"feature":"voice_stt"') ||
    !JSON.stringify(inserts).includes('"feature":"voice_tts"')
  ) {
    throw new Error(`Expected voice usage to write unified usage events, got ${JSON.stringify(inserts)}`);
  }
}
