import {
  DEFAULT_USAGE_LIMITS,
  getUsageLimitsConfig,
  usageLimitEnvName,
} from './config';

export function assertUsageLimitDefaultsMatchBetaBudget() {
  const config = getUsageLimitsConfig({});

  if (config.lockdown !== false) {
    throw new Error('Beta cost lockdown must default to disabled.');
  }
  if (config.features.realtime_voice.perUser.window !== 'month') {
    throw new Error('Realtime voice must use a per-user monthly window.');
  }
  if (
    config.features.realtime_voice.perUser.quantity !== 5 ||
    config.features.realtime_voice.globalMonthly.quantity !== 80 ||
    config.features.chat.perUser.quantity !== 40 ||
    config.features.revision_generation.globalMonthly.quantity !== 400
  ) {
    throw new Error(`Unexpected default usage limits: ${JSON.stringify(config.features)}`);
  }
  if (config.features.embeddings_upload.unit !== 'documents') {
    throw new Error(`Expected embeddings_upload to count documents, got ${config.features.embeddings_upload.unit}`);
  }
}

export function assertUsageLimitEnvOverridesAreParsed() {
  const config = getUsageLimitsConfig({
    BETA_COST_LOCKDOWN: 'true',
    BETA_LIMIT_CHAT_PER_USER_DAILY: '12',
    BETA_LIMIT_CHAT_GLOBAL_MONTHLY: '120',
    BETA_LIMIT_REALTIME_VOICE_PER_USER_MONTHLY: '3',
    BETA_LIMIT_REALTIME_VOICE_GLOBAL_MONTHLY: '30',
  });

  if (!config.lockdown) {
    throw new Error('Expected BETA_COST_LOCKDOWN=true to enable lockdown.');
  }
  if (config.features.chat.perUser.quantity !== 12 || config.features.chat.globalMonthly.quantity !== 120) {
    throw new Error(`Expected chat overrides to parse, got ${JSON.stringify(config.features.chat)}`);
  }
  if (
    config.features.realtime_voice.perUser.quantity !== 3 ||
    config.features.realtime_voice.perUser.window !== 'month' ||
    config.features.realtime_voice.globalMonthly.quantity !== 30
  ) {
    throw new Error(`Expected realtime overrides to parse, got ${JSON.stringify(config.features.realtime_voice)}`);
  }
}

export function assertUsageLimitEnvNamesAreStable() {
  if (usageLimitEnvName('voice_tts', 'perUser') !== 'BETA_LIMIT_VOICE_TTS_PER_USER_DAILY') {
    throw new Error('Expected daily voice_tts per-user env var name.');
  }
  if (usageLimitEnvName('realtime_voice', 'perUser') !== 'BETA_LIMIT_REALTIME_VOICE_PER_USER_MONTHLY') {
    throw new Error('Expected monthly realtime per-user env var name.');
  }
  if (usageLimitEnvName('chat', 'globalMonthly') !== 'BETA_LIMIT_CHAT_GLOBAL_MONTHLY') {
    throw new Error('Expected chat global env var name.');
  }
}

export function assertUsageLimitWorstCaseDefaultEstimateStaysUnderOpenAiHardCap() {
  const total = Object.values(DEFAULT_USAGE_LIMITS).reduce(
    (sum, limit) => sum + limit.globalMonthly.quantity * limit.estimatedUnitCostUsd,
    0,
  );

  if (total > 50) {
    throw new Error(`Default global caps exceed the $50 hard-cap envelope: ${total}`);
  }
}
