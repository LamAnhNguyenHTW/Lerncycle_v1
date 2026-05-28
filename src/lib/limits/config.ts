import 'server-only';

export type UsageFeature =
  | 'realtime_voice'
  | 'voice_stt'
  | 'voice_tts'
  | 'chat'
  | 'active_learning'
  | 'revision_generation'
  | 'embeddings_upload';

export type UsageUnit = 'minutes' | 'responses' | 'messages' | 'generations' | 'documents';
export type UsageWindow = 'day' | 'month';

export type UsageFeatureLimit = {
  unit: UsageUnit;
  estimatedUnitCostUsd: number;
  perUser: {
    quantity: number;
    window: UsageWindow;
  };
  globalMonthly: {
    quantity: number;
  };
};

export type UsageLimitsConfig = {
  lockdown: boolean;
  features: Record<UsageFeature, UsageFeatureLimit>;
};

export const DEFAULT_USAGE_LIMITS: Record<UsageFeature, UsageFeatureLimit> = {
  realtime_voice: {
    unit: 'minutes',
    estimatedUnitCostUsd: 0.3,
    perUser: {quantity: 5, window: 'month'},
    globalMonthly: {quantity: 80},
  },
  voice_stt: {
    unit: 'minutes',
    estimatedUnitCostUsd: 0.003,
    perUser: {quantity: 10, window: 'day'},
    globalMonthly: {quantity: 600},
  },
  voice_tts: {
    unit: 'responses',
    estimatedUnitCostUsd: 0.015,
    perUser: {quantity: 20, window: 'day'},
    globalMonthly: {quantity: 400},
  },
  chat: {
    unit: 'messages',
    estimatedUnitCostUsd: 0.002,
    perUser: {quantity: 40, window: 'day'},
    globalMonthly: {quantity: 4000},
  },
  active_learning: {
    unit: 'messages',
    estimatedUnitCostUsd: 0.002,
    perUser: {quantity: 30, window: 'day'},
    globalMonthly: {quantity: 2000},
  },
  revision_generation: {
    unit: 'generations',
    estimatedUnitCostUsd: 0.01,
    perUser: {quantity: 5, window: 'day'},
    globalMonthly: {quantity: 400},
  },
  embeddings_upload: {
    unit: 'documents',
    estimatedUnitCostUsd: 0.0002,
    perUser: {quantity: 20, window: 'day'},
    globalMonthly: {quantity: 600},
  },
};

type UsageEnv = Record<string, string | undefined>;

export function getUsageLimitsConfig(env: UsageEnv = process.env): UsageLimitsConfig {
  const features = Object.fromEntries(
    Object.entries(DEFAULT_USAGE_LIMITS).map(([feature, defaults]) => {
      const usageFeature = feature as UsageFeature;
      return [
        usageFeature,
        {
          ...defaults,
          perUser: {
            ...defaults.perUser,
            quantity: parsePositiveNumber(env[usageLimitEnvName(usageFeature, 'perUser')], defaults.perUser.quantity),
          },
          globalMonthly: {
            quantity: parsePositiveNumber(
              env[usageLimitEnvName(usageFeature, 'globalMonthly')],
              defaults.globalMonthly.quantity,
            ),
          },
        },
      ];
    }),
  ) as Record<UsageFeature, UsageFeatureLimit>;

  return {
    lockdown: parseBool(env.BETA_COST_LOCKDOWN, false),
    features,
  };
}

export function usageLimitEnvName(feature: UsageFeature, scope: 'perUser' | 'globalMonthly') {
  const prefix = `BETA_LIMIT_${feature.toUpperCase()}`;
  if (scope === 'globalMonthly') {
    return `${prefix}_GLOBAL_MONTHLY`;
  }
  const window = DEFAULT_USAGE_LIMITS[feature].perUser.window === 'month' ? 'MONTHLY' : 'DAILY';
  return `${prefix}_PER_USER_${window}`;
}

function parseBool(value: string | undefined, fallback: boolean) {
  if (value === undefined || value.trim() === '') {
    return fallback;
  }
  return !['0', 'false', 'no', 'off'].includes(value.trim().toLowerCase());
}

function parsePositiveNumber(value: string | undefined, fallback: number) {
  if (value === undefined || value.trim() === '') {
    return fallback;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : fallback;
}
