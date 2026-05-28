import 'server-only';

import {
  DEFAULT_USAGE_LIMITS,
  getUsageLimitsConfig,
  type UsageFeature,
  type UsageFeatureLimit,
  type UsageLimitsConfig,
} from './config';

export type {UsageFeature} from './config';

type UsageRow = {
  quantity: number | string | null;
};

type UsageQueryResult = PromiseLike<{
  data: UsageRow[] | null;
  error: {message: string} | null;
}>;

type UsageInsertResult = PromiseLike<{
  error: {message: string} | null;
}>;

type UsageQuery = {
  eq: (column: string, value: string) => UsageQuery;
  gte: (column: string, value: string) => UsageQueryResult;
};

type UsageTable = {
  select: (columns: string) => UsageQuery;
  insert: (payload: Record<string, unknown>) => UsageInsertResult;
};

export type UsageClient = {
  from: (table: string) => UsageTable;
};

export type UsageQuotaReason =
  | 'lockdown'
  | 'per_user_limit'
  | 'global_monthly_limit'
  | 'usage_read_error';

type UsageQuotaParams = {
  supabase: UsageClient;
  userId: string;
  feature: UsageFeature;
  requested?: number;
  config?: UsageLimitsConfig | {lockdown: boolean; features: Partial<Record<UsageFeature, UsageFeatureLimit>>};
  now?: Date;
};

type RecordUsageParams = {
  supabase: UsageClient;
  userId: string;
  feature: UsageFeature;
  quantity: number;
  model?: string | null;
  sessionId?: string | null;
  createdAt?: Date;
  config?: UsageLimitsConfig | {lockdown: boolean; features: Partial<Record<UsageFeature, UsageFeatureLimit>>};
};

const DEFAULT_LIMIT_MESSAGE =
  'Dieses Beta-Limit ist erreicht. Du kannst andere Funktionen weiter nutzen. / This beta limit has been reached. You can keep using other features.';

export class UsageQuotaError extends Error {
  readonly status = 429;

  constructor(readonly feature: UsageFeature, readonly reason: UsageQuotaReason, message = DEFAULT_LIMIT_MESSAGE) {
    super(message);
    this.name = 'UsageQuotaError';
  }
}

export async function assertUsageQuota(params: UsageQuotaParams): Promise<void> {
  const config = params.config ?? getUsageLimitsConfig();
  if (config.lockdown) {
    throw new UsageQuotaError(params.feature, 'lockdown');
  }

  const limit = resolveFeatureLimit(config, params.feature);
  const requested = normalizeQuantity(params.requested ?? 1);
  const now = params.now ?? new Date();
  const perUserStart = limit.perUser.window === 'month'
    ? startOfMonth(now)
    : new Date(now.getTime() - 24 * 60 * 60 * 1000);
  const monthStart = startOfMonth(now);

  const [perUserUsed, globalUsed] = await Promise.all([
    sumUsage(params.supabase, {
      feature: params.feature,
      userId: params.userId,
      createdAfter: perUserStart,
    }),
    sumUsage(params.supabase, {
      feature: params.feature,
      createdAfter: monthStart,
    }),
  ]);

  if (perUserUsed === null || globalUsed === null) {
    throw new UsageQuotaError(params.feature, 'usage_read_error');
  }
  if (perUserUsed + requested > limit.perUser.quantity) {
    throw new UsageQuotaError(params.feature, 'per_user_limit');
  }
  if (globalUsed + requested > limit.globalMonthly.quantity) {
    throw new UsageQuotaError(params.feature, 'global_monthly_limit');
  }
}

export async function recordUsage(params: RecordUsageParams): Promise<void> {
  const limit = resolveFeatureLimit(params.config ?? getUsageLimitsConfig(), params.feature);
  const quantity = normalizeQuantity(params.quantity);
  const {error} = await params.supabase.from('usage_events').insert({
    user_id: params.userId,
    feature: params.feature,
    quantity,
    unit: limit.unit,
    model: params.model ?? null,
    estimated_cost_usd: estimateCost(params.feature, quantity, params.config),
    session_id: params.sessionId ?? null,
    ...(params.createdAt ? {created_at: params.createdAt.toISOString()} : {}),
  });
  if (error) {
    throw new Error('Failed to write usage event.');
  }
}

export function estimateCost(
  feature: UsageFeature,
  quantity: number,
  config?: UsageLimitsConfig | {lockdown: boolean; features: Partial<Record<UsageFeature, UsageFeatureLimit>>},
) {
  const limit = resolveFeatureLimit(config ?? getUsageLimitsConfig(), feature);
  return roundUsd(normalizeQuantity(quantity) * limit.estimatedUnitCostUsd);
}

function resolveFeatureLimit(
  config: UsageLimitsConfig | {lockdown: boolean; features: Partial<Record<UsageFeature, UsageFeatureLimit>>},
  feature: UsageFeature,
) {
  return config.features[feature] ?? DEFAULT_USAGE_LIMITS[feature];
}

async function sumUsage(
  supabase: UsageClient,
  filters: {
    feature: UsageFeature;
    userId?: string;
    createdAfter: Date;
  },
) {
  try {
    let query = supabase
      .from('usage_events')
      .select('quantity')
      .eq('feature', filters.feature);
    if (filters.userId) {
      query = query.eq('user_id', filters.userId);
    }
    const {data, error} = await query.gte('created_at', filters.createdAfter.toISOString());
    if (error) {
      return null;
    }
    return (data ?? []).reduce((sum, row) => sum + normalizeQuantity(row.quantity), 0);
  } catch {
    return null;
  }
}

function normalizeQuantity(value: unknown) {
  const parsed = Number(value ?? 0);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 0;
}

function startOfMonth(now: Date) {
  return new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), 1, 0, 0, 0, 0));
}

function roundUsd(value: number) {
  return Math.round(value * 1_000_000) / 1_000_000;
}
