import {
  UsageQuotaError,
  assertUsageQuota,
  estimateCost,
  recordUsage,
} from './guard';

function usageClient(rows: Array<{quantity: number}> = [], options: {readError?: boolean; writeError?: boolean} = {}) {
  const inserts: unknown[] = [];
  const query = {
    eq: () => query,
    gte: () => Promise.resolve({
      data: options.readError ? null : rows,
      error: options.readError ? {message: 'read failed'} : null,
    }),
  };
  return {
    inserts,
    from: () => ({
      select: () => query,
      insert: (payload: unknown) => {
        inserts.push(payload);
        return Promise.resolve({error: options.writeError ? {message: 'write failed'} : null});
      },
    }),
  };
}

export async function assertUsageQuotaAllowsUnderCapUsage() {
  await assertUsageQuota({
    supabase: usageClient([{quantity: 1}]),
    userId: 'user-1',
    feature: 'chat',
    requested: 1,
    config: {
      lockdown: false,
      features: {
        chat: {
          unit: 'messages',
          estimatedUnitCostUsd: 0.002,
          perUser: {quantity: 40, window: 'day'},
          globalMonthly: {quantity: 4000},
        },
      },
    },
  });
}

export async function assertUsageQuotaDeniesPerUserOverCap() {
  try {
    await assertUsageQuota({
      supabase: usageClient([{quantity: 40}]),
      userId: 'user-1',
      feature: 'chat',
      requested: 1,
    });
  } catch (error) {
    if (error instanceof UsageQuotaError && error.reason === 'per_user_limit') {
      return;
    }
    throw error;
  }
  throw new Error('Expected per-user cap denial.');
}

export async function assertUsageQuotaDeniesOnReadError() {
  try {
    await assertUsageQuota({
      supabase: usageClient([], {readError: true}),
      userId: 'user-1',
      feature: 'chat',
      requested: 1,
    });
  } catch (error) {
    if (error instanceof UsageQuotaError && error.reason === 'usage_read_error') {
      return;
    }
    throw error;
  }
  throw new Error('Expected budget-safe read-error denial.');
}

export async function assertUsageQuotaDeniesDuringLockdown() {
  try {
    await assertUsageQuota({
      supabase: usageClient([]),
      userId: 'user-1',
      feature: 'chat',
      requested: 1,
      config: {
        lockdown: true,
        features: {},
      },
    });
  } catch (error) {
    if (error instanceof UsageQuotaError && error.reason === 'lockdown') {
      return;
    }
    throw error;
  }
  throw new Error('Expected lockdown denial.');
}

export async function assertRecordUsageStampsEstimatedCost() {
  const client = usageClient();
  await recordUsage({
    supabase: client,
    userId: 'user-1',
    feature: 'voice_tts',
    quantity: 2,
    model: 'tts-model',
    sessionId: '00000000-0000-0000-0000-000000000001',
  });

  const payload = client.inserts[0] as {estimated_cost_usd?: number; unit?: string};
  if (payload.estimated_cost_usd !== 0.03 || payload.unit !== 'responses') {
    throw new Error(`Expected cost-stamped usage insert, got ${JSON.stringify(payload)}`);
  }
}

export function assertEstimateCostUsesFeatureUnitCost() {
  if (estimateCost('realtime_voice', 10) !== 3) {
    throw new Error('Expected realtime estimate to use $0.30/min.');
  }
}
