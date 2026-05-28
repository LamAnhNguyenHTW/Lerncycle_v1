import 'server-only';

import {createServiceClient} from '@/lib/supabase/service';

type UsageCostRow = {
  user_id: string | null;
  feature: string | null;
  estimated_cost_usd: number | string | null;
};

export type MonthlyUsageCostReport = {
  monthStart: string;
  totalEstimatedCostUsd: number;
  byFeature: Record<string, number>;
  byUser: Record<string, number>;
};

export function startOfUtcMonth(now = new Date()) {
  return new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), 1, 0, 0, 0, 0));
}

export function buildMonthlyUsageCostReport(
  rows: UsageCostRow[],
  monthStart = startOfUtcMonth(),
): MonthlyUsageCostReport {
  const byFeature: Record<string, number> = {};
  const byUser: Record<string, number> = {};
  let totalEstimatedCostUsd = 0;

  for (const row of rows) {
    const cost = normalizeCost(row.estimated_cost_usd);
    if (cost <= 0) {
      continue;
    }
    const feature = row.feature ?? 'unknown';
    const userId = row.user_id ?? 'unknown';
    byFeature[feature] = roundUsd((byFeature[feature] ?? 0) + cost);
    byUser[userId] = roundUsd((byUser[userId] ?? 0) + cost);
    totalEstimatedCostUsd = roundUsd(totalEstimatedCostUsd + cost);
  }

  return {
    monthStart: monthStart.toISOString(),
    totalEstimatedCostUsd,
    byFeature,
    byUser,
  };
}

export async function getCurrentMonthUsageCostReport(now = new Date()): Promise<MonthlyUsageCostReport> {
  const monthStart = startOfUtcMonth(now);
  const {data, error} = await createServiceClient()
    .from('usage_events')
    .select('user_id, feature, estimated_cost_usd')
    .gte('created_at', monthStart.toISOString());
  if (error) {
    throw new Error('Failed to load monthly usage cost report.');
  }
  return buildMonthlyUsageCostReport((data ?? []) as UsageCostRow[], monthStart);
}

function normalizeCost(value: unknown) {
  const parsed = Number(value ?? 0);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 0;
}

function roundUsd(value: number) {
  return Math.round(value * 1_000_000) / 1_000_000;
}
