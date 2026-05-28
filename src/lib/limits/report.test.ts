import {buildMonthlyUsageCostReport, startOfUtcMonth} from './report';

export function assertMonthlyUsageCostReportGroupsByFeatureAndUser() {
  const report = buildMonthlyUsageCostReport(
    [
      {user_id: 'user-1', feature: 'chat', estimated_cost_usd: 0.002},
      {user_id: 'user-1', feature: 'chat', estimated_cost_usd: '0.003'},
      {user_id: 'user-2', feature: 'realtime_voice', estimated_cost_usd: 0.3},
      {user_id: 'user-3', feature: 'chat', estimated_cost_usd: 0},
    ],
    new Date('2026-05-01T00:00:00.000Z'),
  );
  if (report.totalEstimatedCostUsd !== 0.305) {
    throw new Error(`Expected total cost to sum positive rows, got ${JSON.stringify(report)}`);
  }
  if (report.byFeature.chat !== 0.005 || report.byFeature.realtime_voice !== 0.3) {
    throw new Error(`Expected feature grouping, got ${JSON.stringify(report.byFeature)}`);
  }
  if (report.byUser['user-1'] !== 0.005 || report.byUser['user-2'] !== 0.3) {
    throw new Error(`Expected user grouping, got ${JSON.stringify(report.byUser)}`);
  }
}

export function assertStartOfUtcMonthUsesUtc() {
  const result = startOfUtcMonth(new Date('2026-05-28T23:30:00.000Z')).toISOString();
  if (result !== '2026-05-01T00:00:00.000Z') {
    throw new Error(`Expected UTC month start, got ${result}`);
  }
}
