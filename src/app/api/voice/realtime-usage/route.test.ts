import {clampRealtimeUsageSeconds, validateRealtimeUsageBody} from './route';

export function assertRealtimeUsageBodyValidationRequiresPositiveDuration() {
  const result = validateRealtimeUsageBody({sessionId: 'session-1', durationSeconds: 0});
  if (result.ok || result.status !== 400) {
    throw new Error(`Expected invalid realtime usage duration to fail, got ${JSON.stringify(result)}`);
  }
}

export function assertRealtimeUsageBodyValidationAcceptsSnakeCaseDuration() {
  const result = validateRealtimeUsageBody({sessionId: ' session-1 ', duration_seconds: 12.2});
  if (!result.ok || result.value.sessionId !== 'session-1' || result.value.durationSeconds !== 13) {
    throw new Error(`Expected realtime usage body to normalize duration, got ${JSON.stringify(result)}`);
  }
}

export function assertRealtimeUsageClampsToExpiresAt() {
  const createdAt = new Date('2026-05-28T10:00:00.000Z');
  const expiresAt = new Date('2026-05-28T10:02:00.000Z');
  const now = new Date('2026-05-28T10:05:00.000Z');
  const clamped = clampRealtimeUsageSeconds({
    reportedSeconds: 500,
    createdAt,
    expiresAt,
    now,
  });
  if (clamped !== 120) {
    throw new Error(`Expected realtime usage to clamp to expiry, got ${clamped}`);
  }
}

export function assertRealtimeUsageClampsToElapsedTime() {
  const createdAt = new Date('2026-05-28T10:00:00.000Z');
  const expiresAt = new Date('2026-05-28T10:10:00.000Z');
  const now = new Date('2026-05-28T10:00:31.000Z');
  const clamped = clampRealtimeUsageSeconds({
    reportedSeconds: 120,
    createdAt,
    expiresAt,
    now,
  });
  if (clamped !== 31) {
    throw new Error(`Expected realtime usage to clamp to elapsed time, got ${clamped}`);
  }
}
