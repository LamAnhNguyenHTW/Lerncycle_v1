'use client';

import {useEffect, useMemo, useRef, useState} from 'react';

import type {OverallStage, SourceStatus} from '@/lib/processing-status';

export type ProcessingStatusState = {
  statuses: SourceStatus[];
  ragReadyCount: number;
  totalCount: number;
  allRagReady: boolean;
  allReady: boolean;
  anyFailed: boolean;
  isLoading: boolean;
};

const FAST_POLL_MS = 3000;
const SLOW_POLL_MS = 15000;
const BACKOFF_AFTER_MS = 60000;
const TERMINAL_STAGES = new Set<OverallStage>(['ready', 'partial_ready', 'failed']);

export function useProcessingStatus(pdfIds?: string[]): ProcessingStatusState {
  const rawPdfIdsKey = (pdfIds ?? []).join('\x1f');
  const normalizedPdfIds = useMemo(
    () => normalizeProcessingStatusPdfIds(rawPdfIdsKey ? rawPdfIdsKey.split('\x1f') : []),
    [rawPdfIdsKey],
  );
  const pdfKey = normalizedPdfIds.join(',');
  const [statuses, setStatuses] = useState<SourceStatus[]>([]);
  const [isLoading, setIsLoading] = useState(normalizedPdfIds.length > 0);
  const lastUpdatedAtRef = useRef<string | null>(null);
  const lastSignatureRef = useRef<string>('');
  const lastChangeAtRef = useRef<number>(Date.now());
  const statusesRef = useRef<SourceStatus[]>([]);

  useEffect(() => {
    let cancelled = false;
    let timeoutId: ReturnType<typeof setTimeout> | null = null;

    lastUpdatedAtRef.current = null;
    lastSignatureRef.current = '';
    lastChangeAtRef.current = Date.now();
    statusesRef.current = [];
    setStatuses([]);
    setIsLoading(normalizedPdfIds.length > 0);

    if (normalizedPdfIds.length === 0) {
      setIsLoading(false);
      return () => {
        cancelled = true;
      };
    }

    const schedule = (delay: number) => {
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
      timeoutId = setTimeout(fetchStatuses, delay);
    };

    const fetchStatuses = async () => {
      if (cancelled) {
        return;
      }
      if (document.visibilityState !== 'visible') {
        schedule(FAST_POLL_MS);
        return;
      }

      const params = new URLSearchParams({pdfIds: normalizedPdfIds.join(',')});
      if (lastUpdatedAtRef.current) {
        params.set('since', lastUpdatedAtRef.current);
      }

      try {
        const response = await fetch(`/api/processing-status?${params.toString()}`);
        if (!response.ok) {
          throw new Error(`Processing status request failed: ${response.status}`);
        }
        const incoming = await response.json() as SourceStatus[];
        if (cancelled) {
          return;
        }

        setStatuses((current) => {
          const merged = mergeStatuses(current, incoming, normalizedPdfIds);
          statusesRef.current = merged;
          const signature = statusSignature(merged);
          if (signature !== lastSignatureRef.current) {
            lastSignatureRef.current = signature;
            lastChangeAtRef.current = Date.now();
          }
          lastUpdatedAtRef.current = maxUpdatedAt(lastUpdatedAtRef.current, incoming);
          return merged;
        });
        setIsLoading(false);

        const nextStatuses = statusesRef.current;
        if (nextStatuses.length === normalizedPdfIds.length && nextStatuses.every(isTerminalStatus)) {
          return;
        }
        const delay = Date.now() - lastChangeAtRef.current > BACKOFF_AFTER_MS
          ? SLOW_POLL_MS
          : FAST_POLL_MS;
        schedule(delay);
      } catch {
        setIsLoading(false);
        schedule(SLOW_POLL_MS);
      }
    };

    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible') {
        schedule(0);
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    schedule(0);

    return () => {
      cancelled = true;
      document.removeEventListener('visibilitychange', handleVisibilityChange);
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
    };
  }, [normalizedPdfIds, pdfKey]);

  return summarizeProcessingStatuses(statuses, normalizedPdfIds.length, isLoading);
}

export function normalizeProcessingStatusPdfIds(pdfIds?: string[]): string[] {
  return [...new Set(pdfIds ?? [])].filter(Boolean).sort();
}

export function summarizeProcessingStatuses(
  statuses: SourceStatus[],
  totalCount: number = statuses.length,
  isLoading = false,
): ProcessingStatusState {
  const ragReadyCount = statuses.filter((status) => status.ragReady).length;
  const allRagReady = totalCount > 0 && ragReadyCount === totalCount;
  const allReady = totalCount > 0
    && statuses.length === totalCount
    && statuses.every((status) => status.overallStage === 'ready');
  const anyFailed = statuses.some((status) => status.overallStage === 'failed');

  return {
    statuses,
    ragReadyCount,
    totalCount,
    allRagReady,
    allReady,
    anyFailed,
    isLoading,
  };
}

function mergeStatuses(
  current: SourceStatus[],
  incoming: SourceStatus[],
  pdfIds: string[],
): SourceStatus[] {
  const byId = new Map(current.map((status) => [status.sourceId, status]));
  for (const status of incoming) {
    byId.set(status.sourceId, status);
  }
  return pdfIds.flatMap((id) => {
    const status = byId.get(id);
    return status ? [status] : [];
  });
}

function maxUpdatedAt(current: string | null, statuses: SourceStatus[]): string | null {
  let max = current;
  for (const status of statuses) {
    if (!max || status.updatedAt > max) {
      max = status.updatedAt;
    }
  }
  return max;
}

function statusSignature(statuses: SourceStatus[]): string {
  return statuses
    .map((status) => `${status.sourceId}:${status.overallStage}:${status.updatedAt}`)
    .join('|');
}

function isTerminalStatus(status: SourceStatus): boolean {
  return TERMINAL_STAGES.has(status.overallStage);
}
