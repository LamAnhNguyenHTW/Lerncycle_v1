import {
  normalizeProcessingStatusPdfIds,
  summarizeProcessingStatuses,
} from './useProcessingStatus';
import type {SourceStatus} from '@/lib/processing-status';

export function assertProcessingStatusPdfIdsNormalizeForStablePolling() {
  const normalized = normalizeProcessingStatusPdfIds(['b', 'a', 'b', '', 'c']);
  if (JSON.stringify(normalized) !== JSON.stringify(['a', 'b', 'c'])) {
    throw new Error(`Expected sorted unique PDF IDs, got ${JSON.stringify(normalized)}`);
  }
}

export function assertProcessingStatusSummaryCountsReadiness() {
  const statuses: SourceStatus[] = [
    status('pdf-1', 'ready', true),
    status('pdf-2', 'partial_ready', true),
    status('pdf-3', 'failed', false),
  ];
  const summary = summarizeProcessingStatuses(statuses, 3, false);

  if (summary.ragReadyCount !== 2 || summary.totalCount !== 3) {
    throw new Error(`Expected 2 of 3 ready, got ${summary.ragReadyCount} of ${summary.totalCount}`);
  }
  if (summary.allRagReady || summary.allReady !== false || summary.anyFailed !== true) {
    throw new Error(`Unexpected summary flags: ${JSON.stringify(summary)}`);
  }
}

function status(
  id: string,
  overallStage: SourceStatus['overallStage'],
  ragReady: boolean,
): SourceStatus {
  return {
    sourceType: 'pdf',
    sourceId: id,
    pdfId: id,
    ragStage: ragReady ? 'completed' : 'failed',
    graphStage: 'disabled',
    overallStage,
    ragReady,
    graphReady: false,
    updatedAt: '2026-05-25T10:00:00.000Z',
    userSafeError: null,
  };
}
