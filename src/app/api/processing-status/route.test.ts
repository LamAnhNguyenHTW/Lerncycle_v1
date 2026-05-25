import {
  parseProcessingStatusQuery,
  shouldLoadMissingPdfStatuses,
  sourceStatusFromRawRow,
  synthesizePdfStatus,
} from './route';

export function assertProcessingStatusQueryValidation() {
  const parsedPdfIds = parseProcessingStatusQuery(new URLSearchParams('pdfIds=b,a,a'));
  if (!parsedPdfIds.ok) {
    throw new Error(`Expected pdfIds query to parse: ${parsedPdfIds.error}`);
  }
  if (JSON.stringify(parsedPdfIds.value.pdfIds) !== JSON.stringify(['b', 'a'])) {
    throw new Error(`Expected de-duped pdfIds, got ${JSON.stringify(parsedPdfIds.value.pdfIds)}`);
  }
  if (parsedPdfIds.value.sourceType !== 'pdf') {
    throw new Error('Expected pdfIds to imply sourceType=pdf.');
  }

  const invalidSourceType = parseProcessingStatusQuery(new URLSearchParams('sourceType=web'));
  if (invalidSourceType.ok) {
    throw new Error('Expected invalid sourceType to be rejected.');
  }

  const invalidSince = parseProcessingStatusQuery(new URLSearchParams('since=not-a-date'));
  if (invalidSince.ok) {
    throw new Error('Expected invalid since timestamp to be rejected.');
  }
}

export function assertRawStatusMapsToSafeSourceStatus() {
  const status = sourceStatusFromRawRow(
    {
      user_id: 'user-1',
      source_type: 'pdf',
      source_id: 'pdf-1',
      pdf_id: 'pdf-1',
      rag_stage: 'indexing_dense',
      rag_status: 'processing',
      rag_stage_error: 'OpenAI failed at C:\\secret\\.env',
      graph_stage: null,
      graph_status: null,
      graph_stage_error: null,
      has_chunks: false,
      updated_at: '2026-05-25T10:00:00.000Z',
    },
    {graphEnabled: false, lang: 'de'},
  );

  if (status.ragStage !== 'indexing' || status.overallStage !== 'indexing') {
    throw new Error(`Expected indexing status, got ${status.ragStage}/${status.overallStage}`);
  }
  if (JSON.stringify(status).includes('C:\\secret')) {
    throw new Error('Raw stage_error leaked into response payload.');
  }
  if (status.userSafeError !== 'Indexierung temporaer nicht verfuegbar - bitte erneut versuchen.') {
    throw new Error(`Expected sanitized OpenAI error, got ${status.userSafeError}`);
  }
}

export function assertGraphFailureMapsToPartialReady() {
  const status = sourceStatusFromRawRow(
    {
      user_id: 'user-1',
      source_type: 'pdf',
      source_id: 'pdf-1',
      pdf_id: 'pdf-1',
      rag_stage: 'completed',
      rag_status: 'completed',
      rag_stage_error: null,
      graph_stage: 'failed',
      graph_status: 'failed',
      graph_stage_error: 'Neo4j password=secret',
      has_chunks: true,
      updated_at: '2026-05-25T10:00:00.000Z',
    },
    {graphEnabled: true, lang: 'en'},
  );

  if (status.overallStage !== 'partial_ready' || !status.ragReady || status.graphReady) {
    throw new Error(`Expected partial_ready with RAG ready, got ${JSON.stringify(status)}`);
  }
  if (JSON.stringify(status).includes('password=secret')) {
    throw new Error('Raw graph error leaked into response payload.');
  }
  if (status.userSafeError !== 'Knowledge graph could not be generated.') {
    throw new Error(`Expected sanitized graph error, got ${status.userSafeError}`);
  }
}

export function assertMissingPdfSynthesisPolicy() {
  const ready = synthesizePdfStatus(
    {id: 'pdf-1', created_at: '2026-05-25T10:00:00.000Z'},
    true,
    {graphEnabled: false},
  );
  if (ready.overallStage !== 'ready' || !ready.ragReady) {
    throw new Error(`Expected missing PDF with chunks to synthesize ready, got ${JSON.stringify(ready)}`);
  }

  const queued = synthesizePdfStatus(
    {id: 'pdf-2', created_at: '2026-05-25T10:00:00.000Z'},
    false,
    {graphEnabled: false},
  );
  if (queued.overallStage !== 'queued' || queued.ragReady) {
    throw new Error(`Expected missing PDF without chunks to synthesize queued, got ${JSON.stringify(queued)}`);
  }
}

export function assertMissingPdfSynthesisOnlyRunsOnFullFetch() {
  if (!shouldLoadMissingPdfStatuses({pdfIds: ['pdf-1']})) {
    throw new Error('Expected missing PDF fallback on the initial full status fetch.');
  }

  if (shouldLoadMissingPdfStatuses({pdfIds: ['pdf-1'], since: '2026-05-25T10:00:00.000Z'})) {
    throw new Error('Expected since polling to preserve unchanged statuses instead of synthesizing queued fallbacks.');
  }
}
