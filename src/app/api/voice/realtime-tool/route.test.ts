import {
  compactRealtimeSources,
  noRealtimeResultsResponse,
  validateRealtimeToolBody,
} from './route';

export function assertRealtimeToolBodyValidationRequiresSessionAndQuery() {
  const result = validateRealtimeToolBody({sessionId: 'session-1'});
  if (!result.ok || !result.value.query.includes('Ist-Prozess')) {
    throw new Error(`Expected realtime tool body to default missing query, got ${JSON.stringify(result)}`);
  }
}

export function assertRealtimeToolBodyValidationAllowsMissingSession() {
  const result = validateRealtimeToolBody({query: 'Process Mining', pdf_ids: ['pdf-1']});
  if (!result.ok || result.value.sessionId !== '') {
    throw new Error(`Expected realtime tool body validation to leave missing sessionId empty, got ${JSON.stringify(result)}`);
  }
}

export function assertRealtimeToolBodyValidationAcceptsQuestionAlias() {
  const result = validateRealtimeToolBody({question: 'Was startet die Einsatzplanung?'});
  if (!result.ok || result.value.query !== 'Was startet die Einsatzplanung?') {
    throw new Error(`Expected question alias to validate, got ${JSON.stringify(result)}`);
  }
}

export function assertRealtimeToolBodyValidationClampsTopK() {
  const result = validateRealtimeToolBody({
    sessionId: 'session-1',
    query: 'Ist-Prozess',
    top_k: 99,
    pdf_ids: ['pdf-1'],
  });
  if (!result.ok || result.value.topK !== 5 || 'pdfIds' in result.value) {
    throw new Error(`Expected top_k to clamp to 5 and ignore browser pdf_ids, got ${JSON.stringify(result)}`);
  }
}

export function assertRealtimeToolCompactsSources() {
  const sources = compactRealtimeSources([{
    pdf_id: 'pdf-1',
    filename: 'Alpha.pdf',
    page_index: 0,
    content: ' A '.repeat(400),
  }]);
  if (sources.length !== 1 || sources[0].source_id !== 'pdf-1' || sources[0].page !== 1 || sources[0].snippet.length > 500) {
    throw new Error(`Expected compact realtime source, got ${JSON.stringify(sources)}`);
  }
}

export function assertRealtimeToolNoResultsShape() {
  const result = noRealtimeResultsResponse();
  if (!result.no_results || result.sources.length !== 0 || !result.answer.includes('Dokumenten')) {
    throw new Error(`Expected no_results response shape, got ${JSON.stringify(result)}`);
  }
}
