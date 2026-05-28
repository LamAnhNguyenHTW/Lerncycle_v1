import {
  realtimeInstructions,
  realtimeRagToolDefinition,
  validateRealtimeTokenBody,
} from './route';

export function assertRealtimeTokenBodyDefaultsToFeynman() {
  const result = validateRealtimeTokenBody(null);
  if (!result.ok || result.value.mode !== 'feynman') {
    throw new Error(`Expected empty body to default to Feynman, got ${JSON.stringify(result)}`);
  }
}

export function assertRealtimeTokenBodyAcceptsSessionId() {
  const result = validateRealtimeTokenBody({sessionId: ' session-1 ', mode: 'feynman'});
  if (!result.ok || result.value.sessionId !== 'session-1' || result.value.mode !== 'feynman') {
    throw new Error(`Expected sessionId and Feynman mode to validate, got ${JSON.stringify(result)}`);
  }
}

export function assertRealtimeTokenBodyAcceptsCourseIdForNewSession() {
  const result = validateRealtimeTokenBody({course_id: ' course-1 ', mode: 'feynman'});
  if (!result.ok || result.value.courseId !== 'course-1' || result.value.sessionId !== undefined) {
    throw new Error(`Expected courseId for a new realtime session, got ${JSON.stringify(result)}`);
  }
}

export function assertRealtimeTokenBodyRejectsNonFeynmanMode() {
  const result = validateRealtimeTokenBody({sessionId: 'session-1', mode: 'guided_learning'});
  if (result.ok) {
    throw new Error('Expected realtime voice token body to reject non-Feynman modes.');
  }
}

export function assertRealtimeInstructionsRequireRagToolForMaterial() {
  const instructions = realtimeInstructions(['Ist-Prozess.pdf'], 'Preview 1 (page 1): Ist-Prozess bedeutet...');
  if (
    !instructions.includes('search_learncycle_context') ||
    !instructions.includes('Force retrieval rule') ||
    !instructions.includes('Ist-Prozess.pdf') ||
    !instructions.includes('Ist-Prozess bedeutet')
  ) {
    throw new Error(`Expected realtime instructions to require the RAG tool, got ${instructions}`);
  }
}

export function assertRealtimeRagToolHasNarrowSchema() {
  const tool = realtimeRagToolDefinition();
  if (tool.name !== 'search_learncycle_context' || tool.parameters.additionalProperties !== false) {
    throw new Error(`Unexpected realtime RAG tool definition: ${JSON.stringify(tool)}`);
  }
}
