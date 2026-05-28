import {
  buildRealtimeChatMessageRows,
  validateRealtimeSummaryBody,
} from './route';

export function assertRealtimeSummaryValidationRejectsRawAudioLikeBodies() {
  const result = validateRealtimeSummaryBody({
    sessionId: 'session-1',
    audio: 'base64-audio',
    turns: [],
  });

  if (result.ok || result.error !== 'No transcript text to persist.') {
    throw new Error(`Expected empty text-only turns to be rejected, got ${JSON.stringify(result)}`);
  }
}

export function assertRealtimeSummaryValidationKeepsTextTurnsOnly() {
  const result = validateRealtimeSummaryBody({
    sessionId: 'session-1',
    turns: [
      {role: 'user', text: '  Hallo   Welt  '},
      {role: 'assistant', text: 'Was meinst du genau?'},
      {role: 'tool', text: 'must be ignored'},
      {role: 'user', audio: 'must be ignored'},
    ],
  });

  if (!result.ok || result.value.turns.length !== 2 || result.value.turns[0].text !== 'Hallo Welt') {
    throw new Error(`Expected text transcript turns to validate, got ${JSON.stringify(result)}`);
  }
}

export function assertRealtimeSummaryBuildsChatMessageRows() {
  const rows = buildRealtimeChatMessageRows({
    sessionId: 'session-1',
    userId: 'user-1',
    turns: [
      {role: 'user', text: 'Ich erklaere Process Mining.'},
      {role: 'assistant', text: 'Was ist ein Event Log?'},
    ],
  });

  if (
    rows.length !== 2 ||
    rows[0].role !== 'user' ||
    rows[1].role !== 'assistant' ||
    rows[0].content !== 'Ich erklaere Process Mining.'
  ) {
    throw new Error(`Expected individual chat message rows, got ${JSON.stringify(rows)}`);
  }
}

export function assertRealtimeSummaryRowsNeverStoreAudio() {
  const rows = buildRealtimeChatMessageRows({
    sessionId: 'session-1',
    userId: 'user-1',
    turns: [
      {role: 'user', text: 'Ich erklaere Process Mining.'},
      {role: 'assistant', text: 'Was ist ein Event Log?'},
    ],
  });

  if (
    JSON.stringify(rows).includes('audio') ||
    !rows.every((row) => row.input_metadata?.input_type === 'voice' && row.input_metadata.realtime_transcript === true)
  ) {
    throw new Error(`Expected text-only realtime metadata, got ${JSON.stringify(rows)}`);
  }
}

export function assertRealtimeSummaryRowsPreserveFinishCommandText() {
  const rows = buildRealtimeChatMessageRows({
    sessionId: 'session-1',
    userId: 'user-1',
    turns: [
      {role: 'user', text: '/fertig'},
      {role: 'assistant', text: 'Was ist ein Event Log?'},
    ],
  });

  if (rows[0].content !== '/fertig') {
    throw new Error(`Expected finish command to remain available to chat history, got ${JSON.stringify(rows)}`);
  }
}
