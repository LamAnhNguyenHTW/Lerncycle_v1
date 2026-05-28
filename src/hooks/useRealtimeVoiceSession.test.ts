import {
  createRealtimeToolOutputEvent,
  appendRealtimeTranscriptTurn,
  extractRealtimeOutputText,
  extractRealtimeTranscriptEvent,
  extractRealtimeToolCall,
  isLikelyRealtimeTranscriptHallucination,
  isBenignRealtimeDisconnectError,
  parseToolArguments,
} from './useRealtimeVoiceSession';

export function assertRealtimeToolArgumentsParseJson() {
  const parsed = parseToolArguments('{"query":"Process Mining","top_k":4}');
  if (parsed.query !== 'Process Mining' || parsed.top_k !== 4) {
    throw new Error(`Unexpected realtime tool arguments: ${JSON.stringify(parsed)}`);
  }
}

export function assertRealtimeToolCallExtractionSupportsOutputItemEvents() {
  const call = extractRealtimeToolCall({
    type: 'response.output_item.done',
    item: {
      type: 'function_call',
      name: 'search_learncycle_context',
      call_id: 'call-1',
      arguments: '{"query":"Ist-Prozess"}',
    },
  });

  if (!call || call.callId !== 'call-1' || call.arguments.query !== 'Ist-Prozess') {
    throw new Error(`Expected tool call extraction, got ${JSON.stringify(call)}`);
  }
}

export function assertRealtimeToolOutputEventUsesFunctionCallOutput() {
  const event = createRealtimeToolOutputEvent('call-1', {sources: []});
  if (event.item.type !== 'function_call_output' || event.item.call_id !== 'call-1') {
    throw new Error(`Unexpected tool output event: ${JSON.stringify(event)}`);
  }
}

export function assertRealtimeTranscriptExtractionReadsUserAudioTranscripts() {
  const event = extractRealtimeTranscriptEvent({
    type: 'conversation.item.input_audio_transcription.completed',
    transcript: 'Hallo, ich erkläre den Ist-Prozess.',
  });
  if (!event || event.role !== 'user' || event.text !== 'Hallo, ich erkläre den Ist-Prozess.') {
    throw new Error(`Unexpected user transcript event: ${JSON.stringify(event)}`);
  }
}

export function assertRealtimeTranscriptExtractionReadsAssistantDeltas() {
  const event = extractRealtimeTranscriptEvent({
    type: 'response.audio_transcript.delta',
    response_id: 'response-1',
    delta: 'Ohh, ',
  });
  if (!event || event.role !== 'assistant' || event.final || event.responseId !== 'response-1') {
    throw new Error(`Unexpected assistant transcript event: ${JSON.stringify(event)}`);
  }
}

export function assertRealtimeTranscriptExtractionFiltersSilenceHallucinations() {
  const event = extractRealtimeTranscriptEvent({
    type: 'conversation.item.input_audio_transcription.completed',
    transcript: 'Bye.',
  });
  if (event !== null || !isLikelyRealtimeTranscriptHallucination('So this session car was some ant bites')) {
    throw new Error(`Expected silence hallucinations to be filtered, got ${JSON.stringify(event)}`);
  }
}

export function assertRealtimeTranscriptExtractionReadsOutputItemDone() {
  const event = extractRealtimeTranscriptEvent({
    type: 'response.output_item.done',
    item: {
      type: 'message',
      content: [
        {
          type: 'audio',
          transcript: 'Ohh, Verzerrungen sind also Abweichungen?',
        },
      ],
    },
  });
  if (!event || event.role !== 'assistant' || !event.final || !event.text.includes('Verzerrungen')) {
    throw new Error(`Expected assistant output item transcript, got ${JSON.stringify(event)}`);
  }
}

export function assertRealtimeOutputTextReadsResponseDoneShape() {
  const text = extractRealtimeOutputText({
    output: [
      {
        type: 'message',
        content: [
          {type: 'output_audio', transcript: 'Ich glaube, ich verstehe.'},
          {type: 'output_text', text: 'Was ist ein Beispiel?'},
        ],
      },
    ],
  });
  if (!text.includes('Ich glaube') || !text.includes('Beispiel')) {
    throw new Error(`Expected response.done output text, got ${text}`);
  }
}

export function assertRealtimeBenignDisconnectErrorsAreRecognized() {
  if (!isBenignRealtimeDisconnectError('Peer connection is closed')) {
    throw new Error('Expected peer connection close errors to be treated as benign.');
  }
}

export function assertRealtimeTranscriptTurnsAreDedupedAndNormalized() {
  const turns = appendRealtimeTranscriptTurn(
    appendRealtimeTranscriptTurn([], {role: 'user', text: ' Hallo   Welt '}),
    {role: 'user', text: 'Hallo Welt'},
  );

  if (turns.length !== 1 || turns[0].text !== 'Hallo Welt') {
    throw new Error(`Expected realtime transcript turn normalization, got ${JSON.stringify(turns)}`);
  }
}
