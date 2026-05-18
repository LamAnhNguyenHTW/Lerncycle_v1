import {encodeSse, parseSseEvents, ragStreamEndpoint, wantsStreaming} from '../route';

export function assertStreamingOptInUsesAcceptHeaderOrBodyFlag() {
  const jsonRequest = new Request('http://localhost/api/chat', {
    headers: {accept: 'application/json'},
  });
  const sseRequest = new Request('http://localhost/api/chat', {
    headers: {accept: 'text/event-stream'},
  });

  if (wantsStreaming(jsonRequest, {})) {
    throw new Error('JSON chat requests must keep the non-streaming response path.');
  }
  if (!wantsStreaming(sseRequest, {})) {
    throw new Error('SSE Accept header should opt into streaming.');
  }
  if (!wantsStreaming(jsonRequest, {stream: true})) {
    throw new Error('stream: true should opt into streaming.');
  }
}

export function assertStreamEndpointUsesDebugTimingOnlyWhenAllowed() {
  const production = ragStreamEndpoint('http://rag.local/', {NODE_ENV: 'production'} as NodeJS.ProcessEnv);
  if (production.includes('debug_timing')) {
    throw new Error(`Production stream endpoint leaked debug timing: ${production}`);
  }

  const development = ragStreamEndpoint('http://rag.local/', {NODE_ENV: 'development'} as NodeJS.ProcessEnv);
  if (!development.endsWith('/rag/answer/stream?debug_timing=1')) {
    throw new Error(`Development stream endpoint should request debug timing: ${development}`);
  }
}

export function assertSseParserHandlesPartialBlocksAndMultipleEvents() {
  const first = encodeSse({event_type: 'token', content: 'Hel'});
  const second = encodeSse({event_type: 'token', content: 'lo'});
  const partial = `${first}${second.slice(0, -2)}`;

  const parsed = parseSseEvents(partial);
  if (parsed.events.length !== 1 || parsed.events[0]?.event_type !== 'token') {
    throw new Error('Expected parser to emit only complete SSE blocks.');
  }
  if (!parsed.remainder.includes('"lo"')) {
    throw new Error('Expected parser to retain the partial SSE block.');
  }

  const completed = parseSseEvents(`${parsed.remainder}\n\n`);
  if (completed.events.length !== 1 || completed.events[0]?.event_type !== 'token') {
    throw new Error('Expected parser to emit the completed remainder block.');
  }
}
