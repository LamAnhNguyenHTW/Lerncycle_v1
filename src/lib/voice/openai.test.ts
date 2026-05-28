import {
  OpenAIVoiceProvider,
  hashedSafetyIdentifier,
  parseOpenAIRealtimeSessionResponse,
  parseOpenAITranscriptionResponse,
} from './openai';

export function assertOpenAITranscriptionResponseParsing() {
  const parsed = parseOpenAITranscriptionResponse(
    {text: '  Hallo Welt  ', duration: 2.4, language: 'de'},
    'gpt-4o-mini-transcribe',
  );

  if (parsed.text !== 'Hallo Welt' || parsed.durationSeconds !== 3 || parsed.language !== 'de') {
    throw new Error(`Unexpected transcription parse result: ${JSON.stringify(parsed)}`);
  }
  if (parsed.model !== 'gpt-4o-mini-transcribe') {
    throw new Error(`Expected model to be preserved, got ${parsed.model}.`);
  }
}

export function assertOpenAIProviderRequiresApiKey() {
  let threw = false;
  try {
    new OpenAIVoiceProvider({apiKey: '', sttModel: 'stt', ttsModel: 'tts'});
  } catch {
    threw = true;
  }

  if (!threw) {
    throw new Error('OpenAIVoiceProvider must reject missing API keys.');
  }
}

export async function assertOpenAISynthesizeReturnsSpeechResult() {
  const provider = new OpenAIVoiceProvider({
    apiKey: 'sk-test',
    sttModel: 'stt',
    ttsModel: 'tts',
    fetchImpl: async () => new Response(new Blob(['audio'], {type: 'audio/mpeg'}), {
      status: 200,
      headers: {'content-type': 'audio/mpeg'},
    }),
  });

  const result = await provider.synthesize('Hallo Welt', {voice: 'alloy', speed: 1});
  if (result.characterCount !== 10 || result.model !== 'tts' || result.audio.type !== 'audio/mpeg') {
    throw new Error(`Unexpected speech result: ${JSON.stringify({characterCount: result.characterCount, model: result.model, type: result.audio.type})}`);
  }
}

export function assertOpenAIRealtimeSessionResponseParsing() {
  const parsed = parseOpenAIRealtimeSessionResponse(
    {
      value: 'ek_test',
      expires_at: 1_800_000_000,
      session: {model: 'gpt-realtime-2'},
    },
    'gpt-realtime-2',
    [],
  );

  if (parsed.clientSecret !== 'ek_test' || parsed.model !== 'gpt-realtime-2') {
    throw new Error(`Unexpected realtime parse result: ${JSON.stringify(parsed)}`);
  }
}

export async function assertOpenAIRealtimeSessionCreationUsesClientSecretsEndpoint() {
  let requestUrl = '';
  let requestInit: RequestInit | undefined;
  const provider = new OpenAIVoiceProvider({
    apiKey: 'sk-test',
    sttModel: 'stt',
    ttsModel: 'tts',
    realtimeModel: 'gpt-realtime-2',
    fetchImpl: async (url, init) => {
      requestUrl = String(url);
      requestInit = init;
      return Response.json({value: 'ek_test', expires_at: 1_800_000_000});
    },
  });

  const result = await provider.createRealtimeSession({
    userId: 'user-1',
    voice: 'alloy',
    instructions: 'Continue the LearnCycle session.',
    tools: [],
    expiresAfterSeconds: 42,
  });

  const headers = requestInit?.headers as Record<string, string>;
  const body = JSON.parse(String(requestInit?.body)) as {
    expires_after: {anchor: string; seconds: number};
    session: {
      model: string;
      type: string;
      audio: {input: {turn_detection: {type: string; eagerness: string; interrupt_response: boolean}}};
    };
  };
  if (!requestUrl.endsWith('/v1/realtime/client_secrets')) {
    throw new Error(`Unexpected realtime URL: ${requestUrl}`);
  }
  if (headers.Authorization !== 'Bearer sk-test') {
    throw new Error('Expected OpenAI API key to be used only server-side.');
  }
  if (headers['OpenAI-Safety-Identifier'] !== await hashedSafetyIdentifier('user-1')) {
    throw new Error('Expected hashed safety identifier header.');
  }
  if (body.session.type !== 'realtime' || body.session.model !== 'gpt-realtime-2') {
    throw new Error(`Unexpected realtime session body: ${JSON.stringify(body)}`);
  }
  if (body.expires_after.seconds !== 42) {
    throw new Error(`Expected realtime expiry to be clamped from quota, got ${JSON.stringify(body.expires_after)}`);
  }
  if (
    body.session.audio.input.turn_detection.type !== 'semantic_vad' ||
    body.session.audio.input.turn_detection.eagerness !== 'medium' ||
    body.session.audio.input.turn_detection.interrupt_response !== true
  ) {
    throw new Error(`Expected semantic VAD realtime turn detection, got ${JSON.stringify(body.session.audio.input.turn_detection)}`);
  }
  if (result.clientSecret !== 'ek_test') {
    throw new Error(`Unexpected realtime session result: ${JSON.stringify(result)}`);
  }
}
