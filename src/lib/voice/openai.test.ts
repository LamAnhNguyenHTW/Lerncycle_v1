import {OpenAIVoiceProvider, parseOpenAITranscriptionResponse} from './openai';

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
