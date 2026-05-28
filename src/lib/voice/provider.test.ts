import type {RealtimeSessionConfig, VoiceProvider} from './provider';
import {GeminiVoiceProvider} from './gemini';

export function assertVoiceProviderContractAcceptsCoreMethods() {
  const provider: VoiceProvider = {
    async transcribe() {
      return {
        text: 'Hallo Welt',
        durationSeconds: 2,
        model: 'test-stt',
      };
    },
    async synthesize() {
      return {
        audio: new Blob(['audio'], {type: 'audio/mpeg'}),
        characterCount: 10,
        model: 'test-tts',
      };
    },
  };

  if (typeof provider.transcribe !== 'function' || typeof provider.synthesize !== 'function') {
    throw new Error('VoiceProvider must expose transcribe and synthesize.');
  }
}

export function assertRealtimeSessionConfigShape() {
  const config: RealtimeSessionConfig = {
    clientSecret: 'ephemeral',
    expiresAt: '2026-05-25T12:00:00.000Z',
    model: 'realtime-model',
    tools: [],
  };

  if (!config.clientSecret || !config.expiresAt || !config.model || !Array.isArray(config.tools)) {
    throw new Error('RealtimeSessionConfig must carry clientSecret, expiresAt, model, and tools.');
  }
}

export function assertGeminiVoiceProviderSatisfiesInterfaceAsStub() {
  const provider: VoiceProvider = new GeminiVoiceProvider();
  if (
    typeof provider.transcribe !== 'function' ||
    typeof provider.synthesize !== 'function' ||
    typeof provider.createRealtimeSession !== 'function'
  ) {
    throw new Error('GeminiVoiceProvider must satisfy VoiceProvider as a stub.');
  }
}
