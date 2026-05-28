import {getVoiceConfig, getVoiceServerConfig} from './config';

export function assertVoiceConfigDefaultsAreSafe() {
  const publicConfig = getVoiceConfig({});

  if (publicConfig.enabled !== false) {
    throw new Error('Voice mode must default to disabled.');
  }
  if (publicConfig.maxRecordingSeconds !== 60) {
    throw new Error(`Expected default max recording seconds to be 60, got ${publicConfig.maxRecordingSeconds}.`);
  }
  if (publicConfig.maxUploadBytes !== 5_242_880) {
    throw new Error(`Expected default max upload bytes to be 5242880, got ${publicConfig.maxUploadBytes}.`);
  }
  if (JSON.stringify(publicConfig.enabledModes) !== JSON.stringify(['feynman'])) {
    throw new Error(`Expected MVP voice mode to be feynman-only, got ${JSON.stringify(publicConfig.enabledModes)}.`);
  }
  if (publicConfig.ttsMaxChars !== 1000) {
    throw new Error(`Expected default TTS max chars to be 1000, got ${publicConfig.ttsMaxChars}.`);
  }
  assertNoServerOnlyVoiceConfigLeak(publicConfig);
}

export function assertVoiceConfigParsesEnvOverrides() {
  const env = {
    VOICE_MODE_ENABLED: 'true',
    VOICE_PROVIDER: 'openai',
    VOICE_STT_MODEL: 'custom-stt',
    VOICE_TTS_MODEL: 'custom-tts',
    VOICE_MAX_RECORDING_SECONDS: '45',
    VOICE_MAX_UPLOAD_BYTES: '123456',
    VOICE_DAILY_MINUTES_PER_USER: '7',
    VOICE_DAILY_TTS_RESPONSES_PER_USER: '9',
    VOICE_TTS_MAX_CHARS: '800',
    OPENAI_API_KEY: 'sk-test-secret',
  };

  const publicConfig = getVoiceConfig(env);
  const serverConfig = getVoiceServerConfig(env);

  if (!publicConfig.enabled || !serverConfig.enabled) {
    throw new Error('VOICE_MODE_ENABLED=true must enable both public and server config.');
  }
  if (serverConfig.provider !== 'openai') {
    throw new Error(`Expected provider openai, got ${serverConfig.provider}.`);
  }
  if (serverConfig.sttModel !== 'custom-stt' || serverConfig.ttsModel !== 'custom-tts') {
    throw new Error(`Expected model overrides, got ${serverConfig.sttModel}/${serverConfig.ttsModel}.`);
  }
  if (serverConfig.dailyMinutesPerUser !== 7 || serverConfig.dailyTtsResponsesPerUser !== 9) {
    throw new Error('Expected daily quota overrides to be parsed.');
  }
  if (serverConfig.openaiApiKey !== 'sk-test-secret') {
    throw new Error('Expected server config to keep the server-only OpenAI API key.');
  }
  assertNoServerOnlyVoiceConfigLeak(publicConfig);
}

function assertNoServerOnlyVoiceConfigLeak(value: object) {
  const serialized = JSON.stringify(value);
  for (const forbidden of ['model', 'custom-stt', 'custom-tts', 'sk-test-secret', 'dailyMinutesPerUser']) {
    if (serialized.includes(forbidden)) {
      throw new Error(`Public voice config leaked server-only value: ${forbidden}`);
    }
  }
}
