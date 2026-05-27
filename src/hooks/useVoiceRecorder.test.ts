import {chooseVoiceRecorderMimeType} from './useVoiceRecorder';

export function assertVoiceRecorderPrefersWebmOpus() {
  const selected = chooseVoiceRecorderMimeType({
    isTypeSupported: (mimeType: string) => mimeType === 'audio/webm;codecs=opus',
  });

  if (selected !== 'audio/webm;codecs=opus') {
    throw new Error(`Expected webm/opus to be preferred, got ${selected ?? 'none'}.`);
  }
}

export function assertVoiceRecorderFallsBackForSafari() {
  const selected = chooseVoiceRecorderMimeType({
    isTypeSupported: (mimeType: string) => mimeType === 'audio/mp4',
  });

  if (selected !== 'audio/mp4') {
    throw new Error(`Expected Safari mp4 fallback, got ${selected ?? 'none'}.`);
  }
}

export function assertVoiceRecorderHandlesMissingMediaRecorderSupport() {
  const selected = chooseVoiceRecorderMimeType(undefined);

  if (selected !== null) {
    throw new Error(`Expected missing MediaRecorder support to return null, got ${selected}.`);
  }
}
