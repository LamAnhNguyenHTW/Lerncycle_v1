import {
  AUDIO_MIME_TYPES,
  parseRecordingSeconds,
  validateVoiceUpload,
} from './route';

export function assertVoiceTranscribeMimeAllowlist() {
  if (!AUDIO_MIME_TYPES.has('audio/webm') || !AUDIO_MIME_TYPES.has('audio/mp4')) {
    throw new Error('Expected webm and mp4 audio MIME types to be allowed.');
  }
  const accepted = validateVoiceUpload({
    contentLength: 10,
    maxUploadBytes: 100,
    file: new File(['audio'], 'voice.webm', {type: 'audio/webm'}),
  });
  if (!accepted.ok) {
    throw new Error(`Expected valid upload, got ${accepted.error}`);
  }
}

export function assertVoiceTranscribeRejectsOversizedUpload() {
  const accepted = validateVoiceUpload({
    contentLength: 101,
    maxUploadBytes: 100,
    file: new File(['audio'], 'voice.webm', {type: 'audio/webm'}),
  });
  if (accepted.ok || accepted.status !== 413) {
    throw new Error(`Expected 413 oversized rejection, got ${JSON.stringify(accepted)}`);
  }
}

export function assertVoiceTranscribeRejectsDisallowedMime() {
  const accepted = validateVoiceUpload({
    contentLength: 10,
    maxUploadBytes: 100,
    file: new File(['not audio'], 'voice.txt', {type: 'text/plain'}),
  });
  if (accepted.ok || accepted.status !== 415) {
    throw new Error(`Expected 415 MIME rejection, got ${JSON.stringify(accepted)}`);
  }
}

export function assertVoiceTranscribeTreatsRecordingSecondsAsAdvisory() {
  if (parseRecordingSeconds('12.5') !== 12.5) {
    throw new Error('Expected numeric recording_seconds form field to parse.');
  }
  if (parseRecordingSeconds('nope') !== undefined) {
    throw new Error('Expected invalid recording_seconds to be ignored.');
  }
}
