import {
  resolveVoiceStyle,
  truncateTextForSpeech,
  validateSpeechBody,
} from './route';

export function assertSpeechBodyValidationAcceptsMessageId() {
  const result = validateSpeechBody({messageId: 'message-1'});
  if (!result.ok || !('messageId' in result.value) || result.value.messageId !== 'message-1') {
    throw new Error(`Expected messageId body to validate, got ${JSON.stringify(result)}`);
  }
}

export function assertSpeechBodyValidationAcceptsTextFallback() {
  const result = validateSpeechBody({text: 'Hallo', mode: 'feynman', sessionId: 'session-1'});
  if (!result.ok || !('text' in result.value) || result.value.text !== 'Hallo') {
    throw new Error(`Expected fallback body to validate, got ${JSON.stringify(result)}`);
  }
}

export function assertSpeechTruncatesAtSentenceBoundary() {
  const text = 'Erster Satz. Zweiter Satz ist viel zu lang und wird abgeschnitten.';
  const truncated = truncateTextForSpeech(text, 20);
  if (truncated !== 'Hier eine kurze gesprochene Zusammenfassung. Erster Satz.') {
    throw new Error(`Unexpected truncated text: ${truncated}`);
  }
}

export function assertSpeechVoiceStyleByMode() {
  if (resolveVoiceStyle('feynman').speed >= resolveVoiceStyle('normal').speed) {
    throw new Error('Expected Feynman voice style to use a calmer cadence than normal.');
  }
}
