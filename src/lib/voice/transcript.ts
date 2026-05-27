import type {ChatMode} from '@/types/chat';

const FEYNMAN_SPOKEN_FINISH_PATTERN = /^(fertig|ich bin fertig|bin fertig|done|i am done|i'm done)$/iu;

export function normalizeVoiceTranscriptForChat(text: string, mode: ChatMode) {
  const normalized = text.trim().replace(/[.!?]+$/g, '').trim();
  if (mode === 'feynman' && FEYNMAN_SPOKEN_FINISH_PATTERN.test(normalized)) {
    return normalized.toLowerCase().includes('done') ? '/done' : '/fertig';
  }
  return text.trim();
}
