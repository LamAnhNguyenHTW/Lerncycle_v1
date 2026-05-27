import {normalizeVoiceTranscriptForChat} from './transcript';

export function assertVoiceFinishTranscriptMapsToFeynmanCommand() {
  if (normalizeVoiceTranscriptForChat('ich bin fertig.', 'feynman') !== '/fertig') {
    throw new Error('Expected German spoken finish phrase to map to /fertig.');
  }
  if (normalizeVoiceTranscriptForChat("I'm done", 'feynman') !== '/done') {
    throw new Error('Expected English spoken finish phrase to map to /done.');
  }
}

export function assertVoiceTranscriptDoesNotRewriteNormalChat() {
  const transcript = 'ich bin fertig';
  if (normalizeVoiceTranscriptForChat(transcript, 'normal') !== transcript) {
    throw new Error('Expected normal chat transcripts to remain unchanged.');
  }
}
