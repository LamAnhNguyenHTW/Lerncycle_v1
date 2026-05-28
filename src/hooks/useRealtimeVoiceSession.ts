'use client';

import {useCallback, useRef, useState} from 'react';
import type {VoiceSessionState} from '@/types/voice';

const OPENAI_REALTIME_CALLS_URL = 'https://api.openai.com/v1/realtime/calls';

export type RealtimeToolCall = {
  callId: string;
  name: string;
  arguments: Record<string, unknown>;
};

export type RealtimeTranscriptEvent =
  | {role: 'user'; text: string; final: true}
  | {role: 'assistant'; text: string; final: boolean; responseId?: string};

export type RealtimeVoiceSessionOptions = {
  sessionId?: string;
  courseId?: string;
  selectedPdfIds?: string[];
  selectedPdfNames?: string[];
  mode?: 'feynman';
  onEvent?: (event: unknown) => void;
  onTranscript?: (event: RealtimeTranscriptEvent) => void;
  onToolCall?: (call: RealtimeToolCall) => Promise<unknown>;
  onSessionCreated?: (sessionId: string) => void;
};

export type RealtimeVoiceSessionResult = {
  state: VoiceSessionState;
  error: string | null;
  isConnected: boolean;
  connect: (options?: RealtimeVoiceSessionOptions) => Promise<void>;
  disconnect: () => void;
  sendEvent: (event: unknown) => void;
};

type PersistedRealtimeTurn = {
  role: 'user' | 'assistant';
  text: string;
};

export function extractRealtimeToolCall(event: unknown): RealtimeToolCall | null {
  if (!event || typeof event !== 'object') {
    return null;
  }
  const value = event as {
    type?: unknown;
    item?: unknown;
    response?: {output?: unknown};
  };
  const candidates: unknown[] = [];
  if (value.item) {
    candidates.push(value.item);
  }
  if (Array.isArray(value.response?.output)) {
    candidates.push(...value.response.output);
  }
  for (const candidate of candidates) {
    if (!candidate || typeof candidate !== 'object') {
      continue;
    }
    const item = candidate as {type?: unknown; name?: unknown; call_id?: unknown; arguments?: unknown};
    if (item.type !== 'function_call' || item.name !== 'search_learncycle_context' || typeof item.call_id !== 'string') {
      continue;
    }
    return {
      callId: item.call_id,
      name: item.name,
      arguments: parseToolArguments(item.arguments),
    };
  }
  return null;
}

export function createRealtimeToolOutputEvent(callId: string, output: unknown) {
  return {
    type: 'conversation.item.create',
    item: {
      type: 'function_call_output',
      call_id: callId,
      output: JSON.stringify(output),
    },
  };
}

export function extractRealtimeTranscriptEvent(event: unknown): RealtimeTranscriptEvent | null {
  if (!event || typeof event !== 'object') {
    return null;
  }
  const value = event as {
    type?: unknown;
    transcript?: unknown;
    delta?: unknown;
    text?: unknown;
    response_id?: unknown;
    item?: unknown;
    response?: unknown;
  };
  if (
    value.type === 'conversation.item.input_audio_transcription.completed' &&
    typeof value.transcript === 'string' &&
    value.transcript.trim() &&
    !isLikelyRealtimeTranscriptHallucination(value.transcript)
  ) {
    return {role: 'user', text: value.transcript.trim(), final: true};
  }
  if (
    (
      value.type === 'response.audio_transcript.delta' ||
      value.type === 'response.text.delta' ||
      value.type === 'response.output_text.delta'
    ) &&
    typeof value.delta === 'string' &&
    value.delta
  ) {
    return {
      role: 'assistant',
      text: value.delta,
      final: false,
      ...(typeof value.response_id === 'string' ? {responseId: value.response_id} : {}),
    };
  }
  if (
    (
      value.type === 'response.audio_transcript.done' ||
      value.type === 'response.text.done' ||
      value.type === 'response.output_text.done'
    ) &&
    typeof (value.transcript ?? value.text) === 'string'
  ) {
    return {
      role: 'assistant',
      text: String(value.transcript ?? value.text),
      final: true,
      ...(typeof value.response_id === 'string' ? {responseId: value.response_id} : {}),
    };
  }
  if (value.type === 'response.output_item.done' && value.item) {
    const text = extractRealtimeOutputText(value.item);
    if (text) {
      return {
        role: 'assistant',
        text,
        final: true,
        ...(typeof value.response_id === 'string' ? {responseId: value.response_id} : {}),
      };
    }
  }
  if (value.type === 'response.done' && value.response) {
    const text = extractRealtimeOutputText(value.response);
    if (text) {
      return {
        role: 'assistant',
        text,
        final: true,
        ...(typeof value.response_id === 'string' ? {responseId: value.response_id} : {}),
      };
    }
  }
  return null;
}

export function extractRealtimeOutputText(value: unknown): string {
  const collected: string[] = [];

  function visit(node: unknown) {
    if (!node || typeof node !== 'object') {
      return;
    }
    if (Array.isArray(node)) {
      node.forEach(visit);
      return;
    }
    const record = node as Record<string, unknown>;
    const type = typeof record.type === 'string' ? record.type : '';
    if (
      (type.includes('audio') || type.includes('text') || type === 'message') &&
      typeof record.transcript === 'string' &&
      record.transcript.trim()
    ) {
      collected.push(record.transcript.trim());
    }
    if (
      (type.includes('text') || type === 'message') &&
      typeof record.text === 'string' &&
      record.text.trim()
    ) {
      collected.push(record.text.trim());
    }
    if (Array.isArray(record.content)) {
      visit(record.content);
    }
    if (Array.isArray(record.output)) {
      visit(record.output);
    }
  }

  visit(value);
  return collected.join(' ').trim();
}

export function isLikelyRealtimeTranscriptHallucination(text: string) {
  const normalized = text
    .trim()
    .toLowerCase()
    .replace(/[.!?,;:"'`´’‘“”()[\]{}]/g, '')
    .replace(/\s+/g, ' ');
  if (!normalized) {
    return true;
  }
  const commonSilenceHallucinations = new Set([
    'bye',
    'goodbye',
    'thank you',
    'thanks',
    'you',
    'untertitel',
    'untertitel im auftrag des zdf',
    'subtitles',
    'subtitles by the amaraorg community',
    'so this session car was some ant bites',
  ]);
  return commonSilenceHallucinations.has(normalized);
}

export function parseToolArguments(value: unknown): Record<string, unknown> {
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  if (typeof value !== 'string' || !value.trim()) {
    return {};
  }
  try {
    const parsed = JSON.parse(value);
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed as Record<string, unknown> : {};
  } catch {
    return {};
  }
}

export function useRealtimeVoiceSession(): RealtimeVoiceSessionResult {
  const [state, setState] = useState<VoiceSessionState>('idle');
  const [error, setError] = useState<string | null>(null);
  const peerConnectionRef = useRef<RTCPeerConnection | null>(null);
  const dataChannelRef = useRef<RTCDataChannel | null>(null);
  const localStreamRef = useRef<MediaStream | null>(null);
  const remoteAudioRef = useRef<HTMLAudioElement | null>(null);
  const intentionalDisconnectRef = useRef(false);
  const activeSessionIdRef = useRef<string | undefined>(undefined);
  const sessionStartedAtRef = useRef<number | null>(null);
  const transcriptTurnsRef = useRef<PersistedRealtimeTurn[]>([]);
  const persistedTurnCountRef = useRef(0);

  const disconnect = useCallback(() => {
    intentionalDisconnectRef.current = true;
    const sessionId = activeSessionIdRef.current;
    const startedAt = sessionStartedAtRef.current;
    const turns = transcriptTurnsRef.current.slice(persistedTurnCountRef.current);
    if (sessionId && turns.length > 0) {
      void persistRealtimeTurns(sessionId, turns);
    }
    if (sessionId && startedAt) {
      void reportRealtimeUsage(sessionId, Math.ceil((Date.now() - startedAt) / 1000));
    }
    dataChannelRef.current?.close();
    peerConnectionRef.current?.close();
    localStreamRef.current?.getTracks().forEach((track) => track.stop());
    remoteAudioRef.current?.remove();
    dataChannelRef.current = null;
    peerConnectionRef.current = null;
    localStreamRef.current = null;
    remoteAudioRef.current = null;
    activeSessionIdRef.current = undefined;
    sessionStartedAtRef.current = null;
    transcriptTurnsRef.current = [];
    persistedTurnCountRef.current = 0;
    setState('idle');
  }, []);

  const sendEvent = useCallback((event: unknown) => {
    const channel = dataChannelRef.current;
    if (!channel || channel.readyState !== 'open') {
      return;
    }
    channel.send(JSON.stringify(event));
  }, []);

  const connect = useCallback(async (options: RealtimeVoiceSessionOptions = {}) => {
    intentionalDisconnectRef.current = false;
    activeSessionIdRef.current = options.sessionId;
    sessionStartedAtRef.current = null;
    transcriptTurnsRef.current = [];
    persistedTurnCountRef.current = 0;
    setError(null);
    setState('connecting');
    try {
      const tokenResponse = await fetch('/api/voice/realtime-token', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          ...(options.sessionId ? {sessionId: options.sessionId} : {}),
          ...(options.courseId ? {course_id: options.courseId} : {}),
          ...(options.selectedPdfIds ? {pdf_ids: options.selectedPdfIds} : {}),
          ...(options.selectedPdfNames ? {pdf_names: options.selectedPdfNames} : {}),
          mode: options.mode ?? 'feynman',
        }),
      });
      if (!tokenResponse.ok) {
        throw new Error('Realtime voice is not available.');
      }
      const token = await tokenResponse.json() as {clientSecret?: string; sessionId?: string};
      if (!token.clientSecret) {
        throw new Error('Realtime voice token is missing.');
      }
      const resolvedSessionId = token.sessionId ?? options.sessionId;
      activeSessionIdRef.current = resolvedSessionId;
      if (resolvedSessionId && resolvedSessionId !== options.sessionId) {
        options.onSessionCreated?.(resolvedSessionId);
      }

      const pc = new RTCPeerConnection();
      peerConnectionRef.current = pc;

      const audio = document.createElement('audio');
      audio.autoplay = true;
      remoteAudioRef.current = audio;
      pc.ontrack = (event) => {
        audio.srcObject = event.streams[0] ?? null;
      };

      const localStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      localStreamRef.current = localStream;
      localStream.getAudioTracks().forEach((track) => pc.addTrack(track, localStream));

      const dataChannel = pc.createDataChannel('oai-events');
      dataChannelRef.current = dataChannel;
      dataChannel.addEventListener('open', () => setState('live'));
      dataChannel.addEventListener('message', async (message) => {
        const event = parseRealtimeEvent(message.data);
        options.onEvent?.(event);
        const transcript = extractRealtimeTranscriptEvent(event);
        if (transcript) {
          if (transcript.final) {
            const beforeCount = transcriptTurnsRef.current.length;
            transcriptTurnsRef.current = appendRealtimeTranscriptTurn(transcriptTurnsRef.current, {
              role: transcript.role,
              text: transcript.text,
            });
            const newTurns = transcriptTurnsRef.current.slice(beforeCount);
            const sessionId = activeSessionIdRef.current;
            if (sessionId && newTurns.length > 0) {
              await persistRealtimeTurns(sessionId, newTurns);
              persistedTurnCountRef.current = transcriptTurnsRef.current.length;
            }
          }
          options.onTranscript?.(transcript);
        }
        const toolCall = extractRealtimeToolCall(event);
        if (toolCall) {
          try {
            const output = options.onToolCall
              ? await options.onToolCall(toolCall)
              : await callRealtimeRagTool(activeSessionIdRef.current, options.selectedPdfIds, toolCall);
            dataChannel.send(JSON.stringify(createRealtimeToolOutputEvent(toolCall.callId, output)));
            dataChannel.send(JSON.stringify({type: 'response.create'}));
          } catch {
            dataChannel.send(JSON.stringify(createRealtimeToolOutputEvent(toolCall.callId, {
              answer: 'LearnCycle retrieval is unavailable right now.',
              sources: [],
            })));
            dataChannel.send(JSON.stringify({type: 'response.create'}));
          }
        }
      });

      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      const sdpResponse = await fetch(OPENAI_REALTIME_CALLS_URL, {
        method: 'POST',
        body: offer.sdp,
        headers: {
          Authorization: `Bearer ${token.clientSecret}`,
          'Content-Type': 'application/sdp',
        },
      });
      if (!sdpResponse.ok) {
        throw new Error('Realtime voice connection failed.');
      }
      await pc.setRemoteDescription({type: 'answer', sdp: await sdpResponse.text()});
      sessionStartedAtRef.current = Date.now();
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : 'Realtime voice failed.';
      if (intentionalDisconnectRef.current || isBenignRealtimeDisconnectError(message)) {
        disconnect();
        setError(null);
        return;
      }
      disconnect();
      setState('error');
      setError(message);
    }
  }, [disconnect]);

  return {
    state,
    error,
    isConnected: state === 'live',
    connect,
    disconnect,
    sendEvent,
  };
}

function parseRealtimeEvent(value: unknown): unknown {
  if (typeof value !== 'string') {
    return value;
  }
  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
}

export function isBenignRealtimeDisconnectError(message: string) {
  const normalized = message.toLowerCase();
  return (
    normalized.includes('peer connection is closed') ||
    normalized.includes('data channel is closed') ||
    normalized.includes('connection is closed')
  );
}

export function appendRealtimeTranscriptTurn(
  current: PersistedRealtimeTurn[],
  next: PersistedRealtimeTurn,
): PersistedRealtimeTurn[] {
  const text = next.text.replace(/\s+/g, ' ').trim();
  if (!text) {
    return current;
  }
  const last = current[current.length - 1];
  if (last?.role === next.role && last.text === text) {
    return current;
  }
  return [...current, {...next, text}].slice(-80);
}

async function persistRealtimeTurns(sessionId: string, turns: PersistedRealtimeTurn[]) {
  try {
    await fetch('/api/voice/realtime-summary', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      keepalive: true,
      body: JSON.stringify({sessionId, turns}),
    });
  } catch {
    // Realtime summary persistence is best-effort; the audio was never stored.
  }
}

async function reportRealtimeUsage(sessionId: string, durationSeconds: number) {
  if (durationSeconds <= 0) {
    return;
  }
  try {
    await fetch('/api/voice/realtime-usage', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      keepalive: true,
      body: JSON.stringify({sessionId, durationSeconds}),
    });
  } catch {
    // Realtime usage reporting is retried best-effort on explicit disconnect paths only.
  }
}

async function callRealtimeRagTool(
  sessionId: string | undefined,
  selectedPdfIds: string[] | undefined,
  call: RealtimeToolCall,
) {
  try {
    const response = await fetch('/api/voice/realtime-tool', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        ...(sessionId ? {sessionId} : {}),
        query: typeof call.arguments.query === 'string' ? call.arguments.query : '',
        top_k: typeof call.arguments.top_k === 'number' ? call.arguments.top_k : 4,
        pdf_ids: selectedPdfIds ?? [],
      }),
    });
    if (!response.ok) {
      return {answer: 'LearnCycle retrieval is unavailable right now.', sources: []};
    }
    return response.json();
  } catch {
    return {answer: 'LearnCycle retrieval is unavailable right now.', sources: []};
  }
}
