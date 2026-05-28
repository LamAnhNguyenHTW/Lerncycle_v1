'use client';

import {useCallback, useEffect, useRef, useState} from 'react';
import type {VoiceSessionState} from '@/types/voice';

const MIME_TYPE_PREFERENCES = [
  'audio/webm;codecs=opus',
  'audio/webm',
  'audio/ogg;codecs=opus',
  'audio/ogg',
  'audio/mp4',
  'audio/aac',
] as const;

type MediaRecorderSupport = {
  isTypeSupported: (mimeType: string) => boolean;
};

export type VoiceRecorderResult = {
  state: VoiceSessionState;
  audioBlob: Blob | null;
  error: string | null;
  recordingSeconds: number;
  maxLengthReached: boolean;
  start: () => Promise<void>;
  stop: () => void;
  cancel: () => void;
};

export function chooseVoiceRecorderMimeType(mediaRecorder: MediaRecorderSupport | undefined): string | null {
  if (!mediaRecorder) {
    return null;
  }
  return MIME_TYPE_PREFERENCES.find((mimeType) => mediaRecorder.isTypeSupported(mimeType)) ?? null;
}

/**
 * Voice recorder state machine:
 * idle -> recording -> idle after stop, or error after unsupported/denied mic access.
 * Later chat integration advances idle transcripts through transcribing/thinking/speaking.
 */
export function useVoiceRecorder(maxRecordingSeconds: number): VoiceRecorderResult {
  const [state, setState] = useState<VoiceSessionState>('idle');
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [maxLengthReached, setMaxLengthReached] = useState(false);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const startedAtRef = useRef<number>(0);
  const autoStopTimerRef = useRef<number | null>(null);
  const tickTimerRef = useRef<number | null>(null);
  const cancelledRef = useRef(false);

  const clearTimers = useCallback(() => {
    if (autoStopTimerRef.current !== null) {
      window.clearTimeout(autoStopTimerRef.current);
      autoStopTimerRef.current = null;
    }
    if (tickTimerRef.current !== null) {
      window.clearInterval(tickTimerRef.current);
      tickTimerRef.current = null;
    }
  }, []);

  const releaseStream = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
  }, []);

  const resetRecorder = useCallback(() => {
    clearTimers();
    // Do not call releaseStream() here so the microphone remains warm for subsequent recordings
    recorderRef.current = null;
    chunksRef.current = [];
    startedAtRef.current = 0;
  }, [clearTimers]);

  const stop = useCallback(() => {
    const recorder = recorderRef.current;
    if (recorder && recorder.state !== 'inactive') {
      recorder.stop();
      return;
    }
    resetRecorder();
    setState('idle');
  }, [resetRecorder]);

  const cancel = useCallback(() => {
    cancelledRef.current = true;
    setAudioBlob(null);
    setMaxLengthReached(false);
    stop();
  }, [stop]);

  const start = useCallback(async () => {
    setError(null);
    setAudioBlob(null);
    setMaxLengthReached(false);
    cancelledRef.current = false;

    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
      setState('error');
      setError('Voice recording is not supported in this browser.');
      return;
    }

    const mimeType = chooseVoiceRecorderMimeType(MediaRecorder);
    if (!mimeType) {
      setState('error');
      setError('Voice recording is not supported in this browser.');
      return;
    }

    try {
      let stream = streamRef.current;
      if (!stream || !stream.active) {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          },
        });
        streamRef.current = stream;
      }
      const recorder = new MediaRecorder(stream, {mimeType});
      recorderRef.current = recorder;
      chunksRef.current = [];
      startedAtRef.current = Date.now();
      setRecordingSeconds(0);

      recorder.addEventListener('dataavailable', (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      });

      recorder.addEventListener('stop', () => {
        clearTimers();
        const elapsed = Math.max(0, Math.round((Date.now() - startedAtRef.current) / 1000));
        setRecordingSeconds(elapsed);
        if (!cancelledRef.current && chunksRef.current.length > 0) {
          setAudioBlob(new Blob(chunksRef.current, {type: mimeType}));
        }
        resetRecorder();
        setState('idle');
      });

      recorder.start();
      setState('recording');
      tickTimerRef.current = window.setInterval(() => {
        setRecordingSeconds(Math.max(0, Math.round((Date.now() - startedAtRef.current) / 1000)));
      }, 500);
      autoStopTimerRef.current = window.setTimeout(() => {
        setMaxLengthReached(true);
        stop();
      }, Math.max(1, maxRecordingSeconds) * 1000);
    } catch {
      resetRecorder();
      releaseStream();
      setState('error');
      setError('Microphone access was denied.');
    }
  }, [clearTimers, maxRecordingSeconds, resetRecorder, releaseStream, stop]);

  useEffect(() => {
    return () => {
      cancelledRef.current = true;
      resetRecorder();
      releaseStream(); // Ensure we release microphone tracks when the component is unmounted
    };
  }, [resetRecorder, releaseStream]);

  return {
    state,
    audioBlob,
    error,
    recordingSeconds,
    maxLengthReached,
    start,
    stop,
    cancel,
  };
}
