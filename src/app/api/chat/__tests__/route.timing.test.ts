import {ragAnswerEndpoint, shouldRequestRagDebugTiming} from '../route';

export function assertProductionDoesNotRequestDebugTimingByDefault() {
  const env = {NODE_ENV: 'production'} as NodeJS.ProcessEnv;

  if (shouldRequestRagDebugTiming(env)) {
    throw new Error('Production must not request RAG debug timing by default.');
  }

  const endpoint = ragAnswerEndpoint('http://rag.local/', env);
  if (endpoint.includes('debug_timing')) {
    throw new Error(`Production endpoint leaked debug_timing: ${endpoint}`);
  }
}

export function assertProductionCanRequestDebugTimingOnlyWithExplicitEnvFlag() {
  const env = {
    NODE_ENV: 'production',
    RAG_DEBUG_TIMING_ENABLED: 'true',
  } as NodeJS.ProcessEnv;

  const endpoint = ragAnswerEndpoint('http://rag.local/', env);
  if (!endpoint.endsWith('/rag/answer?debug_timing=1')) {
    throw new Error(`Expected explicit debug timing endpoint, received ${endpoint}`);
  }
}

export function assertDevelopmentRequestsDebugTiming() {
  const env = {NODE_ENV: 'development'} as NodeJS.ProcessEnv;

  const endpoint = ragAnswerEndpoint('http://rag.local/', env);
  if (!endpoint.endsWith('/rag/answer?debug_timing=1')) {
    throw new Error(`Expected development debug timing endpoint, received ${endpoint}`);
  }
}
