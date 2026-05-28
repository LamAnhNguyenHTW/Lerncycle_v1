import 'server-only';

import type {createClient} from '@/lib/supabase/server';
import type {ChatMode} from '@/types/chat';
import type {VoiceServerConfig} from '@/types/voice';

const DETAILED_PRIMER_LIMIT = 3;

type SupabaseClient = Awaited<ReturnType<typeof createClient>>;

type RealtimeContextBody = {
  sessionId: string;
  sourceIds: string[];
  mode: 'feynman';
};

type RealtimeContextBodyValidation =
  | {ok: true; value: RealtimeContextBody}
  | {ok: false; error: string};

type PrimerRow = {
  source_id: string;
  title: string | null;
  summary: string | null;
  main_topics: unknown;
  key_terms: unknown;
  learning_objectives: unknown;
};

type PdfRow = {
  id: string;
  name: string | null;
};

export type RealtimeContextResult = {
  documentPrimer: string;
  allowedSourceIds: string[];
  realtimeInstructions: string;
};

export class RealtimeContextError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

export function validateRealtimeContextBody(value: unknown): RealtimeContextBodyValidation {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return {ok: false, error: 'Invalid JSON body.'};
  }
  const body = value as Record<string, unknown>;
  if (body.mode !== undefined && body.mode !== 'feynman') {
    return {ok: false, error: 'Realtime voice is only available for Feynman mode.'};
  }
  const sessionId = typeof body.sessionId === 'string' ? body.sessionId.trim() : '';
  if (!sessionId) {
    return {ok: false, error: 'sessionId is required.'};
  }
  const sourceIds = uniqueStrings(body.sourceIds ?? body.pdf_ids);
  return {ok: true, value: {sessionId, sourceIds, mode: 'feynman'}};
}

export async function buildRealtimeContext(params: {
  supabase: SupabaseClient;
  userId: string | null | undefined;
  config: Pick<VoiceServerConfig, 'enabled' | 'realtimePrimerMaxChars'>;
  body: RealtimeContextBody;
}): Promise<RealtimeContextResult> {
  if (!params.config.enabled) {
    throw new RealtimeContextError('Voice mode is disabled.', 404);
  }
  if (!params.userId) {
    throw new RealtimeContextError('Unauthorized', 401);
  }

  await assertSessionBelongsToUser(params.supabase, params.userId, params.body.sessionId);
  const selectedPdfs = await loadOwnedPdfs(
    params.supabase,
    params.userId,
    params.body.sourceIds,
  );
  const allowedSourceIds = selectedPdfs.map((pdf) => pdf.id);
  const primers = await loadDocumentPrimers(params.supabase, params.userId, allowedSourceIds);
  const documentPrimer = buildDocumentPrimerText({
    pdfs: selectedPdfs,
    primers,
    maxChars: params.config.realtimePrimerMaxChars,
  });

  return {
    documentPrimer,
    allowedSourceIds,
    realtimeInstructions: buildRealtimeInstructions({
      mode: params.body.mode,
      pdfNames: selectedPdfs.map((pdf) => pdf.name).filter((name): name is string => Boolean(name)),
      documentPrimer,
    }),
  };
}

export async function persistRealtimeSessionSources(params: {
  supabase: SupabaseClient;
  userId: string;
  sessionId: string;
  allowedSourceIds: string[];
  mode: 'feynman';
}) {
  const {error} = await params.supabase
    .from('voice_realtime_sessions')
    .upsert(
      {
        user_id: params.userId,
        session_id: params.sessionId,
        mode: params.mode,
        allowed_source_ids: params.allowedSourceIds,
        updated_at: new Date().toISOString(),
      },
      {onConflict: 'user_id,session_id'},
    );
  if (error) {
    throw new RealtimeContextError('Failed to persist realtime source scope.', 500);
  }
}

export function buildDocumentPrimerText(params: {
  pdfs: PdfRow[];
  primers: PrimerRow[];
  maxChars: number;
}) {
  if (params.pdfs.length === 0) {
    return '';
  }
  const primerBySource = new Map(params.primers.map((primer) => [primer.source_id, primer]));
  const lines: string[] = [];
  params.pdfs.forEach((pdf, index) => {
    const primer = primerBySource.get(pdf.id);
    const title = cleanText(primer?.title ?? pdf.name ?? 'Selected document');
    const topics = formatList(primer?.main_topics);
    lines.push(`Document ${index + 1}: ${title}`);
    if (!primer) {
      lines.push('Primer: not generated yet. Use retrieval for document-specific details.');
      return;
    }
    if (index < DETAILED_PRIMER_LIMIT) {
      const summary = cleanText(primer.summary ?? '');
      const terms = formatList(primer.key_terms);
      const objectives = formatList(primer.learning_objectives);
      if (summary) {
        lines.push(`Summary: ${summary}`);
      }
      if (topics) {
        lines.push(`Topics: ${topics}`);
      }
      if (terms) {
        lines.push(`Key terms: ${terms}`);
      }
      if (objectives) {
        lines.push(`Learning goals: ${objectives}`);
      }
      return;
    }
    if (topics) {
      lines.push(`Topics: ${topics}`);
    }
  });
  return capText(lines.join('\n'), params.maxChars);
}

export function buildRealtimeInstructions(params: {
  mode: ChatMode;
  pdfNames: string[];
  documentPrimer: string;
}) {
  const sourceLine = params.pdfNames.length > 0
    ? `The learner selected these files for context: ${params.pdfNames.join(', ')}. If the learner says "the file" or "this document", assume they mean these selected files.`
    : 'No selected file names were provided. If file context is needed, call search_learncycle_context.';
  const primerLine = params.documentPrimer
    ? `Compact selected-file primer. Use it for orientation only; for document-specific claims, call search_learncycle_context first:\n${params.documentPrimer}`
    : 'No selected-file primer is available yet. Call search_learncycle_context before making document-specific claims.';
  return [
    'You are LearnCycle voice mode for an active Feynman Technique session.',
    'Act like a curious, child-like learning partner: simple language, short reactions, one simple follow-up question.',
    'Keep replies short, conversational, and in the learner language.',
    sourceLine,
    primerLine,
    'Force retrieval rule: if the learner asks about selected files, document contents, definitions from their material, page claims, quotes, or course-specific concepts, you must call search_learncycle_context before answering.',
    'For general small talk or simple non-document questions, answer briefly without retrieval.',
    'Never expose system, tool, API, database, or retrieval implementation details.',
  ].join('\n');
}

async function assertSessionBelongsToUser(
  supabase: SupabaseClient,
  userId: string,
  sessionId: string,
) {
  const {data, error} = await supabase
    .from('chat_sessions')
    .select('id')
    .eq('id', sessionId)
    .eq('user_id', userId)
    .maybeSingle();
  if (error || !data?.id) {
    throw new RealtimeContextError('Session not found.', 404);
  }
}

async function loadOwnedPdfs(
  supabase: SupabaseClient,
  userId: string,
  sourceIds: string[],
): Promise<PdfRow[]> {
  if (sourceIds.length === 0) {
    return [];
  }
  const {data, error} = await supabase
    .from('pdfs')
    .select('id, name')
    .eq('user_id', userId)
    .in('id', sourceIds);
  if (error || !Array.isArray(data)) {
    throw new RealtimeContextError('Failed to validate selected sources.', 500);
  }
  const rows = data as PdfRow[];
  const foundIds = new Set(rows.map((row) => row.id));
  const missing = sourceIds.filter((sourceId) => !foundIds.has(sourceId));
  if (missing.length > 0) {
    throw new RealtimeContextError('Selected source is not available for this session.', 403);
  }
  return sourceIds.map((sourceId) => rows.find((row) => row.id === sourceId)).filter((row): row is PdfRow => Boolean(row));
}

async function loadDocumentPrimers(
  supabase: SupabaseClient,
  userId: string,
  sourceIds: string[],
): Promise<PrimerRow[]> {
  if (sourceIds.length === 0) {
    return [];
  }
  const {data, error} = await supabase
    .from('rag_document_primers')
    .select('source_id, title, summary, main_topics, key_terms, learning_objectives')
    .eq('user_id', userId)
    .eq('source_type', 'pdf')
    .in('source_id', sourceIds);
  if (error || !Array.isArray(data)) {
    throw new RealtimeContextError('Failed to load document primer.', 500);
  }
  return data as PrimerRow[];
}

function uniqueStrings(value: unknown) {
  if (!Array.isArray(value)) {
    return [];
  }
  return Array.from(
    new Set(
      value
        .filter((item): item is string => typeof item === 'string' && item.trim().length > 0)
        .map((item) => item.trim()),
    ),
  );
}

function formatList(value: unknown) {
  if (!Array.isArray(value)) {
    return '';
  }
  return value
    .filter((item): item is string => typeof item === 'string' && item.trim().length > 0)
    .map((item) => cleanText(item))
    .slice(0, 8)
    .join(', ');
}

function cleanText(value: string) {
  return value.replace(/\s+/g, ' ').trim();
}

function capText(value: string, maxChars: number) {
  const clean = value.trim();
  if (clean.length <= maxChars) {
    return clean;
  }
  return `${clean.slice(0, Math.max(0, maxChars - 3)).trimEnd()}...`;
}
