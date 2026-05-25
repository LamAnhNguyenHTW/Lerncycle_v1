import {NextResponse} from 'next/server';

import {
  deriveOverallStage,
  sanitizeStageError,
  type GraphStage,
  type RagStage,
  type SourceStatus,
} from '@/lib/processing-status';
import {createClient} from '@/lib/supabase/server';

type SourceType = SourceStatus['sourceType'];

type ProcessingStatusQuery = {
  sourceType?: SourceType;
  sourceIds?: string[];
  pdfIds?: string[];
  since?: string;
};

type RawStatusRow = {
  user_id: string;
  source_type: SourceType;
  source_id: string;
  pdf_id: string | null;
  rag_stage: string | null;
  rag_status: string | null;
  rag_stage_error: string | null;
  graph_stage: string | null;
  graph_status: string | null;
  graph_stage_error: string | null;
  has_chunks: boolean | null;
  updated_at: string;
};

type PdfRow = {
  id: string;
  created_at?: string | null;
};

type ChunkSourceRow = {
  source_id: string;
};

type SupabaseQueryResult<T> = PromiseLike<{
  data: T[] | null;
  error: unknown;
}>;

type SupabaseTableQuery<T> = {
  select: (columns: string) => SupabaseTableQuery<T>;
  eq: (column: string, value: string) => SupabaseTableQuery<T>;
  in: (column: string, values: string[]) => SupabaseTableQuery<T>;
} & SupabaseQueryResult<T>;

type SupabaseReader = {
  from: {
    (table: 'pdfs'): SupabaseTableQuery<PdfRow>;
    (table: 'rag_chunks'): SupabaseTableQuery<ChunkSourceRow>;
  };
};

const SOURCE_TYPES = new Set<SourceType>(['pdf', 'note', 'annotation_comment']);

export async function GET(request: Request) {
  const supabase = await createClient();
  const {data: {user}} = await supabase.auth.getUser();

  if (!user) {
    return NextResponse.json({error: 'Unauthorized'}, {status: 401});
  }

  const parsed = parseProcessingStatusQuery(new URL(request.url).searchParams);
  if (!parsed.ok) {
    return NextResponse.json({error: parsed.error}, {status: 400});
  }

  const query = parsed.value;
  const graphEnabled = process.env.GRAPH_EXTRACTION_ENABLED === 'true';
  const lang = languageFromAcceptHeader(request.headers.get('accept-language'));

  let rawQuery = supabase
    .from('v_source_processing_status_raw')
    .select('*')
    .eq('user_id', user.id);

  if (query.pdfIds) {
    rawQuery = rawQuery.eq('source_type', 'pdf').in('source_id', query.pdfIds);
  } else {
    if (query.sourceType) {
      rawQuery = rawQuery.eq('source_type', query.sourceType);
    }
    if (query.sourceIds) {
      rawQuery = rawQuery.in('source_id', query.sourceIds);
    }
  }
  if (query.since) {
    rawQuery = rawQuery.gt('updated_at', query.since);
  }

  const {data, error} = await rawQuery;
  if (error) {
    return NextResponse.json({error: 'Could not load processing status'}, {status: 500});
  }

  const rawRows = (data ?? []) as RawStatusRow[];
  const statuses = rawRows.map((row) => sourceStatusFromRawRow(row, {
    graphEnabled,
    lang,
  }));

  if (shouldLoadMissingPdfStatuses(query)) {
    const existingIds = new Set(statuses.map((status) => status.sourceId));
    const missingIds = query.pdfIds.filter((id) => !existingIds.has(id));
    try {
      statuses.push(...await loadMissingPdfStatuses({
        supabase: supabase as unknown as SupabaseReader,
        userId: user.id,
        pdfIds: missingIds,
        graphEnabled,
      }));
    } catch {
      return NextResponse.json({error: 'Could not load processing status'}, {status: 500});
    }
  }

  return NextResponse.json(statuses);
}

export function shouldLoadMissingPdfStatuses(query: ProcessingStatusQuery): query is ProcessingStatusQuery & {pdfIds: string[]} {
  return Boolean(query.pdfIds && !query.since);
}

export function parseProcessingStatusQuery(
  params: URLSearchParams,
): {ok: true; value: ProcessingStatusQuery} | {ok: false; error: string} {
  const sourceType = params.get('sourceType');
  const sourceIds = splitCsvParam(params.get('sourceIds'));
  const pdfIds = splitCsvParam(params.get('pdfIds'));
  const since = params.get('since')?.trim() || undefined;

  if (sourceType && !SOURCE_TYPES.has(sourceType as SourceType)) {
    return {ok: false, error: 'Invalid sourceType'};
  }
  if (pdfIds && sourceType && sourceType !== 'pdf') {
    return {ok: false, error: 'pdfIds can only be combined with sourceType=pdf'};
  }
  if (pdfIds && sourceIds) {
    return {ok: false, error: 'Use either pdfIds or sourceIds, not both'};
  }
  if (since && Number.isNaN(Date.parse(since))) {
    return {ok: false, error: 'Invalid since timestamp'};
  }

  return {
    ok: true,
    value: {
      sourceType: pdfIds ? 'pdf' : (sourceType as SourceType | undefined),
      sourceIds,
      pdfIds,
      since,
    },
  };
}

export function sourceStatusFromRawRow(
  row: RawStatusRow,
  options: {graphEnabled: boolean; lang: 'de' | 'en'},
): SourceStatus {
  const ragStage = normalizeRagStage(row.rag_stage, row.rag_status);
  const graphStage = normalizeGraphStage(row.graph_stage, row.graph_status, options.graphEnabled);
  const derived = deriveOverallStage({
    ragStage,
    graphStage,
    graphEnabled: options.graphEnabled,
    hasChunks: row.has_chunks === true,
  });
  const rawError = derived.overallStage === 'partial_ready'
    ? row.graph_stage_error
    : row.rag_stage_error ?? row.graph_stage_error;

  return {
    sourceType: row.source_type,
    sourceId: row.source_id,
    pdfId: row.pdf_id,
    ragStage: ragStage ?? (row.has_chunks ? 'completed' : 'queued'),
    graphStage,
    ...derived,
    updatedAt: row.updated_at,
    userSafeError: sanitizeStageError(rawError, options.lang),
  };
}

export function synthesizePdfStatus(
  pdf: PdfRow,
  hasChunks: boolean,
  options: {graphEnabled: boolean},
): SourceStatus {
  const ragStage: RagStage = hasChunks ? 'completed' : 'queued';
  const graphStage: GraphStage = options.graphEnabled && hasChunks ? 'queued' : 'disabled';
  const derived = deriveOverallStage({
    ragStage,
    graphStage,
    graphEnabled: options.graphEnabled,
    hasChunks,
  });

  return {
    sourceType: 'pdf',
    sourceId: pdf.id,
    pdfId: pdf.id,
    ragStage,
    graphStage,
    ...derived,
    updatedAt: pdf.created_at ?? new Date(0).toISOString(),
    userSafeError: null,
  };
}

async function loadMissingPdfStatuses({
  supabase,
  userId,
  pdfIds,
  graphEnabled,
}: {
  supabase: SupabaseReader;
  userId: string;
  pdfIds: string[];
  graphEnabled: boolean;
}): Promise<SourceStatus[]> {
  if (pdfIds.length === 0) {
    return [];
  }

  const {data: pdfRows, error: pdfError} = await supabase
    .from('pdfs')
    .select('id, created_at')
    .eq('user_id', userId)
    .in('id', pdfIds);

  if (pdfError) {
    throw pdfError;
  }

  const existingPdfIds = ((pdfRows ?? []) as PdfRow[]).map((pdf) => pdf.id);
  if (existingPdfIds.length === 0) {
    return [];
  }

  const {data: chunkRows, error: chunkError} = await supabase
    .from('rag_chunks')
    .select('source_id')
    .eq('user_id', userId)
    .eq('source_type', 'pdf')
    .in('source_id', existingPdfIds);

  if (chunkError) {
    throw chunkError;
  }

  const chunkSourceIds = new Set(((chunkRows ?? []) as ChunkSourceRow[]).map((row) => row.source_id));
  return ((pdfRows ?? []) as PdfRow[]).map((pdf) => synthesizePdfStatus(
    pdf,
    chunkSourceIds.has(pdf.id),
    {graphEnabled},
  ));
}

function normalizeRagStage(stage: string | null, status: string | null): RagStage | null {
  if (stage === 'indexing_dense' || stage === 'indexing_sparse') {
    return 'indexing';
  }
  if (stage === 'queued' || stage === 'parsing' || stage === 'completed' || stage === 'failed') {
    return stage;
  }
  if (status === 'completed') {
    return 'completed';
  }
  if (status === 'failed') {
    return 'failed';
  }
  if (status === 'pending' || status === 'processing') {
    return 'queued';
  }
  return null;
}

function normalizeGraphStage(
  stage: string | null,
  status: string | null,
  graphEnabled: boolean,
): GraphStage {
  if (!graphEnabled) {
    return 'disabled';
  }
  if (stage === 'graph_extracting') {
    return 'extracting';
  }
  if (stage === 'queued' || stage === 'completed' || stage === 'failed') {
    return stage;
  }
  if (status === 'completed') {
    return 'completed';
  }
  if (status === 'failed') {
    return 'failed';
  }
  return 'queued';
}

function splitCsvParam(value: string | null): string[] | undefined {
  if (!value?.trim()) {
    return undefined;
  }

  const items = [...new Set(value.split(',').map((item) => item.trim()).filter(Boolean))];
  return items.length > 0 ? items : undefined;
}

function languageFromAcceptHeader(value: string | null): 'de' | 'en' {
  return value?.toLowerCase().startsWith('en') ? 'en' : 'de';
}
