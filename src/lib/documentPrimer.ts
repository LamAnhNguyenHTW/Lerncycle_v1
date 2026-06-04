import 'server-only';

import type {createClient} from '@/lib/supabase/server';

const DETAILED_PRIMER_LIMIT = 3;

type SupabaseClient = Awaited<ReturnType<typeof createClient>>;

export type PrimerRow = {
  source_id: string;
  title: string | null;
  summary: string | null;
  main_topics: unknown;
  key_terms: unknown;
  learning_objectives: unknown;
};

export type PdfRow = {
  id: string;
  name: string | null;
};

export class DocumentPrimerError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

export async function loadOwnedPdfs(
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
    throw new DocumentPrimerError('Failed to validate selected sources.', 500);
  }
  const rows = data as PdfRow[];
  const foundIds = new Set(rows.map((row) => row.id));
  const missing = sourceIds.filter((sourceId) => !foundIds.has(sourceId));
  if (missing.length > 0) {
    throw new DocumentPrimerError('Selected source is not available for this session.', 403);
  }
  return sourceIds.map((sourceId) => rows.find((row) => row.id === sourceId)).filter((row): row is PdfRow => Boolean(row));
}

export async function loadDocumentPrimers(
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
    throw new DocumentPrimerError('Failed to load document primer.', 500);
  }
  return data as PrimerRow[];
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

export function formatList(value: unknown) {
  if (!Array.isArray(value)) {
    return '';
  }
  return value
    .filter((item): item is string => typeof item === 'string' && item.trim().length > 0)
    .map((item) => cleanText(item))
    .slice(0, 8)
    .join(', ');
}

export function cleanText(value: string) {
  return value.replace(/\s+/g, ' ').trim();
}

export function capText(value: string, maxChars: number) {
  const clean = value.trim();
  if (clean.length <= maxChars) {
    return clean;
  }
  return `${clean.slice(0, Math.max(0, maxChars - 3)).trimEnd()}...`;
}
