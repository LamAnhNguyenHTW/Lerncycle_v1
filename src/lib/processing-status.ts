export type RagStage = 'queued' | 'parsing' | 'indexing' | 'completed' | 'failed';
export type GraphStage = 'disabled' | 'queued' | 'extracting' | 'completed' | 'failed';
export type OverallStage =
  | 'queued'
  | 'parsing'
  | 'indexing'
  | 'graph'
  | 'ready'
  | 'partial_ready'
  | 'failed';

export type SourceStatus = {
  sourceType: 'pdf' | 'note' | 'annotation_comment';
  sourceId: string;
  pdfId: string | null;
  ragStage: RagStage;
  graphStage: GraphStage;
  overallStage: OverallStage;
  ragReady: boolean;
  graphReady: boolean;
  updatedAt: string;
  userSafeError: string | null;
};

type DeriveOverallStageInput = {
  ragStage: RagStage | null;
  graphStage: GraphStage | null;
  graphEnabled: boolean;
  hasChunks: boolean;
};

type DerivedOverallStage = Pick<SourceStatus, 'overallStage' | 'ragReady' | 'graphReady'>;

const DE_MESSAGES = {
  parsing: 'Dokument konnte nicht gelesen werden.',
  embedding: 'Indexierung temporaer nicht verfuegbar - bitte erneut versuchen.',
  qdrant: 'Suchindex nicht erreichbar.',
  graph: 'Wissensgraph konnte nicht erzeugt werden.',
  fallback: 'Verarbeitung fehlgeschlagen.',
} as const;

const EN_MESSAGES = {
  parsing: 'Document could not be read.',
  embedding: 'Indexing is temporarily unavailable - please try again.',
  qdrant: 'Search index is unreachable.',
  graph: 'Knowledge graph could not be generated.',
  fallback: 'Processing failed.',
} as const;

export function deriveOverallStage({
  ragStage,
  graphStage,
  graphEnabled,
  hasChunks,
}: DeriveOverallStageInput): DerivedOverallStage {
  if (ragStage === null) {
    return hasChunks
      ? {overallStage: 'ready', ragReady: true, graphReady: false}
      : {overallStage: 'queued', ragReady: false, graphReady: false};
  }

  if (ragStage === 'failed') {
    return {overallStage: 'failed', ragReady: false, graphReady: false};
  }

  if (ragStage === 'queued') {
    return {overallStage: 'queued', ragReady: false, graphReady: false};
  }

  if (ragStage === 'parsing') {
    return {overallStage: 'parsing', ragReady: false, graphReady: false};
  }

  if (ragStage === 'indexing') {
    return {overallStage: 'indexing', ragReady: false, graphReady: false};
  }

  if (!graphEnabled || graphStage === 'disabled') {
    return {overallStage: 'ready', ragReady: true, graphReady: false};
  }

  if (graphStage === 'completed') {
    return {overallStage: 'ready', ragReady: true, graphReady: true};
  }

  if (graphStage === 'failed') {
    return {overallStage: 'partial_ready', ragReady: true, graphReady: false};
  }

  return {overallStage: 'graph', ragReady: true, graphReady: false};
}

export function sanitizeStageError(raw: string | null, lang: 'de' | 'en'): string | null {
  if (!raw?.trim()) {
    return null;
  }

  const messages = lang === 'en' ? EN_MESSAGES : DE_MESSAGES;
  const normalized = raw.toLowerCase();

  if (/\b(docling|parsing|parse|pdf|document)\b/.test(normalized)) {
    return messages.parsing;
  }

  if (/\b(openai|embedding|embeddings|embed|rate limit|429|503)\b/.test(normalized)) {
    return messages.embedding;
  }

  if (/\b(qdrant|vector|upsert|search index|connection refused)\b/.test(normalized)) {
    return messages.qdrant;
  }

  if (/\b(neo4j|graph|cypher|knowledge graph)\b/.test(normalized)) {
    return messages.graph;
  }

  return messages.fallback;
}
