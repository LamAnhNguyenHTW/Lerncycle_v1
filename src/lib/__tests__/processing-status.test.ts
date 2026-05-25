import {
  deriveOverallStage,
  sanitizeStageError,
  type GraphStage,
  type RagStage,
} from '../processing-status';

type StageCase = {
  name: string;
  ragStage: RagStage | null;
  graphStage: GraphStage | null;
  graphEnabled: boolean;
  hasChunks: boolean;
  expected: ReturnType<typeof deriveOverallStage>;
};

const stageCases: StageCase[] = [
  {
    name: 'legacy missing job is ready when chunks exist',
    ragStage: null,
    graphStage: null,
    graphEnabled: false,
    hasChunks: true,
    expected: {overallStage: 'ready', ragReady: true, graphReady: false},
  },
  {
    name: 'legacy missing job is queued without chunks',
    ragStage: null,
    graphStage: null,
    graphEnabled: false,
    hasChunks: false,
    expected: {overallStage: 'queued', ragReady: false, graphReady: false},
  },
  {
    name: 'queued rag job is not ready',
    ragStage: 'queued',
    graphStage: null,
    graphEnabled: true,
    hasChunks: false,
    expected: {overallStage: 'queued', ragReady: false, graphReady: false},
  },
  {
    name: 'parsing rag job maps to parsing',
    ragStage: 'parsing',
    graphStage: null,
    graphEnabled: false,
    hasChunks: false,
    expected: {overallStage: 'parsing', ragReady: false, graphReady: false},
  },
  {
    name: 'indexing rag job maps to indexing',
    ragStage: 'indexing',
    graphStage: null,
    graphEnabled: false,
    hasChunks: false,
    expected: {overallStage: 'indexing', ragReady: false, graphReady: false},
  },
  {
    name: 'completed rag job is ready when graph is disabled',
    ragStage: 'completed',
    graphStage: 'disabled',
    graphEnabled: false,
    hasChunks: true,
    expected: {overallStage: 'ready', ragReady: true, graphReady: false},
  },
  {
    name: 'completed rag job waits for graph when graph job is missing',
    ragStage: 'completed',
    graphStage: null,
    graphEnabled: true,
    hasChunks: true,
    expected: {overallStage: 'graph', ragReady: true, graphReady: false},
  },
  {
    name: 'completed rag job waits for queued graph job',
    ragStage: 'completed',
    graphStage: 'queued',
    graphEnabled: true,
    hasChunks: true,
    expected: {overallStage: 'graph', ragReady: true, graphReady: false},
  },
  {
    name: 'completed rag job waits for extracting graph job',
    ragStage: 'completed',
    graphStage: 'extracting',
    graphEnabled: true,
    hasChunks: true,
    expected: {overallStage: 'graph', ragReady: true, graphReady: false},
  },
  {
    name: 'completed rag and graph jobs are fully ready',
    ragStage: 'completed',
    graphStage: 'completed',
    graphEnabled: true,
    hasChunks: true,
    expected: {overallStage: 'ready', ragReady: true, graphReady: true},
  },
  {
    name: 'failed graph job keeps chat partially ready',
    ragStage: 'completed',
    graphStage: 'failed',
    graphEnabled: true,
    hasChunks: true,
    expected: {overallStage: 'partial_ready', ragReady: true, graphReady: false},
  },
  {
    name: 'failed rag job is failed',
    ragStage: 'failed',
    graphStage: 'completed',
    graphEnabled: true,
    hasChunks: true,
    expected: {overallStage: 'failed', ragReady: false, graphReady: false},
  },
];

export function assertProcessingStatusStageMapping() {
  for (const testCase of stageCases) {
    const actual = deriveOverallStage(testCase);

    if (JSON.stringify(actual) !== JSON.stringify(testCase.expected)) {
      throw new Error(
        `${testCase.name}: expected ${JSON.stringify(testCase.expected)}, received ${JSON.stringify(actual)}`,
      );
    }
  }
}

export function assertProcessingStatusErrorSanitizerGermanMessages() {
  const cases = [
    {
      raw: 'Docling parsing failed at C:\\Users\\o\\secret.pdf\nTraceback line 42',
      expected: 'Dokument konnte nicht gelesen werden.',
    },
    {
      raw: 'OpenAI embeddings API returned 503 for /home/me/.env',
      expected: 'Indexierung temporaer nicht verfuegbar - bitte erneut versuchen.',
    },
    {
      raw: 'Qdrant upsert connection refused at http://localhost:6333',
      expected: 'Suchindex nicht erreichbar.',
    },
    {
      raw: 'Neo4j graph extraction failed with password=secret',
      expected: 'Wissensgraph konnte nicht erzeugt werden.',
    },
    {
      raw: 'Unexpected stack frame at D:\\private\\file.ts:10',
      expected: 'Verarbeitung fehlgeschlagen.',
    },
  ];

  for (const testCase of cases) {
    const actual = sanitizeStageError(testCase.raw, 'de');
    if (actual !== testCase.expected) {
      throw new Error(`Expected "${testCase.expected}", received "${actual}"`);
    }
    assertNoRawErrorLeak(actual);
  }
}

export function assertProcessingStatusErrorSanitizerEnglishMessages() {
  const cases = [
    {
      raw: 'Docling parsing failed',
      expected: 'Document could not be read.',
    },
    {
      raw: 'embedding API unavailable',
      expected: 'Indexing is temporarily unavailable - please try again.',
    },
    {
      raw: 'Qdrant upsert failed',
      expected: 'Search index is unreachable.',
    },
    {
      raw: 'Neo4j connection failed',
      expected: 'Knowledge graph could not be generated.',
    },
    {
      raw: 'Unexpected error',
      expected: 'Processing failed.',
    },
  ];

  for (const testCase of cases) {
    const actual = sanitizeStageError(testCase.raw, 'en');
    if (actual !== testCase.expected) {
      throw new Error(`Expected "${testCase.expected}", received "${actual}"`);
    }
    assertNoRawErrorLeak(actual);
  }
}

export function assertProcessingStatusErrorSanitizerHandlesEmptyErrors() {
  if (sanitizeStageError(null, 'de') !== null) {
    throw new Error('Expected null raw errors to remain null.');
  }
  if (sanitizeStageError('', 'en') !== null) {
    throw new Error('Expected empty raw errors to remain null.');
  }
}

function assertNoRawErrorLeak(value: string | null) {
  if (!value) {
    return;
  }

  const leakedRawPattern = /(C:\\|D:\\|\/home\/|Traceback|password=|secret|localhost:\d+)/i;
  if (leakedRawPattern.test(value)) {
    throw new Error(`Sanitized message leaked raw details: ${value}`);
  }
}
