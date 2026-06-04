import {buildRagRequestBody, loadGuidedLearningDocumentPrimer} from '../route';

type TableRow = Record<string, unknown>;

class FakeQuery {
  private filters = new Map<string, unknown>();
  private inFilters = new Map<string, string[]>();

  constructor(private readonly rows: TableRow[], private readonly error: unknown = null) {}

  select() {
    return this;
  }

  eq(key: string, value: unknown) {
    this.filters.set(key, value);
    return this;
  }

  in(key: string, value: string[]) {
    this.inFilters.set(key, value);
    return this;
  }

  then<TResult1 = {data: TableRow[]; error: unknown}, TResult2 = never>(
    onfulfilled?: ((value: {data: TableRow[]; error: unknown}) => TResult1 | PromiseLike<TResult1>) | null,
    onrejected?: ((reason: unknown) => TResult2 | PromiseLike<TResult2>) | null,
  ) {
    return Promise.resolve({data: this.matchingRows(), error: this.error}).then(onfulfilled, onrejected);
  }

  private matchingRows() {
    return this.rows.filter((row) => {
      for (const [key, value] of this.filters) {
        if (row[key] !== value) {
          return false;
        }
      }
      for (const [key, values] of this.inFilters) {
        if (!values.includes(String(row[key]))) {
          return false;
        }
      }
      return true;
    });
  }
}

class FakeSupabase {
  constructor(private readonly tables: Record<string, TableRow[]>) {}

  from(table: string) {
    return new FakeQuery(this.tables[table] ?? []);
  }
}

function baseRagRequestParams(overrides: Partial<Parameters<typeof buildRagRequestBody>[0]> = {}) {
  return {
    trimmedMessage: 'Können wir die Themen gemeinsam durchgehen?',
    userId: 'user-1',
    sourceTypes: ['pdf' as const],
    topK: 8,
    pdfIds: ['pdf-1'],
    recentMessages: [],
    sessionId: 'session-1',
    memorySourceIds: [],
    promptContext: {
      courseId: null,
      contextSummary: null,
      mode: 'guided_learning' as const,
      activeLearningState: {},
    },
    webMode: 'off' as const,
    useIntentClassifier: false,
    useRetrievalPlanner: false,
    sessionMode: 'guided_learning' as const,
    requestActiveLearningState: {mode: 'guided_learning' as const},
    activeLearningControl: undefined,
    language: 'de' as const,
    ...overrides,
  };
}

export async function assertGuidedLearningPrimerIsLoadedForSelectedPdfs() {
  const supabase = new FakeSupabase({
    pdfs: [{id: 'pdf-1', user_id: 'user-1', name: 'Alpha.pdf'}],
    rag_document_primers: [{
      source_id: 'pdf-1',
      source_type: 'pdf',
      user_id: 'user-1',
      title: 'Alpha',
      summary: 'Die Datei behandelt Ist-Prozesse.',
      main_topics: ['Ist-Prozess', 'BPMN'],
      key_terms: [],
      learning_objectives: [],
    }],
  });

  const primer = await loadGuidedLearningDocumentPrimer({
    supabase: supabase as never,
    userId: 'user-1',
    sessionMode: 'guided_learning',
    pdfIds: ['pdf-1'],
    maxChars: 1200,
  });

  if (!primer?.includes('Ist-Prozess')) {
    throw new Error(`Expected guided-learning primer text, got ${primer}`);
  }
}

export async function assertPrimerIsNotLoadedForFeynmanOrNormal() {
  const supabase = new FakeSupabase({
    pdfs: [{id: 'pdf-1', user_id: 'user-1', name: 'Alpha.pdf'}],
    rag_document_primers: [],
  });

  const feynmanPrimer = await loadGuidedLearningDocumentPrimer({
    supabase: supabase as never,
    userId: 'user-1',
    sessionMode: 'feynman',
    pdfIds: ['pdf-1'],
    maxChars: 1200,
  });
  const normalPrimer = await loadGuidedLearningDocumentPrimer({
    supabase: supabase as never,
    userId: 'user-1',
    sessionMode: 'normal',
    pdfIds: ['pdf-1'],
    maxChars: 1200,
  });

  if (feynmanPrimer !== undefined || normalPrimer !== undefined) {
    throw new Error('Expected primer loading to be disabled for feynman and normal modes.');
  }
}

export function assertRagBodySendsOnlyServerGeneratedPrimerForGuidedLearning() {
  const body = buildRagRequestBody(baseRagRequestParams({
    documentPrimer: 'Server generated primer',
  }));

  if (body.document_primer !== 'Server generated primer') {
    throw new Error(`Expected server-generated primer in RAG body, got ${JSON.stringify(body)}`);
  }
}

export function assertRagBodyOmitsPrimerForFeynmanNormalAndNoPdfCases() {
  const feynmanBody = buildRagRequestBody(baseRagRequestParams({
    sessionMode: 'feynman',
    requestActiveLearningState: {mode: 'feynman'},
    documentPrimer: 'Should not be sent',
  }));
  const normalBody = buildRagRequestBody(baseRagRequestParams({
    sessionMode: 'normal',
    requestActiveLearningState: {},
    documentPrimer: 'Should not be sent',
  }));
  const noPdfBody = buildRagRequestBody(baseRagRequestParams({
    pdfIds: [],
    documentPrimer: undefined,
  }));

  if (feynmanBody.document_primer || normalBody.document_primer || noPdfBody.document_primer) {
    throw new Error('Expected RAG body to omit document_primer when no server-generated guided primer exists.');
  }
}
