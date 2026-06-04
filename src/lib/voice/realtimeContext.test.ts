import {
  RealtimeContextError,
  buildRealtimeContext,
  persistRealtimeSessionSources,
  validateRealtimeContextBody,
} from './realtimeContext';

type TableRow = Record<string, unknown>;

class FakeQuery {
  private filters = new Map<string, unknown>();
  private inFilters = new Map<string, string[]>();

  constructor(
    private readonly rows: TableRow[],
    private readonly upserts: TableRow[] = [],
    private readonly error: unknown = null,
  ) {}

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

  upsert(value: TableRow) {
    this.upserts.push(value);
    return this;
  }

  maybeSingle() {
    return Promise.resolve({data: this.matchingRows()[0] ?? null, error: this.error});
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
  readonly upserts: Record<string, TableRow[]> = {};

  constructor(private readonly tables: Record<string, TableRow[]>) {}

  from(table: string) {
    this.upserts[table] ??= [];
    return new FakeQuery(this.tables[table] ?? [], this.upserts[table]);
  }
}

export function assertRealtimeContextBodyRequiresSession() {
  const result = validateRealtimeContextBody({sourceIds: ['pdf-1']});
  if (result.ok || !result.error.includes('sessionId')) {
    throw new Error(`Expected missing sessionId to fail, got ${JSON.stringify(result)}`);
  }
}

export function assertRealtimeContextBodyDedupesSourceIds() {
  const result = validateRealtimeContextBody({
    sessionId: ' session-1 ',
    sourceIds: ['pdf-1', 'pdf-1', ' pdf-2 '],
  });
  if (!result.ok || result.value.sessionId !== 'session-1' || result.value.sourceIds.join(',') !== 'pdf-1,pdf-2') {
    throw new Error(`Expected source IDs to validate and dedupe, got ${JSON.stringify(result)}`);
  }
}

export async function assertRealtimeContextRejectsForeignSource() {
  const supabase = new FakeSupabase({
    chat_sessions: [{id: 'session-1', user_id: 'user-1'}],
    pdfs: [{id: 'pdf-1', user_id: 'user-1', name: 'Alpha.pdf'}],
    rag_document_primers: [],
  });
  try {
    await buildRealtimeContext({
      supabase: supabase as never,
      userId: 'user-1',
      config: {enabled: true, realtimePrimerMaxChars: 4000},
      body: {sessionId: 'session-1', sourceIds: ['pdf-1', 'foreign-pdf'], mode: 'feynman'},
    });
  } catch (error) {
    if (error instanceof RealtimeContextError && error.status === 403) {
      return;
    }
    throw error;
  }
  throw new Error('Expected realtime context to reject a source not owned by the user.');
}

export async function assertRealtimeContextBuildsPrimerForOwnedSources() {
  const supabase = new FakeSupabase({
    chat_sessions: [{id: 'session-1', user_id: 'user-1'}],
    pdfs: [{id: 'pdf-1', user_id: 'user-1', name: 'Alpha.pdf'}],
    rag_document_primers: [{
      source_id: 'pdf-1',
      user_id: 'user-1',
      source_type: 'pdf',
      title: 'Alpha',
      summary: 'Compact summary',
      main_topics: ['Ist-Prozess'],
      key_terms: ['BPMN'],
      learning_objectives: ['Explain the current process'],
    }],
  });
  const context = await buildRealtimeContext({
    supabase: supabase as never,
    userId: 'user-1',
    config: {enabled: true, realtimePrimerMaxChars: 4000},
    body: {sessionId: 'session-1', sourceIds: ['pdf-1'], mode: 'feynman'},
  });
  if (
    context.allowedSourceIds.join(',') !== 'pdf-1' ||
    !context.documentPrimer.includes('Compact summary') ||
    !context.realtimeInstructions.includes('Force retrieval rule')
  ) {
    throw new Error(`Expected realtime context to build source-bound primer, got ${JSON.stringify(context)}`);
  }
}

export async function assertRealtimeSessionSourcesPersistAllowedSourcesOnly() {
  const supabase = new FakeSupabase({});
  await persistRealtimeSessionSources({
    supabase: supabase as never,
    userId: 'user-1',
    sessionId: 'session-1',
    allowedSourceIds: ['pdf-1'],
    mode: 'feynman',
  });
  const stored = supabase.upserts.voice_realtime_sessions?.[0];
  if (
    stored?.user_id !== 'user-1' ||
    stored.session_id !== 'session-1' ||
    !Array.isArray(stored.allowed_source_ids) ||
    stored.allowed_source_ids.join(',') !== 'pdf-1'
  ) {
    throw new Error(`Expected allowed sources to be persisted, got ${JSON.stringify(stored)}`);
  }
}
