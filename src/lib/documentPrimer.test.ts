import {buildDocumentPrimerText, loadDocumentPrimers, loadOwnedPdfs} from './documentPrimer';

type TableRow = Record<string, unknown>;

class FakeQuery {
  private filters = new Map<string, unknown>();
  private inFilters = new Map<string, string[]>();

  constructor(private readonly rows: TableRow[]) {}

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

  then<TResult1 = {data: TableRow[]; error: null}, TResult2 = never>(
    onfulfilled?: ((value: {data: TableRow[]; error: null}) => TResult1 | PromiseLike<TResult1>) | null,
    onrejected?: ((reason: unknown) => TResult2 | PromiseLike<TResult2>) | null,
  ) {
    return Promise.resolve({data: this.matchingRows(), error: null}).then(onfulfilled, onrejected);
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

export function assertDocumentPrimerTextIsCappedAndBounded() {
  const primer = buildDocumentPrimerText({
    pdfs: [
      {id: 'pdf-1', name: 'Alpha.pdf'},
      {id: 'pdf-2', name: 'Beta.pdf'},
      {id: 'pdf-3', name: 'Gamma.pdf'},
      {id: 'pdf-4', name: 'Delta.pdf'},
    ],
    primers: [
      {
        source_id: 'pdf-1',
        title: 'Alpha',
        summary: 'A'.repeat(80),
        main_topics: ['Ist-Prozess', 'Einsatzplanung'],
        key_terms: ['BPMN'],
        learning_objectives: ['Explain the current process'],
      },
      {
        source_id: 'pdf-4',
        title: 'Delta',
        summary: 'This should not appear for the fourth document.',
        main_topics: ['Late topic'],
        key_terms: ['Hidden term'],
        learning_objectives: [],
      },
    ],
    maxChars: 220,
  });
  if (primer.length > 220 || !primer.includes('Alpha') || !primer.includes('Document 4: Delta')) {
    throw new Error(`Expected capped multi-document primer, got ${primer}`);
  }
  if (primer.includes('This should not appear')) {
    throw new Error(`Expected documents after the third to omit detailed summaries, got ${primer}`);
  }
}

export async function assertDocumentPrimerLoadersAreUserScoped() {
  const supabase = new FakeSupabase({
    pdfs: [
      {id: 'pdf-1', user_id: 'user-1', name: 'Alpha.pdf'},
      {id: 'pdf-2', user_id: 'user-2', name: 'Beta.pdf'},
    ],
    rag_document_primers: [
      {
        source_id: 'pdf-1',
        source_type: 'pdf',
        user_id: 'user-1',
        title: 'Alpha',
        summary: 'Summary',
        main_topics: ['Topic'],
        key_terms: [],
        learning_objectives: [],
      },
      {
        source_id: 'pdf-2',
        source_type: 'pdf',
        user_id: 'user-2',
        title: 'Beta',
        summary: 'Foreign summary',
        main_topics: ['Foreign'],
        key_terms: [],
        learning_objectives: [],
      },
    ],
  });

  const pdfs = await loadOwnedPdfs(supabase as never, 'user-1', ['pdf-1']);
  const primers = await loadDocumentPrimers(supabase as never, 'user-1', ['pdf-1', 'pdf-2']);

  if (pdfs.length !== 1 || pdfs[0]?.id !== 'pdf-1') {
    throw new Error(`Expected user-scoped PDFs, got ${JSON.stringify(pdfs)}`);
  }
  if (primers.length !== 1 || primers[0]?.source_id !== 'pdf-1') {
    throw new Error(`Expected user-scoped primers, got ${JSON.stringify(primers)}`);
  }
}
