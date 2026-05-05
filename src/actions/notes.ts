'use server';

import {createClient} from '@/lib/supabase/server';

// NoteContent matches TipTap's JSONContent schema stored as jsonb
export type NoteContent = Record<string, unknown>;

async function enqueueNoteIndexJob(
  userId: string,
  noteId: string,
  pdfId: string,
): Promise<{error?: string}> {
  const supabase = await createClient();

  const {data: existing, error: existingError} = await supabase
    .from('rag_index_jobs')
    .select('id')
    .eq('user_id', userId)
    .eq('source_type', 'note')
    .eq('source_id', noteId)
    .in('status', ['pending', 'processing'])
    .maybeSingle();

  if (existingError) return {error: existingError.message};
  if (existing) return {};

  const {error} = await supabase.from('rag_index_jobs').insert({
    user_id: userId,
    source_type: 'note',
    source_id: noteId,
    note_id: noteId,
    pdf_id: pdfId,
    status: 'pending',
  });

  if (error && error.code !== '23505') return {error: error.message};
  return {};
}

/** Fetches the note document for a specific PDF owned by the current user. */
export async function getNote(
  pdfId: string,
): Promise<{content?: NoteContent; error?: string}> {
  const supabase = await createClient();
  const {data: {user}} = await supabase.auth.getUser();

  if (!user) return {error: 'Not authenticated.'};

  const {data, error} = await supabase
    .from('notes')
    .select('content')
    .eq('pdf_id', pdfId)
    .eq('user_id', user.id)
    .maybeSingle();

  if (error) return {error: error.message};
  return {content: (data?.content ?? null) as NoteContent | undefined};
}

/** Upserts (insert or update) the single note document for a PDF. */
export async function upsertNote(
  pdfId: string,
  content: NoteContent,
): Promise<{error?: string}> {
  const supabase = await createClient();
  const {data: {user}} = await supabase.auth.getUser();

  if (!user) return {error: 'Not authenticated.'};

  const {data: note, error} = await supabase
    .from('notes')
    .upsert(
      {
        pdf_id: pdfId,
        user_id: user.id,
        content,
        updated_at: new Date().toISOString(),
      },
      {onConflict: 'user_id,pdf_id'},
    )
    .select('id, pdf_id')
    .single();

  if (error) return {error: error.message};

  const {error: jobError} = await enqueueNoteIndexJob(
    user.id,
    note.id,
    note.pdf_id,
  );
  if (jobError) return {error: jobError};

  // No revalidatePath — note content is managed in client state;
  // revalidating on every auto-save (800 ms debounce) would be excessive.
  return {};
}
