'use server';

import {createClient} from '@/lib/supabase/server';

// NoteContent matches TipTap's JSONContent schema stored as jsonb
export type NoteContent = Record<string, unknown>;

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

  const {error} = await supabase.from('notes').upsert(
    {
      pdf_id: pdfId,
      user_id: user.id,
      content,
      updated_at: new Date().toISOString(),
    },
    {onConflict: 'user_id,pdf_id'},
  );

  if (error) return {error: error.message};

  // No revalidatePath — note content is managed in client state;
  // revalidating on every auto-save (800 ms debounce) would be excessive.
  return {};
}
