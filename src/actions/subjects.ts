'use server';

import {createClient} from '@/lib/supabase/server';
import {revalidatePath} from 'next/cache';

/** Creates a new subject within a semester. */
export async function createSubject(
  semesterId: string,
  name: string,
): Promise<{error?: string}> {
  const supabase = await createClient();
  const {data: {user}} = await supabase.auth.getUser();

  if (!user) {
    return {error: 'Not authenticated.'};
  }

  const {error} = await supabase
    .from('subjects')
    .insert({name, semester_id: semesterId, user_id: user.id});

  if (error) {
    return {error: error.message};
  }

  revalidatePath('/');
  return {};
}

/** Deletes a subject (and cascades to weeks/pdfs). */
export async function deleteSubject(id: string): Promise<{error?: string}> {
  const supabase = await createClient();
  const {error} = await supabase.from('subjects').delete().eq('id', id);

  if (error) {
    return {error: error.message};
  }

  revalidatePath('/');
  return {};
}
