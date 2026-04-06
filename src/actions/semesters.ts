'use server';

import {createClient} from '@/lib/supabase/server';
import {revalidatePath} from 'next/cache';

/** Creates a new semester for the authenticated user. */
export async function createSemester(name: string): Promise<{error?: string}> {
  const supabase = await createClient();
  const {data: {user}} = await supabase.auth.getUser();

  if (!user) {
    return {error: 'Not authenticated.'};
  }

  const {error} = await supabase.from('semesters').insert({name, user_id: user.id});

  if (error) {
    return {error: error.message};
  }

  revalidatePath('/');
  return {};
}

/** Deletes a semester (and cascades to subjects/weeks/pdfs). */
export async function deleteSemester(id: string): Promise<{error?: string}> {
  const supabase = await createClient();
  const {error} = await supabase.from('semesters').delete().eq('id', id);

  if (error) {
    return {error: error.message};
  }

  revalidatePath('/');
  return {};
}
