'use server';

import {createClient} from '@/lib/supabase/server';
import {revalidatePath} from 'next/cache';

/** Creates a new week within a subject. */
export async function createWeek(
  subjectId: string,
  name: string,
): Promise<{error?: string}> {
  const supabase = await createClient();
  const {data: {user}} = await supabase.auth.getUser();

  if (!user) {
    return {error: 'Not authenticated.'};
  }

  const {error} = await supabase
    .from('weeks')
    .insert({name, subject_id: subjectId, user_id: user.id});

  if (error) {
    return {error: error.message};
  }

  revalidatePath('/');
  return {};
}

/** Deletes a week (and cascades to pdfs). */
export async function deleteWeek(id: string): Promise<{error?: string}> {
  const supabase = await createClient();
  const {error} = await supabase.from('weeks').delete().eq('id', id);

  if (error) {
    return {error: error.message};
  }

  revalidatePath('/');
  return {};
}
