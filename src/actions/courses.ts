'use server';

import {enqueueSourceDeleteJob} from '@/lib/rag-cleanup';
import {createClient} from '@/lib/supabase/server';
import {revalidatePath} from 'next/cache';

export async function createCourse(name: string): Promise<{error?: string; id?: string}> {
  const supabase = await createClient();
  const {data: {user}} = await supabase.auth.getUser();

  if (!user) return {error: 'Not authenticated.'};
  const trimmedName = name.trim();
  if (!trimmedName) return {error: 'Course name is required.'};

  const {data, error} = await supabase
    .from('courses')
    .insert({name: trimmedName, user_id: user.id})
    .select('id')
    .single();

  if (error) return {error: error.message};

  revalidatePath('/');
  return {id: data.id};
}

export async function deleteCourse(id: string): Promise<{error?: string}> {
  const supabase = await createClient();
  const {data: {user}} = await supabase.auth.getUser();

  if (!user) return {error: 'Not authenticated.'};

  // pdfs.course_id cascades on course deletion — capture the PDFs first so
  // storage objects and index data can be cleaned up as well.
  const {data: pdfs} = await supabase
    .from('pdfs')
    .select('id, storage_path')
    .eq('course_id', id)
    .eq('user_id', user.id);

  if (pdfs && pdfs.length > 0) {
    await supabase.storage.from('pdfs').remove(pdfs.map((p) => p.storage_path));
  }

  const {error} = await supabase
    .from('courses')
    .delete()
    .eq('id', id)
    .eq('user_id', user.id);

  if (error) return {error: error.message};

  for (const pdf of pdfs ?? []) {
    const {error: cleanupError} = await enqueueSourceDeleteJob({
      userId: user.id,
      sourceType: 'pdf',
      sourceId: pdf.id,
      deletedPdfId: pdf.id,
    });
    if (cleanupError) {
      console.error(
        `deleteCourse: cleanup job for pdf ${pdf.id} not queued: ${cleanupError}`,
      );
    }
  }

  revalidatePath('/');
  return {};
}

export async function updateCourse(id: string, name: string): Promise<{error?: string}> {
  const supabase = await createClient();
  const {data: {user}} = await supabase.auth.getUser();

  if (!user) return {error: 'Not authenticated.'};
  const trimmedName = name.trim();
  if (!trimmedName) return {error: 'Course name is required.'};

  const {error} = await supabase
    .from('courses')
    .update({name: trimmedName})
    .eq('id', id)
    .eq('user_id', user.id);

  if (error) return {error: error.message};

  revalidatePath('/');
  return {};
}
