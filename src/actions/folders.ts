'use server';

import {createClient} from '@/lib/supabase/server';
import {revalidatePath} from 'next/cache';

export async function createFolder(
  courseId: string,
  name: string,
): Promise<{error?: string}> {
  const supabase = await createClient();
  const {data: {user}} = await supabase.auth.getUser();

  if (!user) return {error: 'Not authenticated.'};

  const {error} = await supabase
    .from('folders')
    .insert({name, course_id: courseId, user_id: user.id});

  if (error) return {error: error.message};

  revalidatePath('/');
  return {};
}

/**
 * Deletes a folder.
 * @param keepFiles - If true, PDFs inside become loose (attached to the course).
 *                    If false, PDFs are permanently deleted from storage and the DB.
 */
export async function deleteFolder(
  id: string,
  keepFiles: boolean,
): Promise<{error?: string}> {
  const supabase = await createClient();

  if (!keepFiles) {
    const {data: pdfs} = await supabase
      .from('pdfs')
      .select('id, storage_path')
      .eq('folder_id', id);

    if (pdfs && pdfs.length > 0) {
      await supabase.storage.from('pdfs').remove(pdfs.map((p) => p.storage_path));
      await supabase.from('pdfs').delete().eq('folder_id', id);
    }
  }
  // When keepFiles=true the ON DELETE SET NULL constraint on folder_id
  // automatically nullifies folder_id on all PDFs when the folder row is deleted.

  const {error} = await supabase.from('folders').delete().eq('id', id);
  if (error) return {error: error.message};

  revalidatePath('/');
  return {};
}
