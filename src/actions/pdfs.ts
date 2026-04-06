'use server';

import {createClient} from '@/lib/supabase/server';
import {revalidatePath} from 'next/cache';

const BUCKET = 'pdfs';

/** Uploads a PDF to Supabase Storage and stores its metadata in the database. */
export async function uploadPdf(
  weekId: string,
  formData: FormData,
): Promise<{error?: string}> {
  const supabase = await createClient();
  const {data: {user}} = await supabase.auth.getUser();

  if (!user) {
    return {error: 'Not authenticated.'};
  }

  const file = formData.get('file');

  if (!(file instanceof File)) {
    return {error: 'No file provided.'};
  }

  if (file.type !== 'application/pdf') {
    return {error: 'Only PDF files are allowed.'};
  }

  const storagePath = `${user.id}/${weekId}/${Date.now()}_${file.name}`;

  const {error: uploadError} = await supabase.storage
    .from(BUCKET)
    .upload(storagePath, file, {contentType: 'application/pdf', upsert: false});

  if (uploadError) {
    return {error: uploadError.message};
  }

  const {error: dbError} = await supabase.from('pdfs').insert({
    week_id: weekId,
    user_id: user.id,
    name: file.name,
    storage_path: storagePath,
    size_bytes: file.size,
  });

  if (dbError) {
    // Roll back the storage upload to avoid orphaned files.
    await supabase.storage.from(BUCKET).remove([storagePath]);
    return {error: dbError.message};
  }

  revalidatePath('/');
  return {};
}

/** Generates a short-lived signed URL for viewing a private PDF. */
export async function getPdfSignedUrl(
  storagePath: string,
  expiresInSeconds = 3600,
): Promise<{url?: string; error?: string}> {
  const supabase = await createClient();
  const {data, error} = await supabase.storage
    .from(BUCKET)
    .createSignedUrl(storagePath, expiresInSeconds);

  if (error) {
    return {error: error.message};
  }

  return {url: data.signedUrl};
}

/** Deletes a PDF from Storage and removes its metadata row. */
export async function deletePdf(
  id: string,
  storagePath: string,
): Promise<{error?: string}> {
  const supabase = await createClient();

  const {error: storageError} = await supabase.storage
    .from(BUCKET)
    .remove([storagePath]);

  if (storageError) {
    return {error: storageError.message};
  }

  const {error: dbError} = await supabase.from('pdfs').delete().eq('id', id);

  if (dbError) {
    return {error: dbError.message};
  }

  revalidatePath('/');
  return {};
}
