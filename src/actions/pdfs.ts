'use server';

import {createClient} from '@/lib/supabase/server';
import {revalidatePath} from 'next/cache';

const BUCKET = 'pdfs';

/** Uploads a PDF to Supabase Storage and stores its metadata in the database. */
export async function uploadPdf(
  targetId: string,
  targetType: 'course' | 'folder',
  formData: FormData,
): Promise<{error?: string}> {
  try {
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

    let courseId: string;
    let folderId: string | null = null;

    if (targetType === 'course') {
      courseId = targetId;
    } else {
      const {data: folder, error: folderError} = await supabase
        .from('folders')
        .select('course_id')
        .eq('id', targetId)
        .single();

      if (folderError || !folder) {
        return {error: 'Folder not found.'};
      }

      courseId = folder.course_id;
      folderId = targetId;
    }

    const safeName = file.name.replace(/[^\w.\-]+/g, '_');
    const storagePath = `${user.id}/${targetId}/${Date.now()}_${safeName}`;
    const fileBytes = new Uint8Array(await file.arrayBuffer());

    let uploadErrorMessage: string | null = null;
    for (let attempt = 1; attempt <= 2; attempt += 1) {
      const {error: uploadError} = await supabase.storage
        .from(BUCKET)
        .upload(storagePath, fileBytes, {
          contentType: 'application/pdf',
          upsert: false,
        });

      if (!uploadError) {
        uploadErrorMessage = null;
        break;
      }

      uploadErrorMessage = uploadError.message;
      const isRetryable = /fetch failed|network|timeout|connection/i.test(
        uploadError.message,
      );
      if (!isRetryable || attempt === 2) {
        break;
      }
    }

    if (uploadErrorMessage) {
      return {error: `Upload failed: ${uploadErrorMessage}`};
    }

    const {data: pdfRow, error: dbError} = await supabase
      .from('pdfs')
      .insert({
        user_id: user.id,
        course_id: courseId,
        folder_id: folderId,
        name: file.name,
        storage_path: storagePath,
        size_bytes: file.size,
      })
      .select('id')
      .single();

    if (dbError) {
      // Roll back the storage upload to avoid orphaned files.
      await supabase.storage.from(BUCKET).remove([storagePath]);
      return {error: dbError.message};
    }

    const {error: jobError} = await supabase.from('rag_index_jobs').insert({
      user_id: user.id,
      source_type: 'pdf',
      source_id: pdfRow.id,
      pdf_id: pdfRow.id,
      job_kind: 'index_source',
      status: 'pending',
    });

    if (jobError) {
      return {
        error: `PDF uploaded, but indexing could not be queued: ${jobError.message}`,
      };
    }

    revalidatePath('/');
    return {};
  } catch (error) {
    return {
      error:
        error instanceof Error
          ? `Upload failed: ${error.message}`
          : 'Upload failed unexpectedly.',
    };
  }
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
