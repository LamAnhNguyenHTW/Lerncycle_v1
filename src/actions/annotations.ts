'use server';

import {createClient} from '@/lib/supabase/server';
import {revalidatePath} from 'next/cache';

export interface HighlightArea {
  height: number;
  left: number;
  pageIndex: number;
  top: number;
  width: number;
}

export interface Annotation {
  id: string;
  pdf_id: string;
  user_id: string;
  page_index: number;
  highlight_areas: HighlightArea[];
  quote: string | null;
  comment: string | null;
  color: string;
  created_at: string;
}

async function enqueueAnnotationIndexJob(
  userId: string,
  annotationId: string,
  pdfId: string,
): Promise<{error?: string}> {
  const supabase = await createClient();

  const {data: existing, error: existingError} = await supabase
    .from('rag_index_jobs')
    .select('id')
    .eq('user_id', userId)
    .eq('source_type', 'annotation_comment')
    .eq('source_id', annotationId)
    .in('status', ['pending', 'processing'])
    .maybeSingle();

  if (existingError) return {error: existingError.message};
  if (existing) return {};

  const {error} = await supabase.from('rag_index_jobs').insert({
    user_id: userId,
    source_type: 'annotation_comment',
    source_id: annotationId,
    annotation_id: annotationId,
    pdf_id: pdfId,
    job_kind: 'index_source',
    status: 'pending',
  });

  if (error && error.code !== '23505') return {error: error.message};
  return {};
}

/** Fetches all annotations for a PDF owned by the current user. */
export async function getAnnotations(
  pdfId: string,
): Promise<{annotations?: Annotation[]; error?: string}> {
  const supabase = await createClient();
  const {data: {user}} = await supabase.auth.getUser();

  if (!user) return {error: 'Not authenticated.'};

  const {data, error} = await supabase
    .from('pdf_annotations')
    .select('*')
    .eq('pdf_id', pdfId)
    .eq('user_id', user.id)
    .order('created_at', {ascending: true});

  if (error) return {error: error.message};
  return {annotations: (data ?? []) as Annotation[]};
}

/** Creates a new highlight annotation. */
export async function createAnnotation(
  pdfId: string,
  data: {
    page_index: number;
    highlight_areas: HighlightArea[];
    quote: string;
    comment: string;
    color?: string;
  },
): Promise<{annotation?: Annotation; error?: string}> {
  const supabase = await createClient();
  const {data: {user}} = await supabase.auth.getUser();

  if (!user) return {error: 'Not authenticated.'};

  const {data: row, error} = await supabase
    .from('pdf_annotations')
    .insert({
      pdf_id: pdfId,
      user_id: user.id,
      page_index: data.page_index,
      highlight_areas: data.highlight_areas,
      quote: data.quote,
      comment: data.comment,
      color: data.color ?? 'yellow',
    })
    .select()
    .single();

  if (error) return {error: error.message};

  const {error: jobError} = await enqueueAnnotationIndexJob(
    user.id,
    row.id,
    row.pdf_id,
  );
  if (jobError) return {error: jobError};

  revalidatePath('/');
  return {annotation: row as Annotation};
}

/** Deletes an annotation by id. */
export async function deleteAnnotation(
  id: string,
): Promise<{error?: string}> {
  const supabase = await createClient();
  const {data: {user}} = await supabase.auth.getUser();

  if (!user) return {error: 'Not authenticated.'};

  const {error} = await supabase
    .from('pdf_annotations')
    .delete()
    .eq('id', id)
    .eq('user_id', user.id);

  if (error) return {error: error.message};

  await supabase
    .from('rag_chunks')
    .delete()
    .eq('source_type', 'annotation_comment')
    .eq('source_id', id)
    .eq('user_id', user.id);

  revalidatePath('/');
  return {};
}

/** Updates editable fields of an annotation (currently comment and color). */
export async function updateAnnotation(
  id: string,
  data: {
    comment?: string | null;
    color?: string;
  },
): Promise<{annotation?: Annotation; error?: string}> {
  const supabase = await createClient();
  const {data: {user}} = await supabase.auth.getUser();

  if (!user) return {error: 'Not authenticated.'};

  const updates: Record<string, unknown> = {};
  if (data.comment !== undefined) updates.comment = data.comment;
  if (data.color !== undefined) updates.color = data.color;

  const {data: row, error} = await supabase
    .from('pdf_annotations')
    .update(updates)
    .eq('id', id)
    .eq('user_id', user.id)
    .select()
    .single();

  if (error) return {error: error.message};

  const {error: jobError} = await enqueueAnnotationIndexJob(
    user.id,
    row.id,
    row.pdf_id,
  );
  if (jobError) return {error: jobError};

  revalidatePath('/');
  return {annotation: row as Annotation};
}
