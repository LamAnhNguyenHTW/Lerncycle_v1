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

  revalidatePath('/');
  return {annotation: row as Annotation};
}
