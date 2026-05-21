'use server';

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

  const {error} = await supabase
    .from('courses')
    .delete()
    .eq('id', id)
    .eq('user_id', user.id);

  if (error) return {error: error.message};

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
