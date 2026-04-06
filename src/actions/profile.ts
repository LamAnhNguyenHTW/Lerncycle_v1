'use server';

import {createClient} from '@/lib/supabase/server';
import {revalidatePath} from 'next/cache';

export async function getProfile() {
  const supabase = await createClient();
  const {data: {user}} = await supabase.auth.getUser();

  if (!user) return null;

  const {data, error} = await supabase
    .from('profiles')
    .select('id, display_name, avatar_name, avatar_url')
    .eq('id', user.id)
    .single();

  if (error) {
    console.error('Error fetching profile:', error);
    return null;
  }

  return data;
}

/**
 * Updates display_name and optionally switches avatar mode.
 * - When avatar_name is provided, it switches to preset mode (clears avatar_url).
 * - When avatar_name is null/empty, only display_name is updated.
 */
export async function updateProfile(formData: FormData): Promise<{error?: string}> {
  const supabase = await createClient();
  const {data: {user}} = await supabase.auth.getUser();

  if (!user) return {error: 'Not authenticated.'};

  const display_name = formData.get('display_name') as string;
  const avatar_name = (formData.get('avatar_name') as string) || null;
  const usePreset = formData.get('use_preset') === 'true';

  const updates: Record<string, unknown> = {
    display_name,
    updated_at: new Date().toISOString(),
  };

  if (usePreset && avatar_name) {
    updates.avatar_name = avatar_name;
    updates.avatar_url = null;
  }

  const {error} = await supabase
    .from('profiles')
    .update(updates)
    .eq('id', user.id);

  if (error) return {error: error.message};

  revalidatePath('/');
  return {};
}

/**
 * Uploads a profile picture to the public 'avatars' bucket and stores
 * the public URL in profiles.avatar_url. Clears avatar_name.
 */
export async function uploadAvatar(formData: FormData): Promise<{url?: string; error?: string}> {
  const supabase = await createClient();
  const {data: {user}} = await supabase.auth.getUser();

  if (!user) return {error: 'Not authenticated.'};

  const file = formData.get('avatar');
  if (!(file instanceof File)) return {error: 'No file provided.'};
  if (!file.type.startsWith('image/')) return {error: 'Only image files are allowed.'};

  const ext = file.name.split('.').pop() ?? 'jpg';
  const path = `${user.id}/avatar.${ext}`;

  const {error: uploadError} = await supabase.storage
    .from('avatars')
    .upload(path, file, {contentType: file.type, upsert: true});

  if (uploadError) return {error: uploadError.message};

  const {data: {publicUrl}} = supabase.storage.from('avatars').getPublicUrl(path);

  // Append cache-buster so the browser picks up the new image immediately
  const urlWithBuster = `${publicUrl}?v=${Date.now()}`;

  const {error: dbError} = await supabase
    .from('profiles')
    .update({avatar_url: urlWithBuster, avatar_name: null, updated_at: new Date().toISOString()})
    .eq('id', user.id);

  if (dbError) return {error: dbError.message};

  revalidatePath('/');
  return {url: urlWithBuster};
}
