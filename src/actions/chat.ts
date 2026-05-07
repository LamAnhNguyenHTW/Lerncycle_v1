'use server';

import {createClient} from '@/lib/supabase/server';
import {revalidatePath} from 'next/cache';

export async function deleteChatSession(sessionId: string) {
  const supabase = await createClient();
  const {data: {user}} = await supabase.auth.getUser();

  if (!user) {
    throw new Error('Unauthorized');
  }

  const {error} = await supabase
    .from('chat_sessions')
    .delete()
    .eq('id', sessionId)
    .eq('user_id', user.id);

  if (error) {
    console.error('Failed to delete chat session:', error);
    throw new Error('Failed to delete chat session');
  }
}

export async function renameChatSession(sessionId: string, newTitle: string) {
  const supabase = await createClient();
  const {data: {user}} = await supabase.auth.getUser();

  if (!user) {
    throw new Error('Unauthorized');
  }

  const {error} = await supabase
    .from('chat_sessions')
    .update({title: newTitle.trim().slice(0, 80) || 'Untitled chat'})
    .eq('id', sessionId)
    .eq('user_id', user.id);

  if (error) {
    console.error('Failed to rename chat session:', error);
    throw new Error('Failed to rename chat session');
  }
}
