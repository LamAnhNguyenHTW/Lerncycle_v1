'use server';

import {createClient} from '@/lib/supabase/server';
import {redirect} from 'next/navigation';

/** Sends a magic link to the provided email address. */
export async function signInWithEmail(formData: FormData): Promise<{error?: string}> {
  const email = formData.get('email');

  if (typeof email !== 'string' || !email.includes('@')) {
    return {error: 'Please enter a valid email address.'};
  }

  const supabase = await createClient();
  const {error} = await supabase.auth.signInWithOtp({
    email,
    options: {
      emailRedirectTo: `${process.env.NEXT_PUBLIC_SITE_URL ?? 'http://localhost:3000'}/auth/callback`,
    },
  });

  if (error) {
    return {error: error.message};
  }

  redirect('/login?sent=true');
}

/** Signs the current user out. */
export async function signOut(): Promise<void> {
  const supabase = await createClient();
  await supabase.auth.signOut();
  redirect('/login');
}
