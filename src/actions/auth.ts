'use server';

import {createClient} from '@/lib/supabase/server';
import {redirect} from 'next/navigation';

/**
 * Signs the user in with email + password.
 *
 * Beta auth: accounts are admin-created in the Supabase dashboard. Self-service
 * signup, magic link, and password reset are out of scope until a custom domain
 * + SMTP provider are configured. Errors are returned generically to avoid
 * leaking which emails are registered.
 */
export async function signInWithPassword(formData: FormData): Promise<{error?: string}> {
  const email = formData.get('email');
  const password = formData.get('password');

  if (
    typeof email !== 'string' ||
    typeof password !== 'string' ||
    email.length === 0 ||
    password.length === 0 ||
    !email.includes('@')
  ) {
    return {error: 'Invalid email or password.'};
  }

  const supabase = await createClient();
  const {error} = await supabase.auth.signInWithPassword({email, password});

  if (error) {
    return {error: 'Invalid email or password.'};
  }

  redirect('/app');
}

/** Signs the current user out. */
export async function signOut(): Promise<void> {
  const supabase = await createClient();
  await supabase.auth.signOut();
  redirect('/login');
}
