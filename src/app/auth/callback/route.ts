import {createClient} from '@/lib/supabase/server';
import {redirect} from 'next/navigation';

/**
 * Auth callback route handler for Supabase magic link and OAuth flows.
 * Exchanges the code for a session and redirects to the app.
 */
export async function GET(request: Request) {
  const {searchParams, origin} = new URL(request.url);
  const code = searchParams.get('code');
  const next = searchParams.get('next') ?? '/';

  if (code) {
    const supabase = await createClient();
    const {error} = await supabase.auth.exchangeCodeForSession(code);
    if (!error) {
      return Response.redirect(`${origin}${next}`);
    }
  }

  // Auth failed — redirect to login with error state.
  return Response.redirect(`${origin}/login?error=auth_callback_failed`);
}
