/**
 * Neutralized magic-link callback. Beta auth uses email + password
 * (`supabase.auth.signInWithPassword`); no OTP exchange happens here anymore.
 * The route is kept (rather than deleted) so old links, browser history, and
 * residual Supabase Redirect URL entries land softly on /login instead of 404.
 */
export async function GET(request: Request) {
  const {origin} = new URL(request.url);
  return Response.redirect(`${origin}/login`);
}
