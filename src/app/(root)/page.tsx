import {headers} from 'next/headers';
import {redirect} from 'next/navigation';
import {createClient} from '@/lib/supabase/server';
import {pickLocaleFromAcceptLanguage} from '@/lib/locale';

/**
 * Root redirector at `/`.
 *  - Authenticated session → `/app`
 *  - Otherwise → `/de` or `/en` based on the request's `Accept-Language` header
 *    (defaulting to `/de`).
 */
export default async function RootPage() {
  const supabase = await createClient();
  const {data: {user}} = await supabase.auth.getUser();

  if (user) {
    redirect('/app');
  }

  const headerList = await headers();
  const locale = pickLocaleFromAcceptLanguage(headerList.get('accept-language'));
  redirect(`/${locale}`);
}
