import Link from 'next/link';
import type {Locale} from '@/lib/locale';
import {oppositeLocaleHref, type MarketingRouteKey} from '@/locales/routes';
import {Globe} from 'lucide-react';

interface Props {
  currentLocale: Locale;
  routeKey: MarketingRouteKey;
  label: string;
}

/**
 * Server Component locale switcher. Renders a plain `<Link>` to the
 * equivalent path in the other locale — no client-side state needed.
 */
export function LocaleSwitcher({currentLocale, routeKey, label}: Props) {
  const nextLocale: Locale = currentLocale === 'de' ? 'en' : 'de';
  const href = oppositeLocaleHref(currentLocale, routeKey);

  return (
    <Link
      href={href}
      aria-label={`${label}: ${nextLocale.toUpperCase()}`}
      className="inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-sm font-medium text-muted-foreground hover:bg-foreground/5 hover:text-foreground transition-colors"
      hrefLang={nextLocale}
    >
      <Globe className="h-4 w-4" aria-hidden="true" />
      <span>{nextLocale.toUpperCase()}</span>
    </Link>
  );
}
