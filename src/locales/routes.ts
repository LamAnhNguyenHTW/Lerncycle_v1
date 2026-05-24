import type {Locale} from '@/lib/locale';

/**
 * Marketing route keys whose URL slug differs per locale.
 * `landing` is the bare `/[locale]` page and has no slug.
 */
export type MarketingRouteKey = 'landing' | 'privacy' | 'imprint' | 'terms';

/**
 * Per-locale slug for each marketing route. `landing` is empty so
 * `oppositeLocaleHref` resolves it to `/de` / `/en`.
 */
export const localeRoutes: Record<Exclude<MarketingRouteKey, 'landing'>, Record<Locale, string>> = {
  privacy: {de: 'datenschutz', en: 'privacy'},
  imprint: {de: 'impressum', en: 'imprint'},
  terms: {de: 'nutzungsbedingungen', en: 'terms'},
};

/**
 * Inverse lookup: given a localised slug, return the canonical route key.
 * Used by the dynamic `[legalSlug]` page added in Phase 4.
 */
export function routeKeyFromSlug(locale: Locale, slug: string): MarketingRouteKey | null {
  for (const key of Object.keys(localeRoutes) as Array<Exclude<MarketingRouteKey, 'landing'>>) {
    if (localeRoutes[key][locale] === slug) return key;
  }
  return null;
}

/**
 * Build the href of the same logical page in the opposite locale.
 * `oppositeLocaleHref('de', 'privacy')` → `/en/privacy`.
 */
export function oppositeLocaleHref(currentLocale: Locale, routeKey: MarketingRouteKey): string {
  const nextLocale: Locale = currentLocale === 'de' ? 'en' : 'de';
  if (routeKey === 'landing') return `/${nextLocale}`;
  return `/${nextLocale}/${localeRoutes[routeKey][nextLocale]}`;
}
