/**
 * Supported public-facing locales for the marketing site.
 * `de` is the default (primary launch market).
 */
export const SUPPORTED_LOCALES = ['de', 'en'] as const;
export type Locale = (typeof SUPPORTED_LOCALES)[number];
export const DEFAULT_LOCALE: Locale = 'de';

export function isLocale(value: string | undefined | null): value is Locale {
  return value === 'de' || value === 'en';
}

/**
 * Picks the best matching marketing locale from an `Accept-Language` header.
 *
 * Lightweight in-house matcher — no third-party negotiation library.
 * Rules:
 *  - empty / missing header → `DEFAULT_LOCALE` (`de`)
 *  - returns the first listed tag whose primary subtag is `de` or `en`
 *  - quality values (`;q=…`) are ignored; the header's listed order is honoured
 *    (browsers already emit tags in preference order)
 *  - unsupported tags fall back to `DEFAULT_LOCALE`
 *
 * Examples:
 *   'de-DE,de;q=0.9,en;q=0.8' → 'de'
 *   'en-US,en;q=0.9'          → 'en'
 *   'fr-FR'                   → 'de'
 *   ''                        → 'de'
 *   undefined                 → 'de'
 */
export function pickLocaleFromAcceptLanguage(header: string | null | undefined): Locale {
  if (!header) return DEFAULT_LOCALE;

  const tags = header
    .split(',')
    .map((part) => part.split(';')[0].trim().toLowerCase())
    .filter(Boolean);

  for (const tag of tags) {
    const primary = tag.split('-')[0];
    if (primary === 'de' || primary === 'en') {
      return primary;
    }
  }

  return DEFAULT_LOCALE;
}
