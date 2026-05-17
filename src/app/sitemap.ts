import type {MetadataRoute} from 'next';
import {SUPPORTED_LOCALES, type Locale} from '@/lib/locale';
import {localeRoutes} from '@/locales/routes';
import {siteUrl} from '@/lib/site';

/**
 * All 8 public marketing URLs (2 locales × 4 pages):
 *   /de, /en — landings
 *   /de/datenschutz, /en/privacy
 *   /de/impressum,   /en/imprint
 *   /de/nutzungsbedingungen, /en/terms
 *
 * Each entry carries `alternates.languages` so search engines can
 * surface the correct locale.
 */
export default function sitemap(): MetadataRoute.Sitemap {
  const base = siteUrl();
  const now = new Date();

  const landings = SUPPORTED_LOCALES.map((locale: Locale) => ({
    url: `${base}/${locale}`,
    lastModified: now,
    alternates: {
      languages: {
        de: `${base}/de`,
        en: `${base}/en`,
        'x-default': `${base}/de`,
      },
    },
  }));

  const legal = (['privacy', 'imprint', 'terms'] as const).flatMap((key) =>
    SUPPORTED_LOCALES.map((locale) => ({
      url: `${base}/${locale}/${localeRoutes[key][locale]}`,
      lastModified: now,
      alternates: {
        languages: {
          de: `${base}/de/${localeRoutes[key].de}`,
          en: `${base}/en/${localeRoutes[key].en}`,
          'x-default': `${base}/de/${localeRoutes[key].de}`,
        },
      },
    })),
  );

  return [...landings, ...legal];
}
