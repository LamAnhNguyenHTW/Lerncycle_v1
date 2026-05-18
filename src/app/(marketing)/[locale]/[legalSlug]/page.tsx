import type {Metadata} from 'next';
import {notFound} from 'next/navigation';
import {isLocale, SUPPORTED_LOCALES, type Locale} from '@/lib/locale';
import {getDictionary} from '@/locales';
import {localeRoutes, routeKeyFromSlug, type MarketingRouteKey} from '@/locales/routes';
import {LegalPage} from '@/components/marketing/LegalPage';

interface PageProps {
  params: Promise<{locale: string; legalSlug: string}>;
}

type LegalRouteKey = Extract<MarketingRouteKey, 'privacy' | 'imprint' | 'terms'>;

function resolve(locale: string, slug: string): {locale: Locale; key: LegalRouteKey} | null {
  if (!isLocale(locale)) return null;
  const key = routeKeyFromSlug(locale, slug);
  if (key === null || key === 'landing') return null;
  return {locale, key};
}

export default async function LegalRoute({params}: PageProps) {
  const {locale, legalSlug} = await params;
  const resolved = resolve(locale, legalSlug);
  if (!resolved) notFound();

  const dict = getDictionary(resolved.locale);
  const copy = dict.legal[resolved.key];

  return (
    <LegalPage locale={resolved.locale} dict={dict} routeKey={resolved.key} copy={copy} />
  );
}

export function generateStaticParams(): {locale: Locale; legalSlug: string}[] {
  const out: {locale: Locale; legalSlug: string}[] = [];
  for (const locale of SUPPORTED_LOCALES) {
    for (const key of Object.keys(localeRoutes) as LegalRouteKey[]) {
      out.push({locale, legalSlug: localeRoutes[key][locale]});
    }
  }
  return out;
}

export async function generateMetadata({params}: PageProps): Promise<Metadata> {
  const {locale, legalSlug} = await params;
  const resolved = resolve(locale, legalSlug);
  if (!resolved) return {};

  const dict = getDictionary(resolved.locale);
  const copy = dict.legal[resolved.key];

  const languages: Record<string, string> = {
    'x-default': `/de/${localeRoutes[resolved.key].de}`,
  };
  for (const l of SUPPORTED_LOCALES) {
    languages[l] = `/${l}/${localeRoutes[resolved.key][l]}`;
  }

  return {
    title: `${copy.title} — Lerncycle`,
    description: dict.legal.draftBanner,
    alternates: {languages},
  };
}
