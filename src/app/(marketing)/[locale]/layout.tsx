import type {Metadata} from 'next';
import {Inter} from 'next/font/google';
import {notFound} from 'next/navigation';
import {isLocale, SUPPORTED_LOCALES, type Locale} from '@/lib/locale';
import {siteUrl} from '@/lib/site';
import '../../globals.css';

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-sans',
});

interface LayoutProps {
  children: React.ReactNode;
  params: Promise<{locale: string}>;
}

/**
 * Marketing root layout — owns `<html>`/`<body>` for every public-facing
 * marketing page (landing + legal). Drives `<html lang>` from the
 * `[locale]` segment. No `LanguageProvider`: marketing copy is pulled
 * statically per request from `src/locales/`.
 */
export default async function MarketingLayout({children, params}: LayoutProps) {
  const {locale} = await params;
  if (!isLocale(locale)) notFound();

  return (
    <html lang={locale} className={`${inter.variable} h-full antialiased`}>
      <body className="min-h-full bg-background text-foreground font-sans">{children}</body>
    </html>
  );
}

export function generateStaticParams(): {locale: Locale}[] {
  return SUPPORTED_LOCALES.map((locale) => ({locale}));
}

/**
 * Default marketing metadata. Pages may override `title`/`description`
 * via their own `generateMetadata`. `alternates.languages` provides
 * `hreflang` between `/de` and `/en` plus `x-default` → `/de`.
 */
export async function generateMetadata({params}: {params: Promise<{locale: string}>}): Promise<Metadata> {
  const {locale} = await params;
  if (!isLocale(locale)) return {};

  const title = locale === 'de'
    ? 'LearnCycle — Dein KI-Lernsystem'
    : 'LearnCycle — Your AI learning system';
  const description = locale === 'de'
    ? 'LearnCycle verbindet PDFs, Notizen, KI-Chat und Wiederholung in einem geschlossenen Lernzyklus.'
    : 'LearnCycle connects PDFs, notes, AI chat, and review into one closed learning cycle.';

  return {
    metadataBase: new URL(siteUrl()),
    title,
    description,
    alternates: {
      languages: {
        de: '/de',
        en: '/en',
        'x-default': '/de',
      },
    },
    openGraph: {
      type: 'website',
      locale: locale === 'de' ? 'de_DE' : 'en_US',
      url: `/${locale}`,
      siteName: 'LearnCycle',
      title,
      description,
    },
    twitter: {
      card: 'summary_large_image',
      title,
      description,
    },
  };
}
