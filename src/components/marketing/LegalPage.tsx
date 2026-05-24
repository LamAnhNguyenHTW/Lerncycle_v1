import type {Locale} from '@/lib/locale';
import type {Dictionary, LegalPageCopy} from '@/locales/types';
import {MarketingTopNav} from './MarketingTopNav';
import {MarketingFooter} from './MarketingFooter';
import type {MarketingRouteKey} from '@/locales/routes';

interface Props {
  locale: Locale;
  dict: Dictionary;
  routeKey: Extract<MarketingRouteKey, 'privacy' | 'imprint' | 'terms'>;
  copy: LegalPageCopy;
}

/**
 * Shared layout for Datenschutz / Impressum / Nutzungshinweis.
 * The visible draft banner is mandatory until lawyer-reviewed text lands.
 */
export function LegalPage({locale, dict, copy}: Props) {
  return (
    <div className="flex min-h-screen flex-col">
      <MarketingTopNav locale={locale} dict={dict} />
      <main className="flex-1">
        <article className="mx-auto w-full max-w-3xl px-4 py-16 sm:px-6 sm:py-20">
          <header className="mb-10">
            <h1 className="text-3xl sm:text-4xl font-bold tracking-tight text-foreground">
              {copy.title}
            </h1>
            <p className="mt-3 text-sm text-muted-foreground">{copy.lastUpdated}</p>
            <p
              role="note"
              className="mt-6 rounded-lg border border-amber-300/40 bg-amber-50/60 px-4 py-3 text-sm text-amber-900"
            >
              {dict.legal.draftBanner}
            </p>
          </header>

          <div className="flex flex-col gap-8">
            {copy.sections.map((section, idx) => (
              <section key={idx}>
                <h2 className="text-lg font-semibold text-foreground">{section.heading}</h2>
                <p className="mt-2 whitespace-pre-line text-base text-muted-foreground leading-relaxed">
                  {section.body}
                </p>
              </section>
            ))}
          </div>
        </article>
      </main>
      <MarketingFooter locale={locale} dict={dict} />
    </div>
  );
}
