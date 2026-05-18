import Link from 'next/link';
import type {Locale} from '@/lib/locale';
import type {Dictionary} from '@/locales/types';
import {localeRoutes} from '@/locales/routes';
import {LocaleSwitcher} from './LocaleSwitcher';

interface Props {
  locale: Locale;
  dict: Dictionary;
}

export function MarketingFooter({locale, dict}: Props) {
  const legalLinks = [
    {key: 'imprint' as const, label: dict.footer.imprint},
    {key: 'privacy' as const, label: dict.footer.privacy},
    {key: 'terms' as const, label: dict.footer.terms},
  ];

  return (
    <footer className="border-t border-border bg-background">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-4 px-4 py-10 sm:flex-row sm:items-center sm:justify-between sm:px-6">
        <p className="text-sm text-muted-foreground">{dict.footer.copyright}</p>

        <nav className="flex flex-wrap items-center gap-x-4 gap-y-2 text-sm">
          {legalLinks.map((item) => (
            <Link
              key={item.key}
              href={`/${locale}/${localeRoutes[item.key][locale]}`}
              className="text-muted-foreground hover:text-foreground transition-colors"
            >
              {item.label}
            </Link>
          ))}
          <LocaleSwitcher
            currentLocale={locale}
            routeKey="landing"
            label={dict.footer.languageLabel}
          />
        </nav>
      </div>
    </footer>
  );
}
