import Link from 'next/link';
import type {Locale} from '@/lib/locale';
import type {Dictionary} from '@/locales/types';
import {LocaleSwitcher} from './LocaleSwitcher';

interface Props {
  locale: Locale;
  dict: Dictionary;
}

export function MarketingTopNav({locale, dict}: Props) {
  return (
    <header className="sticky top-0 z-40 w-full border-b border-border bg-background/80 backdrop-blur">
      <div className="mx-auto flex h-16 w-full max-w-6xl items-center justify-between px-4 sm:px-6">
        <Link href={`/${locale}`} className="text-lg font-semibold tracking-tight text-foreground">
          LearnCycle
        </Link>

        <div className="flex items-center gap-2">
          <LocaleSwitcher
            currentLocale={locale}
            routeKey="landing"
            label={dict.footer.languageLabel}
          />
          <Link
            href="#beta"
            className="inline-flex h-9 items-center justify-center rounded-md bg-primary px-4 text-sm font-medium text-primary-foreground hover:opacity-90 transition-opacity"
          >
            {dict.nav.cta}
          </Link>
        </div>
      </div>
    </header>
  );
}
