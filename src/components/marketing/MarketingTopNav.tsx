import Link from 'next/link';
import type {Locale} from '@/lib/locale';
import type {Dictionary} from '@/locales/types';
import {Logo} from '@/components/Logo';
import {ThemeToggle} from '@/components/theme/ThemeToggle';
import {LocaleSwitcher} from './LocaleSwitcher';

interface Props {
  locale: Locale;
  dict: Dictionary;
}

export function MarketingTopNav({locale, dict}: Props) {
  return (
    <header className="sticky top-0 z-40 w-full border-b border-border bg-background/80 backdrop-blur">
      <div className="mx-auto flex h-16 sm:h-20 w-full max-w-6xl items-center justify-between gap-2 px-3 sm:px-6">
        <Link href={`/${locale}`} aria-label="LearnCycle" className="inline-flex items-center shrink-0">
          <Logo variant="horizontal" priority className="h-9 w-auto sm:h-11" />
        </Link>

        <div className="flex items-center gap-1 sm:gap-2">
          <ThemeToggle
            labels={{
              label: dict.theme.label,
              light: dict.theme.light,
              dark: dict.theme.dark,
              system: dict.theme.system,
              cycle: dict.theme.cycle,
            }}
          />
          <LocaleSwitcher
            currentLocale={locale}
            routeKey="landing"
            label={dict.footer.languageLabel}
          />
          <Link
            href="#beta"
            className="inline-flex h-10 sm:h-9 items-center justify-center whitespace-nowrap rounded-md bg-primary px-3 sm:px-4 text-xs sm:text-sm font-medium text-primary-foreground hover:opacity-90 transition-opacity"
          >
            {dict.nav.cta}
          </Link>
        </div>
      </div>
    </header>
  );
}
