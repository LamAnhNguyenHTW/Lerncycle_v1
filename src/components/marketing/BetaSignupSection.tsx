import type { Locale } from '@/lib/locale';
import type { Dictionary } from '@/locales/types';
import { localeRoutes } from '@/locales/routes';
import { BetaSignupForm } from './BetaSignupForm';
import { ScrollReveal } from '@/components/marketing/ScrollReveal';

interface Props {
  locale: Locale;
  dict: Dictionary;
}

export function BetaSignupSection({ locale, dict }: Props) {
  const privacyHref = `/${locale}/${localeRoutes.privacy[locale]}`;
  const termsHref = `/${locale}/${localeRoutes.terms[locale]}`;
  return (
    <section
      id="beta"
      className="w-full px-4 py-24 sm:px-6 sm:py-32"
    >
      <ScrollReveal className="mx-auto w-full max-w-3xl text-center">
        <h2 className="text-3xl sm:text-4xl font-bold tracking-tight text-foreground">
          {dict.signup.headline}
        </h2>
        <p className="mt-6 text-lg sm:text-xl text-muted-foreground leading-relaxed max-w-xl mx-auto text-balance">
          {dict.signup.body}
        </p>
        <div className="mt-10 max-w-md mx-auto">
          <BetaSignupForm
            dict={dict}
            locale={locale}
            privacyHref={privacyHref}
            termsHref={termsHref}
          />
        </div>
      </ScrollReveal>
    </section>
  );
}
