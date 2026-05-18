'use client';

import Link from 'next/link';
import {useActionState} from 'react';
import {useFormStatus} from 'react-dom';
import {subscribeBeta, type SubscribeBetaResult} from '@/actions/beta';
import type {Dictionary} from '@/locales/types';

interface Props {
  dict: Dictionary;
  locale: 'de' | 'en';
  privacyHref: string;
  termsHref: string;
}

type State = SubscribeBetaResult | null;

async function action(_prev: State, formData: FormData): Promise<State> {
  return subscribeBeta(formData);
}

export function BetaSignupForm({dict, locale, privacyHref, termsHref}: Props) {
  const [state, formAction] = useActionState<State, FormData>(action, null);

  if (state?.ok) {
    return (
      <p
        role="status"
        className="rounded-lg border border-border bg-card p-4 text-sm text-foreground"
      >
        {state.alreadySubscribed ? dict.signup.alreadySubscribedHint : dict.signup.successHint}
      </p>
    );
  }

  return (
    <form action={formAction} className="flex flex-col gap-3">
      {/* Honeypot — invisible to humans, filled by spam bots. */}
      <label className="sr-only" aria-hidden="true">
        Company
        <input
          type="text"
          name="company"
          tabIndex={-1}
          autoComplete="off"
          defaultValue=""
        />
      </label>

      <input type="hidden" name="locale" value={locale} />
      <input type="hidden" name="source" value="landing_hero" />

      <label className="flex flex-col gap-1.5 text-left">
        <span className="text-sm font-medium text-foreground">{dict.signup.emailLabel}</span>
        <input
          type="email"
          name="email"
          required
          maxLength={254}
          autoComplete="email"
          placeholder={dict.signup.emailPlaceholder}
          className="h-11 rounded-md border border-border bg-background px-4 text-base text-foreground outline-none focus:border-primary transition-colors"
        />
      </label>

      <label className="flex items-start gap-2 text-left text-sm text-muted-foreground">
        <input
          type="checkbox"
          name="consent"
          required
          className="mt-1 h-4 w-4 rounded border-border"
        />
        <span>
          {dict.signup.consent}{' '}
          <Link href={privacyHref} className="underline hover:text-foreground">
            {dict.footer.privacy}
          </Link>
          {' · '}
          <Link href={termsHref} className="underline hover:text-foreground">
            {dict.footer.terms}
          </Link>
        </span>
      </label>

      {state && !state.ok && (
        <p
          role="alert"
          className="rounded-md border border-destructive/30 bg-destructive/5 px-3 py-2 text-sm text-destructive"
        >
          {state.error || dict.signup.errorHint}
        </p>
      )}

      <SubmitButton labelIdle={dict.signup.cta} labelBusy={dict.signup.submitting} />
    </form>
  );
}

function SubmitButton({labelIdle, labelBusy}: {labelIdle: string; labelBusy: string}) {
  const {pending} = useFormStatus();
  return (
    <button
      type="submit"
      disabled={pending}
      className="mt-1 inline-flex h-11 items-center justify-center rounded-md bg-primary px-6 text-base font-medium text-primary-foreground hover:opacity-90 transition-opacity disabled:opacity-60 disabled:cursor-not-allowed"
    >
      {pending ? labelBusy : labelIdle}
    </button>
  );
}
