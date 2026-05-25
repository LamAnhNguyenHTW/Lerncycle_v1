'use client';

import {signInWithPassword} from '@/actions/auth';
import {useSearchParams} from 'next/navigation';
import {useActionState, useState} from 'react';
import {Logo} from '@/components/Logo';
import {ThemeToggle} from '@/components/theme/ThemeToggle';

type State = {error?: string} | undefined;
type LoginLanguage = 'de' | 'en';

const copy = {
  de: {
    welcomeTitle: 'Willkommen zurück',
    welcomeBody:
      'Deine Lernmaterialien, sauber strukturiert und jederzeit griffbereit.',
    emailPlaceholder: 'E-Mail eingeben...',
    passwordPlaceholder: 'Passwort eingeben...',
    authFailed: 'Authentifizierung fehlgeschlagen.',
    loading: 'Lade...',
    submit: 'Einloggen',
    languageLabel: 'Sprache wechseln',
  },
  en: {
    welcomeTitle: 'Welcome back',
    welcomeBody:
      'Your learning materials, well-structured and always within reach.',
    emailPlaceholder: 'Enter email...',
    passwordPlaceholder: 'Enter password...',
    authFailed: 'Authentication failed.',
    loading: 'Loading...',
    submit: 'Log in',
    languageLabel: 'Switch language',
  },
} satisfies Record<LoginLanguage, Record<string, string>>;

export function LoginForm() {
  const searchParams = useSearchParams();
  const callbackError = searchParams.get('error');
  const [language, setLanguage] = useState<LoginLanguage>('de');
  const text = copy[language];

  const [state, action, pending] = useActionState(
    (_prev: State, formData: FormData) => signInWithPassword(formData),
    undefined,
  );

  return (
    <div className="flex h-screen w-full items-center justify-center bg-muted">
      <div className="card-notion w-full max-w-md mx-4 p-12 flex flex-col items-center">
        <div className="self-end flex items-center gap-1">
          <ThemeToggle />
          <LanguageToggle
            language={language}
            label={text.languageLabel}
            onToggle={() => setLanguage(language === 'de' ? 'en' : 'de')}
          />
        </div>
        <div className="mb-12 flex items-center justify-center">
          <Logo variant="horizontal" priority className="h-14 w-auto" />
        </div>

        <h1 className="text-3xl font-bold mb-4 text-center text-foreground">
          {text.welcomeTitle}
        </h1>
        <p className="text-muted-foreground text-center mb-10 text-base leading-relaxed">
          {text.welcomeBody}
        </p>

        <form action={action} className="w-full flex flex-col gap-4">
          <input
            id="email"
            name="email"
            type="email"
            placeholder={text.emailPlaceholder}
            required
            autoComplete="email"
            className="w-full rounded-md border border-border px-4 py-3 text-base outline-none focus:border-primary transition-colors"
          />

          <input
            id="password"
            name="password"
            type="password"
            placeholder={text.passwordPlaceholder}
            required
            autoComplete="current-password"
            className="w-full rounded-md border border-border px-4 py-3 text-base outline-none focus:border-primary transition-colors"
          />

          {(state?.error ?? callbackError) && (
            <p className="text-sm font-medium text-center rounded-md py-2 text-red-600 bg-red-50 dark:text-red-300 dark:bg-red-950/30" role="alert">
              {state?.error ?? text.authFailed}
            </p>
          )}

          <button type="submit" disabled={pending} className="button-notion button-notion-primary w-full justify-center py-3 text-base font-medium mt-2">
            {pending ? text.loading : text.submit}
          </button>
        </form>
      </div>
    </div>
  );
}

function LanguageToggle({
  language,
  label,
  onToggle,
}: {
  language: LoginLanguage;
  label: string;
  onToggle: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onToggle}
      className="h-8 rounded-md px-2 text-xs font-semibold text-muted-foreground hover:bg-foreground/5 hover:text-foreground cursor-pointer"
      aria-label={label}
    >
      {language === 'de' ? 'EN' : 'DE'}
    </button>
  );
}
