'use client';

import {signInWithEmail} from '@/actions/auth';
import {useSearchParams} from 'next/navigation';
import {useActionState, useState} from 'react';
import {NotionIcon} from '@/components/NotionIcon';
import {BookOpen, Send} from 'lucide-react';

type State = {error?: string} | undefined;
type LoginLanguage = 'de' | 'en';

const copy = {
  de: {
    checkInboxTitle: 'Postfach prüfen',
    checkInboxBody:
      'Wir haben dir einen Magic Link geschickt. Klicke darauf, um dich ohne Passwort einzuloggen.',
    welcomeTitle: 'Willkommen zurück',
    welcomeBody:
      'Deine Lernmaterialien, sauber strukturiert und jederzeit griffbereit.',
    emailPlaceholder: 'E-Mail eingeben...',
    authFailed: 'Authentifizierung fehlgeschlagen.',
    loading: 'Lade...',
    submit: 'Mit Magic Link einloggen',
    languageLabel: 'Sprache wechseln',
  },
  en: {
    checkInboxTitle: 'Check your inbox',
    checkInboxBody:
      "We've sent you a Magic Link. Click it to log in without a password.",
    welcomeTitle: 'Welcome back',
    welcomeBody:
      'Your learning materials, well-structured and always within reach.',
    emailPlaceholder: 'Enter email...',
    authFailed: 'Authentication failed.',
    loading: 'Loading...',
    submit: 'Log in with Magic Link',
    languageLabel: 'Switch language',
  },
} satisfies Record<LoginLanguage, Record<string, string>>;

export function LoginForm() {
  const searchParams = useSearchParams();
  const sent = searchParams.get('sent') === 'true';
  const callbackError = searchParams.get('error');
  const [language, setLanguage] = useState<LoginLanguage>('de');
  const text = copy[language];

  const [state, action, pending] = useActionState(
    (_prev: State, formData: FormData) => signInWithEmail(formData),
    undefined,
  );

  if (sent) {
    return (
      <div className="flex h-screen w-full items-center justify-center bg-[#F7F7F5]">
        <div className="card-notion w-full max-w-md mx-4 p-10 flex flex-col items-center text-center">
          <LanguageToggle
            language={language}
            label={text.languageLabel}
            onToggle={() => setLanguage(language === 'de' ? 'en' : 'de')}
          />
          <div className="flex size-20 items-center justify-center rounded-full bg-black/5 text-foreground mb-8">
            <Send className="w-[40px] h-[40px]" strokeWidth={1.5} />
          </div>
          <h1 className="text-3xl font-bold mb-4 text-foreground">
            {text.checkInboxTitle}
          </h1>
          <p className="text-muted-foreground text-lg leading-relaxed">
            {text.checkInboxBody}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen w-full items-center justify-center bg-[#F7F7F5]">
      <div className="card-notion w-full max-w-md mx-4 p-12 flex flex-col items-center">
        <LanguageToggle
          language={language}
          label={text.languageLabel}
          onToggle={() => setLanguage(language === 'de' ? 'en' : 'de')}
        />
        <div className="flex items-center gap-3 mb-12">
           <div className="flex size-12 items-center justify-center rounded-xl bg-black/5 text-foreground">
             <BookOpen className="w-[24px] h-[24px]" strokeWidth={1.75} />
           </div>
           <span className="font-bold text-2xl tracking-tight text-foreground">Learncycle</span>
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

          {(state?.error ?? callbackError) && (
            <p className="text-sm text-red-500 font-medium text-center bg-red-50 py-2 rounded-md" role="alert">
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
      className="self-end rounded-md px-2 py-1 text-xs font-semibold text-muted-foreground hover:bg-black/5 hover:text-foreground"
      aria-label={label}
    >
      {language === 'de' ? 'EN' : 'DE'}
    </button>
  );
}
