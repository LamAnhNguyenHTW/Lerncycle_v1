'use client';

import {signInWithEmail} from '@/actions/auth';
import {useSearchParams} from 'next/navigation';
import {useActionState} from 'react';
import {NotionIcon} from '@/components/NotionIcon';

type State = {error?: string} | undefined;

export function LoginForm() {
  const searchParams = useSearchParams();
  const sent = searchParams.get('sent') === 'true';
  const callbackError = searchParams.get('error');

  const [state, action, pending] = useActionState(
    (_prev: State, formData: FormData) => signInWithEmail(formData),
    undefined,
  );

  if (sent) {
    return (
      <div className="flex h-screen w-full items-center justify-center bg-[#F7F7F5]">
        <div className="card-notion w-full max-w-md mx-4 p-10 flex flex-col items-center text-center">
          <div className="flex size-20 items-center justify-center rounded-full bg-black/5 text-foreground mb-8">
            <NotionIcon name="ni-paper-plane" className="w-[40px] h-[40px]" />
          </div>
          <h1 className="text-3xl font-bold mb-4 text-foreground">Check your inbox!</h1>
          <p className="text-muted-foreground text-lg leading-relaxed">
            We've sent you a Magic Link. Click it to log in (no password required).
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen w-full items-center justify-center bg-[#F7F7F5]">
      <div className="card-notion w-full max-w-md mx-4 p-12 flex flex-col items-center">
        <div className="flex items-center gap-3 mb-12">
           <div className="flex size-12 items-center justify-center rounded-xl bg-black/5 text-foreground">
             <NotionIcon name="ni-folders" className="w-[28px] h-[28px]" />
           </div>
           <span className="font-bold text-2xl tracking-tight text-foreground">Learncycle</span>
        </div>
        
        <h1 className="text-3xl font-bold mb-4 text-center text-foreground">Welcome back</h1>
        <p className="text-muted-foreground text-center mb-10 text-base leading-relaxed">
          Your learning materials, well-structured and always within reach.
        </p>

        <form action={action} className="w-full flex flex-col gap-4">
          <input
            id="email"
            name="email"
            type="email"
            placeholder="E-Mail eingeben..."
            required
            autoComplete="email"
            className="w-full rounded-md border border-border px-4 py-3 text-base outline-none focus:border-primary transition-colors"
          />

          {(state?.error ?? callbackError) && (
            <p className="text-sm text-red-500 font-medium text-center bg-red-50 py-2 rounded-md" role="alert">
              {state?.error ?? 'Authentifizierung fehlgeschlagen.'}
            </p>
          )}

          <button type="submit" disabled={pending} className="button-notion button-notion-primary w-full justify-center py-3 text-base font-medium mt-2">
            {pending ? 'Lade...' : 'Mit Magic Link einloggen'}
          </button>
        </form>
      </div>
    </div>
  );
}
