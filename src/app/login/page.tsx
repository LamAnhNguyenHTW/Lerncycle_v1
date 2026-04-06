'use client';

import {signInWithEmail} from '@/actions/auth';
import {useSearchParams} from 'next/navigation';
import {useActionState} from 'react';
import {Input} from '@/components/ui/input';
import {Button} from '@/components/ui/button';

type State = {error?: string} | undefined;

function initialState(): State {
  return undefined;
}

/** Magic-link login page — Notion-style minimal design. */
export default function LoginPage() {
  const searchParams = useSearchParams();
  const sent = searchParams.get('sent') === 'true';
  const callbackError = searchParams.get('error');

  const [state, action, pending] = useActionState(
    async (_prev: State, formData: FormData) => signInWithEmail(formData),
    undefined,
  );

  if (sent) {
    return (
      <div className="lc-auth-shell">
        <div className="lc-auth-card">
          <div className="lc-auth-icon">✉</div>
          <h1 className="lc-auth-title">Check your inbox</h1>
          <p className="lc-auth-subtitle">
            We sent you a magic link. Click it to sign in — no password needed.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="lc-auth-shell">
      <div className="lc-auth-card">
        <div className="lc-auth-brand">
          <span className="lc-brand-dot" />
          <span className="lc-brand-name">Lerncycle</span>
        </div>

        <h1 className="lc-auth-title">Welcome back</h1>
        <p className="lc-auth-subtitle">
          Enter your email and we'll send you a magic link to sign in.
        </p>

        <form action={action} className="lc-auth-form">
          <Input
            id="email"
            name="email"
            type="email"
            placeholder="you@example.com"
            required
            autoComplete="email"
            className="lc-auth-input"
          />

          {(state?.error ?? callbackError) && (
            <p className="lc-auth-error" role="alert">
              {state?.error ?? 'Authentication failed. Please try again.'}
            </p>
          )}

          <Button type="submit" disabled={pending} className="lc-auth-btn">
            {pending ? 'Sending…' : 'Send magic link'}
          </Button>
        </form>

        <p className="lc-auth-footer">
          Your learning, organized. No password required.
        </p>
      </div>
    </div>
  );
}
