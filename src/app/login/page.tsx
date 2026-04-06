import {Suspense} from 'react';
import {LoginForm} from './LoginForm';

/** Magic-link login page — Notion-style minimal design. */
export default function LoginPage() {
  return (
    <Suspense>
      <LoginForm />
    </Suspense>
  );
}
