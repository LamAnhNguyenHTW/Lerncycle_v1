'use server';

import {headers} from 'next/headers';
import {createClient} from '@/lib/supabase/server';
import {isLocale} from '@/lib/locale';

export type SubscribeBetaResult =
  | {ok: true; alreadySubscribed?: boolean}
  | {ok: false; error: string};

/**
 * Best-effort in-memory rate limit. Keyed by `ip|email`, 5 attempts /
 * 60 s. Process-local — fine for the beta landing page since traffic is
 * low and a real attacker can still be stopped by Supabase's unique
 * index. Resets on every deploy.
 */
const RATE_LIMIT_WINDOW_MS = 60_000;
const RATE_LIMIT_MAX = 5;
const rateLimitBuckets = new Map<string, number[]>();

function isRateLimited(key: string): boolean {
  const now = Date.now();
  const bucket = (rateLimitBuckets.get(key) ?? []).filter(
    (ts) => now - ts < RATE_LIMIT_WINDOW_MS,
  );
  if (bucket.length >= RATE_LIMIT_MAX) {
    rateLimitBuckets.set(key, bucket);
    return true;
  }
  bucket.push(now);
  rateLimitBuckets.set(key, bucket);
  return false;
}

// RFC-5322-ish strict-enough check: local@domain.tld, no spaces, single `@`.
const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const MAX_EMAIL_LEN = 254;

const NEUTRAL_ERROR_DE = 'Bitte versuche es später erneut.';

/**
 * Captures a beta signup from the public landing page.
 *  - normalises the email to lowercase
 *  - silently returns success on honeypot fill (no DB write)
 *  - returns `alreadySubscribed: true` on Postgres unique violation
 *  - best-effort rate-limited per IP+email
 *
 * Always returns a neutral message on failure — never echoes the
 * underlying Postgres / Supabase error to the client.
 */
export async function subscribeBeta(formData: FormData): Promise<SubscribeBetaResult> {
  const rawEmail = formData.get('email');
  const rawLocale = formData.get('locale');
  const honeypot = formData.get('company');
  const source = formData.get('source');

  // Honeypot triggered — silently succeed without touching the DB.
  if (typeof honeypot === 'string' && honeypot.trim().length > 0) {
    return {ok: true};
  }

  if (typeof rawEmail !== 'string' || typeof rawLocale !== 'string') {
    return {ok: false, error: NEUTRAL_ERROR_DE};
  }

  const email = rawEmail.trim().toLowerCase();
  if (!email || email.length > MAX_EMAIL_LEN || !EMAIL_RE.test(email)) {
    return {ok: false, error: NEUTRAL_ERROR_DE};
  }

  if (!isLocale(rawLocale)) {
    return {ok: false, error: NEUTRAL_ERROR_DE};
  }

  const headerList = await headers();
  const ip =
    headerList.get('x-forwarded-for')?.split(',')[0]?.trim() ||
    headerList.get('x-real-ip') ||
    'unknown';
  const userAgent = headerList.get('user-agent') ?? null;

  if (isRateLimited(`${ip}|${email}`)) {
    return {ok: false, error: NEUTRAL_ERROR_DE};
  }

  const supabase = await createClient();
  const {error} = await supabase.from('beta_signups').insert({
    email,
    locale: rawLocale,
    source: typeof source === 'string' && source.length > 0 ? source : 'landing_hero',
    user_agent: userAgent,
  });

  if (error) {
    // 23505 = unique_violation → email already on the list.
    if (error.code === '23505') {
      return {ok: true, alreadySubscribed: true};
    }
    return {ok: false, error: NEUTRAL_ERROR_DE};
  }

  return {ok: true};
}
