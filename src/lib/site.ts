/**
 * Canonical absolute site URL — read from `NEXT_PUBLIC_SITE_URL`,
 * defaulting to `http://localhost:3000` for local dev. Used by
 * `robots.ts`, `sitemap.ts`, and any future absolute-URL emitter
 * (Open Graph images in Phase 5, etc.).
 *
 * Always returns the value WITHOUT a trailing slash so callers can
 * append `/path` safely.
 */
export function siteUrl(): string {
  const raw = process.env.NEXT_PUBLIC_SITE_URL ?? 'http://localhost:3000';
  return raw.replace(/\/+$/, '');
}
