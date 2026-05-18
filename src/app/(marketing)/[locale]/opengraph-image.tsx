import {ImageResponse} from 'next/og';
import {isLocale} from '@/lib/locale';

export const alt = 'LearnCycle — AI learning system';
export const size = {width: 1200, height: 630};
export const contentType = 'image/png';

/**
 * Programmatic Open Graph image for every marketing page under
 * `/[locale]`. Generated at build/runtime via `ImageResponse` — no
 * binary asset checked into the repo. Phase 5 ships an English
 * default; a per-locale variant can be added by inspecting `locale`.
 */
export default async function OgImage({params}: {params: {locale: string}}) {
  const locale = isLocale(params.locale) ? params.locale : 'de';

  const headline = locale === 'de'
    ? 'Dein KI-Lernsystem'
    : 'Your AI learning system';
  const subline = locale === 'de'
    ? 'PDFs, Notizen, KI-Chat und Wiederholung — ein Lernzyklus.'
    : 'PDFs, notes, AI chat, and review — one learning cycle.';

  return new ImageResponse(
    (
      <div
        style={{
          height: '100%',
          width: '100%',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-between',
          padding: '80px',
          backgroundColor: '#ffffff',
          fontFamily: 'system-ui, -apple-system, "Segoe UI", sans-serif',
          color: '#37352f',
        }}
      >
        <div style={{fontSize: 36, fontWeight: 600, letterSpacing: '-0.01em'}}>
          LearnCycle
        </div>
        <div style={{display: 'flex', flexDirection: 'column', gap: 24}}>
          <div style={{fontSize: 72, fontWeight: 700, lineHeight: 1.1, letterSpacing: '-0.02em'}}>
            {headline}
          </div>
          <div style={{fontSize: 32, color: '#37352fa6', lineHeight: 1.4, maxWidth: 900}}>
            {subline}
          </div>
        </div>
        <div style={{fontSize: 24, color: '#37352f73'}}>
          {locale === 'de' ? 'Beta startet bald' : 'Launching soon'}
        </div>
      </div>
    ),
    {...size},
  );
}
