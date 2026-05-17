import Link from 'next/link';
import type {Dictionary} from '@/locales/types';
import {ScrollReveal} from '@/components/marketing/ScrollReveal';

interface Props {
  dict: Dictionary;
}

export function Hero({dict}: Props) {
  return (
    <section className="relative flex min-h-[90vh] items-center justify-center overflow-hidden">
      {/* Background Video */}
      <div className="absolute inset-0 z-0">
        <video
          autoPlay
          loop
          muted
          playsInline
          className="h-full w-full object-cover"
          poster="/images/marketing/app_mockup.png"
        >
          {/* Fallback to user's real screen recording */}
          <source src="/videos/hero-screencast.mp4" type="video/mp4" />
        </video>
        {/* Dark overlay to ensure text readability */}
        <div className="absolute inset-0 bg-black/60 backdrop-blur-[2px]"></div>
      </div>

      <ScrollReveal className="relative z-10 mx-auto w-full max-w-4xl px-4 py-32 text-center sm:px-6">
        <h1 className="text-4xl sm:text-6xl lg:text-7xl font-bold tracking-tight text-white leading-[1.1] drop-shadow-md">
          {dict.hero.h1}
        </h1>
        <p className="mt-8 text-lg sm:text-xl md:text-2xl text-slate-200/90 leading-relaxed max-w-3xl mx-auto font-light">
          {dict.hero.subhead}
        </p>
        <div className="mt-12 flex flex-col sm:flex-row items-center justify-center gap-4">
          <Link
            href="#beta"
            className="inline-flex h-14 items-center justify-center rounded-full bg-primary px-8 text-lg font-medium text-primary-foreground hover:opacity-90 hover:scale-105 transition-all duration-300 shadow-xl"
          >
            {dict.hero.ctaPrimary}
          </Link>
          <Link
            href="/login"
            className="inline-flex h-14 items-center justify-center rounded-full border border-white/20 bg-white/10 backdrop-blur-md px-8 text-lg font-medium text-white hover:bg-white/20 transition-all duration-300"
          >
            {dict.hero.ctaSecondary}
          </Link>
        </div>
      </ScrollReveal>
    </section>
  );
}
