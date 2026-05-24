import type {Dictionary} from '@/locales/types';
import {ScrollReveal} from '@/components/marketing/ScrollReveal';
import Image from 'next/image';
import chaoticDeskImg from '../../../public/images/marketing/chaotic_desk.png';

interface Props {
  dict: Dictionary;
}

export function ProblemSection({dict}: Props) {
  return (
    <section className="bg-muted/30 w-full border-y border-border/40">
      <div className="mx-auto w-full max-w-6xl px-4 py-20 sm:px-6 sm:py-28 lg:py-32">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 lg:gap-20 items-center">
          <ScrollReveal animation="slide-right" className="order-2 lg:order-1 relative rounded-2xl overflow-hidden shadow-2xl border border-border aspect-[4/3]">
            <Image
              src={chaoticDeskImg}
              alt="Chaotic study desk"
              fill
              className="object-cover"
              sizes="(max-width: 1024px) 100vw, 50vw"
              placeholder="blur"
            />
          </ScrollReveal>
          
          <ScrollReveal animation="slide-left" className="order-1 lg:order-2 flex flex-col justify-center">
            <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight text-foreground leading-[1.15]">
              {dict.problem.headline}
            </h2>
            <p className="mt-6 text-lg sm:text-xl text-muted-foreground leading-relaxed">
              {dict.problem.body}
            </p>
          </ScrollReveal>
        </div>
      </div>
    </section>
  );
}
