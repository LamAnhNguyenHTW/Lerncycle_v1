import type {Dictionary} from '@/locales/types';
import {ScrollReveal} from '@/components/marketing/ScrollReveal';
import Image from 'next/image';
import studentLearningImg from '../../../public/images/marketing/student_learning.png';

interface Props {
  dict: Dictionary;
}

export function SolutionSection({dict}: Props) {
  return (
    <section className="w-full bg-background">
      <div className="mx-auto w-full max-w-6xl px-3 py-20 sm:px-6 sm:py-28 lg:py-32">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 lg:gap-20 items-center">
          <ScrollReveal animation="slide-right" className="flex flex-col justify-center">
            <h2 className="text-3xl md:text-4xl font-semibold tracking-tight text-foreground leading-tight">
              {dict.solution.headline}
            </h2>
            <p className="mt-5 text-base sm:text-lg text-muted-foreground leading-relaxed">
              {dict.solution.body}
            </p>
          </ScrollReveal>

          <ScrollReveal animation="slide-left" className="relative rounded-2xl overflow-hidden shadow-2xl border border-border aspect-[4/3] bg-muted/10">
            <Image
              src={studentLearningImg}
              alt="Student learning with Lerncycle"
              fill
              className="object-cover"
              sizes="(max-width: 1024px) 100vw, 50vw"
              placeholder="blur"
            />
          </ScrollReveal>
        </div>
      </div>
    </section>
  );
}
