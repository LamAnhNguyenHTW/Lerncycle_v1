import type {Dictionary} from '@/locales/types';
import {ScrollReveal} from '@/components/marketing/ScrollReveal';

interface Props {
  dict: Dictionary;
}

export function HowItWorks({dict}: Props) {
  return (
    <section className="w-full bg-background relative overflow-hidden">
      <div className="mx-auto w-full max-w-6xl px-4 py-20 sm:px-6 sm:py-28 lg:py-32 relative z-10">
        <ScrollReveal className="text-center max-w-3xl mx-auto">
          <h2 className="text-3xl md:text-4xl font-semibold tracking-tight text-foreground">
            {dict.howItWorks.sectionTitle}
          </h2>
        </ScrollReveal>
        <ol className="mt-14 grid grid-cols-1 gap-6 md:grid-cols-3 relative">
          <div className="hidden md:block absolute top-10 left-[10%] right-[10%] h-px bg-border z-[-1]" />

          {dict.howItWorks.steps.map((step, idx) => (
            <ScrollReveal
              key={idx}
              animation="fade-up"
              delay={idx * 0.15}
              className="relative rounded-2xl border border-border bg-card p-6 shadow-sm hover:shadow-md transition-all duration-300"
            >
              <div className="mx-auto flex h-10 w-10 items-center justify-center rounded-full bg-primary text-sm font-semibold text-primary-foreground shadow-md mb-4 ring-4 ring-background">
                {idx + 1}
              </div>
              <h3 className="text-base font-medium text-foreground text-center">{step.title}</h3>
              <p className="mt-2 text-sm text-muted-foreground leading-relaxed text-center">
                {step.body}
              </p>
            </ScrollReveal>
          ))}
        </ol>
      </div>
    </section>
  );
}
