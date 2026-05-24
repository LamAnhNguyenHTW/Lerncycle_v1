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
          <h2 className="text-3xl sm:text-4xl font-bold tracking-tight text-foreground">
            {dict.howItWorks.sectionTitle}
          </h2>
        </ScrollReveal>
        <ol className="mt-16 grid grid-cols-1 gap-8 md:grid-cols-3 relative">
          {/* Connecting line for desktop */}
          <div className="hidden md:block absolute top-12 left-[10%] right-[10%] h-[2px] bg-border z-[-1]" />
          
          {dict.howItWorks.steps.map((step, idx) => (
            <ScrollReveal
              key={idx}
              animation="fade-up"
              delay={idx * 0.15}
              className="relative rounded-2xl border border-border bg-card p-8 shadow-sm hover:shadow-xl transition-all duration-300"
            >
              <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-full bg-primary text-xl font-bold text-primary-foreground shadow-lg mb-6 ring-4 ring-background">
                {idx + 1}
              </div>
              <h3 className="text-xl font-semibold text-foreground text-center">{step.title}</h3>
              <p className="mt-4 text-base text-muted-foreground leading-relaxed text-center">
                {step.body}
              </p>
            </ScrollReveal>
          ))}
        </ol>
      </div>
    </section>
  );
}
