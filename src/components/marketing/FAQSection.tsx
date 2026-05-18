import type {Dictionary} from '@/locales/types';
import {ScrollReveal} from '@/components/marketing/ScrollReveal';

interface Props {
  dict: Dictionary;
}

/**
 * Accessible FAQ using native `<details>` / `<summary>` — zero client
 * JS, works without hydration, keyboard-navigable for free.
 */
export function FAQSection({dict}: Props) {
  return (
    <section className="bg-muted/30 w-full border-y border-border/40">
      <div className="mx-auto w-full max-w-3xl px-4 py-20 sm:px-6 sm:py-28 lg:py-32">
        <ScrollReveal className="text-center">
          <h2 className="text-3xl sm:text-4xl font-bold tracking-tight text-foreground">
            {dict.faq.sectionTitle}
          </h2>
        </ScrollReveal>
        <ul className="mt-16 flex flex-col gap-4">
          {dict.faq.items.map((item, idx) => (
            <ScrollReveal key={idx} animation="fade-up" delay={idx * 0.1}>
              <details className="group rounded-2xl border border-border bg-card px-6 py-5 shadow-sm hover:shadow-md transition-shadow open:bg-card">
                <summary className="flex cursor-pointer list-none items-center justify-between gap-4 text-lg font-semibold text-foreground">
                  <span>{item.question}</span>
                  <span
                    aria-hidden="true"
                    className="ml-2 flex h-8 w-8 items-center justify-center rounded-full bg-primary/10 text-xl font-light text-primary transition-transform duration-300 group-open:rotate-45 group-open:bg-primary group-open:text-primary-foreground shrink-0"
                  >
                    +
                  </span>
                </summary>
                <p className="mt-4 text-base text-muted-foreground leading-relaxed pl-2 border-l-2 border-primary/20">
                  {item.answer}
                </p>
              </details>
            </ScrollReveal>
          ))}
        </ul>
      </div>
    </section>
  );
}
