import {FileText, Brain, RotateCw, MessageSquare} from 'lucide-react';
import type {Dictionary, FeatureIcon} from '@/locales/types';
import {ScrollReveal} from '@/components/marketing/ScrollReveal';

interface Props {
  dict: Dictionary;
}

const iconMap: Record<FeatureIcon, typeof FileText> = {
  pdf: FileText,
  active: Brain,
  revision: RotateCw,
  chat: MessageSquare,
};

export function FeatureGrid({dict}: Props) {
  return (
    <section className="bg-muted/30 w-full border-y border-border/40">
      <div className="mx-auto w-full max-w-6xl px-4 py-20 sm:px-6 sm:py-28 lg:py-32">
        <ScrollReveal className="text-center max-w-3xl mx-auto">
          <h2 className="text-3xl sm:text-4xl font-bold tracking-tight text-foreground">
            {dict.features.sectionTitle}
          </h2>
        </ScrollReveal>
        
        <div className="mt-16 grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
          {dict.features.items.map((item, idx) => {
            const Icon = iconMap[item.icon];
            return (
              <ScrollReveal
                key={idx}
                animation="fade-up"
                delay={idx * 0.1}
                className="group rounded-2xl border border-border bg-card p-8 shadow-sm hover:shadow-xl hover:-translate-y-1 transition-all duration-300"
              >
                <div className="flex h-14 w-14 items-center justify-center rounded-xl bg-primary/10 text-primary group-hover:bg-primary group-hover:text-primary-foreground transition-colors duration-300">
                  <Icon className="h-7 w-7" strokeWidth={1.75} />
                </div>
                <h3 className="mt-6 text-xl font-semibold text-foreground">{item.title}</h3>
                <p className="mt-3 text-base text-muted-foreground leading-relaxed">{item.body}</p>
              </ScrollReveal>
            );
          })}
        </div>
      </div>
    </section>
  );
}
