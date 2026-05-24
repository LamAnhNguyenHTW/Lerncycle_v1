/**
 * Typed dictionary shape for the public marketing site.
 * Both `de.ts` and `en.ts` MUST satisfy this interface — TypeScript
 * enforces key parity across locales.
 */
export interface Dictionary {
  nav: {
    cta: string;
  };
  hero: {
    h1: string;
    subhead: string;
    ctaPrimary: string;
    ctaSecondary: string;
    mockupCoreLabel: string;
    mockupInputPdf: string;
    mockupInputNote: string;
    mockupInputHighlight: string;
    mockupInputAudio: string;
    mockupOutputChat: string;
    mockupOutputFlashcards: string;
    mockupOutputFeynman: string;
  };
  problem: {
    headline: string;
    body: string;
  };
  solution: {
    headline: string;
    body: string;
  };
  features: {
    sectionTitle: string;
    items: Array<{title: string; body: string; icon: FeatureIcon}>;
  };
  howItWorks: {
    sectionTitle: string;
    steps: Array<{title: string; body: string}>;
  };
  faq: {
    sectionTitle: string;
    items: Array<{question: string; answer: string}>;
  };
  signup: {
    headline: string;
    body: string;
    cta: string;
    consent: string;
    emailPlaceholder: string;
    emailLabel: string;
    successHint: string;
    alreadySubscribedHint: string;
    errorHint: string;
    submitting: string;
  };
  footer: {
    legal: string;
    imprint: string;
    privacy: string;
    terms: string;
    copyright: string;
    languageLabel: string;
  };
  legal: {
    draftBanner: string;
    privacy: LegalPageCopy;
    imprint: LegalPageCopy;
    terms: LegalPageCopy;
  };
}

export interface LegalPageCopy {
  title: string;
  lastUpdated: string;
  sections: Array<{heading: string; body: string}>;
}

export type FeatureIcon = 'pdf' | 'active' | 'revision' | 'chat';
