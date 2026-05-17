import type {Dictionary} from './types';

export const en: Dictionary = {
  nav: {
    cta: 'Get beta access',
  },
  hero: {
    h1: 'Your AI learning system for PDFs, notes, and active understanding',
    subhead:
      "LearnCycle connects your study materials, your notes, and an AI tutor that actually understands what you're reading.",
    ctaPrimary: 'Get beta access',
    ctaSecondary: 'Already invited? Sign in',
  },
  problem: {
    headline: 'Learning today is fragmented.',
    body:
      'PDFs sit in your downloads folder, notes live in Notion, questions go to ChatGPT — without any context from your own material. Too much is lost between sources, understanding, and review.',
  },
  solution: {
    headline: 'One place. One learning cycle.',
    body:
      'LearnCycle brings documents, notes, annotations, contextual AI chat, Feynman sessions, and review into one closed loop — all grounded in your own materials.',
  },
  features: {
    sectionTitle: 'What you get',
    items: [
      {
        icon: 'pdf',
        title: 'PDFs & notes',
        body: 'Upload, highlight, annotate, and take notes side-by-side. Auto-save included.',
      },
      {
        icon: 'active',
        title: 'Active learning',
        body: 'Feynman technique and guided learning — the AI probes until you really understand the material.',
      },
      {
        icon: 'revision',
        title: 'Review',
        body: 'Flashcards with SM-2 spaced repetition, multiple-choice mocktests, and a mindmap of your topics.',
      },
      {
        icon: 'chat',
        title: 'Contextual AI chat',
        body: 'Ask questions against your own materials — with citations from your PDFs, notes, and annotations.',
      },
    ],
  },
  howItWorks: {
    sectionTitle: 'How it works',
    steps: [
      {title: 'Upload', body: 'Create a course, drop in your PDFs.'},
      {
        title: 'Learn actively',
        body: 'Highlight, take notes, ask questions, verify in Feynman mode.',
      },
      {
        title: 'Review',
        body: 'Flashcards, mocktests, and mindmap keep your knowledge fresh.',
      },
    ],
  },
  faq: {
    sectionTitle: 'Frequently asked questions',
    items: [
      {
        question: 'How does the app work?',
        answer:
          "You create a course, drop in your PDFs, and then work through a closed learning cycle: read and highlight your PDF, take notes side-by-side, ask questions about your material in chat, test your understanding with the Feynman technique, and finally review with flashcards, mocktests, and a mindmap. Every answer comes with citations from your own material.",
      },
      {
        question: 'How is this different from ChatGPT?',
        answer:
          "ChatGPT answers from its general training data — it doesn't know your lecture slides, your notes, or your annotations. LearnCycle is an AI tutor that works exclusively on your own materials: every answer cites the exact passage from your PDF or note. On top of that come pedagogical modes (Feynman, guided learning) and review tools (spaced-repetition flashcards, mocktests, mindmap) that ChatGPT doesn't have.",
      },
      {
        question: 'Is the beta free?',
        answer: 'Yes — the beta is free for the entire test phase.',
      },
      {
        question: 'What data do you store?',
        answer:
          'Your email, uploaded PDFs, notes, annotations, and chat history. Everything is encrypted at Supabase (EU) and Vercel. The full list of sub-processors is in the Privacy Policy.',
      },
      {
        question: 'When is the public launch?',
        answer:
          "We invite testers in waves and currently have no fixed launch date. We'll reach out by email as soon as we can onboard you.",
      },
      {
        question: 'Does it work on mobile?',
        answer:
          'The beta is desktop-first — PDF reading and note-taking work best there. Mobile is usable but will only be further optimised after the beta.',
      },
      {
        question: 'Can I delete my data?',
        answer:
          'Yes, anytime. Email privacy@lerncycle.app and we delete your account including all PDFs, notes, and chats.',
      },
    ],
  },
  signup: {
    headline: 'Get beta access',
    body: "We invite testers in waves. Sign up and we'll reach out as soon as we can onboard you.",
    cta: 'Sign up',
    consent:
      'I agree that my email is stored for sending the beta invitation. See the Privacy Policy for details.',
    emailPlaceholder: 'you@university.edu',
    emailLabel: 'Email address',
    successHint: "Thanks! We'll be in touch as soon as a beta slot opens up.",
    alreadySubscribedHint: "You're already on the list — we'll be in touch.",
    errorHint: 'Please try again in a moment.',
    submitting: 'Submitting …',
  },
  footer: {
    legal: 'Imprint · Privacy · Beta notice',
    imprint: 'Imprint',
    privacy: 'Privacy',
    terms: 'Beta notice',
    copyright: '© 2026 LearnCycle',
    languageLabel: 'Language',
  },
  legal: {
    draftBanner: 'This is a beta draft, not legal advice.',
    privacy: {
      title: 'Privacy Policy',
      lastUpdated: 'Last updated: May 17, 2026',
      sections: [
        {
          heading: 'Data controller',
          body: 'Lam Anh Nguyen (placeholder — will be replaced before public launch). Contact: privacy@lerncycle.app',
        },
        {
          heading: 'Data we collect',
          body:
            'Email address (beta signup), Supabase auth identifiers, uploaded PDFs, your notes and annotations, chat history, generated flashcards and mocktests.',
        },
        {
          heading: 'Sub-processors',
          body:
            'Supabase (database/auth/storage), Vercel (hosting), OpenAI (embeddings + chat), Qdrant Cloud (retrieval index), optionally Tavily (web search) and Neo4j Aura (knowledge graph).',
        },
        {
          heading: 'Legal basis and retention',
          body:
            'Processing is based on Art. 6(1)(b) and (f) GDPR (contract performance and legitimate interest in operating the tool). Beta data is retained until you withdraw consent or until the beta ends.',
        },
        {
          heading: 'Your rights',
          body:
            'You may request access, rectification, deletion, restriction, portability, and object to processing at any time. Requests: privacy@lerncycle.app',
        },
      ],
    },
    imprint: {
      title: 'Imprint',
      lastUpdated: 'Last updated: May 17, 2026',
      sections: [
        {
          heading: 'Details per § 5 TMG (German Telemedia Act)',
          body: 'Lam Anh Nguyen\n[Address — placeholder]\n[Postcode City]\nGermany',
        },
        {
          heading: 'Contact',
          body: 'Email: hello@lerncycle.app',
        },
        {
          heading: 'Responsible for content per § 18(2) MStV',
          body: 'Lam Anh Nguyen, address as above.',
        },
      ],
    },
    terms: {
      title: 'Beta notice',
      lastUpdated: 'Last updated: May 17, 2026',
      sections: [
        {
          heading: 'Beta status',
          body:
            'LearnCycle is in closed beta. There is no SLA, no guaranteed availability, and no guarantee of data retention — data may be reset during the beta.',
        },
        {
          heading: 'No warranty',
          body:
            'The beta is provided free of charge and without warranty. AI responses may be incorrect and do not replace professional or legal advice.',
        },
        {
          heading: 'Governing law',
          body: 'German law applies (placeholder — final version follows at public launch).',
        },
      ],
    },
  },
};
