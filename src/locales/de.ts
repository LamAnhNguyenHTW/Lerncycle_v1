import type {Dictionary} from './types';

export const de: Dictionary = {
  nav: {
    cta: 'Beta-Zugang sichern',
  },
  hero: {
    h1: 'Dein KI-Lernsystem für PDFs, Notizen und aktives Verstehen',
    subhead:
      'LearnCycle verbindet deine Studienmaterialien, deine Notizen und einen KI-Tutor, der wirklich versteht, was du gerade liest.',
    ctaPrimary: 'Beta-Zugang sichern',
    ctaSecondary: 'Schon eingeladen? Anmelden',
  },
  problem: {
    headline: 'Lernen ist heute zersplittert.',
    body:
      'PDFs liegen im Downloads-Ordner, Notizen in Notion, Fragen landen bei ChatGPT — ohne Kontext zu deinem Material. Zwischen Quellen, Verständnis und Wiederholung geht zu viel verloren.',
  },
  solution: {
    headline: 'Ein Ort. Ein Lernzyklus.',
    body:
      'LearnCycle bringt Dokumente, Notizen, Annotationen, kontextuellen KI-Chat, Feynman-Sessions und Wiederholung in einen geschlossenen Kreislauf — alles auf deinen eigenen Materialien.',
  },
  features: {
    sectionTitle: 'Was du bekommst',
    items: [
      {
        icon: 'pdf',
        title: 'PDFs & Notizen',
        body: 'Hochladen, hervorheben, kommentieren und parallel Notizen schreiben. Auto-Save inklusive.',
      },
      {
        icon: 'active',
        title: 'Aktives Lernen',
        body: 'Feynman-Technik und geführtes Lernen — die KI fragt nach, bis du den Stoff wirklich verstanden hast.',
      },
      {
        icon: 'revision',
        title: 'Wiederholung',
        body: 'Karteikarten mit SM-2-Spaced-Repetition, Multiple-Choice-Mocktests und eine Mindmap deiner Themen.',
      },
      {
        icon: 'chat',
        title: 'Kontextueller KI-Chat',
        body: 'Fragen stellen an deine eigenen Materialien — mit Quellenangaben aus deinen PDFs, Notizen und Annotationen.',
      },
    ],
  },
  howItWorks: {
    sectionTitle: 'So funktioniert es',
    steps: [
      {title: 'Hochladen', body: 'Kurs anlegen, PDFs hineinziehen.'},
      {
        title: 'Aktiv lernen',
        body: 'Markieren, Notizen schreiben, Fragen stellen, im Feynman-Modus überprüfen.',
      },
      {
        title: 'Wiederholen',
        body: 'Karteikarten, Mocktest und Mindmap halten dein Wissen frisch.',
      },
    ],
  },
  faq: {
    sectionTitle: 'Häufige Fragen',
    items: [
      {
        question: 'Wie funktioniert die App?',
        answer:
          'Du legst einen Kurs an, ziehst deine PDFs hinein und arbeitest dann in einem geschlossenen Lernzyklus: PDF lesen und markieren, parallel Notizen schreiben, im Chat Fragen an dein Material stellen, mit der Feynman-Technik dein Verständnis testen und am Ende mit Karteikarten, Mocktest und Mindmap wiederholen. Alle Antworten kommen mit Quellenangabe aus deinen eigenen Materialien.',
      },
      {
        question: 'Was ist der Unterschied zu ChatGPT?',
        answer:
          'ChatGPT antwortet aus seinem allgemeinen Trainingswissen — es kennt deine Vorlesungsfolien, deine Notizen oder deine Annotationen nicht. LearnCycle ist ein KI-Tutor, der ausschließlich auf deinen eigenen Materialien arbeitet: jede Antwort zitiert die konkrete Stelle in deiner PDF oder Notiz. Dazu kommen pädagogische Modi (Feynman, geführtes Lernen) und Wiederholungs-Werkzeuge (Spaced-Repetition-Karten, Mocktest, Mindmap), die ChatGPT nicht hat.',
      },
      {
        question: 'Was kostet die Beta?',
        answer: 'Die Beta ist während der gesamten Testphase kostenlos.',
      },
      {
        question: 'Welche Daten speichert ihr?',
        answer:
          'Deine E-Mail, deine hochgeladenen PDFs, deine Notizen, Annotationen und Chat-Verläufe. Alles liegt verschlüsselt bei Supabase (EU) und Vercel. Details und die vollständige Liste der Sub-Auftragsverarbeiter findest du in der Datenschutzerklärung.',
      },
      {
        question: 'Wann kommt der öffentliche Launch?',
        answer:
          'Wir laden in Wellen ein und es gibt aktuell kein festes Datum. Sobald wir dich onboarden können, melden wir uns per Mail.',
      },
      {
        question: 'Funktioniert es auf dem Handy?',
        answer:
          'Die Beta ist Desktop-first ausgelegt — PDF-Lesen und Notizen-Schreiben funktioniert dort am besten. Die mobile Ansicht ist grundsätzlich nutzbar, wird aber erst nach der Beta weiter optimiert.',
      },
      {
        question: 'Kann ich meine Daten löschen?',
        answer:
          'Ja, jederzeit. Schreib uns an privacy@lerncycle.app und wir löschen deinen Account inklusive aller PDFs, Notizen und Chats.',
      },
    ],
  },
  signup: {
    headline: 'Beta-Zugang sichern',
    body: 'Wir vergeben den Zugang aktuell in Wellen. Trag dich einfach ein und wir melden uns bei dir, sobald ein Platz für dich frei wird.',
    cta: 'Eintragen',
    consent:
      'Ich bin einverstanden, dass meine E-Mail zum Versand der Beta-Einladung gespeichert wird. Mehr dazu in der Datenschutzerklärung.',
    emailPlaceholder: 'deine@uni-mail.de',
    emailLabel: 'E-Mail-Adresse',
    successHint: 'Danke! Wir melden uns, sobald ein Beta-Platz frei wird.',
    alreadySubscribedHint: 'Du stehst schon auf der Liste — wir melden uns.',
    errorHint: 'Bitte versuche es später erneut.',
    submitting: 'Eintrag läuft …',
  },
  footer: {
    legal: 'Impressum · Datenschutz · Nutzungshinweis',
    imprint: 'Impressum',
    privacy: 'Datenschutz',
    terms: 'Nutzungshinweis',
    copyright: '© 2026 LearnCycle',
    languageLabel: 'Sprache',
  },
  legal: {
    draftBanner: 'Diese Version ist ein Beta-Entwurf und ersetzt keine Rechtsberatung.',
    privacy: {
      title: 'Datenschutzerklärung',
      lastUpdated: 'Stand: 17. Mai 2026',
      sections: [
        {
          heading: 'Verantwortlicher',
          body: 'Lam Anh Nguyen (Platzhalter — wird vor dem Beta-Launch ersetzt). Kontakt: privacy@lerncycle.app',
        },
        {
          heading: 'Welche Daten wir erheben',
          body:
            'E-Mail-Adresse (Beta-Anmeldung), Supabase-Auth-Daten, hochgeladene PDFs, deine Notizen und Annotationen, Chat-Verläufe, generierte Karteikarten und Mocktests.',
        },
        {
          heading: 'Sub-Auftragsverarbeiter',
          body:
            'Supabase (Datenbank/Auth/Storage), Vercel (Hosting), OpenAI (Embeddings + Chat), Qdrant Cloud (Retrieval-Index), optional Tavily (Websuche) und Neo4j Aura (Knowledge Graph).',
        },
        {
          heading: 'Rechtsgrundlage und Speicherdauer',
          body:
            'Verarbeitung erfolgt auf Grundlage von Art. 6 Abs. 1 lit. b und f DSGVO (Vertragserfüllung und berechtigtes Interesse am Betrieb des Lerntools). Beta-Daten werden bis zum Widerruf oder bis zum Ende der Beta gespeichert.',
        },
        {
          heading: 'Deine Rechte',
          body:
            'Du hast jederzeit Recht auf Auskunft, Berichtigung, Löschung, Einschränkung der Verarbeitung, Datenübertragbarkeit und Widerspruch. Anfragen an: privacy@lerncycle.app',
        },
      ],
    },
    imprint: {
      title: 'Impressum',
      lastUpdated: 'Stand: 17. Mai 2026',
      sections: [
        {
          heading: 'Angaben gemäß § 5 TMG',
          body: 'Lam Anh Nguyen\n[Adresse — Platzhalter]\n[PLZ Ort]\nDeutschland',
        },
        {
          heading: 'Kontakt',
          body: 'E-Mail: hello@lerncycle.app',
        },
        {
          heading: 'Verantwortlich für den Inhalt nach § 18 Abs. 2 MStV',
          body: 'Lam Anh Nguyen, Adresse wie oben.',
        },
      ],
    },
    terms: {
      title: 'Nutzungshinweis (Beta)',
      lastUpdated: 'Stand: 17. Mai 2026',
      sections: [
        {
          heading: 'Beta-Status',
          body:
            'LearnCycle befindet sich in einer geschlossenen Beta-Phase. Es besteht kein Anspruch auf Verfügbarkeit, keine zugesicherten Service Levels und keine Garantie auf Datenerhalt — Daten können im Rahmen der Beta zurückgesetzt werden.',
        },
        {
          heading: 'Keine Gewährleistung',
          body:
            'Die Beta wird unentgeltlich und ohne Gewährleistung zur Verfügung gestellt. Antworten der KI können fehlerhaft sein und ersetzen keine Fach- oder Rechtsberatung.',
        },
        {
          heading: 'Anwendbares Recht',
          body: 'Es gilt deutsches Recht (Platzhalter — finale Fassung folgt zum Produktivstart).',
        },
      ],
    },
  },
};
