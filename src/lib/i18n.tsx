'use client';

import {createContext, useContext, useEffect, useMemo, useState} from 'react';
import type React from 'react';

export type Language = 'de' | 'en';

export type TranslationKey =
  | 'nav.home'
  | 'nav.notetaking'
  | 'nav.learn'
  | 'nav.activeLearning'
  | 'nav.revision'
  | 'nav.openSidebar'
  | 'nav.collapseSidebar'
  | 'nav.logout'
  | 'language.label'
  | 'language.de'
  | 'language.en'
  | 'dashboard.headline'
  | 'dashboard.subtitle'
  | 'dashboard.upload'
  | 'dashboard.uploadSubtitle'
  | 'dashboard.insert'
  | 'dashboard.insertSubtitle'
  | 'dashboard.record'
  | 'dashboard.recordSubtitle'
  | 'chat.courseMaterials'
  | 'chat.useAllMaterials'
  | 'chat.recentChats'
  | 'chat.newChat'
  | 'chat.learnEmptyTitle'
  | 'chat.learnEmptyDescription'
  | 'chat.learnPlaceholder'
  | 'chat.disclaimer'
  | 'chat.thinking'
  | 'chat.sourceDetails'
  | 'chat.hideDetails'
  | 'chat.showLess'
  | 'chat.more'
  | 'active.settings'
  | 'active.suggestions'
  | 'active.topicPlaceholder'
  | 'active.guided'
  | 'active.feynman'
  | 'active.guidedEmptyTitle'
  | 'active.guidedEmptyDescription'
  | 'active.guidedPlaceholder'
  | 'active.feynmanEmptyTitle'
  | 'active.feynmanEmptyDescription'
  | 'active.feynmanPlaceholder'
  | 'active.difficulty'
  | 'active.auto'
  | 'active.beginner'
  | 'active.intermediate'
  | 'active.advanced'
  | 'materials.title'
  | 'materials.newFolder'
  | 'materials.folderPlaceholder'
  | 'materials.save'
  | 'materials.cancel'
  | 'materials.noFolders'
  | 'materials.noFoldersHint'
  | 'materials.directUploads'
  | 'materials.unfiledDocuments'
  | 'materials.close'
  | 'materials.uploadDirect'
  | 'materials.uploadPdf'
  | 'materials.deleteFolder'
  | 'materials.deleteFolderTitle'
  | 'materials.deleteFolderWithFiles'
  | 'materials.deleteEmptyFolder'
  | 'materials.keepFiles'
  | 'materials.deleteEverything'
  | 'materials.delete'
  | 'materials.deleting'
  | 'materials.chat'
  | 'upload.uploading'
  | 'upload.onlyPdf'
  | 'upload.dropToUpload'
  | 'upload.dropOrClick'
  | 'upload.pdfOnly'
  | 'upload.failedUnexpectedly'
  | 'study.loadingViewer'
  | 'study.loadingEditor'
  | 'study.chooseDifferentPdf'
  | 'study.allPdfs'
  | 'study.loading'
  | 'study.choosePdf'
  | 'study.noPdfs'
  | 'study.available'
  | 'study.unfiled'
  | 'note.placeholder'
  | 'note.saving'
  | 'note.saved'
  | 'note.saveFailed'
  | 'note.bold'
  | 'note.italic'
  | 'note.strikethrough'
  | 'note.inlineCode'
  | 'note.highlight'
  | 'note.heading1'
  | 'note.heading2'
  | 'note.heading3'
  | 'note.text'
  | 'note.bulletList'
  | 'note.numberedList'
  | 'note.taskList'
  | 'note.blockquote'
  | 'note.codeBlock'
  | 'note.divider'
  | 'note.undo'
  | 'note.redo'
  | 'note.insertBlock'
  | 'note.plainParagraph'
  | 'note.largeHeading'
  | 'note.mediumHeading'
  | 'note.smallHeading'
  | 'note.unorderedList'
  | 'note.orderedList'
  | 'note.checklist'
  | 'note.multiLineCode'
  | 'note.blockquoteCallout'
  | 'note.horizontalSeparator'
  | 'pdf.addComment'
  | 'pdf.saveHighlight'
  | 'pdf.couldNotLoad'
  | 'pdf.loading'
  | 'pdf.collapseHighlights'
  | 'pdf.expandHighlights'
  | 'pdf.highlights'
  | 'pdf.highlight'
  | 'pdf.noHighlights'
  | 'pdf.jumpToHighlight'
  | 'pdf.page'
  | 'pdf.noText'
  | 'pdf.collapse'
  | 'pdf.expand'
  | 'pdf.highlightOnTitle'
  | 'pdf.highlightOffTitle'
  | 'pdf.highlightOn'
  | 'pdf.highlightOff'
  | 'pdf.color.yellow'
  | 'pdf.color.green'
  | 'pdf.color.blue'
  | 'pdf.color.pink';

const translations: Record<Language, Record<TranslationKey, string>> = {
  en: {
    'nav.home': 'Home',
    'nav.notetaking': 'Notetaking',
    'nav.learn': 'Learn & Research',
    'nav.activeLearning': 'Active Learning',
    'nav.revision': 'Revision',
    'nav.openSidebar': 'Open sidebar',
    'nav.collapseSidebar': 'Collapse sidebar',
    'nav.logout': 'Log out',
    'language.label': 'Language',
    'language.de': 'Deutsch',
    'language.en': 'English',
    'dashboard.headline': 'Hello {name}, what do you want to master in {course}?',
    'dashboard.subtitle': 'Upload everything and get interactive notes, flashcards, quizzes, and more',
    'dashboard.upload': 'Upload',
    'dashboard.uploadSubtitle': 'Add materials directly to the course',
    'dashboard.insert': 'Insert',
    'dashboard.insertSubtitle': 'YouTube, website, text',
    'dashboard.record': 'Record',
    'dashboard.recordSubtitle': 'Record live lecture',
    'chat.courseMaterials': 'Course Materials',
    'chat.useAllMaterials': 'Use all materials',
    'chat.recentChats': 'Recent Chats',
    'chat.newChat': 'New chat',
    'chat.learnEmptyTitle': 'How can I help you learn?',
    'chat.learnEmptyDescription': 'Ask questions across all your course materials, or select specific files to narrow the search.',
    'chat.learnPlaceholder': 'Ask about your materials...',
    'chat.disclaimer': 'AI can make mistakes. Always check course materials for accuracy.',
    'chat.thinking': 'Thinking',
    'chat.sourceDetails': 'Source details',
    'chat.hideDetails': 'Hide details',
    'chat.showLess': 'Show less',
    'chat.more': 'more',
    'active.settings': 'Settings',
    'active.suggestions': 'Suggestions',
    'active.topicPlaceholder': 'Optional topic',
    'active.guided': 'Guided Learning',
    'active.feynman': 'Feynman Technique',
    'active.guidedEmptyTitle': 'What would you like to understand step by step?',
    'active.guidedEmptyDescription': 'Start with a topic. Learncycle guides you through your materials with focused questions.',
    'active.guidedPlaceholder': 'Name a topic you want to learn with guidance...',
    'active.feynmanEmptyTitle': 'Which concept would you like to explain simply?',
    'active.feynmanEmptyDescription': 'Explain a concept in your own words. Learncycle asks follow-up questions until the idea is clear.',
    'active.feynmanPlaceholder': 'Explain a concept or name what you want to practice...',
    'active.difficulty': 'Difficulty',
    'active.auto': 'Auto',
    'active.beginner': 'Beginner',
    'active.intermediate': 'Intermediate',
    'active.advanced': 'Advanced',
    'materials.title': 'Materials & Folders',
    'materials.newFolder': 'New Folder',
    'materials.folderPlaceholder': 'Folder name (e.g. Week 1)',
    'materials.save': 'Save',
    'materials.cancel': 'Cancel',
    'materials.noFolders': 'No folders yet.',
    'materials.noFoldersHint': 'Create a folder to organize your PDFs.',
    'materials.directUploads': 'Direct Uploads',
    'materials.unfiledDocuments': 'Unfiled documents',
    'materials.close': 'Close',
    'materials.uploadDirect': 'Upload PDF directly',
    'materials.uploadPdf': 'Upload PDF',
    'materials.deleteFolder': 'Delete folder',
    'materials.deleteFolderTitle': 'Delete folder "{name}"?',
    'materials.deleteFolderWithFiles': 'This folder contains {count} PDF{plural}. What should happen to the files?',
    'materials.deleteEmptyFolder': 'This empty folder will be permanently deleted.',
    'materials.keepFiles': 'Keep files',
    'materials.deleteEverything': 'Delete everything',
    'materials.delete': 'Delete',
    'materials.deleting': 'Deleting...',
    'materials.chat': 'Chat',
    'upload.uploading': 'Uploading...',
    'upload.onlyPdf': 'Only PDF files are accepted',
    'upload.dropToUpload': 'Drop to upload',
    'upload.dropOrClick': 'Drop a PDF here, or click to select',
    'upload.pdfOnly': 'PDF only · max 50 MB',
    'upload.failedUnexpectedly': 'Upload failed unexpectedly.',
    'study.loadingViewer': 'Loading viewer...',
    'study.loadingEditor': 'Loading editor...',
    'study.chooseDifferentPdf': 'Choose a different PDF',
    'study.allPdfs': 'All PDFs',
    'study.loading': 'Loading...',
    'study.choosePdf': 'Choose a PDF to study',
    'study.noPdfs': 'No PDFs uploaded yet. Go to the Home tab to upload your first file.',
    'study.available': '{count} PDF{plural} available in {course}',
    'study.unfiled': 'Unfiled',
    'note.placeholder': "Start writing, or type '/' for commands...",
    'note.saving': 'Saving...',
    'note.saved': 'Saved',
    'note.saveFailed': 'Save failed',
    'note.bold': 'Bold',
    'note.italic': 'Italic',
    'note.strikethrough': 'Strikethrough',
    'note.inlineCode': 'Inline code',
    'note.highlight': 'Highlight',
    'note.heading1': 'Heading 1',
    'note.heading2': 'Heading 2',
    'note.heading3': 'Heading 3',
    'note.text': 'Text',
    'note.bulletList': 'Bullet list',
    'note.numberedList': 'Numbered list',
    'note.taskList': 'Task list',
    'note.blockquote': 'Blockquote',
    'note.codeBlock': 'Code block',
    'note.divider': 'Divider',
    'note.undo': 'Undo',
    'note.redo': 'Redo',
    'note.insertBlock': 'Insert block',
    'note.plainParagraph': 'Plain paragraph',
    'note.largeHeading': 'Large section heading',
    'note.mediumHeading': 'Medium section heading',
    'note.smallHeading': 'Small section heading',
    'note.unorderedList': 'Unordered list',
    'note.orderedList': 'Ordered numbered list',
    'note.checklist': 'Checklist with checkboxes',
    'note.multiLineCode': 'Multi-line code snippet',
    'note.blockquoteCallout': 'Blockquote callout',
    'note.horizontalSeparator': 'Horizontal separator',
    'pdf.addComment': 'Add a comment (optional)...',
    'pdf.saveHighlight': 'Save highlight',
    'pdf.couldNotLoad': 'Could not load PDF',
    'pdf.loading': 'Loading {name}...',
    'pdf.collapseHighlights': 'Collapse highlights',
    'pdf.expandHighlights': 'Expand highlights',
    'pdf.highlights': 'Highlights',
    'pdf.highlight': 'highlight',
    'pdf.noHighlights': 'No highlights yet.',
    'pdf.jumpToHighlight': 'Jump to highlight',
    'pdf.page': 'Page',
    'pdf.noText': '(No text)',
    'pdf.collapse': 'Collapse',
    'pdf.expand': 'Expand',
    'pdf.highlightOnTitle': 'Highlight mode on: select text to create highlight',
    'pdf.highlightOffTitle': 'Highlight mode off',
    'pdf.highlightOn': 'Highlight: On',
    'pdf.highlightOff': 'Highlight: Off',
    'pdf.color.yellow': 'Yellow',
    'pdf.color.green': 'Green',
    'pdf.color.blue': 'Blue',
    'pdf.color.pink': 'Pink',
  },
  de: {
    'nav.home': 'Home',
    'nav.notetaking': 'Notizen',
    'nav.learn': 'Lernen & Recherchieren',
    'nav.activeLearning': 'Aktives Lernen',
    'nav.revision': 'Wiederholen',
    'nav.openSidebar': 'Sidebar öffnen',
    'nav.collapseSidebar': 'Sidebar einklappen',
    'nav.logout': 'Abmelden',
    'language.label': 'Sprache',
    'language.de': 'Deutsch',
    'language.en': 'English',
    'dashboard.headline': 'Hallo {name}, was möchtest du in {course} meistern?',
    'dashboard.subtitle': 'Lade Materialien hoch und nutze interaktive Notizen, Karteikarten, Quizze und mehr',
    'dashboard.upload': 'Hochladen',
    'dashboard.uploadSubtitle': 'Materialien direkt zum Kurs hinzufügen',
    'dashboard.insert': 'Einfügen',
    'dashboard.insertSubtitle': 'YouTube, Website, Text',
    'dashboard.record': 'Aufnehmen',
    'dashboard.recordSubtitle': 'Live-Vorlesung aufnehmen',
    'chat.courseMaterials': 'Kursmaterialien',
    'chat.useAllMaterials': 'Alle Materialien nutzen',
    'chat.recentChats': 'Letzte Chats',
    'chat.newChat': 'Neuer Chat',
    'chat.learnEmptyTitle': 'Wie kann ich dir beim Lernen helfen?',
    'chat.learnEmptyDescription': 'Stelle Fragen zu deinen Kursmaterialien oder wähle bestimmte Dateien aus.',
    'chat.learnPlaceholder': 'Frage zu deinen Materialien...',
    'chat.disclaimer': 'KI kann Fehler machen. Prüfe wichtige Aussagen in deinen Kursmaterialien.',
    'chat.thinking': 'Denkt nach',
    'chat.sourceDetails': 'Quellendetails',
    'chat.hideDetails': 'Details ausblenden',
    'chat.showLess': 'Weniger anzeigen',
    'chat.more': 'weitere',
    'active.settings': 'Einstellungen',
    'active.suggestions': 'Vorschläge',
    'active.topicPlaceholder': 'Thema optional eingeben',
    'active.guided': 'Geführtes Lernen',
    'active.feynman': 'Feynman-Technik',
    'active.guidedEmptyTitle': 'Was möchtest du Schritt für Schritt verstehen?',
    'active.guidedEmptyDescription': 'Starte mit einem Thema. Learncycle führt dich mit gezielten Fragen durch deine Materialien.',
    'active.guidedPlaceholder': 'Nenne ein Thema, das du geführt lernen willst...',
    'active.feynmanEmptyTitle': 'Welches Konzept möchtest du einfach erklären?',
    'active.feynmanEmptyDescription': 'Erkläre ein Konzept in deinen Worten. Learncycle fragt nach, bis die Idee klar ist.',
    'active.feynmanPlaceholder': 'Erkläre ein Konzept oder nenne, was du üben willst...',
    'active.difficulty': 'Niveau',
    'active.auto': 'Auto',
    'active.beginner': 'Anfänger',
    'active.intermediate': 'Fortgeschritten',
    'active.advanced': 'Experte',
    'materials.title': 'Materialien & Ordner',
    'materials.newFolder': 'Neuer Ordner',
    'materials.folderPlaceholder': 'Ordnername (z.B. Woche 1)',
    'materials.save': 'Speichern',
    'materials.cancel': 'Abbrechen',
    'materials.noFolders': 'Noch keine Ordner.',
    'materials.noFoldersHint': 'Erstelle einen Ordner, um deine PDFs zu organisieren.',
    'materials.directUploads': 'Direkte Uploads',
    'materials.unfiledDocuments': 'Nicht einsortierte Dokumente',
    'materials.close': 'Schließen',
    'materials.uploadDirect': 'PDF direkt hochladen',
    'materials.uploadPdf': 'PDF hochladen',
    'materials.deleteFolder': 'Ordner löschen',
    'materials.deleteFolderTitle': 'Ordner "{name}" löschen?',
    'materials.deleteFolderWithFiles': 'Dieser Ordner enthält {count} PDF{plural}. Was soll mit den Dateien passieren?',
    'materials.deleteEmptyFolder': 'Dieser leere Ordner wird dauerhaft gelöscht.',
    'materials.keepFiles': 'Dateien behalten',
    'materials.deleteEverything': 'Alles löschen',
    'materials.delete': 'Löschen',
    'materials.deleting': 'Lösche...',
    'materials.chat': 'Chat',
    'upload.uploading': 'Lade hoch...',
    'upload.onlyPdf': 'Nur PDF-Dateien werden akzeptiert',
    'upload.dropToUpload': 'Loslassen zum Hochladen',
    'upload.dropOrClick': 'PDF hier ablegen oder klicken zum Auswählen',
    'upload.pdfOnly': 'Nur PDF · max. 50 MB',
    'upload.failedUnexpectedly': 'Upload unerwartet fehlgeschlagen.',
    'study.loadingViewer': 'Viewer wird geladen...',
    'study.loadingEditor': 'Editor wird geladen...',
    'study.chooseDifferentPdf': 'Anderes PDF auswählen',
    'study.allPdfs': 'Alle PDFs',
    'study.loading': 'Lädt...',
    'study.choosePdf': 'Wähle ein PDF zum Lernen',
    'study.noPdfs': 'Noch keine PDFs hochgeladen. Lade im Home-Tab deine erste Datei hoch.',
    'study.available': '{count} PDF{plural} verfügbar in {course}',
    'study.unfiled': 'Nicht einsortiert',
    'note.placeholder': "Schreibe los oder tippe '/' für Befehle...",
    'note.saving': 'Speichert...',
    'note.saved': 'Gespeichert',
    'note.saveFailed': 'Speichern fehlgeschlagen',
    'note.bold': 'Fett',
    'note.italic': 'Kursiv',
    'note.strikethrough': 'Durchgestrichen',
    'note.inlineCode': 'Inline-Code',
    'note.highlight': 'Highlight',
    'note.heading1': 'Überschrift 1',
    'note.heading2': 'Überschrift 2',
    'note.heading3': 'Überschrift 3',
    'note.text': 'Text',
    'note.bulletList': 'Aufzählung',
    'note.numberedList': 'Nummerierte Liste',
    'note.taskList': 'Checkliste',
    'note.blockquote': 'Zitat',
    'note.codeBlock': 'Codeblock',
    'note.divider': 'Trennlinie',
    'note.undo': 'Rückgängig',
    'note.redo': 'Wiederholen',
    'note.insertBlock': 'Block einfügen',
    'note.plainParagraph': 'Normaler Absatz',
    'note.largeHeading': 'Große Abschnittsüberschrift',
    'note.mediumHeading': 'Mittlere Abschnittsüberschrift',
    'note.smallHeading': 'Kleine Abschnittsüberschrift',
    'note.unorderedList': 'Ungeordnete Liste',
    'note.orderedList': 'Geordnete nummerierte Liste',
    'note.checklist': 'Checkliste mit Checkboxen',
    'note.multiLineCode': 'Mehrzeiliger Codeblock',
    'note.blockquoteCallout': 'Zitat-Block',
    'note.horizontalSeparator': 'Horizontale Trennlinie',
    'pdf.addComment': 'Kommentar hinzufügen (optional)...',
    'pdf.saveHighlight': 'Highlight speichern',
    'pdf.couldNotLoad': 'PDF konnte nicht geladen werden',
    'pdf.loading': '{name} wird geladen...',
    'pdf.collapseHighlights': 'Highlights einklappen',
    'pdf.expandHighlights': 'Highlights ausklappen',
    'pdf.highlights': 'Highlights',
    'pdf.highlight': 'Highlight',
    'pdf.noHighlights': 'Noch keine Highlights.',
    'pdf.jumpToHighlight': 'Zum Highlight springen',
    'pdf.page': 'Seite',
    'pdf.noText': '(Kein Text)',
    'pdf.collapse': 'Einklappen',
    'pdf.expand': 'Ausklappen',
    'pdf.highlightOnTitle': 'Highlight-Modus an: Text markieren, um ein Highlight zu erstellen',
    'pdf.highlightOffTitle': 'Highlight-Modus aus',
    'pdf.highlightOn': 'Highlight: An',
    'pdf.highlightOff': 'Highlight: Aus',
    'pdf.color.yellow': 'Gelb',
    'pdf.color.green': 'Grün',
    'pdf.color.blue': 'Blau',
    'pdf.color.pink': 'Pink',
  },
};

const LanguageContext = createContext<{
  language: Language;
  setLanguage: (language: Language) => void;
  t: (key: TranslationKey, values?: Record<string, string>) => string;
} | null>(null);

export function LanguageProvider({children}: {children: React.ReactNode}) {
  const [language, setLanguageState] = useState<Language>('de');

  useEffect(() => {
    const stored = window.localStorage.getItem('learncycle-language');
    if (stored === 'de' || stored === 'en') {
      setLanguageState(stored);
    }
  }, []);

  const value = useMemo(() => ({
    language,
    setLanguage: (next: Language) => {
      setLanguageState(next);
      window.localStorage.setItem('learncycle-language', next);
      document.documentElement.lang = next;
    },
    t: (key: TranslationKey, values?: Record<string, string>) => {
      let text = translations[language][key] ?? translations.en[key] ?? key;
      for (const [name, replacement] of Object.entries(values ?? {})) {
        text = text.replaceAll(`{${name}}`, replacement);
      }
      return text;
    },
  }), [language]);

  useEffect(() => {
    document.documentElement.lang = language;
  }, [language]);

  return <LanguageContext.Provider value={value}>{children}</LanguageContext.Provider>;
}

export function useLanguage() {
  const context = useContext(LanguageContext);
  if (!context) {
    throw new Error('useLanguage must be used inside LanguageProvider');
  }
  return context;
}
