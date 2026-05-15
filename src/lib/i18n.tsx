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
  | 'pdf.color.pink'
  | 'revision.tabs.flashcards'
  | 'revision.tabs.mindmap'
  | 'revision.tabs.mocktest'
  | 'revision.headline'
  | 'revision.subtitle'
  | 'revision.flashcards.newDeck'
  | 'revision.flashcards.title'
  | 'revision.flashcards.titlePlaceholder'
  | 'revision.flashcards.count'
  | 'revision.flashcards.sources'
  | 'revision.flashcards.generate'
  | 'revision.flashcards.generating'
  | 'revision.flashcards.empty'
  | 'revision.flashcards.failed'
  | 'revision.flashcards.review'
  | 'revision.flashcards.reviewEmpty'
  | 'revision.flashcards.flip'
  | 'revision.flashcards.again'
  | 'revision.flashcards.hard'
  | 'revision.flashcards.good'
  | 'revision.flashcards.easy'
  | 'revision.flashcards.cardCount'
  | 'revision.flashcards.dueCount'
  | 'revision.flashcards.delete'
  | 'revision.flashcards.deleteConfirm'
  | 'revision.flashcards.statusReady'
  | 'revision.flashcards.statusPending'
  | 'revision.flashcards.statusFailed'
  | 'revision.flashcards.cardLabel'
  | 'revision.flashcards.question'
  | 'revision.flashcards.answer'
  | 'revision.flashcards.due'
  | 'revision.flashcards.nextDue'
  | 'revision.flashcards.repetitions'
  | 'revision.flashcards.easeFactor'
  | 'revision.flashcards.newCard'
  | 'revision.mocktest.newTest'
  | 'revision.mocktest.title'
  | 'revision.mocktest.count'
  | 'revision.mocktest.questionCount'
  | 'revision.mocktest.start'
  | 'revision.mocktest.submit'
  | 'revision.mocktest.next'
  | 'revision.mocktest.previous'
  | 'revision.mocktest.score'
  | 'revision.mocktest.completed'
  | 'revision.mocktest.empty'
  | 'revision.mocktest.questionOf'
  | 'revision.mocktest.correct'
  | 'revision.mocktest.incorrect'
  | 'revision.mocktest.explanation'
  | 'revision.mocktest.tryAgain'
  | 'revision.mocktest.back'
  | 'revision.mocktest.delete'
  | 'revision.mindmap.selectPdf'
  | 'revision.mindmap.empty'
  | 'revision.mindmap.generatingHint'
  | 'revision.mindmap.loading'
  | 'revision.common.cancel'
  | 'revision.common.create'
  | 'revision.common.close'
  | 'revision.common.back'
  | 'revision.common.pdfSelectAll'
  | 'revision.common.pdfSelectNone';

const translations: Record<Language, Record<TranslationKey, string>> = {
  en: {
    'nav.home': 'Home',
    'nav.notetaking': 'Notes',
    'nav.learn': 'Chat & Research',
    'nav.activeLearning': 'Active Learning',
    'nav.revision': 'Spaced Review',
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
    'revision.tabs.flashcards': 'Flashcards',
    'revision.tabs.mindmap': 'Mindmap',
    'revision.tabs.mocktest': 'Mock Test',
    'revision.headline': 'Revision',
    'revision.subtitle': 'Practice flashcards, explore a mindmap, and take mock tests on your materials.',
    'revision.flashcards.newDeck': 'New deck',
    'revision.flashcards.title': 'Title',
    'revision.flashcards.titlePlaceholder': 'e.g. Chapter 1 — Process Mining',
    'revision.flashcards.count': 'Cards',
    'revision.flashcards.sources': 'Source PDFs',
    'revision.flashcards.generate': 'Generate',
    'revision.flashcards.generating': 'Generating cards...',
    'revision.flashcards.empty': 'No decks yet. Create your first one.',
    'revision.flashcards.failed': 'Generation failed',
    'revision.flashcards.review': 'Review',
    'revision.flashcards.reviewEmpty': 'No cards are due right now. Great job!',
    'revision.flashcards.flip': 'Show answer',
    'revision.flashcards.again': 'Again',
    'revision.flashcards.hard': 'Hard',
    'revision.flashcards.good': 'Good',
    'revision.flashcards.easy': 'Easy',
    'revision.flashcards.cardCount': '{count} card{plural}',
    'revision.flashcards.dueCount': '{count} due',
    'revision.flashcards.delete': 'Delete deck',
    'revision.flashcards.deleteConfirm': 'Delete this deck and all its cards?',
    'revision.flashcards.statusReady': 'Ready',
    'revision.flashcards.statusPending': 'Generating',
    'revision.flashcards.statusFailed': 'Failed',
    'revision.flashcards.cardLabel': 'Card {n}',
    'revision.flashcards.question': 'Question',
    'revision.flashcards.answer': 'Answer',
    'revision.flashcards.due': 'Due',
    'revision.flashcards.nextDue': 'Next due date',
    'revision.flashcards.repetitions': 'Repetitions',
    'revision.flashcards.easeFactor': 'Ease Factor',
    'revision.flashcards.newCard': 'New card',
    'revision.mocktest.newTest': 'New test',
    'revision.mocktest.title': 'Title',
    'revision.mocktest.count': 'Questions',
    'revision.mocktest.questionCount': '{count} question{plural}',
    'revision.mocktest.start': 'Start test',
    'revision.mocktest.submit': 'Submit',
    'revision.mocktest.next': 'Next',
    'revision.mocktest.previous': 'Previous',
    'revision.mocktest.score': 'Score',
    'revision.mocktest.completed': 'Completed ({score}%)',
    'revision.mocktest.empty': 'No tests yet. Generate your first.',
    'revision.mocktest.questionOf': 'Question {n} of {total}',
    'revision.mocktest.correct': 'Correct',
    'revision.mocktest.incorrect': 'Incorrect',
    'revision.mocktest.explanation': 'Explanation',
    'revision.mocktest.tryAgain': 'Try again',
    'revision.mocktest.back': 'Back to tests',
    'revision.mocktest.delete': 'Delete test',
    'revision.mindmap.selectPdf': 'Select a PDF to view its mindmap',
    'revision.mindmap.empty': 'No learning structure has been generated for this PDF yet.',
    'revision.mindmap.generatingHint': 'Mindmaps are built from the Learning Graph extracted by the indexing worker. Once your PDF has been processed, its concept hierarchy will appear here.',
    'revision.mindmap.loading': 'Loading mindmap...',
    'revision.common.cancel': 'Cancel',
    'revision.common.create': 'Create',
    'revision.common.close': 'Close',
    'revision.common.back': 'Back',
    'revision.common.pdfSelectAll': 'Select all',
    'revision.common.pdfSelectNone': 'Clear',
  },
  de: {
    'nav.home': 'Home',
    'nav.notetaking': 'Notizen',
    'nav.learn': 'Chat & Recherche',
    'nav.activeLearning': 'Aktives Lernen',
    'nav.revision': 'Spaced Review',
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
    'revision.tabs.flashcards': 'Karteikarten',
    'revision.tabs.mindmap': 'Mindmap',
    'revision.tabs.mocktest': 'Mocktest',
    'revision.headline': 'Revision',
    'revision.subtitle': 'Übe mit Karteikarten, erkunde eine Mindmap und schreibe Mocktests aus deinen Materialien.',
    'revision.flashcards.newDeck': 'Neues Deck',
    'revision.flashcards.title': 'Titel',
    'revision.flashcards.titlePlaceholder': 'z. B. Kapitel 1 — Process Mining',
    'revision.flashcards.count': 'Karten',
    'revision.flashcards.sources': 'Quell-PDFs',
    'revision.flashcards.generate': 'Erstellen',
    'revision.flashcards.generating': 'Karten werden erstellt...',
    'revision.flashcards.empty': 'Noch keine Decks. Erstelle dein erstes.',
    'revision.flashcards.failed': 'Erstellung fehlgeschlagen',
    'revision.flashcards.review': 'Wiederholen',
    'revision.flashcards.reviewEmpty': 'Aktuell sind keine Karten fällig. Sehr gut!',
    'revision.flashcards.flip': 'Antwort zeigen',
    'revision.flashcards.again': 'Nochmal',
    'revision.flashcards.hard': 'Schwer',
    'revision.flashcards.good': 'Gut',
    'revision.flashcards.easy': 'Einfach',
    'revision.flashcards.cardCount': '{count} Karte{plural}',
    'revision.flashcards.dueCount': '{count} fällig',
    'revision.flashcards.delete': 'Deck löschen',
    'revision.flashcards.deleteConfirm': 'Dieses Deck und alle Karten löschen?',
    'revision.flashcards.statusReady': 'Bereit',
    'revision.flashcards.statusPending': 'Wird erstellt',
    'revision.flashcards.statusFailed': 'Fehlgeschlagen',
    'revision.flashcards.cardLabel': 'Karte {n}',
    'revision.flashcards.question': 'Frage',
    'revision.flashcards.answer': 'Antwort',
    'revision.flashcards.due': 'Fällig',
    'revision.flashcards.nextDue': 'Nächste Fälligkeit',
    'revision.flashcards.repetitions': 'Wiederholungen',
    'revision.flashcards.easeFactor': 'Leichtigkeit',
    'revision.flashcards.newCard': 'Neue Karte',
    'revision.mocktest.newTest': 'Neuer Test',
    'revision.mocktest.title': 'Titel',
    'revision.mocktest.count': 'Fragen',
    'revision.mocktest.questionCount': '{count} Frage{plural}',
    'revision.mocktest.start': 'Test starten',
    'revision.mocktest.submit': 'Abgeben',
    'revision.mocktest.next': 'Weiter',
    'revision.mocktest.previous': 'Zurück',
    'revision.mocktest.score': 'Ergebnis',
    'revision.mocktest.completed': 'Absolviert ({score}%)',
    'revision.mocktest.empty': 'Noch keine Tests. Erstelle deinen ersten.',
    'revision.mocktest.questionOf': 'Frage {n} von {total}',
    'revision.mocktest.correct': 'Richtig',
    'revision.mocktest.incorrect': 'Falsch',
    'revision.mocktest.explanation': 'Erklärung',
    'revision.mocktest.tryAgain': 'Erneut starten',
    'revision.mocktest.back': 'Zurück zur Liste',
    'revision.mocktest.delete': 'Test löschen',
    'revision.mindmap.selectPdf': 'Wähle ein PDF, um die Mindmap anzuzeigen',
    'revision.mindmap.empty': 'Für dieses PDF wurde noch keine Lernstruktur erstellt.',
    'revision.mindmap.generatingHint': 'Mindmaps werden aus dem Learning Graph erzeugt, den der Indexierungs-Worker baut. Sobald dein PDF verarbeitet ist, erscheint hier die Konzepthierarchie.',
    'revision.mindmap.loading': 'Mindmap wird geladen...',
    'revision.common.cancel': 'Abbrechen',
    'revision.common.create': 'Erstellen',
    'revision.common.close': 'Schließen',
    'revision.common.back': 'Zurück',
    'revision.common.pdfSelectAll': 'Alle auswählen',
    'revision.common.pdfSelectNone': 'Auswahl löschen',
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
