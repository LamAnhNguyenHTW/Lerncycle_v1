'use client';

import {useEditor, EditorContent} from '@tiptap/react';
import type {JSONContent} from '@tiptap/core';
import StarterKit from '@tiptap/starter-kit';
import Placeholder from '@tiptap/extension-placeholder';
import Highlight from '@tiptap/extension-highlight';
import TaskList from '@tiptap/extension-task-list';
import TaskItem from '@tiptap/extension-task-item';
import Typography from '@tiptap/extension-typography';
import {useEffect, useRef, useState, useCallback, useMemo} from 'react';
import {
  Bold,
  Italic,
  Strikethrough,
  Heading2,
  Heading3,
  List,
  ListOrdered,
  Quote,
  Code,
  Code2,
  Minus,
  Undo2,
  Redo2,
  Highlighter,
  CheckSquare,
} from 'lucide-react';
import {upsertNote} from '@/actions/notes';
import {createSlashExtension} from './slash-command';
import type {SlashCallbacks, SlashMenuState, SlashItem} from './slash-command';
import {SlashMenu} from './SlashMenu';
import {useLanguage} from '@/lib/i18n';

interface NoteEditorProps {
  pdfId: string;
  initialContent: JSONContent | null;
}

type SaveState = 'idle' | 'saving' | 'saved' | 'error';

interface BubblePos {
  top: number;
  left: number;
}

interface ToolbarButtonProps {
  onClick: () => void;
  active?: boolean;
  title: string;
  children: React.ReactNode;
  wide?: boolean;
}

function ToolbarButton({onClick, active, title, children, wide}: ToolbarButtonProps) {
  return (
    <button
      onMouseDown={(e) => {
        e.preventDefault();
        onClick();
      }}
      className={`note-toolbar-btn${active ? ' active' : ''}${wide ? ' wide' : ''}`}
      title={title}
      type="button"
    >
      {children}
    </button>
  );
}

/** Notion-style rich text editor with bubble menu, highlight, and slash commands. */
export function NoteEditor({pdfId, initialContent}: NoteEditorProps) {
  const {t} = useLanguage();
  const [saveState, setSaveState] = useState<SaveState>('idle');
  const [bubblePos, setBubblePos] = useState<BubblePos | null>(null);
  const [slashMenu, setSlashMenu] = useState<SlashMenuState | null>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const bubbleRef = useRef<HTMLDivElement>(null);

  // Stable ref so the slash extension always calls the latest handlers
  const callbacksRef = useRef<SlashCallbacks>({
    onOpen: () => {},
    onUpdate: () => {},
    onClose: () => {},
    onKeyDown: () => false,
  });

  // Keep callbacksRef current without recreating the extension
  callbacksRef.current = {
    onOpen: (state) => setSlashMenu(state),
    onUpdate: ({items, position}) =>
      setSlashMenu((prev) => (prev ? {...prev, items, position, selectedIndex: 0} : null)),
    onClose: () => setSlashMenu(null),
    onKeyDown: (event) => {
      if (!slashMenu) return false;
      if (event.key === 'ArrowDown') {
        setSlashMenu((prev) =>
          prev ? {...prev, selectedIndex: (prev.selectedIndex + 1) % prev.items.length} : null,
        );
        return true;
      }
      if (event.key === 'ArrowUp') {
        setSlashMenu((prev) =>
          prev
            ? {...prev, selectedIndex: (prev.selectedIndex - 1 + prev.items.length) % prev.items.length}
            : null,
        );
        return true;
      }
      if (event.key === 'Enter') {
        if (slashMenu.items[slashMenu.selectedIndex]) {
          slashMenu.executeCommand(slashMenu.items[slashMenu.selectedIndex]);
          setSlashMenu(null);
        }
        return true;
      }
      if (event.key === 'Escape') {
        setSlashMenu(null);
        return true;
      }
      return false;
    },
  };

  // Created once — uses the ref getter so callbacks are always fresh
  const slashExtension = useMemo(
    () => createSlashExtension(() => callbacksRef.current),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );

  const save = useCallback(
    async (content: JSONContent) => {
      setSaveState('saving');
      const result = await upsertNote(pdfId, content as Record<string, unknown>);
      setSaveState(result.error ? 'error' : 'saved');
      setTimeout(() => setSaveState('idle'), 2000);
    },
    [pdfId],
  );

  const updateBubble = useCallback(() => {
    const sel = window.getSelection();
    if (!sel || sel.isCollapsed || sel.rangeCount === 0) {
      setBubblePos(null);
      return;
    }
    const rect = sel.getRangeAt(0).getBoundingClientRect();
    if (rect.width === 0) {
      setBubblePos(null);
      return;
    }
    setBubblePos({top: rect.top - 44, left: rect.left + rect.width / 2});
  }, []);

  const editor = useEditor({
    immediatelyRender: false,
    extensions: [
      StarterKit,
      Placeholder.configure({placeholder: t('note.placeholder')}),
      Highlight.configure({multicolor: false}),
      TaskList,
      TaskItem.configure({nested: true}),
      Typography,
      slashExtension,
    ],
    content: initialContent ?? {type: 'doc', content: []},
    editorProps: {
      attributes: {class: 'note-editor-content'},
    },
    onUpdate: ({editor: ed}) => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(() => save(ed.getJSON()), 800);
    },
    onSelectionUpdate: updateBubble,
  });

  // Hide bubble on outside click
  useEffect(() => {
    const hide = (e: MouseEvent) => {
      if (bubbleRef.current?.contains(e.target as Node)) return;
      setBubblePos(null);
    };
    document.addEventListener('mousedown', hide);
    return () => document.removeEventListener('mousedown', hide);
  }, []);

  useEffect(() => {
    if (!editor) return;
    editor.commands.setContent(initialContent ?? {type: 'doc', content: []});
    setSaveState('idle');
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pdfId]);

  useEffect(() => {
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, []);

  const saveLabel =
    saveState === 'saving'
      ? t('note.saving')
      : saveState === 'saved'
        ? t('note.saved')
        : saveState === 'error'
          ? t('note.saveFailed')
          : '';

  const handleSlashSelect = useCallback(
    (item: SlashItem) => {
      slashMenu?.executeCommand(item);
      setSlashMenu(null);
    },
    [slashMenu],
  );

  return (
    <div className="note-editor-container">
      {/* Slash command menu */}
      {slashMenu && slashMenu.items.length > 0 && (
        <SlashMenu
          items={slashMenu.items}
          selectedIndex={slashMenu.selectedIndex}
          position={slashMenu.position}
          onSelect={handleSlashSelect}
        />
      )}

      {/* Floating bubble menu */}
      {bubblePos && editor && (
        <div
          ref={bubbleRef}
          className="note-bubble-menu"
          style={{top: bubblePos.top, left: bubblePos.left}}
        >
          <button
            onMouseDown={(e) => { e.preventDefault(); editor.chain().focus().toggleBold().run(); }}
            className={`note-bubble-btn${editor.isActive('bold') ? ' active' : ''}`}
            title={t('note.bold')}
            type="button"
          >
            <Bold size={13} />
          </button>
          <button
            onMouseDown={(e) => { e.preventDefault(); editor.chain().focus().toggleItalic().run(); }}
            className={`note-bubble-btn${editor.isActive('italic') ? ' active' : ''}`}
            title={t('note.italic')}
            type="button"
          >
            <Italic size={13} />
          </button>
          <button
            onMouseDown={(e) => { e.preventDefault(); editor.chain().focus().toggleStrike().run(); }}
            className={`note-bubble-btn${editor.isActive('strike') ? ' active' : ''}`}
            title={t('note.strikethrough')}
            type="button"
          >
            <Strikethrough size={13} />
          </button>
          <button
            onMouseDown={(e) => { e.preventDefault(); editor.chain().focus().toggleCode().run(); }}
            className={`note-bubble-btn${editor.isActive('code') ? ' active' : ''}`}
            title={t('note.inlineCode')}
            type="button"
          >
            <Code size={13} />
          </button>
          <div className="note-bubble-divider" />
          <button
            onMouseDown={(e) => { e.preventDefault(); editor.chain().focus().toggleHighlight().run(); }}
            className={`note-bubble-btn highlight${editor.isActive('highlight') ? ' active' : ''}`}
            title={t('note.highlight')}
            type="button"
          >
            <Highlighter size={13} />
          </button>
        </div>
      )}

      {/* Fixed toolbar */}
      <div className="note-editor-toolbar">
        <div className="note-editor-toolbar-left">
          <ToolbarButton
            onClick={() => editor?.chain().focus().toggleHeading({level: 2}).run()}
            active={editor?.isActive('heading', {level: 2})}
            title={t('note.heading2')}
          >
            <Heading2 size={15} />
          </ToolbarButton>
          <ToolbarButton
            onClick={() => editor?.chain().focus().toggleHeading({level: 3}).run()}
            active={editor?.isActive('heading', {level: 3})}
            title={t('note.heading3')}
          >
            <Heading3 size={15} />
          </ToolbarButton>

          <div className="note-toolbar-divider" />

          <ToolbarButton
            onClick={() => editor?.chain().focus().toggleBold().run()}
            active={editor?.isActive('bold')}
            title={t('note.bold')}
          >
            <Bold size={14} />
          </ToolbarButton>
          <ToolbarButton
            onClick={() => editor?.chain().focus().toggleItalic().run()}
            active={editor?.isActive('italic')}
            title={t('note.italic')}
          >
            <Italic size={14} />
          </ToolbarButton>
          <ToolbarButton
            onClick={() => editor?.chain().focus().toggleStrike().run()}
            active={editor?.isActive('strike')}
            title={t('note.strikethrough')}
          >
            <Strikethrough size={14} />
          </ToolbarButton>
          <ToolbarButton
            onClick={() => editor?.chain().focus().toggleCode().run()}
            active={editor?.isActive('code')}
            title={t('note.inlineCode')}
          >
            <Code size={14} />
          </ToolbarButton>
          <ToolbarButton
            onClick={() => editor?.chain().focus().toggleHighlight().run()}
            active={editor?.isActive('highlight')}
            title={t('note.highlight')}
          >
            <Highlighter size={14} />
          </ToolbarButton>

          <div className="note-toolbar-divider" />

          <ToolbarButton
            onClick={() => editor?.chain().focus().toggleBulletList().run()}
            active={editor?.isActive('bulletList')}
            title={t('note.bulletList')}
          >
            <List size={15} />
          </ToolbarButton>
          <ToolbarButton
            onClick={() => editor?.chain().focus().toggleOrderedList().run()}
            active={editor?.isActive('orderedList')}
            title={t('note.numberedList')}
          >
            <ListOrdered size={15} />
          </ToolbarButton>
          <ToolbarButton
            onClick={() => editor?.chain().focus().toggleTaskList().run()}
            active={editor?.isActive('taskList')}
            title={t('note.taskList')}
          >
            <CheckSquare size={14} />
          </ToolbarButton>
          <ToolbarButton
            onClick={() => editor?.chain().focus().toggleBlockquote().run()}
            active={editor?.isActive('blockquote')}
            title={t('note.blockquote')}
          >
            <Quote size={14} />
          </ToolbarButton>
          <ToolbarButton
            onClick={() => editor?.chain().focus().toggleCodeBlock().run()}
            active={editor?.isActive('codeBlock')}
            title={t('note.codeBlock')}
          >
            <Code2 size={14} />
          </ToolbarButton>
          <ToolbarButton
            onClick={() => editor?.chain().focus().setHorizontalRule().run()}
            title={t('note.divider')}
          >
            <Minus size={14} />
          </ToolbarButton>

          <div className="note-toolbar-divider" />

          <ToolbarButton onClick={() => editor?.chain().focus().undo().run()} title={t('note.undo')}>
            <Undo2 size={14} />
          </ToolbarButton>
          <ToolbarButton onClick={() => editor?.chain().focus().redo().run()} title={t('note.redo')}>
            <Redo2 size={14} />
          </ToolbarButton>
        </div>

        {saveLabel && (
          <span className={`note-save-indicator${saveState === 'error' ? ' error' : ''}`}>
            {saveLabel}
          </span>
        )}
      </div>

      {/* Editor area */}
      <div className="note-editor-body">
        <EditorContent editor={editor} />
      </div>
    </div>
  );
}
