import { Extension } from '@tiptap/core';
import type { Range, Editor } from '@tiptap/core';
import { Suggestion } from '@tiptap/suggestion';
import type {TranslationKey} from '@/lib/i18n';

export interface SlashItem {
  title: string;
  description: string;
  titleKey: TranslationKey;
  descriptionKey: TranslationKey;
  icon: string;
  execute: (editor: Editor, range: Range) => void;
}

const ALL_ITEMS: SlashItem[] = [
  {
    title: 'Text',
    description: 'Plain paragraph',
    titleKey: 'note.text',
    descriptionKey: 'note.plainParagraph',
    icon: '¶',
    execute: (editor, range) =>
      editor.chain().focus().deleteRange(range).setParagraph().run(),
  },
  {
    title: 'Heading 1',
    description: 'Large section heading',
    titleKey: 'note.heading1',
    descriptionKey: 'note.largeHeading',
    icon: 'H1',
    execute: (editor, range) =>
      editor.chain().focus().deleteRange(range).setHeading({ level: 1 }).run(),
  },
  {
    title: 'Heading 2',
    description: 'Medium section heading',
    titleKey: 'note.heading2',
    descriptionKey: 'note.mediumHeading',
    icon: 'H2',
    execute: (editor, range) =>
      editor.chain().focus().deleteRange(range).setHeading({ level: 2 }).run(),
  },
  {
    title: 'Heading 3',
    description: 'Small section heading',
    titleKey: 'note.heading3',
    descriptionKey: 'note.smallHeading',
    icon: 'H3',
    execute: (editor, range) =>
      editor.chain().focus().deleteRange(range).setHeading({ level: 3 }).run(),
  },
  {
    title: 'Bullet List',
    description: 'Unordered list',
    titleKey: 'note.bulletList',
    descriptionKey: 'note.unorderedList',
    icon: '•',
    execute: (editor, range) =>
      editor.chain().focus().deleteRange(range).toggleBulletList().run(),
  },
  {
    title: 'Numbered List',
    description: 'Ordered numbered list',
    titleKey: 'note.numberedList',
    descriptionKey: 'note.orderedList',
    icon: '1.',
    execute: (editor, range) =>
      editor.chain().focus().deleteRange(range).toggleOrderedList().run(),
  },
  {
    title: 'Task List',
    description: 'Checklist with checkboxes',
    titleKey: 'note.taskList',
    descriptionKey: 'note.checklist',
    icon: '☑',
    execute: (editor, range) =>
      editor.chain().focus().deleteRange(range).toggleTaskList().run(),
  },
  {
    title: 'Code Block',
    description: 'Multi-line code snippet',
    titleKey: 'note.codeBlock',
    descriptionKey: 'note.multiLineCode',
    icon: '</>',
    execute: (editor, range) =>
      editor.chain().focus().deleteRange(range).toggleCodeBlock().run(),
  },
  {
    title: 'Quote',
    description: 'Blockquote callout',
    titleKey: 'note.blockquote',
    descriptionKey: 'note.blockquoteCallout',
    icon: '"',
    execute: (editor, range) =>
      editor.chain().focus().deleteRange(range).toggleBlockquote().run(),
  },
  {
    title: 'Divider',
    description: 'Horizontal separator',
    titleKey: 'note.divider',
    descriptionKey: 'note.horizontalSeparator',
    icon: '—',
    execute: (editor, range) =>
      editor.chain().focus().deleteRange(range).setHorizontalRule().run(),
  },
];

export function filterSlashItems(query: string): SlashItem[] {
  if (!query) return ALL_ITEMS;
  const q = query.toLowerCase();
  return ALL_ITEMS.filter(
    (item) =>
      item.title.toLowerCase().includes(q) ||
      item.description.toLowerCase().includes(q),
  );
}

export interface SlashMenuState {
  items: SlashItem[];
  selectedIndex: number;
  position: { top: number; left: number };
  executeCommand: (item: SlashItem) => void;
}

export interface SlashCallbacks {
  onOpen: (state: SlashMenuState) => void;
  onUpdate: (partial: Pick<SlashMenuState, 'items' | 'position'>) => void;
  onClose: () => void;
  onKeyDown: (event: KeyboardEvent) => boolean;
}

/** Creates the slash-command Tiptap extension. Pass a stable ref getter so the
 *  plugin can call the latest React callbacks without being recreated. */
export function createSlashExtension(getCallbacks: () => SlashCallbacks) {
  return Extension.create({
    name: 'slashCommand',

    addProseMirrorPlugins() {
      return [
        Suggestion<SlashItem, SlashItem>({
          editor: this.editor,
          char: '/',
          startOfLine: false,
          allowSpaces: false,

          items: ({ query }) => filterSlashItems(query),

          render: () => ({
            onStart(props) {
              const rect = props.clientRect?.() ?? null;
              getCallbacks().onOpen({
                items: props.items,
                selectedIndex: 0,
                position: rect
                  ? { top: rect.bottom + 6, left: rect.left }
                  : { top: 0, left: 0 },
                executeCommand: (item) => props.command(item),
              });
            },
            onUpdate(props) {
              const rect = props.clientRect?.() ?? null;
              getCallbacks().onUpdate({
                items: props.items,
                position: rect
                  ? { top: rect.bottom + 6, left: rect.left }
                  : { top: 0, left: 0 },
              });
            },
            onExit() {
              getCallbacks().onClose();
            },
            onKeyDown(props) {
              return getCallbacks().onKeyDown(props.event);
            },
          }),

          command: ({ editor, range, props }) => {
            props.execute(editor, range);
          },
        }),
      ];
    },
  });
}
