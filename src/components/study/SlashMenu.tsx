'use client';

import { useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import type { SlashItem } from './slash-command';
import {useLanguage} from '@/lib/i18n';

interface SlashMenuProps {
  items: SlashItem[];
  selectedIndex: number;
  position: { top: number; left: number };
  onSelect: (item: SlashItem) => void;
}

/** Keyboard-navigable slash command dropdown, rendered into document.body. */
export function SlashMenu({ items, selectedIndex, position, onSelect }: SlashMenuProps) {
  const {t} = useLanguage();
  const itemRefs = useRef<(HTMLButtonElement | null)[]>([]);

  // Scroll selected item into view
  useEffect(() => {
    itemRefs.current[selectedIndex]?.scrollIntoView({ block: 'nearest' });
  }, [selectedIndex]);

  if (items.length === 0) return null;

  return createPortal(
    <div
      className="slash-menu"
      style={{ top: position.top, left: position.left }}
      role="listbox"
      aria-label={t('note.insertBlock')}
    >
      {items.map((item, i) => (
        <button
          key={item.title}
          ref={(el) => { itemRefs.current[i] = el; }}
          role="option"
          aria-selected={i === selectedIndex}
          className={`slash-menu-item${i === selectedIndex ? ' selected' : ''}`}
          onMouseDown={(e) => {
            e.preventDefault();
            onSelect(item);
          }}
          type="button"
        >
          <span className="slash-menu-icon">{item.icon}</span>
          <span className="slash-menu-text">
            <span className="slash-menu-title">{t(item.titleKey)}</span>
            <span className="slash-menu-desc">{t(item.descriptionKey)}</span>
          </span>
        </button>
      ))}
    </div>,
    document.body,
  );
}
