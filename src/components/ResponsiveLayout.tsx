'use client';

import {useState} from 'react';
import {cn} from '@/lib/utils';
import {NotionIcon} from './NotionIcon';

interface Props {
  sidebar: React.ReactNode;
  children: React.ReactNode;
}

export function ResponsiveLayout({sidebar, children}: Props) {
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <div className="flex h-full min-h-dvh w-full overflow-hidden bg-var(--bg-soft) text-var(--text)">
      {/* Desktop Sidebar (hidden on mobile) */}
      <div className="hidden md:flex h-full">
        {sidebar}
      </div>

      {/* Mobile Sidebar Overlay */}
      {mobileOpen && (
        <div className="md:hidden fixed inset-0 z-50 flex">
          {/* Backdrop */}
          <div 
            className="fixed inset-0 bg-black/40" 
            onClick={() => setMobileOpen(false)}
          />
          {/* Sidebar container */}
          <div className="relative z-50 flex h-full w-4/5 max-w-sm flex-col bg-background shadow-xl transition-transform animate-in slide-in-from-left duration-200">
            {sidebar}
            <button
              onClick={() => setMobileOpen(false)}
              aria-label="Close menu"
              className="absolute right-3 top-4 flex min-h-[44px] min-w-[44px] items-center justify-center rounded-full text-muted-foreground hover:bg-foreground/5 hover:text-foreground"
            >
              <NotionIcon name="ni-chevron-left-circle" className="w-[24px] h-[24px]" />
            </button>
          </div>
        </div>
      )}

      {/* Main Content Area */}
      <div className="flex flex-1 flex-col overflow-hidden relative">
        {/* Mobile Header */}
        <header className="md:hidden flex h-14 shrink-0 items-center border-b border-border bg-background px-4">
          <button
            onClick={() => setMobileOpen(true)}
            aria-label="Open menu"
            className="flex min-h-[44px] min-w-[44px] -ml-2 items-center justify-center rounded-md text-muted-foreground hover:bg-foreground/5 hover:text-foreground"
          >
            <NotionIcon name="ni-list" className="w-[24px] h-[24px]" />
          </button>
          <span className="ml-2 font-bold text-base tracking-tight text-foreground">Learncycle</span>
        </header>

        {/* Content Scroll Area */}
        {children}
      </div>
    </div>
  );
}
