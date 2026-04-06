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
    <div className="flex h-full w-full overflow-hidden bg-var(--bg-soft) text-var(--text)">
      {/* Desktop Sidebar (hidden on mobile) */}
      <div className="hidden md:flex h-full">
        {sidebar}
      </div>

      {/* Mobile Sidebar Overlay */}
      {mobileOpen && (
        <div className="md:hidden fixed inset-0 z-50 flex">
          {/* Backdrop */}
          <div 
            className="fixed inset-0 bg-black/20" 
            onClick={() => setMobileOpen(false)}
          />
          {/* Sidebar container */}
          <div className="relative z-50 flex h-full w-4/5 max-w-sm flex-col bg-white shadow-xl transition-transform animate-in slide-in-from-left duration-200">
            {sidebar}
            <button 
              onClick={() => setMobileOpen(false)}
              className="absolute right-4 top-6 rounded-full bg-black/5 p-2 text-muted-foreground hover:bg-black/10 hover:text-foreground"
            >
              <NotionIcon name="ni-x" className="w-[24px] h-[24px]" />
            </button>
          </div>
        </div>
      )}

      {/* Main Content Area */}
      <div className="flex flex-1 flex-col overflow-hidden relative">
        {/* Mobile Header */}
        <header className="md:hidden flex h-14 shrink-0 items-center border-b border-border bg-white px-4">
          <button 
            onClick={() => setMobileOpen(true)}
            className="rounded-md p-2 -ml-2 text-muted-foreground hover:bg-black/5 hover:text-foreground"
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
