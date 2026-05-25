'use client';

import {ThemeProvider as NextThemesProvider} from 'next-themes';
import type {ReactNode} from 'react';

export function ThemeProvider({children}: {children: ReactNode}) {
  return (
    <NextThemesProvider
      attribute="class"
      defaultTheme="light"
      enableSystem={false}
      disableTransitionOnChange
      storageKey="learncycle-theme"
    >
      {children}
    </NextThemesProvider>
  );
}
