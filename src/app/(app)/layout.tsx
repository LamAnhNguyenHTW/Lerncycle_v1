import type {Metadata} from 'next';
import {Inter} from 'next/font/google';
import {LanguageProvider} from '@/lib/i18n';
import {ThemeProvider} from '@/components/theme/ThemeProvider';
import '../globals.css';

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-sans',
});

export const metadata: Metadata = {
  title: 'Learncycle',
  description: 'AI-powered learning companion for structured study.',
  icons: {
    icon: [
      {url: '/favicon/favicon.svg', type: 'image/svg+xml'},
      {url: '/favicon/favicon-32x32.png', sizes: '32x32', type: 'image/png'},
      {url: '/favicon/favicon-16x16.png', sizes: '16x16', type: 'image/png'},
    ],
    apple: '/favicon/apple-touch-icon.png',
  },
  manifest: '/site.webmanifest',
};

export default function AppRootLayout({children}: {children: React.ReactNode}) {
  return (
    <html
      lang="de"
      suppressHydrationWarning
      className={`${inter.variable} h-full antialiased`}
    >
      <body className="h-full flex flex-col font-sans">
        <ThemeProvider>
          <LanguageProvider>{children}</LanguageProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
