'use client';

import {useEffect, useState} from 'react';
import {useTheme} from 'next-themes';
import {Sun, Moon, MonitorSmartphone} from 'lucide-react';
import {cn} from '@/lib/utils';

interface Props {
  variant?: 'compact' | 'full';
  className?: string;
  labels?: {
    label: string;
    light: string;
    dark: string;
    system: string;
    cycle: string;
  };
}

const ORDER = ['light', 'dark', 'system'] as const;
type ThemeChoice = (typeof ORDER)[number];

export function ThemeToggle({variant = 'compact', className, labels}: Props) {
  const {theme, setTheme} = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setMounted(true);
  }, []);

  const current = (mounted ? (theme as ThemeChoice) : 'system') ?? 'system';
  const next = ORDER[(ORDER.indexOf(current) + 1) % ORDER.length];

  const Icon = current === 'light' ? Sun : current === 'dark' ? Moon : MonitorSmartphone;
  const currentLabel = labels
    ? current === 'light'
      ? labels.light
      : current === 'dark'
        ? labels.dark
        : labels.system
    : current;

  return (
    <button
      type="button"
      onClick={() => setTheme(next)}
      aria-label={labels?.cycle ?? 'Switch theme'}
      title={labels?.cycle ?? 'Switch theme'}
      suppressHydrationWarning
      className={cn(
        'inline-flex items-center gap-2 rounded-lg text-xs font-medium text-muted-foreground transition-colors hover:bg-black/5 hover:text-foreground dark:hover:bg-white/5 cursor-pointer',
        variant === 'full' ? 'h-8 px-2' : 'h-8 w-8 justify-center',
        className,
      )}
    >
      <Icon className="h-4 w-4" aria-hidden="true" />
      {variant === 'full' && mounted && <span>{currentLabel}</span>}
    </button>
  );
}
