import Image from 'next/image';
import {cn} from '@/lib/utils';

type Variant = 'horizontal' | 'horizontal-dark' | 'mark' | 'mark-dark' | 'app-icon';

const SOURCES: Record<Variant, {src: string; width: number; height: number}> = {
  'horizontal': {src: '/logo/learncycle-logo-horizontal.svg', width: 800, height: 180},
  'horizontal-dark': {src: '/logo/learncycle-logo-horizontal-dark.svg', width: 800, height: 180},
  'mark': {src: '/logo/learncycle-mark.svg', width: 256, height: 256},
  'mark-dark': {src: '/logo/learncycle-mark-dark.svg', width: 256, height: 256},
  'app-icon': {src: '/logo/learncycle-app-icon.svg', width: 256, height: 256},
};

interface Props {
  variant?: Variant;
  className?: string;
  priority?: boolean;
}

export function Logo({variant = 'horizontal', className, priority}: Props) {
  if (variant === 'horizontal') {
    return (
      <ThemedPair
        light={SOURCES['horizontal']}
        dark={SOURCES['horizontal-dark']}
        className={className}
        priority={priority}
      />
    );
  }

  if (variant === 'mark') {
    return (
      <ThemedPair
        light={SOURCES['mark']}
        dark={SOURCES['mark-dark']}
        className={className}
        priority={priority}
      />
    );
  }

  const {src, width, height} = SOURCES[variant];
  return (
    <Image
      src={src}
      width={width}
      height={height}
      alt="LearnCycle"
      priority={priority}
      unoptimized
      className={className}
    />
  );
}

function ThemedPair({
  light,
  dark,
  className,
  priority,
}: {
  light: {src: string; width: number; height: number};
  dark: {src: string; width: number; height: number};
  className?: string;
  priority?: boolean;
}) {
  return (
    <>
      <Image
        src={light.src}
        width={light.width}
        height={light.height}
        alt="LearnCycle"
        priority={priority}
        unoptimized
        className={cn('block dark:hidden', className)}
      />
      <Image
        src={dark.src}
        width={dark.width}
        height={dark.height}
        alt="LearnCycle"
        priority={priority}
        unoptimized
        className={cn('hidden dark:block', className)}
      />
    </>
  );
}
