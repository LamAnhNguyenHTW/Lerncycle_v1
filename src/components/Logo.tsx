import Image from 'next/image';

type Variant = 'horizontal' | 'horizontal-dark' | 'mark' | 'app-icon';

const SOURCES: Record<Variant, {src: string; width: number; height: number}> = {
  'horizontal': {src: '/logo/learncycle-logo-horizontal.svg', width: 800, height: 180},
  'horizontal-dark': {src: '/logo/learncycle-logo-horizontal-dark.svg', width: 800, height: 180},
  'mark': {src: '/logo/learncycle-mark.svg', width: 256, height: 256},
  'app-icon': {src: '/logo/learncycle-app-icon.svg', width: 256, height: 256},
};

interface Props {
  variant?: Variant;
  className?: string;
  priority?: boolean;
}

export function Logo({variant = 'horizontal', className, priority}: Props) {
  const {src, width, height} = SOURCES[variant];
  return (
    <Image
      src={src}
      width={width}
      height={height}
      alt="LearnCycle"
      priority={priority}
      className={className}
    />
  );
}
