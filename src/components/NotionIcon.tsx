import {cn} from '@/lib/utils';

interface Props {
  name: string;
  className?: string;
}

/**
 * A tiny wrapper that loads a Notion SVG from /public/icons/
 * and uses a CSS mask so it perfectly inherits the current text color.
 */
export function NotionIcon({name, className}: Props) {
  return (
    <div
      className={cn('inline-block bg-current', className)}
      style={{
        maskImage: `url(/icons/${name}.svg)`,
        WebkitMaskImage: `url(/icons/${name}.svg)`,
        maskSize: 'contain',
        WebkitMaskSize: 'contain',
        maskRepeat: 'no-repeat',
        WebkitMaskRepeat: 'no-repeat',
        maskPosition: 'center',
        WebkitMaskPosition: 'center',
      }}
    />
  );
}
