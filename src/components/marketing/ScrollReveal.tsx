'use client';

import { motion, HTMLMotionProps } from 'framer-motion';
import { ReactNode } from 'react';

interface ScrollRevealProps extends HTMLMotionProps<"div"> {
  children: ReactNode;
  animation?: 'fade-up' | 'fade-in' | 'slide-left' | 'slide-right';
  delay?: number;
  duration?: number;
}

export function ScrollReveal({
  children,
  animation = 'fade-up',
  delay = 0,
  duration = 0.5,
  className,
  ...rest
}: ScrollRevealProps) {
  const variants = {
    hidden: {
      opacity: 0,
      y: animation === 'fade-up' ? 30 : 0,
      x: animation === 'slide-left' ? 30 : animation === 'slide-right' ? -30 : 0,
    },
    visible: {
      opacity: 1,
      y: 0,
      x: 0,
    },
  };

  return (
    <motion.div
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true, margin: "-10%" }}
      transition={{ duration, delay, ease: 'easeOut' }}
      variants={variants}
      className={className}
      {...rest}
    >
      {children}
    </motion.div>
  );
}
