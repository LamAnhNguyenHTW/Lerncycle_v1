'use client';

import Link from 'next/link';
import {motion} from 'framer-motion';
import {FileText, PenLine, Highlighter, Mic, MessageSquare, Layers, Brain} from 'lucide-react';
import type {LucideIcon} from 'lucide-react';
import type {Dictionary} from '@/locales/types';
import {Logo} from '@/components/Logo';

interface Props {
  dict: Dictionary;
}

export function Hero({dict}: Props) {
  return (
    <section className="relative overflow-hidden bg-gradient-to-b from-white via-white to-slate-50">
      <div className="mx-auto grid w-full max-w-7xl grid-cols-1 items-center gap-12 px-4 py-20 sm:px-6 md:py-28 lg:grid-cols-2 lg:gap-16 lg:py-32">
        <div className="text-center lg:text-left">
          <h1 className="text-4xl font-semibold tracking-tight text-slate-900 sm:text-5xl lg:text-6xl">
            {dict.hero.h1}
          </h1>
          <p className="mx-auto mt-5 max-w-xl text-base leading-relaxed text-slate-600 sm:text-lg lg:mx-0">
            {dict.hero.subhead}
          </p>
          <div className="mt-8 flex flex-col items-center justify-center gap-3 sm:flex-row lg:justify-start">
            <Link
              href="#beta"
              className="inline-flex h-11 cursor-pointer items-center justify-center rounded-full bg-slate-900 px-6 text-sm font-medium text-white shadow-sm transition-colors duration-200 hover:bg-slate-800"
            >
              {dict.hero.ctaPrimary}
            </Link>
            <Link
              href="/login"
              className="inline-flex h-11 cursor-pointer items-center justify-center rounded-full border border-slate-200 bg-white px-6 text-sm font-medium text-slate-700 transition-colors duration-200 hover:bg-slate-50"
            >
              {dict.hero.ctaSecondary}
            </Link>
          </div>
        </div>

        <HeroFunnel dict={dict} />
      </div>
    </section>
  );
}

interface InputNode {
  icon: LucideIcon;
  label: string;
  from: {x: string; y: string};
  delay: number;
}

interface OutputNode {
  icon: LucideIcon;
  label: string;
  to: {x: string; y: string};
  delay: number;
}

function HeroFunnel({dict}: {dict: Dictionary}) {
  const inputs: InputNode[] = [
    {icon: FileText, label: dict.hero.mockupInputPdf, from: {x: '-110%', y: '-90%'}, delay: 0},
    {icon: PenLine, label: dict.hero.mockupInputNote, from: {x: '110%', y: '-90%'}, delay: 0.9},
    {icon: Highlighter, label: dict.hero.mockupInputHighlight, from: {x: '-110%', y: '90%'}, delay: 1.8},
    {icon: Mic, label: dict.hero.mockupInputAudio, from: {x: '110%', y: '90%'}, delay: 2.7},
  ];

  const outputs: OutputNode[] = [
    {icon: MessageSquare, label: dict.hero.mockupOutputChat, to: {x: '125%', y: '-50%'}, delay: 0.4},
    {icon: Layers, label: dict.hero.mockupOutputFlashcards, to: {x: '125%', y: '50%'}, delay: 1.8},
    {icon: Brain, label: dict.hero.mockupOutputFeynman, to: {x: '-125%', y: '70%'}, delay: 3.2},
  ];

  return (
    <div className="relative mx-auto aspect-square w-full max-w-md sm:max-w-lg">
      <div
        className="absolute inset-[12%] rounded-full"
        style={{
          background:
            'radial-gradient(circle, rgba(241,245,249,0.8) 0%, rgba(255,255,255,1) 60%, rgba(255,255,255,0) 100%)',
        }}
      />

      <svg
        className="absolute inset-0 h-full w-full text-slate-200"
        viewBox="0 0 100 100"
        fill="none"
        aria-hidden="true"
      >
        {[
          [10, 12],
          [90, 12],
          [10, 88],
          [90, 88],
        ].map(([x, y], i) => (
          <line
            key={i}
            x1={x}
            y1={y}
            x2="50"
            y2="50"
            stroke="currentColor"
            strokeWidth="0.3"
            strokeDasharray="1.5 1.5"
          />
        ))}
      </svg>

      <motion.div
        className="absolute left-1/2 top-1/2 z-20 -translate-x-1/2 -translate-y-1/2"
        animate={{scale: [1, 1.04, 1]}}
        transition={{duration: 2.4, repeat: Infinity, ease: 'easeInOut'}}
      >
        <div className="relative">
          <motion.div
            className="absolute inset-0 rounded-2xl bg-slate-900/10"
            animate={{scale: [1, 1.6, 1.6], opacity: [0.5, 0, 0]}}
            transition={{duration: 2.4, repeat: Infinity, ease: 'easeOut'}}
          />
          <div className="relative flex h-20 w-20 items-center justify-center rounded-2xl border border-slate-200 bg-white shadow-xl sm:h-24 sm:w-24">
            <Logo variant="mark" className="h-12 w-12 sm:h-14 sm:w-14" />
          </div>
        </div>
      </motion.div>

      {inputs.map((node, i) => (
        <InputParticle key={`in-${i}`} node={node} />
      ))}

      {outputs.map((node, i) => (
        <OutputCard key={`out-${i}`} node={node} />
      ))}
    </div>
  );
}

function InputParticle({node}: {node: InputNode}) {
  const Icon = node.icon;
  return (
    <motion.div
      className="absolute left-1/2 top-1/2 z-10 -translate-x-1/2 -translate-y-1/2"
      initial={{x: node.from.x, y: node.from.y, opacity: 0, scale: 0.8}}
      animate={{
        x: [node.from.x, '0%', '0%'],
        y: [node.from.y, '0%', '0%'],
        opacity: [0, 1, 0],
        scale: [0.8, 1, 0.6],
      }}
      transition={{
        duration: 3.6,
        times: [0, 0.7, 1],
        repeat: Infinity,
        delay: node.delay,
        ease: 'easeInOut',
      }}
    >
      <div className="flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1.5 shadow-md">
        <Icon className="h-3.5 w-3.5 text-slate-700" strokeWidth={2} />
        <span className="text-[11px] font-medium text-slate-700">{node.label}</span>
      </div>
    </motion.div>
  );
}

function OutputCard({node}: {node: OutputNode}) {
  const Icon = node.icon;
  return (
    <motion.div
      className="absolute left-1/2 top-1/2 z-10 -translate-x-1/2 -translate-y-1/2"
      initial={{x: '0%', y: '0%', opacity: 0, scale: 0.6}}
      animate={{
        x: ['0%', node.to.x, node.to.x],
        y: ['0%', node.to.y, node.to.y],
        opacity: [0, 1, 1, 0],
        scale: [0.6, 1, 1, 0.8],
      }}
      transition={{
        duration: 4.2,
        times: [0, 0.4, 0.85, 1],
        repeat: Infinity,
        delay: node.delay,
        ease: 'easeOut',
      }}
    >
      <div className="flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 py-2 shadow-lg">
        <span className="flex h-6 w-6 items-center justify-center rounded-lg bg-emerald-50 text-emerald-700">
          <Icon className="h-3.5 w-3.5" strokeWidth={2} />
        </span>
        <span className="text-xs font-medium text-slate-900">{node.label}</span>
      </div>
    </motion.div>
  );
}
