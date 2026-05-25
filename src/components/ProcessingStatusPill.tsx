'use client';

import {AlertCircle, CheckCircle2, Loader2} from 'lucide-react';

import type {OverallStage} from '@/lib/processing-status';
import {useLanguage} from '@/lib/i18n';

type ProcessingStatusPillProps = {
  stage: OverallStage;
  userSafeError?: string | null;
};

const LABEL_KEYS: Record<OverallStage, {de: string; en: string}> = {
  queued: {de: 'In Warteschlange', en: 'Queued'},
  parsing: {de: 'Dokument wird gelesen', en: 'Reading document'},
  indexing: {de: 'Wird indexiert', en: 'Indexing'},
  graph: {de: 'Wissensgraph wird gebaut', en: 'Building graph'},
  partial_ready: {de: 'Chat bereit - Graph laeuft', en: 'Chat ready - Graph running'},
  ready: {de: 'Bereit', en: 'Ready'},
  failed: {de: 'Fehlgeschlagen', en: 'Failed'},
};

const CLASS_NAMES: Record<OverallStage, string> = {
  queued: 'border-border bg-muted text-muted-foreground',
  parsing: 'border-blue-200 bg-blue-50 text-blue-700',
  indexing: 'border-blue-200 bg-blue-50 text-blue-700',
  graph: 'border-blue-200 bg-blue-50 text-blue-700',
  partial_ready: 'border-amber-200 bg-amber-50 text-amber-800',
  ready: 'border-emerald-200 bg-emerald-50 text-emerald-700',
  failed: 'border-red-200 bg-red-50 text-red-700',
};

export function ProcessingStatusPill({stage, userSafeError}: ProcessingStatusPillProps) {
  const {language} = useLanguage();
  const label = LABEL_KEYS[stage][language];
  const title = stage === 'failed' && userSafeError ? userSafeError : label;

  return (
    <span
      className={`inline-flex h-6 max-w-[180px] shrink-0 items-center gap-1 rounded-md border px-1.5 text-xs font-medium ${CLASS_NAMES[stage]}`}
      title={title}
      aria-label={title}
    >
      {iconForStage(stage)}
      <span className="truncate">{label}</span>
    </span>
  );
}

function iconForStage(stage: OverallStage) {
  if (stage === 'ready' || stage === 'partial_ready') {
    return <CheckCircle2 className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />;
  }
  if (stage === 'failed') {
    return <AlertCircle className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />;
  }
  if (stage === 'queued') {
    return <span className="h-2 w-2 shrink-0 rounded-full bg-current opacity-50" aria-hidden="true" />;
  }
  return <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin" aria-hidden="true" />;
}
