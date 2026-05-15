'use client';

import {useMemo, useState} from 'react';
import {useLanguage} from '@/lib/i18n';
import type {Course} from '@/lib/data';
import {FlashcardsTab} from './flashcards/FlashcardsTab';
import {MockTestTab} from './mocktest/MockTestTab';
import {MindmapTab} from './mindmap/MindmapTab';
import {collectCoursePdfs} from './pdf-utils';

type SubTab = 'flashcards' | 'mindmap' | 'mocktest';

export function RevisionSection({course}: {course: Course}) {
  const {t} = useLanguage();
  const [tab, setTab] = useState<SubTab>('flashcards');
  const pdfOptions = useMemo(() => collectCoursePdfs(course), [course]);

  return (
    <div className="max-w-4xl mx-auto w-full md:pt-4 pb-12 space-y-6">
      <div>
        <h1 className="text-2xl font-bold">{t('revision.headline')}</h1>
        <p className="text-muted-foreground mt-1 text-sm">{t('revision.subtitle')}</p>
      </div>

      <div className="border-b border-border">
        <nav className="flex gap-1 -mb-px">
          {(
            [
              {id: 'flashcards' as const, label: t('revision.tabs.flashcards')},
              {id: 'mindmap' as const, label: t('revision.tabs.mindmap')},
              {id: 'mocktest' as const, label: t('revision.tabs.mocktest')},
            ]
          ).map((item) => (
            <button
              key={item.id}
              type="button"
              onClick={() => setTab(item.id)}
              className={
                'px-4 py-2 text-sm font-medium border-b-2 transition ' +
                (tab === item.id
                  ? 'border-primary text-foreground'
                  : 'border-transparent text-muted-foreground hover:text-foreground')
              }
            >
              {item.label}
            </button>
          ))}
        </nav>
      </div>

      <div>
        {tab === 'flashcards' && <FlashcardsTab courseId={course.id} pdfOptions={pdfOptions} />}
        {tab === 'mindmap' && <MindmapTab pdfOptions={pdfOptions} />}
        {tab === 'mocktest' && <MockTestTab courseId={course.id} pdfOptions={pdfOptions} />}
      </div>
    </div>
  );
}
