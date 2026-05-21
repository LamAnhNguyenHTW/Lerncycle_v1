'use client';

import {useState} from 'react';
import {PdfDropzone} from '@/components/PdfDropzone';
import {NotionIcon} from './NotionIcon';
import {useLanguage} from '@/lib/i18n';
import type {BetaLimits} from '@/lib/beta-limits';

export function DashboardPlaceholder({
  courseId,
  courseName,
  displayName,
  betaLimits,
}: {
  courseId: string;
  courseName: string;
  displayName: string;
  betaLimits: BetaLimits;
}) {
  const [showCourseUpload, setShowCourseUpload] = useState(false);
  const {t} = useLanguage();

  return (
    <div className="flex flex-col items-center max-w-4xl mx-auto pt-6 md:pt-12 w-full gap-6 md:gap-10">
      <div className="text-center space-y-2 md:space-y-3">
        <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight">
          {t('dashboard.headline', {name: displayName, course: courseName})}
        </h1>
        <p className="text-muted-foreground text-sm">
          {t('dashboard.subtitle')}
        </p>
      </div>

      <BetaLimitsNotice limits={betaLimits} />

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 md:gap-5 w-full max-w-3xl mb-8">
        <button 
          onClick={() => setShowCourseUpload(!showCourseUpload)}
          className={`flex flex-col items-start p-5 md:p-6 rounded-xl border transition-all text-left shadow-sm hover:shadow-md hover:border-black/20 ${showCourseUpload ? 'border-primary ring-1 ring-primary' : 'border-border bg-white'}`}
        >
          <div className="mb-3 text-foreground"><NotionIcon name="ni-file-upload" className="w-[24px] h-[24px]" /></div>
          <h3 className="font-medium text-sm mb-1">{t('dashboard.upload')}</h3>
          <p className="text-xs text-muted-foreground">{t('dashboard.uploadSubtitle')}</p>
        </button>

        <ActionCard
          icon={<NotionIcon name="ni-link" className="w-[24px] h-[24px]" />}
          title={t('dashboard.insert')}
          subtitle={t('dashboard.insertSubtitle')}
          comingSoon
        />
        <ActionCard
          icon={<NotionIcon name="ni-microphone" className="w-[24px] h-[24px]" />}
          title={t('dashboard.record')}
          subtitle={t('dashboard.recordSubtitle')}
          comingSoon
        />
      </div>

      {showCourseUpload && (
        <div className="w-full max-w-4xl -mt-6">
          <PdfDropzone targetId={courseId} targetType="course" onUploaded={() => setShowCourseUpload(false)} />
        </div>
      )}
    </div>
  );
}

function BetaLimitsNotice({limits}: {limits: BetaLimits}) {
  const {t} = useLanguage();
  const items = [
    t('beta.limitPdfSize', {value: String(limits.maxPdfMegabytes)}),
    t('beta.limitPdfCount', {value: String(limits.maxPdfsPerUser)}),
    t('beta.limitChat', {value: String(limits.maxChatMessagesPerDay)}),
    t('beta.limitIndexing', {value: String(limits.maxRagJobsPerDay)}),
  ];

  return (
    <section className="w-full max-w-3xl rounded-lg border border-border bg-white px-4 py-3 text-left shadow-sm">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <div className="flex items-center gap-2 text-sm font-semibold text-foreground">
            <NotionIcon name="ni-rocket" className="h-4 w-4" />
            <span>{t('beta.limitsTitle')}</span>
          </div>
          <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
            {t('beta.limitsDescription')}
          </p>
        </div>
        {limits.frozen && (
          <span className="w-fit rounded-md bg-amber-100 px-2 py-1 text-xs font-semibold text-amber-800">
            {t('beta.paused')}
          </span>
        )}
      </div>
      <div className="mt-3 grid grid-cols-1 gap-2 text-xs text-muted-foreground sm:grid-cols-2">
        {items.map((item) => (
          <div key={item} className="rounded-md bg-gray-50 px-3 py-2">
            {item}
          </div>
        ))}
      </div>
    </section>
  );
}

function ActionCard({
  icon,
  title,
  subtitle,
  comingSoon
}: {
  icon: React.ReactNode;
  title: string;
  subtitle: string;
  comingSoon?: boolean;
}) {
  return (
    <button className={`flex flex-col items-start p-5 md:p-6 rounded-xl border transition-all text-left relative overflow-hidden ${comingSoon ? 'border-border/50 bg-gray-50/50 cursor-not-allowed opacity-70 hover:opacity-100 group' : 'border-border bg-white shadow-sm hover:shadow-md hover:border-black/10'}`}>
      <div className={`mb-3 ${comingSoon ? 'text-muted-foreground/60' : 'text-muted-foreground'}`}>{icon}</div>
      <h3 className={`font-medium text-sm mb-1 ${comingSoon ? 'text-muted-foreground' : ''}`}>{title}</h3>
      <p className="text-xs text-muted-foreground">{subtitle}</p>
      
      {comingSoon && (
        <div className="absolute top-4 right-4 rotate-12 transition-transform group-hover:scale-110">
          <span className="bg-gray-200 text-gray-600 text-xs font-bold px-2 py-0.5 rounded shadow-sm border border-gray-300">
            SOON
          </span>
        </div>
      )}
    </button>
  );
}
