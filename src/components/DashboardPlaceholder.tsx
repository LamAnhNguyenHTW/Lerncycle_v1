'use client';

import {useState} from 'react';
import {PdfDropzone} from '@/components/PdfDropzone';
import {NotionIcon} from './NotionIcon';
import {useLanguage} from '@/lib/i18n';

export function DashboardPlaceholder({courseId, courseName, displayName}: {courseId: string; courseName: string; displayName: string}) {
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
