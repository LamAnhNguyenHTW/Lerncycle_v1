'use client';

import {useState} from 'react';
import {PdfDropzone} from '@/components/PdfDropzone';
import {NotionIcon} from './NotionIcon';

export function DashboardPlaceholder({courseId, courseName, displayName}: {courseId: string; courseName: string; displayName: string}) {
  const [showCourseUpload, setShowCourseUpload] = useState(false);

  return (
    <div className="flex flex-col items-center max-w-5xl mx-auto pt-8 md:pt-20 w-full gap-8 md:gap-14">
      <div className="text-center space-y-3 md:space-y-5">
        <h1 className="text-3xl sm:text-4xl font-semibold tracking-tight">
          Hello {displayName}, what do you want to master in {courseName}?
        </h1>
        <p className="text-muted-foreground text-base">
          Upload everything and get interactive notes, flashcards, quizzes, and more
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 md:gap-8 w-full max-w-4xl mb-10">
        <button 
          onClick={() => setShowCourseUpload(!showCourseUpload)}
          className={`flex flex-col items-start p-6 md:p-8 rounded-2xl border transition-all text-left shadow-sm hover:shadow-md hover:border-black/20 ${showCourseUpload ? 'border-primary ring-1 ring-primary' : 'border-border bg-white'}`}
        >
          <div className="mb-4 text-foreground"><NotionIcon name="ni-file-upload" className="w-[32px] h-[32px]" /></div>
          <h3 className="font-medium text-base mb-1">Upload</h3>
          <p className="text-sm text-muted-foreground">Add materials directly to the course</p>
        </button>

        <ActionCard
          icon={<NotionIcon name="ni-link" className="w-[32px] h-[32px]" />}
          title="Insert"
          subtitle="YouTube, Website, Text"
          comingSoon
        />
        <ActionCard
          icon={<NotionIcon name="ni-microphone" className="w-[32px] h-[32px]" />}
          title="Record"
          subtitle="Record live lecture"
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
    <button className={`flex flex-col items-start p-6 md:p-8 rounded-2xl border transition-all text-left relative overflow-hidden ${comingSoon ? 'border-border/50 bg-gray-50/50 cursor-not-allowed opacity-70 hover:opacity-100 group' : 'border-border bg-white shadow-sm hover:shadow-md hover:border-black/10'}`}>
      <div className={`mb-4 ${comingSoon ? 'text-muted-foreground/60' : 'text-muted-foreground'}`}>{icon}</div>
      <h3 className={`font-medium text-base mb-1 ${comingSoon ? 'text-muted-foreground' : ''}`}>{title}</h3>
      <p className="text-sm text-muted-foreground">{subtitle}</p>
      
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
