import {getCourses} from '@/lib/data';
import {getProfile} from '@/actions/profile';
import {LeftSidebar} from '@/components/sidebar/LeftSidebar';
import {DashboardPlaceholder} from '@/components/DashboardPlaceholder';
import {FolderList} from '@/components/FolderList';
import {ResponsiveLayout} from '@/components/ResponsiveLayout';
import {ProfileView} from '@/components/ProfileView';
import {StudyInterface} from '@/components/study/StudyInterface';
import {ChatInterface} from '@/components/learn/ChatInterface';
import {ActiveLearningSection} from '@/components/active-learning/ActiveLearningSection';
import {RevisionSection} from '@/components/revision/RevisionSection';

interface Props {
  searchParams: Promise<{courseId?: string; tab?: string; pdfId?: string; sessionId?: string}>;
}

export default async function Page({searchParams}: Props) {
  const {courseId, tab = 'home', pdfId, sessionId} = await searchParams;
  const courses = await getCourses();
  const profile = await getProfile();

  const activeCourse = courseId 
    ? courses.find(c => c.id === courseId) || courses[0] 
    : courses[0];

  return (
    <ResponsiveLayout
      sidebar={
        <LeftSidebar 
          courses={courses} 
          activeCourseId={activeCourse?.id} 
          activeTab={tab} 
          activePdfId={pdfId}
          profile={profile}
        />
      }
    >
      <main className={`flex flex-1 flex-col w-full position-relative ${(tab === 'notetaking' || tab === 'learn' || tab === 'feynman') ? 'overflow-hidden p-0' : 'overflow-y-auto px-4 sm:px-6 md:px-8'}`}>
        <header className="hidden md:flex h-14 shrink-0 items-center justify-end border-b border-transparent px-4">
           {/* Top nav empty for now, maybe profile later */}
        </header>

        {tab === 'profile' && profile ? (
          <ProfileView profile={profile} />
        ) : activeCourse ? (
          <>
            {tab === 'home' && (
              <div className="max-w-4xl mx-auto w-full md:pt-4">
                <DashboardPlaceholder courseId={activeCourse.id} courseName={activeCourse.name} displayName={profile?.display_name ?? 'there'} />
                <FolderList course={activeCourse} />
              </div>
            )}
            {tab === 'notetaking' && activeCourse && (
              <StudyInterface course={activeCourse} initialPdfId={pdfId} />
            )}
            {tab === 'learn' && (
              <>
                {/* RAG chat: src/app/api/chat/route.ts -> RAG service: rag_pipeline/api.py */}
                <ChatInterface course={activeCourse} initialPdfId={pdfId} initialSessionId={sessionId} profile={profile} />
              </>
            )}
            {tab === 'feynman' && <ActiveLearningSection course={activeCourse} initialPdfId={pdfId} initialSessionId={sessionId} profile={profile} />}
            {tab === 'revision' && <RevisionSection course={activeCourse} />}
          </>
        ) : (
          <div className="flex flex-1 items-center justify-center flex-col text-center mt-20">
            <h1 className="text-2xl font-semibold mb-2">Welcome to Learncycle</h1>
            <p className="text-muted-foreground">Create your first course in the sidebar to get started.</p>
          </div>
        )}
      </main>
    </ResponsiveLayout>
  );
}
