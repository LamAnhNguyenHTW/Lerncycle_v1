import {Suspense} from 'react';
import {getCourses} from '@/lib/data';
import {getProfile} from '@/actions/profile';
import {LeftSidebar} from '@/components/sidebar/LeftSidebar';
import {DashboardPlaceholder} from '@/components/DashboardPlaceholder';
import {FolderList} from '@/components/FolderList';
import {ResponsiveLayout} from '@/components/ResponsiveLayout';
import {ProfileView} from '@/components/ProfileView';

interface Props {
  searchParams: Promise<{courseId?: string; tab?: string}>;
}

export default async function Page({searchParams}: Props) {
  const {courseId, tab = 'home'} = await searchParams;
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
          profile={profile}
        />
      }
    >
      <main className="flex flex-1 flex-col overflow-y-auto px-4 sm:px-6 md:px-8 w-full position-relative">
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
            {tab === 'notetaking' && <div className="py-8"><h1 className="text-2xl font-bold">Notetaking für {activeCourse.name}</h1><p className="text-muted-foreground mt-2">Hier kommt das Notizen-Tool hin.</p></div>}
            {tab === 'learn' && <div className="py-8"><h1 className="text-2xl font-bold">Learn & Research</h1><p className="text-muted-foreground mt-2">Chatbot und Q&A Interface.</p></div>}
            {tab === 'feynman' && <div className="py-8"><h1 className="text-2xl font-bold">Feynman Technique</h1><p className="text-muted-foreground mt-2">Lehre es einem 5-Jährigen.</p></div>}
            {tab === 'revision' && <div className="py-8"><h1 className="text-2xl font-bold">Revision</h1><p className="text-muted-foreground mt-2">Active Recall und Karteikarten.</p></div>}
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

