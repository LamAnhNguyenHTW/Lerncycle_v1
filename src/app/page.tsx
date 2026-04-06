import {Suspense} from 'react';
import {getSemesterTree} from '@/lib/data';
import {SidebarTree} from '@/components/sidebar/SidebarTree';
import {WeekContent} from '@/components/WeekContent';
import {SignOutButton} from '@/components/SignOutButton';
import {Skeleton} from '@/components/ui/skeleton';

interface Props {
  searchParams: Promise<{week?: string}>;
}

export default async function DashboardPage({searchParams}: Props) {
  const {week: weekId} = await searchParams;
  const tree = await getSemesterTree();

  // Resolve the active week name from the tree for display.
  let activeWeekName: string | null = null;
  if (weekId) {
    outer: for (const semester of tree) {
      for (const subject of semester.subjects) {
        for (const week of subject.weeks) {
          if (week.id === weekId) {
            activeWeekName = week.name;
            break outer;
          }
        }
      }
    }
  }

  return (
    <div className="flex h-full flex-col">
      {/* Top bar */}
      <header className="flex h-12 shrink-0 items-center justify-between border-b border-border px-4">
        <div className="flex items-center gap-2">
          <span className="size-2 rounded-full bg-foreground" />
          <span className="text-sm font-semibold tracking-tight">Lerncycle</span>
        </div>
        <SignOutButton />
      </header>

      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <aside className="flex w-60 shrink-0 flex-col gap-4 overflow-y-auto border-r border-border px-3 py-4">
          <p className="px-1 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Library
          </p>
          <SidebarTree tree={tree} />
        </aside>

        {/* Main content */}
        <main className="flex flex-1 flex-col overflow-y-auto px-8 py-8">
          {weekId && activeWeekName ? (
            <Suspense fallback={<WeekContentSkeleton />}>
              <WeekContent weekId={weekId} weekName={activeWeekName} />
            </Suspense>
          ) : (
            <EmptyState hasSemesters={tree.length > 0} />
          )}
        </main>
      </div>
    </div>
  );
}

function EmptyState({hasSemesters}: {hasSemesters: boolean}) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-3 text-center">
      <div className="flex size-12 items-center justify-center rounded-xl bg-muted">
        <BookIcon className="size-6 text-muted-foreground" />
      </div>
      <div>
        <p className="text-sm font-medium">
          {hasSemesters ? 'Select a week to get started' : 'No semesters yet'}
        </p>
        <p className="mt-1 text-xs text-muted-foreground">
          {hasSemesters
            ? 'Click a week in the sidebar to view its PDFs.'
            : 'Add your first semester in the sidebar to begin organizing your studies.'}
        </p>
      </div>
    </div>
  );
}

function WeekContentSkeleton() {
  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-2">
        <Skeleton className="h-6 w-40" />
        <Skeleton className="h-4 w-24" />
      </div>
      <div className="flex flex-col gap-2">
        {[1, 2, 3].map((i) => (
          <Skeleton key={i} className="h-14 w-full rounded-lg" />
        ))}
      </div>
      <Skeleton className="h-36 w-full rounded-lg" />
    </div>
  );
}

function BookIcon({className}: {className?: string}) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden="true"
    >
      <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
      <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
    </svg>
  );
}
