'use client';

import {useState} from 'react';
import {Course} from '@/lib/data';
import {NotionIcon} from '../NotionIcon';
import {createCourse, deleteCourse, updateCourse} from '@/actions/courses';
import {useRouter} from 'next/navigation';

interface Props {
  courses: Course[];
  activeCourseId?: string;
  collapsed?: boolean;
}

export function CourseSwitcher({courses, activeCourseId, collapsed = false}: Props) {
  const router = useRouter();
  const [isOpen, setIsOpen] = useState(false);
  const [isCreating, setIsCreating] = useState(false);
  const [newCourseName, setNewCourseName] = useState('');
  const [editingCourseId, setEditingCourseId] = useState<string | null>(null);
  const [editingCourseName, setEditingCourseName] = useState('');
  const [courseToDelete, setCourseToDelete] = useState<Course | null>(null);

  const activeCourse = courses.find((c) => c.id === activeCourseId) || courses[0];

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newCourseName.trim()) return;
    const {id, error} = await createCourse(newCourseName);
    if (!error && id) {
      setNewCourseName('');
      setIsCreating(false);
      setIsOpen(false);
      router.push(`/?courseId=${id}`);
    }
  };

  const handleStartEdit = (course: Course) => {
    setEditingCourseId(course.id);
    setEditingCourseName(course.name);
  };

  const handleSaveEdit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!editingCourseId) return;
    const trimmed = editingCourseName.trim();
    if (!trimmed) return;

    const {error} = await updateCourse(editingCourseId, trimmed);
    if (!error) {
      setEditingCourseId(null);
      setEditingCourseName('');
      router.refresh();
    }
  };

  const handleDeleteConfirm = async () => {
    if (!courseToDelete) return;
    const {error} = await deleteCourse(courseToDelete.id);
    if (error) return;

    const remainingCourses = courses.filter((c) => c.id !== courseToDelete.id);
    const nextCourse = remainingCourses[0];
    setCourseToDelete(null);
    router.push(nextCourse ? `/?courseId=${nextCourse.id}` : '/');
    router.refresh();
  };

  return (
    <div className="relative w-full">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex w-full items-center justify-between rounded-xl px-3 py-3 text-base font-medium hover:bg-black/5"
        title={activeCourse?.name || 'Select Course'}
      >
        <div className="flex items-center gap-3 min-w-0">
          <div className="flex size-10 shrink-0 items-center justify-center rounded-md bg-black/5 text-foreground">
            <NotionIcon name="ni-folders" className="w-[24px] h-[24px]" />
          </div>
          {!collapsed && <span className="truncate pr-2">{activeCourse?.name || 'Select Course'}</span>}
        </div>
        {!collapsed && (
          <NotionIcon name="ni-chevron-down-circle" className="w-[22px] h-[22px] shrink-0 text-muted-foreground" />
        )}
      </button>

      {isOpen && (
        <>
          <div
            className="fixed inset-0 z-40"
            onClick={() => setIsOpen(false)}
          ></div>
          <div
            className={`absolute top-full z-50 mt-2 rounded-xl border border-border bg-white p-1.5 shadow-lg ${
              collapsed ? 'left-full ml-2 w-72' : 'left-0 w-full'
            }`}
          >
            <div className="px-3 py-2 text-xs uppercase tracking-wider font-semibold text-muted-foreground">
              Private Courses
            </div>
            {courses.map((course) => (
              <div
                key={course.id}
                className={`flex w-full items-center justify-between rounded-lg px-3 py-2.5 text-base ${
                  course.id === activeCourse?.id ? 'bg-black/5' : 'hover:bg-black/5'
                }`}
              >
                {editingCourseId === course.id ? (
                  <form onSubmit={handleSaveEdit} className="flex w-full items-center gap-2">
                    <input
                      autoFocus
                      type="text"
                      className="w-full rounded-md border border-border px-2 py-1.5 text-sm outline-none focus:border-primary"
                      value={editingCourseName}
                      onChange={(e) => setEditingCourseName(e.target.value)}
                    />
                    <button
                      type="submit"
                      className="rounded-md p-1.5 text-muted-foreground hover:bg-black/5 hover:text-foreground"
                      title="Save"
                    >
                      <NotionIcon name="ni-check" className="w-[18px] h-[18px]" />
                    </button>
                    <button
                      type="button"
                      onClick={() => setEditingCourseId(null)}
                      className="rounded-md p-1.5 text-muted-foreground hover:bg-black/5 hover:text-foreground"
                      title="Cancel"
                    >
                      <NotionIcon name="ni-x" className="w-[18px] h-[18px]" />
                    </button>
                  </form>
                ) : (
                  <>
                    <button
                      onClick={() => {
                        setIsOpen(false);
                        router.push(`/?courseId=${course.id}`);
                      }}
                      className="flex min-w-0 flex-1 items-center gap-3 text-left"
                    >
                      <div className="flex size-9 shrink-0 items-center justify-center rounded-md bg-black/5 text-foreground">
                        <NotionIcon name="ni-folders" className="w-[20px] h-[20px]" />
                      </div>
                      <span className="truncate">{course.name}</span>
                    </button>
                    <div className="ml-2 flex items-center gap-1">
                      <button
                        type="button"
                        onClick={() => handleStartEdit(course)}
                        className="rounded-md p-1.5 text-muted-foreground hover:bg-black/5 hover:text-foreground"
                        title="Edit course"
                      >
                        <NotionIcon name="ni-pencil" className="w-[18px] h-[18px]" />
                      </button>
                      <button
                        type="button"
                        onClick={() => setCourseToDelete(course)}
                        className="rounded-md p-1.5 text-muted-foreground hover:bg-red-50 hover:text-red-600"
                        title="Delete course"
                      >
                        <NotionIcon name="ni-x" className="w-[18px] h-[18px]" />
                      </button>
                    </div>
                  </>
                )}
              </div>
            ))}

            <div className="my-1 h-px bg-border"></div>

            {isCreating ? (
              <form onSubmit={handleCreate} className="px-2 py-1.5">
                <input
                  autoFocus
                  type="text"
                  placeholder="Course name..."
                  className="w-full rounded-md border border-border px-3 py-2 text-base outline-none focus:border-primary"
                  value={newCourseName}
                  onChange={(e) => setNewCourseName(e.target.value)}
                />
              </form>
            ) : (
              <button
                onClick={() => setIsCreating(true)}
                className="flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-base text-muted-foreground hover:bg-black/5 hover:text-foreground mt-1"
              >
                <NotionIcon name="ni-plus" className="w-[22px] h-[22px]" />
                Create a Course
              </button>
            )}
          </div>
        </>
      )}

      {courseToDelete && (
        <div className="fixed inset-0 z-[60] flex items-center justify-center p-4">
          <div className="absolute inset-0 bg-black/35" onClick={() => setCourseToDelete(null)} />
          <div className="relative w-full max-w-md rounded-2xl border border-border bg-white p-6 shadow-2xl">
            <h3 className="text-lg font-semibold text-foreground">Delete course?</h3>
            <p className="mt-2 text-sm text-muted-foreground">
              This will remove <span className="font-medium text-foreground">{courseToDelete.name}</span> and all
              folders and PDFs inside it.
            </p>
            <div className="mt-6 flex items-center justify-end gap-2">
              <button
                type="button"
                onClick={() => setCourseToDelete(null)}
                className="rounded-lg border border-border px-4 py-2 text-sm font-medium hover:bg-black/5"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleDeleteConfirm}
                className="rounded-lg bg-red-600 px-4 py-2 text-sm font-medium text-white hover:bg-red-700"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
