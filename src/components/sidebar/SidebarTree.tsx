'use client';

import {useState, useTransition} from 'react';
import {useRouter, useSearchParams} from 'next/navigation';
import {createSemester, deleteSemester} from '@/actions/semesters';
import {createSubject, deleteSubject} from '@/actions/subjects';
import {createWeek, deleteWeek} from '@/actions/weeks';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '@/components/ui/alert-dialog';
import type {SemesterTree} from '@/types';
import {cn} from '@/lib/utils';

interface Props {
  tree: SemesterTree[];
}

/** Interactive sidebar tree for Semester → Subject → Week navigation. */
export function SidebarTree({tree}: Props) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const activeWeekId = searchParams.get('week');

  const [openSemesters, setOpenSemesters] = useState<Set<string>>(new Set());
  const [openSubjects, setOpenSubjects] = useState<Set<string>>(new Set());

  function toggleSemester(id: string) {
    setOpenSemesters((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function toggleSubject(id: string) {
    setOpenSubjects((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function selectWeek(id: string) {
    const params = new URLSearchParams(searchParams.toString());
    params.set('week', id);
    router.push(`/?${params.toString()}`);
  }

  return (
    <nav className="flex flex-col gap-0.5">
      {tree.map((semester) => (
        <div key={semester.id}>
          {/* Semester row */}
          <div className="group flex items-center gap-1 rounded-md pr-1 hover:bg-muted/60">
            <button
              type="button"
              onClick={() => toggleSemester(semester.id)}
              className="flex flex-1 items-center gap-1.5 truncate py-1.5 pl-2 text-left text-sm font-medium"
            >
              <ChevronIcon open={openSemesters.has(semester.id)} />
              <span className="truncate">{semester.name}</span>
            </button>
            <CreateButton
              label="Add subject"
              placeholder="Subject name"
              onSubmit={(name) => createSubject(semester.id, name)}
              onCreated={() => {
                setOpenSemesters((prev) => new Set([...prev, semester.id]));
                router.refresh();
              }}
            />
            <DeleteButton
              label={semester.name}
              onDelete={async () => {
                await deleteSemester(semester.id);
                router.refresh();
              }}
            />
          </div>

          {/* Subjects */}
          {openSemesters.has(semester.id) && (
            <div className="ml-3 flex flex-col gap-0.5 border-l border-border pl-2">
              {semester.subjects.length === 0 && (
                <p className="py-1 text-xs text-muted-foreground">No subjects yet</p>
              )}
              {semester.subjects.map((subject) => (
                <div key={subject.id}>
                  {/* Subject row */}
                  <div className="group flex items-center gap-1 rounded-md pr-1 hover:bg-muted/60">
                    <button
                      type="button"
                      onClick={() => toggleSubject(subject.id)}
                      className="flex flex-1 items-center gap-1.5 truncate py-1 pl-1 text-left text-sm"
                    >
                      <ChevronIcon open={openSubjects.has(subject.id)} />
                      <span className="truncate">{subject.name}</span>
                    </button>
                    <CreateButton
                      label="Add week"
                      placeholder="Week name"
                      onSubmit={(name) => createWeek(subject.id, name)}
                      onCreated={() => {
                        setOpenSubjects((prev) => new Set([...prev, subject.id]));
                        router.refresh();
                      }}
                    />
                    <DeleteButton
                      label={subject.name}
                      onDelete={async () => {
                        await deleteSubject(subject.id);
                        router.refresh();
                      }}
                    />
                  </div>

                  {/* Weeks */}
                  {openSubjects.has(subject.id) && (
                    <div className="ml-3 flex flex-col gap-0.5 border-l border-border pl-2">
                      {subject.weeks.length === 0 && (
                        <p className="py-1 text-xs text-muted-foreground">No weeks yet</p>
                      )}
                      {subject.weeks.map((week) => (
                        <div
                          key={week.id}
                          className="group flex items-center gap-1 rounded-md pr-1 hover:bg-muted/60"
                        >
                          <button
                            type="button"
                            onClick={() => selectWeek(week.id)}
                            className={cn(
                              'flex flex-1 truncate py-1 pl-1 text-left text-sm',
                              activeWeekId === week.id
                                ? 'font-medium text-foreground'
                                : 'text-muted-foreground',
                            )}
                          >
                            <span className="truncate">{week.name}</span>
                          </button>
                          <DeleteButton
                            label={week.name}
                            onDelete={async () => {
                              if (activeWeekId === week.id) {
                                const params = new URLSearchParams(searchParams.toString());
                                params.delete('week');
                                router.push(`/?${params.toString()}`);
                              }
                              await deleteWeek(week.id);
                              router.refresh();
                            }}
                          />
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      ))}

      {/* Add semester */}
      <div className="mt-2">
        <CreateButton
          label="Add semester"
          placeholder="Semester name"
          onSubmit={createSemester}
          onCreated={() => router.refresh()}
          showLabel
        />
      </div>
    </nav>
  );
}

// ─── Sub-components ───────────────────────────────────────────────────────────

interface CreateButtonProps {
  label: string;
  placeholder: string;
  onSubmit: (name: string) => Promise<{error?: string}>;
  onCreated: () => void;
  showLabel?: boolean;
}

function CreateButton({label, placeholder, onSubmit, onCreated, showLabel}: CreateButtonProps) {
  const [open, setOpen] = useState(false);
  const [value, setValue] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [pending, startTransition] = useTransition();

  function handleSubmit() {
    const trimmed = value.trim();
    if (!trimmed) return;

    startTransition(async () => {
      const result = await onSubmit(trimmed);
      if (result?.error) {
        setError(result.error);
      } else {
        setValue('');
        setError(null);
        setOpen(false);
        onCreated();
      }
    });
  }

  if (!open) {
    return (
      <button
        type="button"
        onClick={() => setOpen(true)}
        title={label}
        className={cn(
          'flex items-center gap-1 rounded p-0.5 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground',
          showLabel ? 'px-2 py-1 text-xs' : 'opacity-0 group-hover:opacity-100',
        )}
      >
        <PlusIcon className="size-3.5 shrink-0" />
        {showLabel && <span>{label}</span>}
      </button>
    );
  }

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        handleSubmit();
      }}
      className="flex items-center gap-1 py-0.5"
    >
      <input
        autoFocus
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => e.key === 'Escape' && setOpen(false)}
        placeholder={placeholder}
        className="h-6 flex-1 rounded border border-input bg-background px-2 text-xs outline-none focus:border-ring"
        disabled={pending}
      />
      <button
        type="submit"
        disabled={pending || !value.trim()}
        className="rounded bg-primary px-2 py-0.5 text-xs text-primary-foreground disabled:opacity-50"
      >
        {pending ? '…' : 'Add'}
      </button>
      <button
        type="button"
        onClick={() => {
          setValue('');
          setError(null);
          setOpen(false);
        }}
        className="rounded px-1 py-0.5 text-xs text-muted-foreground hover:text-foreground"
      >
        ✕
      </button>
      {error && <p className="text-xs text-destructive">{error}</p>}
    </form>
  );
}

interface DeleteButtonProps {
  label: string;
  onDelete: () => Promise<void>;
}

function DeleteButton({label, onDelete}: DeleteButtonProps) {
  const [pending, startTransition] = useTransition();

  return (
    <AlertDialog>
      <AlertDialogTrigger
        className="rounded p-0.5 text-muted-foreground opacity-0 transition-opacity hover:bg-muted hover:text-destructive group-hover:opacity-100"
        title={`Delete ${label}`}
      >
        <TrashIcon className="size-3.5" />
      </AlertDialogTrigger>
      <AlertDialogContent size="sm">
        <AlertDialogHeader>
          <AlertDialogTitle>Delete &quot;{label}&quot;?</AlertDialogTitle>
          <AlertDialogDescription>
            This will permanently delete everything inside it. This action cannot be undone.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          <AlertDialogAction
            variant="destructive"
            disabled={pending}
            onClick={() => startTransition(onDelete)}
          >
            {pending ? 'Deleting…' : 'Delete'}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}

function ChevronIcon({open}: {open: boolean}) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={cn('size-3.5 shrink-0 text-muted-foreground transition-transform', open && 'rotate-90')}
      aria-hidden="true"
    >
      <polyline points="9 18 15 12 9 6" />
    </svg>
  );
}

function PlusIcon({className}: {className?: string}) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden="true"
    >
      <line x1="12" x2="12" y1="5" y2="19" />
      <line x1="5" x2="19" y1="12" y2="12" />
    </svg>
  );
}

function TrashIcon({className}: {className?: string}) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden="true"
    >
      <polyline points="3 6 5 6 21 6" />
      <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
      <path d="M10 11v6M14 11v6" />
      <path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2" />
    </svg>
  );
}
