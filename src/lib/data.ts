import {createClient} from '@/lib/supabase/server';
import type {Pdf, SemesterTree} from '@/types';

/**
 * Fetches the full semester → subject → week tree for the current user.
 * Returns an empty array when no user is authenticated.
 */
export async function getSemesterTree(): Promise<SemesterTree[]> {
  const supabase = await createClient();
  const {data: {user}} = await supabase.auth.getUser();

  if (!user) return [];

  const {data: semesters} = await supabase
    .from('semesters')
    .select('*')
    .order('created_at', {ascending: true});

  if (!semesters || semesters.length === 0) return [];

  const semesterIds = semesters.map((s) => s.id);

  const {data: subjects} = await supabase
    .from('subjects')
    .select('*')
    .in('semester_id', semesterIds)
    .order('created_at', {ascending: true});

  const subjectIds = (subjects ?? []).map((s) => s.id);

  const {data: weeks} =
    subjectIds.length > 0
      ? await supabase
          .from('weeks')
          .select('*')
          .in('subject_id', subjectIds)
          .order('created_at', {ascending: true})
      : {data: []};

  return semesters.map((semester) => ({
    ...semester,
    subjects: (subjects ?? [])
      .filter((s) => s.semester_id === semester.id)
      .map((subject) => ({
        ...subject,
        weeks: (weeks ?? []).filter((w) => w.subject_id === subject.id),
      })),
  }));
}

/** Fetches all PDFs for a given week. Returns empty array if not authenticated. */
export async function getPdfsForWeek(weekId: string): Promise<Pdf[]> {
  const supabase = await createClient();
  const {data: {user}} = await supabase.auth.getUser();

  if (!user) return [];

  const {data} = await supabase
    .from('pdfs')
    .select('*')
    .eq('week_id', weekId)
    .order('created_at', {ascending: true});

  return data ?? [];
}
