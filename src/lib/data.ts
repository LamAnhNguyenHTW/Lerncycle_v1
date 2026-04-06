import {createClient} from '@/lib/supabase/server';

export interface PdfFile {
  id: string;
  name: string;
  storage_path: string;
  size_bytes: number;
  created_at: string;
}

export interface Folder {
  id: string;
  name: string;
  created_at: string;
  pdfs: PdfFile[];
}

export interface Course {
  id: string;
  name: string;
  created_at: string;
  /** PDFs uploaded directly to the course (not inside any folder). */
  loose_pdfs: PdfFile[];
  folders: Folder[];
}

export async function getCourses(): Promise<Course[]> {
  const supabase = await createClient();
  const {data: {user}} = await supabase.auth.getUser();

  if (!user) {
    return [];
  }

  const [coursesResult, loosePdfsResult] = await Promise.all([
    supabase
      .from('courses')
      .select(`
        id,
        name,
        created_at,
        folders (
          id,
          name,
          created_at,
          pdfs (
            id,
            name,
            storage_path,
            size_bytes,
            created_at
          )
        )
      `)
      .eq('user_id', user.id)
      .order('created_at', {ascending: true}),
    supabase
      .from('pdfs')
      .select('id, name, storage_path, size_bytes, created_at, course_id')
      .eq('user_id', user.id)
      .is('folder_id', null)
      .order('created_at', {ascending: true}),
  ]);

  if (!coursesResult.data) {
    return [];
  }

  const loosePdfs = loosePdfsResult.data ?? [];

  return coursesResult.data.map((course) => ({
    ...course,
    loose_pdfs: loosePdfs
      .filter((p) => p.course_id === course.id)
      .map(({course_id: _cid, ...p}) => p),
    folders: (course.folders ?? []) as Folder[],
  })) as Course[];
}
