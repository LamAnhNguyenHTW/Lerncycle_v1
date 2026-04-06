export interface Semester {
  id: string;
  user_id: string;
  name: string;
  created_at: string;
}

export interface Subject {
  id: string;
  user_id: string;
  semester_id: string;
  name: string;
  created_at: string;
}

export interface Week {
  id: string;
  user_id: string;
  subject_id: string;
  name: string;
  created_at: string;
}

export interface Pdf {
  id: string;
  user_id: string;
  week_id: string;
  name: string;
  storage_path: string;
  size_bytes: number;
  created_at: string;
}

/** Full tree structure for the sidebar. */
export interface SemesterTree extends Semester {
  subjects: SubjectTree[];
}

export interface SubjectTree extends Subject {
  weeks: Week[];
}
