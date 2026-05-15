import type {Course, PdfFile} from '@/lib/data';

export interface PdfOption {
  id: string;
  name: string;
  folderName: string | null;
}

export function collectCoursePdfs(course: Course): PdfOption[] {
  const fromFolders = course.folders.flatMap((folder) =>
    folder.pdfs.map((pdf) => ({
      id: pdf.id,
      name: pdf.name,
      folderName: folder.name,
    })),
  );
  const loose = course.loose_pdfs.map((pdf: PdfFile) => ({
    id: pdf.id,
    name: pdf.name,
    folderName: null,
  }));
  return [...fromFolders, ...loose];
}
