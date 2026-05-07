'use client';

import { type Course, type PdfFile, type Folder } from '@/lib/data';
import { NotionIcon } from '@/components/NotionIcon';

interface PdfPickerProps {
  course: Course;
  onSelect: (pdf: PdfFile) => void;
}

export function PdfPicker({ course, onSelect }: PdfPickerProps) {
  const totalPdfs =
    course.loose_pdfs.length +
    course.folders.reduce((acc, f) => acc + f.pdfs.length, 0);

  return (
    <div className="pdf-picker">
      <div className="pdf-picker-header">
        <NotionIcon name="ni-book-open" className="w-[32px] h-[32px] mb-3 opacity-50" />
        <h2 className="pdf-picker-title">Choose a PDF to study</h2>
        <p className="pdf-picker-subtitle">
          {totalPdfs === 0
            ? 'No PDFs uploaded yet. Go to the Home tab to upload your first file.'
            : `${totalPdfs} PDF${totalPdfs === 1 ? '' : 's'} available in ${course.name}`}
        </p>
      </div>

      {totalPdfs > 0 && (
        <div className="pdf-picker-list">
          {/* Direct uploads */}
          {course.loose_pdfs.length > 0 && (
            <PdfGroup label="Unfiled" pdfs={course.loose_pdfs} onSelect={onSelect} />
          )}

          {/* Folders */}
          {course.folders.map((folder: Folder) =>
            folder.pdfs.length > 0 ? (
              <PdfGroup
                key={folder.id}
                label={folder.name}
                pdfs={folder.pdfs}
                onSelect={onSelect}
              />
            ) : null,
          )}
        </div>
      )}
    </div>
  );
}

function PdfGroup({
  label,
  pdfs,
  onSelect,
}: {
  label: string;
  pdfs: PdfFile[];
  onSelect: (pdf: PdfFile) => void;
}) {
  return (
    <div className="pdf-picker-group">
      <div className="pdf-picker-group-label">{label}</div>
      {pdfs.map((pdf) => (
        <button
          key={pdf.id}
          className="pdf-picker-item"
          onClick={() => onSelect(pdf)}
        >
          <NotionIcon name="ni-file-text" className="w-[20px] h-[20px] flex-shrink-0" />
          <span className="pdf-picker-item-name">{pdf.name}</span>
          <span className="pdf-picker-item-size">
            {(pdf.size_bytes / 1024 / 1024).toFixed(1)} MB
          </span>
        </button>
      ))}
    </div>
  );
}
