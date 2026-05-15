'use client';

import { type Course, type PdfFile, type Folder } from '@/lib/data';
import { NotionIcon } from '@/components/NotionIcon';
import {useLanguage} from '@/lib/i18n';

interface PdfPickerProps {
  course: Course;
  onSelect: (pdf: PdfFile) => void;
}

export function PdfPicker({ course, onSelect }: PdfPickerProps) {
  const {t} = useLanguage();
  const totalPdfs =
    course.loose_pdfs.length +
    course.folders.reduce((acc, f) => acc + f.pdfs.length, 0);

  return (
    <div className="pdf-picker">
      <div className="pdf-picker-header">
        <NotionIcon name="ni-book-open" className="w-[32px] h-[32px] mb-3 opacity-50" />
        <h2 className="pdf-picker-title">{t('study.choosePdf')}</h2>
        <p className="pdf-picker-subtitle">
          {totalPdfs === 0
            ? t('study.noPdfs')
            : t('study.available', {count: String(totalPdfs), plural: totalPdfs === 1 ? '' : 's', course: course.name})}
        </p>
      </div>

      {totalPdfs > 0 && (
        <div className="pdf-picker-list">
          {/* Direct uploads */}
          {course.loose_pdfs.length > 0 && (
            <PdfGroup label={t('study.unfiled')} pdfs={course.loose_pdfs} onSelect={onSelect} />
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
