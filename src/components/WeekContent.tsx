import {getPdfsForWeek} from '@/lib/data';
import {PdfDropzone} from '@/components/PdfDropzone';
import {PdfList} from '@/components/PdfList';

interface Props {
  weekId: string;
  weekName: string;
}

/** Server Component: renders the PDF list + upload zone for a selected week. */
export async function WeekContent({weekId, weekName}: Props) {
  const pdfs = await getPdfsForWeek(weekId);

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h2 className="text-lg font-semibold tracking-tight">{weekName}</h2>
        <p className="mt-0.5 text-sm text-muted-foreground">
          {pdfs.length === 0 ? 'No PDFs yet' : `${pdfs.length} PDF${pdfs.length === 1 ? '' : 's'}`}
        </p>
      </div>

      <PdfList pdfs={pdfs} weekId={weekId} />

      <div>
        <h3 className="mb-3 text-sm font-medium text-muted-foreground uppercase tracking-wider">
          Upload
        </h3>
        <PdfDropzone weekId={weekId} />
      </div>
    </div>
  );
}
