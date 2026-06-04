import 'server-only';

import {readPositiveInteger} from '@/lib/beta-limits';

export const BETA_FROZEN_ERROR_DE =
  'Die Beta ist vor\u00fcbergehend pausiert. Bitte versuche es sp\u00e4ter erneut.';
export const BETA_FROZEN_ERROR_EN =
  'The Beta is temporarily paused. Please try again later.';
export const BETA_FROZEN_ERROR = `${BETA_FROZEN_ERROR_DE} / ${BETA_FROZEN_ERROR_EN}`;

const PDF_TYPE_ERROR =
  'Nur PDF-Dateien sind erlaubt. / Only PDF files are allowed.';
const PDF_SIZE_ERROR =
  'Die PDF-Datei ist zu gro\u00df. / The PDF file is too large.';
const PDF_PAGES_ERROR =
  'Die PDF-Datei hat zu viele Seiten f\u00fcr die Beta. / Too many pages for the Beta.';
const PDF_UNREADABLE_ERROR =
  'Die PDF-Datei konnte nicht gelesen werden. / The PDF file could not be read.';
const PDF_COUNT_ERROR =
  'Du hast das PDF-Limit f\u00fcr die Beta erreicht. / You have reached the Beta PDF limit.';
const CHAT_DAILY_LIMIT_ERROR =
  'Du hast das t\u00e4gliche Chat-Limit f\u00fcr die Beta erreicht. / You have reached the daily Beta chat limit.';
const RAG_JOBS_DAILY_LIMIT_ERROR =
  'Du hast das t\u00e4gliche Indexierungs-Limit f\u00fcr die Beta erreicht. / You have reached the daily Beta indexing limit.';

type CountClient = {
  from: (table: string) => any;
};

export class BetaGuardError extends Error {
  constructor(message: string, readonly status = 429) {
    super(message);
    this.name = 'BetaGuardError';
  }
}

function sinceLast24Hours(): string {
  return new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString();
}

async function exactCount(
  query: PromiseLike<{count: number | null; error: {message: string} | null}>,
): Promise<number> {
  const {count, error} = await query;
  if (error) {
    throw new Error(error.message);
  }
  return count ?? 0;
}

export function assertBetaNotFrozen(): void {
  if (process.env.BETA_FROZEN === 'true') {
    throw new BetaGuardError(BETA_FROZEN_ERROR, 503);
  }
}

export function assertPdfWithinLimits(file: File): void {
  if (file.type !== 'application/pdf') {
    throw new BetaGuardError(PDF_TYPE_ERROR, 400);
  }

  const maxBytes = readPositiveInteger('BETA_MAX_PDF_BYTES', 25 * 1024 * 1024);
  if (file.size > maxBytes) {
    throw new BetaGuardError(PDF_SIZE_ERROR, 413);
  }
}

export async function assertPdfPagesBelowLimit(file: File): Promise<void> {
  const maxPages = readPositiveInteger('BETA_MAX_PDF_PAGES', 30);
  let numPages: number;
  try {
    const buffer = new Uint8Array(await file.arrayBuffer());
    const pdfjs = await import('pdfjs-dist/legacy/build/pdf.js');
    const doc = await pdfjs.getDocument({
      data: buffer,
      useSystemFonts: false,
      isEvalSupported: false,
    }).promise;
    numPages = doc.numPages;
    await doc.destroy();
  } catch {
    throw new BetaGuardError(PDF_UNREADABLE_ERROR, 400);
  }

  if (numPages > maxPages) {
    throw new BetaGuardError(PDF_PAGES_ERROR, 413);
  }
}

export async function assertUserPdfCountBelowLimit(
  supabase: CountClient,
  userId: string,
): Promise<void> {
  const maxPdfs = readPositiveInteger('BETA_MAX_PDFS_PER_USER', 25);
  const count = await exactCount(
    supabase
      .from('pdfs')
      .select('id', {count: 'exact', head: true})
      .eq('user_id', userId),
  );

  if (count >= maxPdfs) {
    throw new BetaGuardError(PDF_COUNT_ERROR, 429);
  }
}

export async function assertUserChatMessagesUnderDailyLimit(
  supabase: CountClient,
  userId: string,
): Promise<void> {
  const maxMessages = readPositiveInteger('BETA_MAX_CHAT_MESSAGES_PER_DAY', 80);
  const count = await exactCount(
    supabase
      .from('chat_messages')
      .select('id', {count: 'exact', head: true})
      .eq('user_id', userId)
      .eq('role', 'user')
      .gte('created_at', sinceLast24Hours()),
  );

  if (count >= maxMessages) {
    throw new BetaGuardError(CHAT_DAILY_LIMIT_ERROR, 429);
  }
}

export async function assertUserRagJobsUnderDailyLimit(
  supabase: CountClient,
  userId: string,
): Promise<void> {
  const maxJobs = readPositiveInteger('BETA_MAX_RAG_JOBS_PER_DAY', 40);
  const count = await exactCount(
    supabase
      .from('rag_index_jobs')
      .select('id', {count: 'exact', head: true})
      .eq('user_id', userId)
      .gte('created_at', sinceLast24Hours()),
  );

  if (count >= maxJobs) {
    throw new BetaGuardError(RAG_JOBS_DAILY_LIMIT_ERROR, 429);
  }
}
