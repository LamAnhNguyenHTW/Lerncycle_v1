export interface BetaLimits {
  frozen: boolean;
  maxPdfBytes: number;
  maxPdfMegabytes: number;
  maxPdfPages: number;
  maxPdfsPerUser: number;
  maxChatMessagesPerDay: number;
  maxRagJobsPerDay: number;
}

export function readPositiveInteger(name: string, fallback: number): number {
  const raw = process.env[name];
  if (!raw) return fallback;

  const parsed = Number.parseInt(raw, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

export function getBetaLimits(): BetaLimits {
  const maxPdfBytes = readPositiveInteger('BETA_MAX_PDF_BYTES', 25 * 1024 * 1024);

  return {
    frozen: process.env.BETA_FROZEN === 'true',
    maxPdfBytes,
    maxPdfMegabytes: Math.round(maxPdfBytes / 1024 / 1024),
    maxPdfPages: readPositiveInteger('BETA_MAX_PDF_PAGES', 30),
    maxPdfsPerUser: readPositiveInteger('BETA_MAX_PDFS_PER_USER', 25),
    maxChatMessagesPerDay: readPositiveInteger(
      'BETA_MAX_CHAT_MESSAGES_PER_DAY',
      80,
    ),
    maxRagJobsPerDay: readPositiveInteger('BETA_MAX_RAG_JOBS_PER_DAY', 40),
  };
}
