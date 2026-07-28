import 'server-only';

import {createServiceClient} from '@/lib/supabase/service';

export type DeletableSourceType = 'pdf' | 'note' | 'annotation_comment';

export interface SourceDeleteParams {
  userId: string;
  sourceType: DeletableSourceType;
  sourceId: string;
  /** PDF the source belonged to; for sourceType 'pdf' this equals sourceId. */
  deletedPdfId?: string;
}

/**
 * Enqueues a `delete_source` cleanup job after a source row was deleted.
 *
 * The worker (service role) removes Qdrant points, `rag_chunks`, both Neo4j
 * graph layers, `rag_documents`, and `rag_document_primers`. The job is
 * inserted with `pdf_id`/`note_id`/`annotation_id` = NULL so it survives the
 * ON DELETE CASCADE of the source row; the target identity lives in
 * `source_type`/`source_id` and `metadata.deleted_pdf_id`.
 *
 * Uses the service client because the authenticated client only holds insert
 * policies on `rag_index_jobs`; `userId` must always come from the server-side
 * session of the deleting user. Failures must not block the deletion itself —
 * callers log the returned error and continue.
 */
export async function enqueueSourceDeleteJob(
  params: SourceDeleteParams,
): Promise<{error?: string}> {
  try {
    const supabase = createServiceClient();

    // The partial unique index rag_index_jobs_one_active_source_idx allows only
    // one active job per (user_id, source_type, source_id). Leftover index jobs
    // for the deleted source are dead work — remove them before inserting.
    const {error: cancelError} = await supabase
      .from('rag_index_jobs')
      .delete()
      .eq('user_id', params.userId)
      .eq('source_type', params.sourceType)
      .eq('source_id', params.sourceId)
      .in('status', ['pending', 'processing']);

    if (cancelError) {
      return {error: `Could not cancel stale index jobs: ${cancelError.message}`};
    }

    const {error: insertError} = await supabase.from('rag_index_jobs').insert({
      user_id: params.userId,
      source_type: params.sourceType,
      source_id: params.sourceId,
      pdf_id: null,
      note_id: null,
      annotation_id: null,
      job_kind: 'delete_source',
      status: 'pending',
      metadata: params.deletedPdfId ? {deleted_pdf_id: params.deletedPdfId} : {},
    });

    if (insertError) {
      return {error: `Could not enqueue cleanup job: ${insertError.message}`};
    }

    return {};
  } catch (error) {
    return {
      error:
        error instanceof Error
          ? `Could not enqueue cleanup job: ${error.message}`
          : 'Could not enqueue cleanup job.',
    };
  }
}
