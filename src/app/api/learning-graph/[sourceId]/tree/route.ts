import {NextResponse} from 'next/server';
import {createClient} from '@/lib/supabase/server';

export type LearningTreeNode = {
  id: string;
  label: string;
  type: 'document' | 'topic' | 'subtopic' | 'concept' | 'objective';
  summary?: string | null;
  pageStart?: number | null;
  pageEnd?: number | null;
  confidence?: number | null;
  chunkIds: string[];
  children: LearningTreeNode[];
};

type PythonLearningTreeNode = {
  id: string;
  label: string;
  type: LearningTreeNode['type'];
  summary?: string | null;
  page_start?: number | null;
  page_end?: number | null;
  confidence?: number | null;
  chunk_ids?: string[];
  children?: PythonLearningTreeNode[];
};

function errorResponse(message: string, status: number) {
  return NextResponse.json({error: message}, {status});
}

function normalizeTreeNode(node: PythonLearningTreeNode): LearningTreeNode {
  return {
    id: node.id,
    label: node.label,
    type: node.type,
    summary: node.summary,
    pageStart: node.page_start,
    pageEnd: node.page_end,
    confidence: node.confidence,
    chunkIds: node.chunk_ids ?? [],
    children: (node.children ?? []).map(normalizeTreeNode),
  };
}

export async function GET(_request: Request, context: RouteContext<'/api/learning-graph/[sourceId]/tree'>) {
  const supabase = await createClient();
  const {data: {user}} = await supabase.auth.getUser();
  if (!user) {
    return errorResponse('Unauthorized', 401);
  }

  const {sourceId} = await context.params;
  const ragApiUrl = process.env.RAG_API_URL;
  const internalApiKey = process.env.RAG_INTERNAL_API_KEY;
  if (!ragApiUrl || !internalApiKey) {
    return errorResponse('Learning graph is not configured.', 500);
  }

  const url = new URL(`${ragApiUrl.replace(/\/$/, '')}/learning-graph/${sourceId}/tree`);
  url.searchParams.set('user_id', user.id);
  const response = await fetch(url, {
    headers: {Authorization: `Bearer ${internalApiKey}`},
  });
  if (response.status === 404) {
    return errorResponse('Learning graph not found.', 404);
  }
  if (!response.ok) {
    return errorResponse('Learning graph service failed.', 500);
  }

  const tree = await response.json() as PythonLearningTreeNode;
  return NextResponse.json(normalizeTreeNode(tree));
}
