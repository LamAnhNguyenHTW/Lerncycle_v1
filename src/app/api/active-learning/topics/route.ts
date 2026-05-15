import {NextResponse} from 'next/server';
import {createClient} from '@/lib/supabase/server';
import {getTopicExtractionFlags, resolveTopicExtractionMode} from '@/lib/server-env';

const MAX_PDFS = 20;
const MAX_CHUNKS = 120;
const MAX_TOPICS = 5;
const DOCUMENT_ORDER_SCORE_WINDOW = 60;

const STOPWORDS = new Set([
  'about',
  'after',
  'also',
  'and',
  'are',
  'based',
  'can',
  'chapter',
  'course',
  'data',
  'der',
  'die',
  'das',
  'ein',
  'eine',
  'for',
  'from',
  'has',
  'have',
  'ist',
  'mit',
  'not',
  'oder',
  'pdf',
  'process',
  'section',
  'slide',
  'slides',
  'that',
  'the',
  'this',
  'und',
  'von',
  'werden',
  'with',
]);

const LOW_VALUE_TOPIC_PHRASES = new Set([
  'article information',
  'architecture overview',
  'contents',
  'overview',
  'themen',
  'table of contents',
]);

const DOMAIN_TOPIC_TERMS = [
  'case oriented process mining',
  'object oriented process mining',
  'event log',
  'event logs',
  'object centric',
  'process mining',
];

type Topic = {
  name: string;
  normalizedName: string;
  score: number;
  evidence: {
    heading?: string;
    sample?: string;
    pdf_id?: string;
  };
};

type RagChunk = {
  pdf_id: string | null;
  content: string;
  heading_path: string[] | null;
  page_index: number | null;
};

type LearningTreeNode = {
  id: string;
  label: string;
  type: 'document' | 'topic' | 'subtopic' | 'concept' | 'objective';
  confidence?: number | null;
  order_index?: number | null;
  orderIndex?: number | null;
  page_start?: number | null;
  pageStart?: number | null;
  chunk_ids?: string[];
  chunkIds?: string[];
  children?: LearningTreeNode[];
};

export function topicExtractionModeForEnv(env: NodeJS.ProcessEnv = process.env) {
  return resolveTopicExtractionMode(getTopicExtractionFlags(env));
}

function errorResponse(message: string, status: number) {
  return NextResponse.json({error: message}, {status});
}

function normalizeName(value: string) {
  return value
    .normalize('NFKC')
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function displayName(value: string) {
  return value
    .replace(/[^\p{L}\p{N}\-+/ ]+/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .split(' ')
    .filter(Boolean)
    .map((part) => {
      const lower = part.toLowerCase();
      if (part.length <= 4 && part === part.toUpperCase()) return part;
      if (['bpmn', 'sql', 'xes', 'erp', 'crm', 'etl', 'api'].includes(lower)) return lower.toUpperCase();
      return lower.charAt(0).toUpperCase() + lower.slice(1);
    })
    .join(' ');
}

function isUsefulToken(token: string) {
  const normalized = normalizeName(token);
  return normalized.length >= 3 && !STOPWORDS.has(normalized) && !/^\d+$/.test(normalized);
}

function addTopic(topics: Map<string, Topic>, rawName: string, score: number, evidence: Topic['evidence']) {
  const name = displayName(rawName);
  const normalizedName = normalizeName(name);
  if (!name || normalizedName.length < 3) return;

  const words = normalizedName.split(' ');
  if (words.length > 5 || words.every((word) => STOPWORDS.has(word))) return;

  const existing = topics.get(normalizedName);
  if (!existing || score > existing.score) {
    topics.set(normalizedName, {name, normalizedName, score, evidence});
  } else {
    existing.score += Math.min(score, 3);
  }
}

function phraseCandidates(text: string) {
  const normalized = text
    .replace(/([a-zäöüß])([A-ZÄÖÜ])/g, '$1 $2')
    .replace(/[^\p{L}\p{N}\-+/ ]+/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim();
  const tokens = normalized.split(' ').filter(isUsefulToken);
  const candidates: string[] = [];

  for (let size = 3; size >= 1; size -= 1) {
    for (let index = 0; index <= tokens.length - size; index += 1) {
      candidates.push(tokens.slice(index, index + size).join(' '));
    }
  }

  return candidates;
}

function extractTopicsFromChunks(chunks: RagChunk[]) {
  const topics = new Map<string, Topic>();

  chunks.forEach((chunk, chunkIndex) => {
    const heading = (chunk.heading_path ?? []).filter(Boolean).at(-1);
    if (heading) {
      addTopic(topics, heading, 80 - chunkIndex * 0.1, {
        heading,
        pdf_id: chunk.pdf_id ?? undefined,
      });
    }

    const sample = chunk.content.slice(0, 900);
    const capitalizedPhrases = sample.match(
      /\b(?:[A-ZÄÖÜ][\p{L}\p{N}+\-/]{2,}|[A-Z]{2,})(?:\s+(?:[A-ZÄÖÜ][\p{L}\p{N}+\-/]{2,}|[A-Z]{2,})){0,3}/gu,
    ) ?? [];

    capitalizedPhrases.forEach((phrase, index) => {
      addTopic(topics, phrase, 55 - index * 0.5, {
        heading,
        sample: sample.slice(0, 220),
        pdf_id: chunk.pdf_id ?? undefined,
      });
    });

    phraseCandidates(sample).slice(0, 30).forEach((phrase, index) => {
      addTopic(topics, phrase, 25 - index * 0.1, {
        heading,
        sample: sample.slice(0, 220),
        pdf_id: chunk.pdf_id ?? undefined,
      });
    });
  });

  return [...topics.values()]
    .sort((a, b) => b.score - a.score || a.name.localeCompare(b.name))
    .slice(0, MAX_TOPICS);
}

export function flattenTopicsFromLearningTrees(trees: LearningTreeNode[]) {
  const topics = new Map<string, Topic>();

  const visit = (node: LearningTreeNode, depth: number, pdfId?: string) => {
    if (node.type === 'topic' || node.type === 'subtopic' || node.type === 'concept') {
      addTopic(topics, node.label, learningTopicScore(node, depth), {
        heading: node.label,
        pdf_id: pdfId,
      });
    }
    for (const child of node.children ?? []) {
      visit(child, depth + 1, pdfId);
    }
  };

  trees.forEach((tree) => visit(tree, 0));
  return [...topics.values()]
    .sort((a, b) => b.score - a.score || a.name.localeCompare(b.name))
    .slice(0, MAX_TOPICS);
}

function learningTopicScore(node: LearningTreeNode, depth: number) {
  const normalized = normalizeName(node.label);
  const confidenceScore = Math.round((node.confidence ?? 0.5) * 100);
  const orderIndex = node.order_index ?? node.orderIndex;
  const pageStart = node.page_start ?? node.pageStart;
  const chunkCount = (node.chunk_ids ?? node.chunkIds ?? []).length;
  const topicChildren = (node.children ?? []).filter((child) => child.type === 'topic' || child.type === 'subtopic').length;
  const evidenceChildren = (node.children ?? []).filter((child) => child.type === 'concept' || child.type === 'objective').length;
  const orderBonus = typeof orderIndex === 'number'
    ? Math.max(0, DOCUMENT_ORDER_SCORE_WINDOW - orderIndex)
    : typeof pageStart === 'number'
      ? Math.max(0, DOCUMENT_ORDER_SCORE_WINDOW - pageStart)
      : 0;
  const rootBonus = node.type === 'topic' ? 25 : 0;
  const conceptPenalty = node.type === 'concept' ? 10 : 0;
  const structureBonus = topicChildren * 18 + evidenceChildren * 6 + Math.min(chunkCount, 4) * 3;
  const domainBonus = DOMAIN_TOPIC_TERMS.some((term) => normalized.includes(term)) ? 35 : 0;
  const depthPenalty = depth * 12;
  const lowValuePenalty = LOW_VALUE_TOPIC_PHRASES.has(normalized) ? 45 : 0;

  return confidenceScore + rootBonus + structureBonus + orderBonus + domainBonus - depthPenalty - lowValuePenalty - conceptPenalty;
}

async function fetchLearningTree(sourceId: string, userId: string): Promise<LearningTreeNode | null> {
  const ragApiUrl = process.env.RAG_API_URL;
  const internalApiKey = process.env.RAG_INTERNAL_API_KEY;
  if (!ragApiUrl || !internalApiKey) {
    return null;
  }
  const url = new URL(`${ragApiUrl.replace(/\/$/, '')}/learning-graph/${sourceId}/tree`);
  url.searchParams.set('user_id', userId);
  const response = await fetch(url, {
    headers: {Authorization: `Bearer ${internalApiKey}`},
  });
  if (response.status === 404) {
    return null;
  }
  if (!response.ok) {
    throw new Error('Learning graph service failed.');
  }
  return await response.json() as LearningTreeNode;
}

export async function POST(request: Request) {
  const extractionMode = topicExtractionModeForEnv();
  if (extractionMode === 'disabled') {
    return NextResponse.json({topics: []});
  }

  const supabase = await createClient();
  const {data: {user}} = await supabase.auth.getUser();

  if (!user) {
    return errorResponse('Not authenticated.', 401);
  }

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return errorResponse('Invalid JSON body.', 400);
  }

  const {courseId, pdfIds} = body as {courseId?: unknown; pdfIds?: unknown};
  if (typeof courseId !== 'string' || !Array.isArray(pdfIds) || pdfIds.some((id) => typeof id !== 'string')) {
    return errorResponse('courseId and pdfIds are required.', 400);
  }

  const scopedPdfIds = [...new Set(pdfIds)].slice(0, MAX_PDFS);
  if (scopedPdfIds.length === 0) {
    return NextResponse.json({topics: []});
  }

  const {data: ownedPdfs, error: pdfError} = await supabase
    .from('pdfs')
    .select('id')
    .eq('user_id', user.id)
    .eq('course_id', courseId)
    .in('id', scopedPdfIds);

  if (pdfError) {
    return errorResponse(pdfError.message, 500);
  }

  const ownedPdfIds = (ownedPdfs ?? []).map((pdf) => pdf.id as string);
  if (ownedPdfIds.length === 0) {
    return NextResponse.json({topics: []});
  }

  if (extractionMode === 'learning_graph') {
    // Temporary compatibility adapter: new learning features should call /api/learning-graph/[sourceId]/tree directly.
    const trees = (
      await Promise.all(ownedPdfIds.map((pdfId) => fetchLearningTree(pdfId, user.id)))
    ).filter((tree): tree is LearningTreeNode => tree !== null);
    return NextResponse.json({
      topics: flattenTopicsFromLearningTrees(trees).map((topic) => ({
        name: topic.name,
        normalizedName: topic.normalizedName,
        score: topic.score,
        masteryState: 'new',
      })),
    });
  }

  const {data: existingTopics} = await supabase
    .from('learning_topics')
    .select('name, normalized_name, score, mastery_state')
    .eq('user_id', user.id)
    .eq('course_id', courseId)
    .in('pdf_id', ownedPdfIds)
    .order('score', {ascending: false})
    .limit(MAX_TOPICS);

  if ((existingTopics?.length ?? 0) >= MAX_TOPICS) {
    return NextResponse.json({
      topics: existingTopics?.map((topic) => ({
        name: topic.name,
        normalizedName: topic.normalized_name,
        score: topic.score,
        masteryState: topic.mastery_state,
      })) ?? [],
    });
  }

  const {data: chunks, error: chunksError} = await supabase
    .from('rag_chunks')
    .select('pdf_id, content, heading_path, page_index')
    .eq('user_id', user.id)
    .eq('source_type', 'pdf')
    .in('pdf_id', ownedPdfIds)
    .order('page_index', {ascending: true})
    .limit(MAX_CHUNKS);

  if (chunksError) {
    return errorResponse(chunksError.message, 500);
  }

  const extracted = extractTopicsFromChunks((chunks ?? []) as RagChunk[]);
  if (extracted.length > 0) {
    await supabase.from('learning_topics').upsert(
      extracted.map((topic) => ({
        user_id: user.id,
        course_id: courseId,
        pdf_id: topic.evidence.pdf_id,
        name: topic.name,
        normalized_name: topic.normalizedName,
        evidence: topic.evidence,
        score: topic.score,
      })),
      {onConflict: 'user_id,course_id,pdf_id,normalized_name'},
    );
  }

  const {data: topics, error: topicsError} = await supabase
    .from('learning_topics')
    .select('name, normalized_name, score, mastery_state')
    .eq('user_id', user.id)
    .eq('course_id', courseId)
    .in('pdf_id', ownedPdfIds)
    .order('score', {ascending: false})
    .limit(MAX_TOPICS);

  if (topicsError) {
    return errorResponse(topicsError.message, 500);
  }

  return NextResponse.json({
    topics: (topics ?? []).map((topic) => ({
      name: topic.name,
      normalizedName: topic.normalized_name,
      score: topic.score,
      masteryState: topic.mastery_state,
    })),
  });
}
