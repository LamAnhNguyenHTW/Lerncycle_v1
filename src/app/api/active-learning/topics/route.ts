import {NextResponse} from 'next/server';
import {createClient} from '@/lib/supabase/server';

const MAX_PDFS = 20;
const MAX_CHUNKS = 120;
const MAX_TOPICS = 5;

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

export async function POST(request: Request) {
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
