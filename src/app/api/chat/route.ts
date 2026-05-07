import {NextResponse} from 'next/server';
import {createClient} from '@/lib/supabase/server';
import type {ChatRequest, ChatResponse, ChatSourceType} from '@/types/chat';

const SOURCE_TYPES: ChatSourceType[] = ['pdf', 'note', 'annotation_comment'];

function errorResponse(message: string, status: number) {
  return NextResponse.json({error: message}, {status});
}

function validateBody(body: Partial<ChatRequest>) {
  if (typeof body.message !== 'string' || body.message.trim().length === 0) {
    return 'Message must be a non-empty string.';
  }
  if (body.message.length > 2000) {
    return 'Message must be at most 2000 characters.';
  }
  if (body.top_k !== undefined && (!Number.isInteger(body.top_k) || body.top_k < 1 || body.top_k > 20)) {
    return 'top_k must be an integer between 1 and 20.';
  }
  if (
    body.source_types !== undefined &&
    (!Array.isArray(body.source_types) || body.source_types.some((sourceType) => !SOURCE_TYPES.includes(sourceType)))
  ) {
    return 'source_types contains an unsupported value.';
  }
  if (body.pdf_ids !== undefined && (!Array.isArray(body.pdf_ids) || body.pdf_ids.some((pdfId) => typeof pdfId !== 'string'))) {
    return 'pdf_ids must be an array of strings.';
  }
  return null;
}

async function getOrCreateSession(
  supabase: Awaited<ReturnType<typeof createClient>>,
  userId: string,
  body: Partial<ChatRequest>,
) {
  if (body.session_id) {
    const {data} = await supabase
      .from('chat_sessions')
      .select('id')
      .eq('id', body.session_id)
      .eq('user_id', userId)
      .maybeSingle();
    if (data?.id) {
      return data.id as string;
    }
  }

  const {data, error} = await supabase
    .from('chat_sessions')
    .insert({
      user_id: userId,
      course_id: body.course_id ?? null,
      title: body.message?.trim().slice(0, 80) ?? null,
    })
    .select('id')
    .single();
  if (error || !data?.id) {
    throw new Error('Failed to create chat session.');
  }
  return data.id as string;
}

export async function POST(request: Request) {
  const supabase = await createClient();
  const {data: {user}} = await supabase.auth.getUser();
  if (!user) {
    return errorResponse('Unauthorized', 401);
  }

  let body: Partial<ChatRequest>;
  try {
    body = await request.json();
  } catch {
    return errorResponse('Invalid JSON body.', 400);
  }

  const validationError = validateBody(body);
  if (validationError) {
    return errorResponse(validationError, 400);
  }
  if (body.use_rag !== true) {
    return errorResponse('Non-RAG chat is not implemented.', 400);
  }

  const ragApiUrl = process.env.RAG_API_URL;
  const internalApiKey = process.env.RAG_INTERNAL_API_KEY;
  if (!ragApiUrl || !internalApiKey) {
    console.error('RAG chat env vars missing.');
    return errorResponse('RAG chat is not configured.', 500);
  }

  const topK = body.top_k ?? 8;
  const sourceTypes = body.source_types ?? SOURCE_TYPES;
  const pdfIds = body.pdf_ids?.filter(Boolean) ?? [];
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 60000);

  try {
    const sessionId = await getOrCreateSession(supabase, user.id, body);
    await supabase.from('chat_messages').insert({
      session_id: sessionId,
      user_id: user.id,
      role: 'user',
      content: body.message!.trim(),
      pdf_ids: pdfIds,
    });

    const response = await fetch(`${ragApiUrl.replace(/\/$/, '')}/rag/answer`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${internalApiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query: body.message!.trim(),
        user_id: user.id,
        source_types: sourceTypes,
        top_k: topK,
        pdf_ids: pdfIds.length > 0 ? pdfIds : undefined,
      }),
      signal: controller.signal,
    });

    if (!response.ok) {
      console.error('RAG service error', {status: response.status, body: await response.text()});
      return errorResponse('RAG service failed to answer.', 500);
    }

    const ragResponse = await response.json();
    await supabase.from('chat_messages').insert({
      session_id: sessionId,
      user_id: user.id,
      role: 'assistant',
      content: ragResponse.answer,
      sources: ragResponse.sources ?? [],
      pdf_ids: pdfIds,
    });
    await supabase
      .from('chat_sessions')
      .update({updated_at: new Date().toISOString()})
      .eq('id', sessionId)
      .eq('user_id', user.id);

    const chatResponse: ChatResponse = {
      session_id: sessionId,
      answer: ragResponse.answer,
      sources: ragResponse.sources ?? [],
      retrieval: {mode: 'hybrid', top_k: topK},
    };
    return NextResponse.json(chatResponse);
  } catch (error) {
    console.error('RAG service request failed', error);
    return errorResponse('RAG service failed to answer.', 500);
  } finally {
    clearTimeout(timeout);
  }
}

export async function GET(request: Request) {
  const supabase = await createClient();
  const {data: {user}} = await supabase.auth.getUser();
  if (!user) {
    return errorResponse('Unauthorized', 401);
  }

  const {searchParams} = new URL(request.url);
  const courseId = searchParams.get('courseId');
  let sessionQuery = supabase
    .from('chat_sessions')
    .select('id, title, course_id, updated_at')
    .eq('user_id', user.id)
    .order('updated_at', {ascending: false})
    .limit(20);
  if (courseId) {
    sessionQuery = sessionQuery.eq('course_id', courseId);
  }

  const {data: sessions, error: sessionsError} = await sessionQuery;
  if (sessionsError) {
    console.error('Failed to load chat sessions', sessionsError);
    return errorResponse('Failed to load chat sessions.', 500);
  }

  const sessionIds = (sessions ?? []).map((session) => session.id);
  if (sessionIds.length === 0) {
    return NextResponse.json({sessions: []});
  }

  const {data: messages, error: messagesError} = await supabase
    .from('chat_messages')
    .select('id, session_id, role, content, sources, pdf_ids, created_at')
    .eq('user_id', user.id)
    .in('session_id', sessionIds)
    .order('created_at', {ascending: true});
  if (messagesError) {
    console.error('Failed to load chat messages', messagesError);
    return errorResponse('Failed to load chat messages.', 500);
  }

  return NextResponse.json({
    sessions: (sessions ?? []).map((session) => ({
      ...session,
      messages: (messages ?? []).filter((message) => message.session_id === session.id),
    })),
  });
}
