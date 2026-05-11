export type ChatSourceType = 'pdf' | 'note' | 'annotation_comment';
export type ChatRole = 'user' | 'assistant';

export interface RecentChatMessage {
  role: ChatRole;
  content: string;
}

export interface ChatRequest {
  message: string;
  source_types?: ChatSourceType[];
  use_rag?: boolean;
  top_k?: number;
  course_id?: string;
  session_id?: string;
  pdf_ids?: string[];
  recent_messages?: RecentChatMessage[];
}

export interface ChatSource {
  chunk_id: string;
  source_type: ChatSourceType;
  source_id: string;
  title: string | null;
  heading: string | null;
  page: number | null;
  score: number | null;
  snippet: string;
  metadata: {
    filename?: string;
  };
}

export interface ChatRetrievalMeta {
  mode: 'hybrid';
  top_k: number;
}

export interface ChatResponse {
  session_id?: string;
  answer: string;
  sources: ChatSource[];
  retrieval: ChatRetrievalMeta;
}

export interface StoredChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources: ChatSource[];
  pdf_ids: string[];
  created_at: string;
}

export interface StoredChatSession {
  id: string;
  title: string | null;
  course_id: string | null;
  updated_at: string;
  messages: StoredChatMessage[];
}
