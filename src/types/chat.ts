export type ChatSourceType = 'pdf' | 'note' | 'annotation_comment' | 'chat_memory' | 'knowledge_graph' | 'web' | 'general_knowledge';
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
  enableWebSearch?: boolean;
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
    session_id?: string;
    memory_kind?: string;
    backing_chunk_ids?: string[];
    node_names?: string[];
    relationship_count?: number;
    url?: string;
    provider?: string;
    published_date?: string;
    retrieved_at?: string;
    rank?: number;
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
  intent?: {
    classifier_used?: boolean;
    fallback_used?: boolean;
    question_type?: string;
    needs_pdf?: boolean;
    needs_notes?: boolean;
    needs_annotations?: boolean;
    needs_chat_memory?: boolean;
    needs_graph?: boolean;
    needs_web?: boolean;
    confidence?: number;
    reasoning_summary?: string;
  } | null;
  retrieval_plan?: {
    planner_used?: boolean;
    fallback_used?: boolean;
    graph_available?: boolean;
    error_type?: string;
    steps?: Array<{
      tool: string;
      status: string;
      top_k?: number;
      reason?: string | null;
      result_count?: number;
      error_type?: string | null;
    }>;
  } | null;
  agentic_retriever?: {
    enabled?: boolean;
    used?: boolean;
    quality_mode?: string;
    refinement_mode?: string;
    refinement_used?: boolean;
    refinement_rounds?: number;
    tool_call_count?: number;
    quality?: {
      status?: string | null;
      missing_aspects?: string[];
    };
    fallback_used?: boolean;
    error_type?: string | null;
  } | null;
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
