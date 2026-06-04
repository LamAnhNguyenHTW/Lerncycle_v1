export type ChatSourceType = 'pdf' | 'note' | 'annotation_comment' | 'chat_memory' | 'knowledge_graph' | 'web' | 'general_knowledge';
export type ChatRole = 'user' | 'assistant';
export type ChatMode = 'normal' | 'guided_learning' | 'feynman';
export type ActiveLearningMode = 'guided_learning' | 'feynman';
export type LearningLanguage = 'de' | 'en';
export type ActiveLearningStep =
  | 'diagnose_prior_knowledge'
  | 'ask_question'
  | 'evaluate_answer'
  | 'give_hint'
  | 'summarize';

export type ExerciseStatus =
  | 'active'
  | 'final_check'
  | 'ready_for_result'
  | 'completed';

export interface ActiveLearningState {
  mode?: ActiveLearningMode;
  topic?: string;
  target_concept?: string;
  difficulty?: 'beginner' | 'intermediate' | 'advanced';
  language?: LearningLanguage;
  learner_name?: string;
  current_step?: ActiveLearningStep;
  covered_concepts?: string[];
  misconceptions?: string[];
  user_understanding_score?: number;
  next_goal?: string;
  exercise_status?: ExerciseStatus;
  completion_readiness?: number;
  remaining_gaps?: string[];
  final_check_question?: string;
  turn_count?: number;
}

export interface ActiveLearningControl {
  finish_requested: boolean;
  should_nudge_completion: boolean;
  generate_final_result: boolean;
}

export interface RecentChatMessage {
  role: ChatRole;
  content: string;
}

export interface ChatRequest {
  message: string;
  mode?: ChatMode;
  topic?: string;
  difficulty?: 'beginner' | 'intermediate' | 'advanced';
  language?: LearningLanguage;
  learner_name?: string;
  source_types?: ChatSourceType[];
  use_rag?: boolean;
  top_k?: number;
  course_id?: string;
  session_id?: string;
  pdf_ids?: string[];
  recent_messages?: RecentChatMessage[];
  enableWebSearch?: boolean;
  stream?: boolean;
  input_metadata?: {
    input_type?: 'voice';
    transcription_model?: string;
    recording_seconds?: number;
  };
}

export interface RagAnswerRequestBody {
  query: string;
  user_id: string;
  source_types: ChatSourceType[];
  top_k: number;
  pdf_ids?: string[];
  recent_messages: RecentChatMessage[];
  session_id: string;
  memory_source_ids: string[];
  memory_mode: 'auto';
  graph_mode: 'auto' | 'off';
  context_summary?: string;
  web_mode: 'on' | 'off';
  use_intent_classifier: boolean;
  use_retrieval_planner: boolean;
  chat_mode: ChatMode;
  active_learning_state: ActiveLearningState;
  active_learning_control?: ActiveLearningControl;
  chat_language?: LearningLanguage;
  document_primer?: string;
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
  active_learning_state?: ActiveLearningState;
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
  input_metadata?: {
    input_type?: 'voice';
    transcription_model?: string;
    recording_seconds?: number;
  } | null;
}

export interface StoredChatSession {
  id: string;
  title: string | null;
  course_id: string | null;
  updated_at: string;
  mode: ChatMode;
  active_learning_state: ActiveLearningState;
  messages: StoredChatMessage[];
}
