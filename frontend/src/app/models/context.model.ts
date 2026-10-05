export interface CompressionMetrics {
  original_words: number;
  compressed_words: number;
  original_tokens_estimate: number;
  compressed_tokens_estimate: number;
  compression_percentage: number;
  original_characters?: number;
  compressed_characters?: number;
}

export interface AnalyzeRequest {
  source_type: string;
  content: string;
}

export interface AnalyzeResponse {
  conversation_type: string;
  primary_goal: string;
  secondary_goals: string[];
  important_facts: string[];
  user_preferences: string[];
  constraints: string[];
  decisions: string[];
  rejected_ideas: string[];
  assumptions: string[];
  important_entities: string[];
  completed_items: string[];
  current_state: string[];
  open_questions: string[];
  recent_context: string[];
  last_user_intent: string;
  portable_context: string;
  metrics: CompressionMetrics;
}
