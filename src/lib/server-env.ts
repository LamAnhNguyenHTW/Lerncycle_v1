export type TopicExtractionMode = 'legacy' | 'learning_graph' | 'disabled';

export type TopicExtractionFlags = {
  legacyTopicExtractionEnabled: boolean;
  learningGraphExtractionEnabled: boolean;
};

export function parseServerBool(value: string | undefined, fallback: boolean) {
  if (value === undefined || value === '') {
    return fallback;
  }
  return !['0', 'false', 'no', 'off'].includes(value.trim().toLowerCase());
}

export function getTopicExtractionFlags(env: NodeJS.ProcessEnv = process.env): TopicExtractionFlags {
  return {
    legacyTopicExtractionEnabled: parseServerBool(env.LEGACY_TOPIC_EXTRACTION_ENABLED, true),
    learningGraphExtractionEnabled: parseServerBool(env.LEARNING_GRAPH_EXTRACTION_ENABLED, false),
  };
}

export function resolveTopicExtractionMode(flags: TopicExtractionFlags): TopicExtractionMode {
  if (flags.legacyTopicExtractionEnabled) {
    return 'legacy';
  }
  if (flags.learningGraphExtractionEnabled) {
    return 'learning_graph';
  }
  return 'disabled';
}
