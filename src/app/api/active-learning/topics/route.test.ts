import {flattenTopicsFromLearningTrees, topicExtractionModeForEnv} from './route';

type FlagCase = {
  name: string;
  env: Record<string, string | undefined>;
  expected: ReturnType<typeof topicExtractionModeForEnv>;
};

const cases: FlagCase[] = [
  {
    name: 'defaults to legacy extraction',
    env: {},
    expected: 'legacy',
  },
  {
    name: 'keeps legacy extraction when both flags are on',
    env: {
      LEGACY_TOPIC_EXTRACTION_ENABLED: 'true',
      LEARNING_GRAPH_EXTRACTION_ENABLED: 'true',
    },
    expected: 'legacy',
  },
  {
    name: 'switches to learning graph when legacy is off and learning graph is on',
    env: {
      LEGACY_TOPIC_EXTRACTION_ENABLED: 'false',
      LEARNING_GRAPH_EXTRACTION_ENABLED: 'true',
    },
    expected: 'learning_graph',
  },
  {
    name: 'disables extraction when both flags are off',
    env: {
      LEGACY_TOPIC_EXTRACTION_ENABLED: 'false',
      LEARNING_GRAPH_EXTRACTION_ENABLED: 'false',
    },
    expected: 'disabled',
  },
];

export function assertTopicExtractionFlagSwitching() {
  for (const testCase of cases) {
    const actual = topicExtractionModeForEnv(testCase.env as NodeJS.ProcessEnv);
    if (actual !== testCase.expected) {
      throw new Error(`${testCase.name}: expected ${testCase.expected}, received ${actual}`);
    }
  }
}

export function assertLearningTreeFlatteningPreservesTopicShape() {
  const topics = flattenTopicsFromLearningTrees([
    {
      id: 'document:pdf-1',
      label: 'Document',
      type: 'document',
      children: [
        {
          id: 'topic-1',
          label: 'Process Mining',
          type: 'topic',
          confidence: 0.9,
          children: [
            {
              id: 'subtopic-1',
              label: 'Event Logs',
              type: 'subtopic',
              confidence: 0.8,
              children: [],
            },
          ],
        },
      ],
    },
  ]);

  if (topics[0]?.name !== 'Process Mining') {
    throw new Error('Expected top learning graph topic to be first.');
  }
  if (typeof topics[0]?.normalizedName !== 'string' || typeof topics[0]?.score !== 'number') {
    throw new Error('Expected legacy topic response shape.');
  }
}

export function assertLearningTreeFlatteningRanksStructuralTopicsFirst() {
  const topics = flattenTopicsFromLearningTrees([
    {
      id: 'document:pdf-1',
      label: 'Document',
      type: 'document',
      children: [
        {
          id: 'detail-1',
          label: 'Abgleich Zwischen Modell Und Realitat',
          type: 'subtopic',
          confidence: 0.95,
          order_index: 45,
          chunk_ids: ['chunk-9'],
          children: [],
        },
        {
          id: 'topic-1',
          label: 'Process Mining',
          type: 'topic',
          confidence: 0.7,
          order_index: 1,
          chunk_ids: ['chunk-1', 'chunk-2'],
          children: [
            {
              id: 'subtopic-1',
              label: 'Event Logs',
              type: 'subtopic',
              confidence: 0.65,
              order_index: 2,
              chunk_ids: ['chunk-3'],
              children: [
                {
                  id: 'concept-1',
                  label: 'XES',
                  type: 'concept',
                  confidence: 0.8,
                  children: [],
                },
              ],
            },
          ],
        },
      ],
    },
  ]);

  if (topics[0]?.name !== 'Process Mining') {
    throw new Error(`Expected structural main topic first, received ${topics[0]?.name}`);
  }
  if (!topics.some((topic) => topic.name === 'Event Logs')) {
    throw new Error('Expected important subtopic to remain in suggestions.');
  }
}

export function assertLearningTreeFlatteningPromotesSpecificConcepts() {
  const topics = flattenTopicsFromLearningTrees([
    {
      id: 'document:pdf-1',
      label: 'Document',
      type: 'document',
      children: [
        {
          id: 'topic-1',
          label: 'Themen',
          type: 'topic',
          confidence: 0.9,
          order_index: 1,
          chunk_ids: ['chunk-1'],
          children: [],
        },
        {
          id: 'topic-2',
          label: 'Process Mining',
          type: 'topic',
          confidence: 0.7,
          order_index: 2,
          chunk_ids: ['chunk-2'],
          children: [
            {
              id: 'concept-1',
              label: 'Case-oriented Process Mining',
              type: 'concept',
              confidence: 0.7,
              order_index: 3,
              chunk_ids: ['chunk-3'],
              children: [],
            },
            {
              id: 'concept-2',
              label: 'Object-oriented Process Mining',
              type: 'concept',
              confidence: 0.7,
              order_index: 4,
              chunk_ids: ['chunk-4'],
              children: [],
            },
          ],
        },
      ],
    },
  ]);

  if (topics.some((topic) => topic.name === 'Themen')) {
    throw new Error('Expected generic agenda heading to be filtered out of the top suggestions.');
  }
  if (!topics.some((topic) => topic.name === 'Case-oriented Process Mining')) {
    throw new Error('Expected specific process-mining concept in suggestions.');
  }
  if (!topics.some((topic) => topic.name === 'Object-oriented Process Mining')) {
    throw new Error('Expected specific process-mining concept in suggestions.');
  }
}
