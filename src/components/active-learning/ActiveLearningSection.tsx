'use client';

import {useState} from 'react';
import {ChatInterface} from '@/components/learn/ChatInterface';
import type {ChatMode} from '@/types/chat';
import type {Course} from '@/lib/data';

type ActiveMode = Extract<ChatMode, 'guided_learning' | 'feynman'>;
type Difficulty = '' | 'beginner' | 'intermediate' | 'advanced';

export function ActiveLearningSection({
  course,
  initialPdfId,
  initialSessionId,
  profile,
}: {
  course: Course;
  initialPdfId?: string;
  initialSessionId?: string;
  profile?: {
    display_name: string | null;
    avatar_name: string | null;
    avatar_url: string | null;
  } | null;
}) {
  const [mode, setMode] = useState<ActiveMode>('guided_learning');
  const [topic, setTopic] = useState('');
  const [difficulty, setDifficulty] = useState<Difficulty>('');

  return (
    <ChatInterface
      course={course}
      initialPdfId={initialPdfId}
      initialSessionId={initialSessionId}
      chatMode={mode}
      showActiveLearningModes
      onChatModeChange={setMode}
      activeLearningTopic={topic}
      activeLearningDifficulty={difficulty}
      onActiveLearningTopicChange={setTopic}
      onActiveLearningDifficultyChange={setDifficulty}
      profile={profile}
    />
  );
}
