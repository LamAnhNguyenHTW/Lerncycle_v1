ALTER TABLE chat_sessions
  ADD COLUMN IF NOT EXISTS mode TEXT NOT NULL DEFAULT 'normal',
  ADD COLUMN IF NOT EXISTS active_learning_state JSONB NOT NULL DEFAULT '{}'::jsonb;

DO $$ BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'chat_sessions_mode_check'
  ) THEN
    ALTER TABLE chat_sessions
      ADD CONSTRAINT chat_sessions_mode_check
      CHECK (mode IN ('normal', 'guided_learning', 'feynman'));
  END IF;
END $$;
