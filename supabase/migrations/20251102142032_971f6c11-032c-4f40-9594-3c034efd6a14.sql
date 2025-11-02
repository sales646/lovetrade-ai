-- Create tables for autonomous RL training

-- Q-Learning state storage
CREATE TABLE IF NOT EXISTS rl_q_state (
  id INTEGER PRIMARY KEY DEFAULT 1,
  q_table JSONB NOT NULL DEFAULT '{}'::jsonb,
  epsilon NUMERIC NOT NULL DEFAULT 0.3,
  alpha NUMERIC NOT NULL DEFAULT 0.01,
  gamma NUMERIC NOT NULL DEFAULT 0.99,
  episode_count INTEGER NOT NULL DEFAULT 0,
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  CONSTRAINT single_q_state CHECK (id = 1)
);

COMMENT ON TABLE rl_q_state IS 'Stores Q-learning state including Q-table and hyperparameters';
COMMENT ON COLUMN rl_q_state.q_table IS 'Q-value table mapping states to action values [sell, hold, buy]';
COMMENT ON COLUMN rl_q_state.epsilon IS 'Exploration rate for epsilon-greedy policy';
COMMENT ON COLUMN rl_q_state.alpha IS 'Learning rate for Q-value updates';
COMMENT ON COLUMN rl_q_state.gamma IS 'Discount factor for future rewards';
COMMENT ON COLUMN rl_q_state.episode_count IS 'Total number of training episodes completed';

-- RL training metrics
CREATE TABLE IF NOT EXISTS rl_training_metrics (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  episodes INTEGER NOT NULL,
  avg_reward NUMERIC NOT NULL,
  avg_steps NUMERIC NOT NULL,
  epsilon NUMERIC NOT NULL,
  total_episodes INTEGER NOT NULL,
  q_table_size INTEGER NOT NULL,
  duration_seconds NUMERIC NOT NULL
);

COMMENT ON TABLE rl_training_metrics IS 'Tracks RL training performance over time';

-- Enable RLS
ALTER TABLE rl_q_state ENABLE ROW LEVEL SECURITY;
ALTER TABLE rl_training_metrics ENABLE ROW LEVEL SECURITY;

-- Service role full access
CREATE POLICY "Service role full access rl_q_state"
  ON rl_q_state FOR ALL
  USING (auth.role() = 'service_role');

CREATE POLICY "Service role full access rl_training_metrics"
  ON rl_training_metrics FOR ALL
  USING (auth.role() = 'service_role');

-- Allow read access for monitoring
CREATE POLICY "Anyone can read rl_training_metrics"
  ON rl_training_metrics FOR SELECT
  USING (true);