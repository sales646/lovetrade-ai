-- Add columns for weighted expert imitation metrics
ALTER TABLE rl_training_metrics 
ADD COLUMN IF NOT EXISTS l_imitation NUMERIC,
ADD COLUMN IF NOT EXISTS l_rl NUMERIC,
ADD COLUMN IF NOT EXISTS l_total NUMERIC,
ADD COLUMN IF NOT EXISTS alpha_mix NUMERIC,
ADD COLUMN IF NOT EXISTS action_buy_pct NUMERIC,
ADD COLUMN IF NOT EXISTS action_sell_pct NUMERIC,
ADD COLUMN IF NOT EXISTS action_hold_pct NUMERIC,
ADD COLUMN IF NOT EXISTS expert_accuracies JSONB,
ADD COLUMN IF NOT EXISTS sharpe_ratio NUMERIC,
ADD COLUMN IF NOT EXISTS sortino_ratio NUMERIC;

-- Create table for per-expert contributions
CREATE TABLE IF NOT EXISTS expert_contributions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  training_metric_id UUID REFERENCES rl_training_metrics(id),
  expert_name TEXT NOT NULL,
  weight NUMERIC NOT NULL,
  loss_contribution NUMERIC NOT NULL,
  accuracy NUMERIC NOT NULL,
  sample_count INTEGER NOT NULL
);

COMMENT ON TABLE expert_contributions IS 'Tracks per-expert loss and accuracy contributions during training';

-- Enable RLS
ALTER TABLE expert_contributions ENABLE ROW LEVEL SECURITY;

-- Service role full access
CREATE POLICY "Service role full access expert_contributions"
  ON expert_contributions FOR ALL
  USING (auth.role() = 'service_role');

-- Allow read access
CREATE POLICY "Anyone can read expert_contributions"
  ON expert_contributions FOR SELECT
  USING (true);