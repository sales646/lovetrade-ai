-- Add win rate tracking to rl_training_metrics table
ALTER TABLE rl_training_metrics 
ADD COLUMN IF NOT EXISTS win_rate_pct DECIMAL(5,2) DEFAULT 0,
ADD COLUMN IF NOT EXISTS total_trades INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS winning_trades INTEGER DEFAULT 0;