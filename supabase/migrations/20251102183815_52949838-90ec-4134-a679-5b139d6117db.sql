-- Add dollar P&L and account equity tracking to rl_training_metrics table
ALTER TABLE rl_training_metrics 
ADD COLUMN IF NOT EXISTS avg_dollar_pnl DECIMAL(12,2) DEFAULT 0,
ADD COLUMN IF NOT EXISTS account_equity DECIMAL(12,2) DEFAULT 100000;