-- Add return percentage tracking to rl_training_metrics table
ALTER TABLE rl_training_metrics 
ADD COLUMN IF NOT EXISTS avg_return_pct DECIMAL(8,3) DEFAULT 0;