-- Add confidence tracking to rl_training_metrics table
ALTER TABLE rl_training_metrics 
ADD COLUMN IF NOT EXISTS avg_confidence DECIMAL(5,3) DEFAULT 0;