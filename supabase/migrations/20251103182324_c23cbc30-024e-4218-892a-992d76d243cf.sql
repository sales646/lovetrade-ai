-- Create pbt_populations table for Population-Based Training tracking
CREATE TABLE public.pbt_populations (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  run_id UUID REFERENCES public.training_runs(id) ON DELETE CASCADE,
  generation INTEGER NOT NULL,
  population_id INTEGER NOT NULL,
  hyperparams JSONB NOT NULL DEFAULT '{}',
  performance NUMERIC NOT NULL DEFAULT 0,
  best_performance NUMERIC NOT NULL DEFAULT 0,
  mean_performance NUMERIC NOT NULL DEFAULT 0,
  performance_variance NUMERIC,
  population_size INTEGER NOT NULL DEFAULT 8,
  exploit_interval INTEGER NOT NULL DEFAULT 5,
  age INTEGER NOT NULL DEFAULT 0,
  is_best BOOLEAN NOT NULL DEFAULT false,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Enable RLS
ALTER TABLE public.pbt_populations ENABLE ROW LEVEL SECURITY;

-- Create policies for read access
CREATE POLICY "Anyone can read pbt_populations"
  ON public.pbt_populations
  FOR SELECT
  USING (true);

-- Service role full access
CREATE POLICY "Service role full access pbt_populations"
  ON public.pbt_populations
  FOR ALL
  USING (auth.role() = 'service_role');

-- Create index for efficient queries
CREATE INDEX idx_pbt_populations_run_id ON public.pbt_populations(run_id);
CREATE INDEX idx_pbt_populations_generation ON public.pbt_populations(generation DESC);

-- Add config column to training_runs for distributed settings
ALTER TABLE public.training_runs 
ADD COLUMN IF NOT EXISTS config JSONB DEFAULT '{}';

COMMENT ON TABLE public.pbt_populations IS 'Tracks Population-Based Training generations and hyperparameter evolution';
COMMENT ON COLUMN public.pbt_populations.generation IS 'PBT generation number';
COMMENT ON COLUMN public.pbt_populations.population_id IS 'Individual ID within the population';
COMMENT ON COLUMN public.pbt_populations.hyperparams IS 'Current hyperparameters for this individual';
COMMENT ON COLUMN public.pbt_populations.performance IS 'Current performance metric (e.g., reward, Sharpe ratio)';
COMMENT ON COLUMN public.pbt_populations.is_best IS 'True if this is the best performer in current generation';