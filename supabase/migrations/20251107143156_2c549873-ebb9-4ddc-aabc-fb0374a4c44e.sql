-- Create storage bucket for trained models
INSERT INTO storage.buckets (id, name, public)
VALUES ('trained-models', 'trained-models', false);

-- RLS policies for trained models bucket
CREATE POLICY "Service role can upload models"
ON storage.objects FOR INSERT
WITH CHECK (bucket_id = 'trained-models' AND auth.role() = 'service_role');

CREATE POLICY "Service role can read models"
ON storage.objects FOR SELECT
USING (bucket_id = 'trained-models' AND auth.role() = 'service_role');

CREATE POLICY "Service role can update models"
ON storage.objects FOR UPDATE
USING (bucket_id = 'trained-models' AND auth.role() = 'service_role');

CREATE POLICY "Service role can delete models"
ON storage.objects FOR DELETE
USING (bucket_id = 'trained-models' AND auth.role() = 'service_role');

-- Create table for model metadata
CREATE TABLE public.trained_models (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  run_id UUID REFERENCES public.training_runs(id),
  model_name TEXT NOT NULL,
  model_type TEXT NOT NULL, -- 'transformer', 'ppo', 'ensemble'
  storage_path TEXT NOT NULL, -- path in storage bucket
  file_size_bytes BIGINT,
  performance_metrics JSONB DEFAULT '{}',
  hyperparameters JSONB DEFAULT '{}',
  trained_on_symbols TEXT[] DEFAULT '{}',
  training_duration_seconds INTEGER,
  final_sharpe_ratio NUMERIC,
  final_win_rate NUMERIC,
  is_best BOOLEAN DEFAULT false,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

-- Enable RLS
ALTER TABLE public.trained_models ENABLE ROW LEVEL SECURITY;

-- RLS policies
CREATE POLICY "Anyone can read trained models"
ON public.trained_models FOR SELECT
USING (true);

CREATE POLICY "Service role can insert trained models"
ON public.trained_models FOR INSERT
WITH CHECK (auth.role() = 'service_role');

CREATE POLICY "Service role can update trained models"
ON public.trained_models FOR UPDATE
USING (auth.role() = 'service_role');

-- Create index for faster queries
CREATE INDEX idx_trained_models_run_id ON public.trained_models(run_id);
CREATE INDEX idx_trained_models_performance ON public.trained_models(final_sharpe_ratio DESC, final_win_rate DESC);
CREATE INDEX idx_trained_models_is_best ON public.trained_models(is_best) WHERE is_best = true;

-- Update timestamp trigger
CREATE TRIGGER update_trained_models_updated_at
BEFORE UPDATE ON public.trained_models
FOR EACH ROW
EXECUTE FUNCTION public.update_updated_at_column();