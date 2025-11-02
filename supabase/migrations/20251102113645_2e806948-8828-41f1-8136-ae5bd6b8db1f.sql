-- Table for storing expert trajectories
CREATE TABLE IF NOT EXISTS public.expert_trajectories (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  symbol TEXT NOT NULL,
  timeframe TEXT NOT NULL,
  tactic_id TEXT NOT NULL, -- VWAP_REVERSION, NEWS_MOMENTUM, etc.
  timestamp TIMESTAMPTZ NOT NULL,
  
  -- State observation
  obs_features JSONB NOT NULL, -- All features as JSON
  
  -- Action taken
  action INTEGER NOT NULL CHECK (action IN (-1, 0, 1)), -- SELL/HOLD/BUY
  
  -- Reward
  reward NUMERIC NOT NULL,
  delta_equity NUMERIC NOT NULL,
  fees NUMERIC NOT NULL,
  slippage NUMERIC NOT NULL,
  
  -- Metadata
  entry_quality NUMERIC, -- 0-1 score
  rr_ratio NUMERIC, -- Risk:Reward ratio
  regime_tag TEXT, -- trend/sideways/high-vol
  
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_trajectories_symbol_time ON public.expert_trajectories(symbol, timestamp DESC);
CREATE INDEX idx_trajectories_tactic ON public.expert_trajectories(tactic_id);

-- Table for training runs
CREATE TABLE IF NOT EXISTS public.training_runs (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  run_name TEXT NOT NULL,
  phase TEXT NOT NULL, -- 'BC' or 'PPO'
  
  -- Hyperparameters
  hyperparams JSONB NOT NULL,
  
  -- Status
  status TEXT NOT NULL DEFAULT 'running', -- running/completed/failed
  current_epoch INTEGER DEFAULT 0,
  total_epochs INTEGER,
  
  -- Best checkpoint info
  best_val_sharpe NUMERIC,
  best_checkpoint_path TEXT,
  
  started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  completed_at TIMESTAMPTZ,
  
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Table for training metrics per epoch
CREATE TABLE IF NOT EXISTS public.training_metrics (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  run_id UUID NOT NULL REFERENCES public.training_runs(id) ON DELETE CASCADE,
  epoch INTEGER NOT NULL,
  split TEXT NOT NULL, -- train/val/test
  
  -- Performance metrics
  mean_reward NUMERIC,
  profit_factor NUMERIC,
  sharpe_ratio NUMERIC,
  win_rate NUMERIC,
  avg_rr NUMERIC,
  max_drawdown NUMERIC,
  
  -- Action distribution
  action_buy_pct NUMERIC,
  action_sell_pct NUMERIC,
  action_hold_pct NUMERIC,
  
  -- Loss metrics (BC/PPO)
  policy_loss NUMERIC,
  value_loss NUMERIC,
  entropy NUMERIC,
  
  -- Counterfactual metrics
  good_skips INTEGER DEFAULT 0,
  missed_winners INTEGER DEFAULT 0,
  
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_metrics_run_epoch ON public.training_metrics(run_id, epoch);

-- Table for technical indicators (cached)
CREATE TABLE IF NOT EXISTS public.technical_indicators (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  symbol TEXT NOT NULL,
  timeframe TEXT NOT NULL,
  timestamp TIMESTAMPTZ NOT NULL,
  
  -- Technical features
  rsi_14 NUMERIC,
  atr_14 NUMERIC,
  ema_20 NUMERIC,
  ema_50 NUMERIC,
  vwap NUMERIC,
  vwap_distance_pct NUMERIC,
  intraday_position NUMERIC, -- 0-1 from day low to high
  range_pct NUMERIC, -- (H-L)/C
  volume_zscore NUMERIC,
  
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(symbol, timeframe, timestamp)
);

CREATE INDEX idx_indicators_symbol_time ON public.technical_indicators(symbol, timeframe, timestamp DESC);

-- Table for news features
CREATE TABLE IF NOT EXISTS public.news_features (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  symbol TEXT NOT NULL,
  headline TEXT NOT NULL,
  timestamp TIMESTAMPTZ NOT NULL,
  source TEXT,
  
  -- AI-derived features
  sentiment NUMERIC CHECK (sentiment >= -1 AND sentiment <= 1), -- -1 to 1
  surprise_score NUMERIC, -- 0-1
  relevance_score NUMERIC, -- 0-1
  freshness_minutes INTEGER,
  
  -- Raw content for reprocessing
  article_snippet TEXT,
  
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_news_symbol_time ON public.news_features(symbol, timestamp DESC);

-- RLS policies
ALTER TABLE public.expert_trajectories ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.training_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.training_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.technical_indicators ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.news_features ENABLE ROW LEVEL SECURITY;

-- Public read access for training data
CREATE POLICY "Anyone can read trajectories"
  ON public.expert_trajectories FOR SELECT USING (true);

CREATE POLICY "Anyone can read training runs"
  ON public.training_runs FOR SELECT USING (true);

CREATE POLICY "Anyone can read metrics"
  ON public.training_metrics FOR SELECT USING (true);

CREATE POLICY "Anyone can read indicators"
  ON public.technical_indicators FOR SELECT USING (true);

CREATE POLICY "Anyone can read news"
  ON public.news_features FOR SELECT USING (true);