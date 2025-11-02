-- Bot configuration och status
CREATE TABLE IF NOT EXISTS public.bot_config (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  is_active BOOLEAN NOT NULL DEFAULT false,
  continuous_learning_enabled BOOLEAN NOT NULL DEFAULT true,
  max_concurrent_positions INTEGER NOT NULL DEFAULT 5,
  max_position_size_pct NUMERIC NOT NULL DEFAULT 20.0,
  risk_per_trade_pct NUMERIC NOT NULL DEFAULT 1.0,
  max_drawdown_pct NUMERIC NOT NULL DEFAULT 10.0,
  loop_interval_minutes INTEGER NOT NULL DEFAULT 5,
  loops_per_cycle INTEGER NOT NULL DEFAULT 12,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Strategy performance tracking
CREATE TABLE IF NOT EXISTS public.strategy_performance (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  strategy_name TEXT NOT NULL,
  total_trades INTEGER NOT NULL DEFAULT 0,
  winning_trades INTEGER NOT NULL DEFAULT 0,
  losing_trades INTEGER NOT NULL DEFAULT 0,
  win_rate NUMERIC NOT NULL DEFAULT 0,
  profit_factor NUMERIC NOT NULL DEFAULT 0,
  sharpe_ratio NUMERIC NOT NULL DEFAULT 0,
  max_drawdown NUMERIC NOT NULL DEFAULT 0,
  avg_rr_ratio NUMERIC NOT NULL DEFAULT 0,
  is_active BOOLEAN NOT NULL DEFAULT true,
  last_trade_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(strategy_name)
);

-- Signal correlations (för diversifiering)
CREATE TABLE IF NOT EXISTS public.signal_correlations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  signal_id UUID REFERENCES public.trading_signals(id),
  position_symbol TEXT NOT NULL,
  correlation NUMERIC NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- RL agent decisions
CREATE TABLE IF NOT EXISTS public.rl_decisions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  signal_id UUID REFERENCES public.trading_signals(id),
  action TEXT NOT NULL CHECK (action IN ('buy', 'sell', 'hold')),
  confidence NUMERIC NOT NULL CHECK (confidence BETWEEN 0 AND 1),
  q_value NUMERIC,
  reasoning TEXT,
  state_features JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Position sizing calculations
CREATE TABLE IF NOT EXISTS public.position_sizing (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  signal_id UUID REFERENCES public.trading_signals(id),
  base_size NUMERIC NOT NULL,
  rl_adjusted_size NUMERIC NOT NULL,
  volatility_adjusted_size NUMERIC NOT NULL,
  drawdown_adjusted_size NUMERIC NOT NULL,
  final_size NUMERIC NOT NULL,
  risk_amount NUMERIC NOT NULL,
  factors JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Online learning results
CREATE TABLE IF NOT EXISTS public.online_learning (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  model_type TEXT NOT NULL,
  learning_rate NUMERIC NOT NULL,
  samples_processed INTEGER NOT NULL,
  loss NUMERIC,
  metrics JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Bot loop execution logs
CREATE TABLE IF NOT EXISTS public.bot_loops (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  loop_number INTEGER NOT NULL,
  signals_generated INTEGER NOT NULL DEFAULT 0,
  trades_placed INTEGER NOT NULL DEFAULT 0,
  trades_skipped INTEGER NOT NULL DEFAULT 0,
  positions_closed INTEGER NOT NULL DEFAULT 0,
  total_pnl NUMERIC NOT NULL DEFAULT 0,
  started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  completed_at TIMESTAMPTZ,
  status TEXT NOT NULL DEFAULT 'running' CHECK (status IN ('running', 'completed', 'failed')),
  error_message TEXT
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_strategy_perf_name ON public.strategy_performance(strategy_name);
CREATE INDEX IF NOT EXISTS idx_strategy_perf_active ON public.strategy_performance(is_active);
CREATE INDEX IF NOT EXISTS idx_correlations_signal ON public.signal_correlations(signal_id);
CREATE INDEX IF NOT EXISTS idx_rl_decisions_signal ON public.rl_decisions(signal_id);
CREATE INDEX IF NOT EXISTS idx_position_sizing_signal ON public.position_sizing(signal_id);
CREATE INDEX IF NOT EXISTS idx_bot_loops_started ON public.bot_loops(started_at DESC);

-- RLS policies
ALTER TABLE public.bot_config ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.strategy_performance ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.signal_correlations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.rl_decisions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.position_sizing ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.online_learning ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.bot_loops ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Service role full access bot_config" ON public.bot_config FOR ALL USING (true);
CREATE POLICY "Service role full access strategy_performance" ON public.strategy_performance FOR ALL USING (true);
CREATE POLICY "Service role full access signal_correlations" ON public.signal_correlations FOR ALL USING (true);
CREATE POLICY "Service role full access rl_decisions" ON public.rl_decisions FOR ALL USING (true);
CREATE POLICY "Service role full access position_sizing" ON public.position_sizing FOR ALL USING (true);
CREATE POLICY "Service role full access online_learning" ON public.online_learning FOR ALL USING (true);
CREATE POLICY "Service role full access bot_loops" ON public.bot_loops FOR ALL USING (true);

-- Triggers för updated_at
CREATE TRIGGER update_bot_config_updated_at
  BEFORE UPDATE ON public.bot_config
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_strategy_perf_updated_at
  BEFORE UPDATE ON public.strategy_performance
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

-- Insert default config
INSERT INTO public.bot_config (
  is_active, 
  continuous_learning_enabled,
  max_concurrent_positions,
  max_position_size_pct,
  risk_per_trade_pct,
  max_drawdown_pct,
  loop_interval_minutes,
  loops_per_cycle
) VALUES (
  false,
  true,
  5,
  20.0,
  1.0,
  10.0,
  5,
  12
) ON CONFLICT DO NOTHING;