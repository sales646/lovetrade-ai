-- Trading signals från strategier eller RL-agent
CREATE TABLE IF NOT EXISTS public.trading_signals (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  symbol TEXT NOT NULL,
  action TEXT NOT NULL CHECK (action IN ('buy', 'sell', 'hold')),
  confidence NUMERIC NOT NULL CHECK (confidence BETWEEN 0 AND 1),
  proposed_size NUMERIC NOT NULL,
  source TEXT NOT NULL, -- 'rl_agent', 'vwap_reversion', 'news_momentum', etc.
  market_data JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Risk assessments
CREATE TABLE IF NOT EXISTS public.risk_assessments (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  signal_id UUID REFERENCES public.trading_signals(id),
  risk_score NUMERIC NOT NULL CHECK (risk_score BETWEEN 0 AND 1),
  adjusted_size NUMERIC NOT NULL,
  should_execute BOOLEAN NOT NULL,
  reason TEXT NOT NULL,
  factors JSONB NOT NULL,
  assessed_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Executed trades
CREATE TABLE IF NOT EXISTS public.trades (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  signal_id UUID REFERENCES public.trading_signals(id),
  risk_assessment_id UUID REFERENCES public.risk_assessments(id),
  symbol TEXT NOT NULL,
  action TEXT NOT NULL CHECK (action IN ('buy', 'sell')),
  size NUMERIC NOT NULL,
  entry_price NUMERIC NOT NULL,
  stop_loss NUMERIC,
  take_profit NUMERIC,
  status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'closed', 'cancelled')),
  exit_price NUMERIC,
  pnl NUMERIC,
  executed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  closed_at TIMESTAMPTZ
);

-- Active positions (materialized view för snabb access)
CREATE TABLE IF NOT EXISTS public.positions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  symbol TEXT NOT NULL UNIQUE,
  side TEXT NOT NULL CHECK (side IN ('long', 'short')),
  size NUMERIC NOT NULL,
  entry_price NUMERIC NOT NULL,
  current_price NUMERIC,
  unrealized_pnl NUMERIC,
  stop_loss NUMERIC,
  take_profit NUMERIC,
  opened_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- System logs för debugging
CREATE TABLE IF NOT EXISTS public.system_logs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  level TEXT NOT NULL CHECK (level IN ('INFO', 'WARN', 'ERROR')),
  source TEXT NOT NULL,
  message TEXT NOT NULL,
  metadata JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes för performance
CREATE INDEX IF NOT EXISTS idx_signals_created ON public.trading_signals(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON public.trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_status ON public.trades(status);
CREATE INDEX IF NOT EXISTS idx_logs_created ON public.system_logs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_logs_level ON public.system_logs(level);

-- RLS policies (alla tabeller tillgängliga för service role)
ALTER TABLE public.trading_signals ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.risk_assessments ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.trades ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.positions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.system_logs ENABLE ROW LEVEL SECURITY;

-- Allow service role full access (för edge functions)
CREATE POLICY "Service role full access signals" ON public.trading_signals FOR ALL USING (true);
CREATE POLICY "Service role full access assessments" ON public.risk_assessments FOR ALL USING (true);
CREATE POLICY "Service role full access trades" ON public.trades FOR ALL USING (true);
CREATE POLICY "Service role full access positions" ON public.positions FOR ALL USING (true);
CREATE POLICY "Service role full access logs" ON public.system_logs FOR ALL USING (true);

-- Function för att uppdatera updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_positions_updated_at
  BEFORE UPDATE ON public.positions
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();