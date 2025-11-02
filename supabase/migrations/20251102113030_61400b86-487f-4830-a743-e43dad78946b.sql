-- Create table for storing historical OHLCV (Open, High, Low, Close, Volume) data
CREATE TABLE IF NOT EXISTS public.historical_bars (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  symbol TEXT NOT NULL,
  timeframe TEXT NOT NULL, -- '1m', '5m', '15m', '1h', '1d'
  timestamp TIMESTAMPTZ NOT NULL,
  open NUMERIC NOT NULL,
  high NUMERIC NOT NULL,
  low NUMERIC NOT NULL,
  close NUMERIC NOT NULL,
  volume BIGINT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(symbol, timeframe, timestamp)
);

-- Create indexes for efficient querying
CREATE INDEX idx_historical_bars_symbol_timeframe ON public.historical_bars(symbol, timeframe);
CREATE INDEX idx_historical_bars_timestamp ON public.historical_bars(timestamp DESC);
CREATE INDEX idx_historical_bars_symbol_timestamp ON public.historical_bars(symbol, timestamp DESC);

-- Table is public-readable for training purposes (no auth required)
ALTER TABLE public.historical_bars ENABLE ROW LEVEL SECURITY;

-- Allow anyone to read historical data (needed for ML training)
CREATE POLICY "Anyone can read historical bars"
  ON public.historical_bars
  FOR SELECT
  USING (true);

-- Only allow server (edge functions) to insert data
CREATE POLICY "Only service role can insert historical bars"
  ON public.historical_bars
  FOR INSERT
  WITH CHECK (false); -- Client inserts disabled, only service role can insert

-- Create table for storing symbol metadata
CREATE TABLE IF NOT EXISTS public.symbols (
  symbol TEXT PRIMARY KEY,
  name TEXT,
  exchange TEXT,
  sector TEXT,
  industry TEXT,
  last_fetched TIMESTAMPTZ,
  is_active BOOLEAN DEFAULT true,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE public.symbols ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Anyone can read symbols"
  ON public.symbols
  FOR SELECT
  USING (true);

-- Create function for updating timestamps
CREATE OR REPLACE FUNCTION public.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SET search_path = public;

-- Create trigger for symbol updates
CREATE TRIGGER update_symbols_updated_at
  BEFORE UPDATE ON public.symbols
  FOR EACH ROW
  EXECUTE FUNCTION public.update_updated_at_column();