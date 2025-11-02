-- Create news_sentiment table for storing analyzed news data
CREATE TABLE IF NOT EXISTS public.news_sentiment (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  symbol TEXT NOT NULL,
  timestamp TIMESTAMPTZ NOT NULL DEFAULT now(),
  overall_sentiment FLOAT NOT NULL DEFAULT 0,
  bullish_count INTEGER NOT NULL DEFAULT 0,
  bearish_count INTEGER NOT NULL DEFAULT 0,
  neutral_count INTEGER NOT NULL DEFAULT 0,
  confidence FLOAT NOT NULL DEFAULT 0,
  key_themes JSONB DEFAULT '[]'::jsonb,
  market_impact TEXT,
  article_count INTEGER NOT NULL DEFAULT 0,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(symbol, timestamp)
);

-- Create macro_data table for storing macro-economic indicators
CREATE TABLE IF NOT EXISTS public.macro_data (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  timestamp TIMESTAMPTZ NOT NULL DEFAULT now(),
  vix FLOAT NOT NULL,
  spy_price FLOAT NOT NULL,
  spy_change_pct FLOAT NOT NULL,
  dxy_price FLOAT NOT NULL,
  dxy_change_pct FLOAT NOT NULL,
  treasury_yield_10y FLOAT NOT NULL,
  market_regime TEXT NOT NULL,
  risk_on BOOLEAN NOT NULL DEFAULT false,
  risk_off BOOLEAN NOT NULL DEFAULT false,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(timestamp)
);

-- Add indexes for faster queries
CREATE INDEX IF NOT EXISTS idx_news_sentiment_symbol_timestamp ON public.news_sentiment(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_macro_data_timestamp ON public.macro_data(timestamp DESC);

-- Enable RLS
ALTER TABLE public.news_sentiment ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.macro_data ENABLE ROW LEVEL SECURITY;

-- Create policies for public read access
CREATE POLICY "News sentiment is viewable by everyone" 
ON public.news_sentiment 
FOR SELECT 
USING (true);

CREATE POLICY "Macro data is viewable by everyone" 
ON public.macro_data 
FOR SELECT 
USING (true);

-- Create policies for service role inserts
CREATE POLICY "Service role can insert news sentiment" 
ON public.news_sentiment 
FOR INSERT 
WITH CHECK (true);

CREATE POLICY "Service role can insert macro data" 
ON public.macro_data 
FOR INSERT 
WITH CHECK (true);