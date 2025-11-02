import "https://deno.land/x/xhr@0.1.0/mod.ts";
import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);

// Multi-strategy signal generation
const strategies = [
  "vwap_reversion",
  "news_momentum",
  "trend_pullback",
  "rsi_divergence",
  "liquidity_sweep",
  "volume_spike_reversal",
  "vwap_delta",
  "afternoon_fade",
  "gap_fill",
  "opening_range_breakout",
];

async function fetchMarketData(symbol: string) {
  try {
    const { data, error } = await supabase.functions.invoke("fetch-market-data", {
      body: { symbol, timeframe: "5m", limit: 100 },
    });

    if (error) throw error;
    return data;
  } catch (error) {
    console.error(`Failed to fetch data for ${symbol}:`, error);
    return null;
  }
}

function calculateIndicators(bars: any[]) {
  if (!bars || bars.length === 0) return null;

  const closes = bars.map((b) => b.close);
  const volumes = bars.map((b) => b.volume);
  const latest = bars[bars.length - 1];

  // Calculate RSI
  let gains = 0, losses = 0;
  for (let i = 1; i < Math.min(15, closes.length); i++) {
    const diff = closes[i] - closes[i - 1];
    if (diff > 0) gains += diff;
    else losses += Math.abs(diff);
  }
  const rs = gains / (losses || 1);
  const rsi = 100 - 100 / (1 + rs);

  // Calculate VWAP
  let vwapSum = 0, volumeSum = 0;
  for (const bar of bars) {
    const typicalPrice = (bar.high + bar.low + bar.close) / 3;
    vwapSum += typicalPrice * bar.volume;
    volumeSum += bar.volume;
  }
  const vwap = vwapSum / volumeSum;
  const vwapDev = ((latest.close - vwap) / vwap) * 100;

  // Calculate ATR
  let atrSum = 0;
  for (let i = 1; i < Math.min(15, bars.length); i++) {
    const high = bars[i].high;
    const low = bars[i].low;
    const prevClose = bars[i - 1].close;
    const tr = Math.max(high - low, Math.abs(high - prevClose), Math.abs(low - prevClose));
    atrSum += tr;
  }
  const atr = atrSum / Math.min(14, bars.length - 1);

  // EMA
  const ema20 = closes.slice(-20).reduce((a, b) => a + b, 0) / 20;
  const ema50 = closes.slice(-50).reduce((a, b) => a + b, 0) / 50;

  // Volume Z-Score
  const avgVolume = volumes.reduce((a, b) => a + b, 0) / volumes.length;
  const volStd = Math.sqrt(
    volumes.reduce((sum, v) => sum + Math.pow(v - avgVolume, 2), 0) / volumes.length
  );
  const volumeZScore = (latest.volume - avgVolume) / (volStd || 1);

  return {
    price: latest.close,
    rsi,
    atr,
    vwap,
    vwapDeviation: vwapDev,
    ema20,
    ema50,
    volumeZScore,
    recentVolatility: (atr / latest.close) * 100,
  };
}

function generateStrategySignals(symbol: string, indicators: any) {
  const signals = [];

  // VWAP Reversion
  if (Math.abs(indicators.vwapDeviation) > 2) {
    signals.push({
      strategy: "vwap_reversion",
      action: indicators.vwapDeviation < 0 ? "buy" : "sell",
      confidence: Math.min(0.9, 0.5 + Math.abs(indicators.vwapDeviation) * 0.1),
      reasoning: `VWAP deviation ${indicators.vwapDeviation.toFixed(2)}% - mean reversion expected`,
    });
  }

  // RSI Divergence
  if (indicators.rsi < 30) {
    signals.push({
      strategy: "rsi_divergence",
      action: "buy",
      confidence: 0.75 + (30 - indicators.rsi) * 0.01,
      reasoning: `RSI oversold at ${indicators.rsi.toFixed(1)}`,
    });
  } else if (indicators.rsi > 70) {
    signals.push({
      strategy: "rsi_divergence",
      action: "sell",
      confidence: 0.75 + (indicators.rsi - 70) * 0.01,
      reasoning: `RSI overbought at ${indicators.rsi.toFixed(1)}`,
    });
  }

  // Volume Spike Reversal
  if (indicators.volumeZScore > 2) {
    signals.push({
      strategy: "volume_spike_reversal",
      action: indicators.price > indicators.ema20 ? "sell" : "buy",
      confidence: 0.7,
      reasoning: `Volume spike detected (z-score: ${indicators.volumeZScore.toFixed(2)})`,
    });
  }

  // Trend Pullback
  if (indicators.ema20 > indicators.ema50 && indicators.price < indicators.ema20) {
    signals.push({
      strategy: "trend_pullback",
      action: "buy",
      confidence: 0.65,
      reasoning: "Bullish trend with pullback to EMA20",
    });
  } else if (indicators.ema20 < indicators.ema50 && indicators.price > indicators.ema20) {
    signals.push({
      strategy: "trend_pullback",
      action: "sell",
      confidence: 0.65,
      reasoning: "Bearish trend with pullback to EMA20",
    });
  }

  return signals;
}

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { watchlist } = await req.json();
    const symbols = watchlist || ["AAPL", "TSLA", "MSFT"];

    const allSignals = [];

    for (const symbol of symbols) {
      const marketData = await fetchMarketData(symbol);
      if (!marketData?.bars || marketData.bars.length === 0) {
        console.log(`No data for ${symbol}`);
        continue;
      }

      const indicators = calculateIndicators(marketData.bars);
      if (!indicators) continue;

      const strategySignals = generateStrategySignals(symbol, indicators);

      // Aggregate signals per symbol (ensemble voting)
      if (strategySignals.length > 0) {
        const buySignals = strategySignals.filter((s) => s.action === "buy");
        const sellSignals = strategySignals.filter((s) => s.action === "sell");

        let finalAction: string;
        let finalConfidence: number;
        let strategies: string[];

        if (buySignals.length > sellSignals.length) {
          finalAction = "buy";
          finalConfidence =
            buySignals.reduce((sum, s) => sum + s.confidence, 0) / buySignals.length;
          strategies = buySignals.map((s) => s.strategy);
        } else if (sellSignals.length > buySignals.length) {
          finalAction = "sell";
          finalConfidence =
            sellSignals.reduce((sum, s) => sum + s.confidence, 0) / sellSignals.length;
          strategies = sellSignals.map((s) => s.strategy);
        } else {
          continue; // Conflicting signals - skip
        }

        allSignals.push({
          symbol,
          action: finalAction,
          confidence: finalConfidence,
          proposed_size: 50,
          source: strategies.join("+"),
          market_data: indicators,
          strategies,
        });
      }
    }

    return new Response(
      JSON.stringify({
        signals: allSignals,
        generated_at: new Date().toISOString(),
      }),
      { headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  } catch (error) {
    console.error("Trading engine error:", error);
    return new Response(
      JSON.stringify({ error: error instanceof Error ? error.message : "Unknown error" }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});
