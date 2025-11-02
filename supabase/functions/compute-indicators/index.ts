import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.78.0";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

// Technical indicator calculations
function calculateRSI(closes: number[], period: number = 14): number | null {
  if (closes.length < period + 1) return null;
  
  let gains = 0, losses = 0;
  for (let i = 1; i <= period; i++) {
    const change = closes[i] - closes[i - 1];
    if (change > 0) gains += change;
    else losses -= change;
  }
  
  const avgGain = gains / period;
  const avgLoss = losses / period;
  
  if (avgLoss === 0) return 100;
  const rs = avgGain / avgLoss;
  return 100 - (100 / (1 + rs));
}

function calculateATR(bars: any[], period: number = 14): number | null {
  if (bars.length < period) return null;
  
  let atr = 0;
  for (let i = 1; i < period + 1; i++) {
    const high = bars[i].high;
    const low = bars[i].low;
    const prevClose = bars[i - 1].close;
    
    const tr = Math.max(
      high - low,
      Math.abs(high - prevClose),
      Math.abs(low - prevClose)
    );
    atr += tr;
  }
  
  return atr / period;
}

function calculateEMA(values: number[], period: number): number | null {
  if (values.length < period) return null;
  
  const multiplier = 2 / (period + 1);
  let ema = values.slice(0, period).reduce((a, b) => a + b) / period;
  
  for (let i = period; i < values.length; i++) {
    ema = (values[i] - ema) * multiplier + ema;
  }
  
  return ema;
}

function calculateVWAP(bars: any[]): number {
  let cumVolPrice = 0;
  let cumVol = 0;
  
  for (const bar of bars) {
    const typical = (bar.high + bar.low + bar.close) / 3;
    cumVolPrice += typical * bar.volume;
    cumVol += bar.volume;
  }
  
  return cumVol > 0 ? cumVolPrice / cumVol : bars[bars.length - 1].close;
}

function calculateVolumeZScore(volumes: number[]): number {
  const mean = volumes.reduce((a, b) => a + b) / volumes.length;
  const variance = volumes.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / volumes.length;
  const std = Math.sqrt(variance);
  
  const currentVol = volumes[volumes.length - 1];
  return std > 0 ? (currentVol - mean) / std : 0;
}

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { symbol, timeframe = "1m", lookback = 200 } = await req.json();

    if (!symbol) {
      return new Response(JSON.stringify({ error: "Symbol is required" }), {
        status: 400,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    console.log(`Computing indicators for ${symbol} ${timeframe}`);

    const supabaseAdmin = createClient(
      Deno.env.get("SUPABASE_URL") ?? "",
      Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "",
    );

    // Fetch historical bars
    const { data: bars, error: barsError } = await supabaseAdmin
      .from("historical_bars")
      .select("*")
      .eq("symbol", symbol)
      .eq("timeframe", timeframe)
      .order("timestamp", { ascending: true })
      .limit(lookback);

    if (barsError || !bars || bars.length === 0) {
      console.error("Error fetching bars:", barsError);
      return new Response(
        JSON.stringify({ error: "No historical data available" }),
        { status: 404, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    console.log(`Processing ${bars.length} bars`);

    // Extract values
    const closes = bars.map(b => Number(b.close));
    const highs = bars.map(b => Number(b.high));
    const lows = bars.map(b => Number(b.low));
    const volumes = bars.map(b => Number(b.volume));

    // Calculate indicators for each bar (need enough history)
    const indicators = [];
    const minRequired = 50; // Need at least 50 bars for EMA50

    for (let i = minRequired; i < bars.length; i++) {
      const bar = bars[i];
      const historySlice = bars.slice(0, i + 1);
      const closesSlice = closes.slice(0, i + 1);
      const volumesSlice = volumes.slice(Math.max(0, i - 20), i + 1); // Last 20 for z-score

      // Day's high/low for intraday position
      const dayBars = historySlice.filter(b => 
        new Date(b.timestamp).toDateString() === new Date(bar.timestamp).toDateString()
      );
      const dayHigh = Math.max(...dayBars.map(b => Number(b.high)));
      const dayLow = Math.min(...dayBars.map(b => Number(b.low)));

      const rsi = calculateRSI(closesSlice.slice(-15), 14);
      const atr = calculateATR(historySlice.slice(-15), 14);
      const ema20 = calculateEMA(closesSlice.slice(-20), 20);
      const ema50 = calculateEMA(closesSlice.slice(-50), 50);
      const vwap = calculateVWAP(dayBars);
      const close = Number(bar.close);

      indicators.push({
        symbol,
        timeframe,
        timestamp: bar.timestamp,
        rsi_14: rsi,
        atr_14: atr,
        ema_20: ema20,
        ema_50: ema50,
        vwap: vwap,
        vwap_distance_pct: vwap > 0 ? ((close - vwap) / vwap) * 100 : 0,
        intraday_position: dayHigh > dayLow ? (close - dayLow) / (dayHigh - dayLow) : 0.5,
        range_pct: close > 0 ? ((Number(bar.high) - Number(bar.low)) / close) * 100 : 0,
        volume_zscore: calculateVolumeZScore(volumesSlice),
      });
    }

    // Upsert to database
    if (indicators.length > 0) {
      const { error: upsertError } = await supabaseAdmin
        .from("technical_indicators")
        .upsert(indicators, { onConflict: "symbol,timeframe,timestamp" });

      if (upsertError) {
        console.error("Error upserting indicators:", upsertError);
        return new Response(
          JSON.stringify({ error: "Failed to store indicators" }),
          { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
        );
      }
    }

    console.log(`Successfully computed ${indicators.length} indicator rows`);

    return new Response(
      JSON.stringify({
        success: true,
        symbol,
        timeframe,
        indicatorsCount: indicators.length,
      }),
      { headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  } catch (error) {
    console.error("Error in compute-indicators:", error);
    return new Response(
      JSON.stringify({ error: error instanceof Error ? error.message : "Unknown error" }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});
