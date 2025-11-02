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

async function log(level: "INFO" | "WARN" | "ERROR", message: string, metadata?: any) {
  console.log(`[${level}] ${message}`);
  await supabase.from("system_logs").insert({
    level,
    source: "auto-data-generator",
    message,
    metadata: metadata || {},
  });
}

// Generate synthetic market data (simulates real trading conditions)
function generateSyntheticBar(basePrice: number, volatility: number, timestamp: Date) {
  const change = (Math.random() - 0.5) * volatility * basePrice;
  const open = basePrice;
  const close = basePrice + change;
  const high = Math.max(open, close) + Math.random() * volatility * basePrice * 0.3;
  const low = Math.min(open, close) - Math.random() * volatility * basePrice * 0.3;
  const volume = Math.floor(1000000 + Math.random() * 5000000);
  
  return { open, high, low, close, volume, timestamp };
}

function calculateTechnicalIndicators(bars: any[]) {
  const indicators = [];
  
  for (let i = 14; i < bars.length; i++) {
    const recentBars = bars.slice(i - 14, i + 1);
    
    // Calculate RSI
    let gains = 0, losses = 0;
    for (let j = 1; j < recentBars.length; j++) {
      const change = recentBars[j].close - recentBars[j - 1].close;
      if (change > 0) gains += change;
      else losses += Math.abs(change);
    }
    const avgGain = gains / 14;
    const avgLoss = losses / 14;
    const rs = avgGain / (avgLoss || 1);
    const rsi_14 = 100 - (100 / (1 + rs));
    
    // Calculate ATR
    let atrSum = 0;
    for (let j = 1; j < recentBars.length; j++) {
      const tr = Math.max(
        recentBars[j].high - recentBars[j].low,
        Math.abs(recentBars[j].high - recentBars[j - 1].close),
        Math.abs(recentBars[j].low - recentBars[j - 1].close)
      );
      atrSum += tr;
    }
    const atr_14 = atrSum / 14;
    
    // Calculate VWAP
    let vwapSum = 0, volumeSum = 0;
    for (const bar of recentBars) {
      const typical = (bar.high + bar.low + bar.close) / 3;
      vwapSum += typical * bar.volume;
      volumeSum += bar.volume;
    }
    const vwap = vwapSum / volumeSum;
    const vwap_distance_pct = ((bars[i].close - vwap) / vwap) * 100;
    
    // Calculate EMA
    const ema_20 = recentBars.slice(-20).reduce((sum, b) => sum + b.close, 0) / 20;
    const ema_50 = recentBars.reduce((sum, b) => sum + b.close, 0) / recentBars.length;
    
    // Volume z-score
    const avgVolume = recentBars.reduce((sum, b) => sum + b.volume, 0) / recentBars.length;
    const stdVolume = Math.sqrt(recentBars.reduce((sum, b) => sum + Math.pow(b.volume - avgVolume, 2), 0) / recentBars.length);
    const volume_zscore = (bars[i].volume - avgVolume) / (stdVolume || 1);
    
    // Intraday position (0 = low, 1 = high)
    const intraday_position = (bars[i].close - bars[i].low) / (bars[i].high - bars[i].low || 1);
    
    // Range %
    const range_pct = ((bars[i].high - bars[i].low) / bars[i].close) * 100;
    
    indicators.push({
      timestamp: bars[i].timestamp,
      rsi_14,
      atr_14,
      vwap,
      vwap_distance_pct,
      ema_20,
      ema_50,
      volume_zscore,
      intraday_position,
      range_pct,
    });
  }
  
  return indicators;
}

function generateExpertTrajectories(symbol: string, bars: any[], indicators: any[]) {
  const trajectories = [];
  
  for (let i = 32; i < indicators.length; i++) {
    const ind = indicators[i];
    const bar = bars[i + 14]; // Offset for indicator calculation
    
    // Frame stack (last 32 bars with indicators)
    const frameStack = [];
    for (let j = i - 32; j < i; j++) {
      frameStack.push({
        close: bars[j + 14].close,
        volume: bars[j + 14].volume,
        rsi_14: indicators[j].rsi_14,
        atr_14: indicators[j].atr_14,
        vwap_distance_pct: indicators[j].vwap_distance_pct,
        volume_zscore: indicators[j].volume_zscore,
      });
    }
    
    // Strategy 1: RSI_EMA (40% weight)
    if (ind.rsi_14 < 30 && ind.ema_20 > ind.ema_50) {
      trajectories.push({
        symbol,
        timeframe: "5m",
        timestamp: ind.timestamp,
        tactic_id: "RSI_EMA",
        action: 1, // BUY
        reward: Math.random() * 2 - 0.5, // Simulated reward
        obs_features: { frame_stack: frameStack, current: ind },
        entry_quality: 0.8,
        rr_ratio: 2.0,
        delta_equity: Math.random() * 100,
        fees: 0.1,
        slippage: 0.05,
      });
    } else if (ind.rsi_14 > 70 && ind.ema_20 < ind.ema_50) {
      trajectories.push({
        symbol,
        timeframe: "5m",
        timestamp: ind.timestamp,
        tactic_id: "RSI_EMA",
        action: -1, // SELL
        reward: Math.random() * 2 - 0.5,
        obs_features: { frame_stack: frameStack, current: ind },
        entry_quality: 0.8,
        rr_ratio: 2.0,
        delta_equity: Math.random() * 100,
        fees: 0.1,
        slippage: 0.05,
      });
    }
    
    // Strategy 2: VWAP_REVERSION (30% weight)
    if (ind.vwap_distance_pct < -1.5 && ind.volume_zscore > 1.5) {
      trajectories.push({
        symbol,
        timeframe: "5m",
        timestamp: ind.timestamp,
        tactic_id: "VWAP_REVERSION",
        action: 1, // BUY
        reward: Math.random() * 2 - 0.5,
        obs_features: { frame_stack: frameStack, current: ind },
        entry_quality: 0.75,
        rr_ratio: 1.8,
        delta_equity: Math.random() * 80,
        fees: 0.1,
        slippage: 0.05,
      });
    } else if (ind.vwap_distance_pct > 1.5 && ind.volume_zscore > 1.5) {
      trajectories.push({
        symbol,
        timeframe: "5m",
        timestamp: ind.timestamp,
        tactic_id: "VWAP_REVERSION",
        action: -1, // SELL
        reward: Math.random() * 2 - 0.5,
        obs_features: { frame_stack: frameStack, current: ind },
        entry_quality: 0.75,
        rr_ratio: 1.8,
        delta_equity: Math.random() * 80,
        fees: 0.1,
        slippage: 0.05,
      });
    }
    
    // Strategy 3: TREND_PULLBACK (10% weight)
    if (ind.ema_20 > ind.ema_50 && ind.rsi_14 < 45) {
      trajectories.push({
        symbol,
        timeframe: "5m",
        timestamp: ind.timestamp,
        tactic_id: "TREND_PULLBACK",
        action: 1, // BUY
        reward: Math.random() * 1.5 - 0.3,
        obs_features: { frame_stack: frameStack, current: ind },
        entry_quality: 0.7,
        rr_ratio: 2.5,
        delta_equity: Math.random() * 120,
        fees: 0.1,
        slippage: 0.05,
      });
    }
    
    // Add some HOLD actions too (important for learning)
    if (Math.random() < 0.3) {
      trajectories.push({
        symbol,
        timeframe: "5m",
        timestamp: ind.timestamp,
        tactic_id: "HOLD_BASELINE",
        action: 0, // HOLD
        reward: -0.05, // Small penalty
        obs_features: { frame_stack: frameStack, current: ind },
        entry_quality: 0.5,
        rr_ratio: 1.0,
        delta_equity: 0,
        fees: 0,
        slippage: 0,
      });
    }
  }
  
  return trajectories;
}

async function generateDataForSymbol(symbol: string, numBars: number = 500) {
  await log("INFO", `📊 Generating ${numBars} bars for ${symbol}...`);
  
  // Base prices for different symbols
  const basePrices: Record<string, number> = {
    AAPL: 180,
    TSLA: 250,
    MSFT: 380,
    GOOGL: 140,
    AMZN: 170,
  };
  
  const basePrice = basePrices[symbol] || 100;
  const volatility = 0.02; // 2% volatility
  
  // Generate synthetic bars
  const bars = [];
  const startTime = new Date();
  startTime.setDate(startTime.getDate() - 7); // Start 7 days ago
  
  for (let i = 0; i < numBars; i++) {
    const timestamp = new Date(startTime);
    timestamp.setMinutes(timestamp.getMinutes() + i * 5); // 5-minute bars
    
    const currentPrice = basePrice * (1 + (Math.sin(i / 20) * 0.05)); // Add some trend
    const bar = generateSyntheticBar(currentPrice, volatility, timestamp);
    bars.push(bar);
  }
  
  // Insert bars into database
  await log("INFO", `💾 Inserting ${bars.length} bars for ${symbol}...`);
  const barInserts = bars.map((bar) => ({
    symbol,
    timeframe: "5m",
    timestamp: bar.timestamp.toISOString(),
    open: bar.open,
    high: bar.high,
    low: bar.low,
    close: bar.close,
    volume: bar.volume,
  }));
  
  // Insert in batches of 100
  for (let i = 0; i < barInserts.length; i += 100) {
    const batch = barInserts.slice(i, i + 100);
    await supabase.from("historical_bars").insert(batch);
  }
  
  // Calculate technical indicators
  await log("INFO", `🔧 Calculating technical indicators for ${symbol}...`);
  const indicators = calculateTechnicalIndicators(bars);
  
  // Insert indicators
  const indicatorInserts = indicators.map((ind) => ({
    symbol,
    timeframe: "5m",
    timestamp: ind.timestamp.toISOString(),
    rsi_14: ind.rsi_14,
    atr_14: ind.atr_14,
    vwap: ind.vwap,
    vwap_distance_pct: ind.vwap_distance_pct,
    ema_20: ind.ema_20,
    ema_50: ind.ema_50,
    volume_zscore: ind.volume_zscore,
    intraday_position: ind.intraday_position,
    range_pct: ind.range_pct,
  }));
  
  for (let i = 0; i < indicatorInserts.length; i += 100) {
    const batch = indicatorInserts.slice(i, i + 100);
    await supabase.from("technical_indicators").insert(batch);
  }
  
  // Generate expert trajectories
  await log("INFO", `🎯 Generating expert trajectories for ${symbol}...`);
  const trajectories = generateExpertTrajectories(symbol, bars, indicators);
  
  // Insert trajectories
  for (let i = 0; i < trajectories.length; i += 100) {
    const batch = trajectories.slice(i, i + 100);
    await supabase.from("expert_trajectories").insert(batch);
  }
  
  // Update symbol metadata
  await supabase.from("symbols").upsert({
    symbol,
    name: symbol,
    exchange: "NASDAQ",
    is_active: true,
    last_fetched: new Date().toISOString(),
  });
  
  await log("INFO", `✅ Generated ${bars.length} bars, ${indicators.length} indicators, ${trajectories.length} trajectories for ${symbol}`);
  
  return {
    bars: bars.length,
    indicators: indicators.length,
    trajectories: trajectories.length,
  };
}

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { symbols = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN"], bars_per_symbol = 500 } = await req.json().catch(() => ({}));
    
    await log("INFO", `🚀 Auto data generator started for ${symbols.length} symbols`);
    
    const results = [];
    
    for (const symbol of symbols) {
      const result = await generateDataForSymbol(symbol, bars_per_symbol);
      results.push({ symbol, ...result });
    }
    
    const totalBars = results.reduce((sum, r) => sum + r.bars, 0);
    const totalIndicators = results.reduce((sum, r) => sum + r.indicators, 0);
    const totalTrajectories = results.reduce((sum, r) => sum + r.trajectories, 0);
    
    await log("INFO", `✅ Data generation complete: ${totalBars} bars, ${totalIndicators} indicators, ${totalTrajectories} trajectories`);

    return new Response(
      JSON.stringify({
        success: true,
        symbols: results,
        totals: {
          bars: totalBars,
          indicators: totalIndicators,
          trajectories: totalTrajectories,
        },
      }),
      { headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  } catch (error) {
    await log("ERROR", "Auto data generator failed", { error });
    return new Response(
      JSON.stringify({ error: error instanceof Error ? error.message : "Unknown error" }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});
