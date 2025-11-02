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

// Market regime types
type MarketRegime = "STRONG_TREND_UP" | "STRONG_TREND_DOWN" | "WEAK_TREND_UP" | "WEAK_TREND_DOWN" | "SIDEWAYS" | "CHOPPY" | "HIGH_VOLATILITY";

// Generate realistic market data with different regimes
function generateRealisticBars(basePrice: number, numBars: number, startTime: Date) {
  const bars = [];
  let currentPrice = basePrice;
  let regime: MarketRegime = "SIDEWAYS";
  let regimeLength = 0;
  let trendStrength = 0;
  
  for (let i = 0; i < numBars; i++) {
    // Change regime every 30-100 bars
    if (regimeLength === 0) {
      const regimes: MarketRegime[] = ["STRONG_TREND_UP", "STRONG_TREND_DOWN", "WEAK_TREND_UP", "WEAK_TREND_DOWN", "SIDEWAYS", "CHOPPY", "HIGH_VOLATILITY"];
      regime = regimes[Math.floor(Math.random() * regimes.length)];
      regimeLength = Math.floor(30 + Math.random() * 70);
      trendStrength = 0.3 + Math.random() * 0.7;
    }
    regimeLength--;
    
    const timestamp = new Date(startTime);
    timestamp.setMinutes(timestamp.getMinutes() + i * 5);
    
    // Generate bar based on regime
    let drift = 0;
    let volatility = 0.015; // Base 1.5%
    
    switch (regime) {
      case "STRONG_TREND_UP":
        drift = 0.002 * trendStrength; // 0.2% per bar
        volatility = 0.01;
        break;
      case "STRONG_TREND_DOWN":
        drift = -0.002 * trendStrength;
        volatility = 0.01;
        break;
      case "WEAK_TREND_UP":
        drift = 0.0005 * trendStrength;
        volatility = 0.015;
        break;
      case "WEAK_TREND_DOWN":
        drift = -0.0005 * trendStrength;
        volatility = 0.015;
        break;
      case "SIDEWAYS":
        drift = (Math.random() - 0.5) * 0.0003; // Very small random drift
        volatility = 0.008; // Low volatility
        break;
      case "CHOPPY":
        drift = (Math.random() - 0.5) * 0.001; // Random whipsaws
        volatility = 0.02; // Higher volatility
        break;
      case "HIGH_VOLATILITY":
        drift = (Math.random() - 0.5) * 0.003; // Large swings
        volatility = 0.03;
        break;
    }
    
    // Add noise
    const noise = (Math.random() - 0.5) * volatility * currentPrice;
    currentPrice = currentPrice * (1 + drift) + noise;
    
    // Generate OHLC
    const open = currentPrice;
    const close = currentPrice + (Math.random() - 0.5) * volatility * currentPrice;
    const high = Math.max(open, close) * (1 + Math.random() * volatility * 0.5);
    const low = Math.min(open, close) * (1 - Math.random() * volatility * 0.5);
    
    // Volume varies with volatility
    const baseVolume = 2000000;
    const volumeMultiplier = regime === "HIGH_VOLATILITY" ? 2.5 : regime === "CHOPPY" ? 1.8 : 1.0;
    const volume = Math.floor(baseVolume * volumeMultiplier * (0.5 + Math.random()));
    
    currentPrice = close;
    
    bars.push({ open, high, low, close, volume, timestamp, regime });
  }
  
  return bars;
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

// Simulate a trade forward through price action to calculate real P&L
function simulateTradeOutcome(
  bars: any[],
  entryIndex: number,
  side: "BUY" | "SELL",
  atr: number,
  maxHoldBars: number = 20
) {
  const entryBar = bars[entryIndex];
  const entryPrice = entryBar.close;
  
  // Risk management parameters
  const stopLossDistance = atr * 2; // 2x ATR stop
  const takeProfitDistance = atr * 4; // 4x ATR target (2:1 R:R)
  const slippagePct = 0.05; // 0.05% slippage
  const feesPct = 0.1; // 0.1% fees
  
  const stopLossPrice = side === "BUY" 
    ? entryPrice - stopLossDistance 
    : entryPrice + stopLossDistance;
  
  const takeProfitPrice = side === "BUY" 
    ? entryPrice + takeProfitDistance 
    : entryPrice - takeProfitDistance;
  
  // Walk forward through bars to find exit
  let exitPrice = entryPrice;
  let exitReason = "TIME"; // TIME, STOP_LOSS, TAKE_PROFIT
  let barsHeld = 0;
  
  for (let i = entryIndex + 1; i < Math.min(entryIndex + maxHoldBars + 1, bars.length); i++) {
    const bar = bars[i];
    barsHeld++;
    
    if (side === "BUY") {
      // Check if stop loss hit
      if (bar.low <= stopLossPrice) {
        exitPrice = stopLossPrice;
        exitReason = "STOP_LOSS";
        break;
      }
      // Check if take profit hit
      if (bar.high >= takeProfitPrice) {
        exitPrice = takeProfitPrice;
        exitReason = "TAKE_PROFIT";
        break;
      }
    } else { // SELL
      // Check if stop loss hit
      if (bar.high >= stopLossPrice) {
        exitPrice = stopLossPrice;
        exitReason = "STOP_LOSS";
        break;
      }
      // Check if take profit hit
      if (bar.low <= takeProfitPrice) {
        exitPrice = takeProfitPrice;
        exitReason = "TAKE_PROFIT";
        break;
      }
    }
    
    // Time-based exit
    if (barsHeld >= maxHoldBars) {
      exitPrice = bar.close;
      exitReason = "TIME";
      break;
    }
  }
  
  // Calculate P&L
  let grossPnlPct = side === "BUY"
    ? ((exitPrice - entryPrice) / entryPrice) * 100
    : ((entryPrice - exitPrice) / entryPrice) * 100;
  
  // Apply slippage and fees
  const netPnlPct = grossPnlPct - slippagePct - feesPct;
  
  // Convert P&L to reward (scale to reasonable range)
  // +5% = +1.0 reward, -2.5% = -0.5 reward
  const reward = netPnlPct / 5.0;
  
  const win = netPnlPct > 0;
  const entryQuality = win ? 0.8 + Math.random() * 0.2 : 0.1 + Math.random() * 0.3;
  const rrRatio = win ? Math.abs(netPnlPct / 2.5) : 0.5; // Actual R:R achieved
  
  return {
    reward,
    pnl_pct: netPnlPct,
    exit_reason: exitReason,
    bars_held: barsHeld,
    win,
    entry_quality: entryQuality,
    rr_ratio: Math.min(rrRatio, 3.0), // Cap at 3:1
    delta_equity: netPnlPct * 10, // Scale for display
    fees: feesPct,
    slippage: slippagePct,
  };
}

function generateExpertTrajectories(symbol: string, bars: any[], indicators: any[]) {
  const trajectories = [];
  
  for (let i = 32; i < indicators.length; i++) {
    const ind = indicators[i];
    const bar = bars[i + 14]; // Offset for indicator calculation
    const regime = bar.regime;
    
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
    
    // Quality score based on regime (0-1)
    let regimeQuality = 0.5; // Default
    if (regime === "STRONG_TREND_UP" || regime === "STRONG_TREND_DOWN") regimeQuality = 0.9;
    else if (regime === "WEAK_TREND_UP" || regime === "WEAK_TREND_DOWN") regimeQuality = 0.7;
    else if (regime === "SIDEWAYS") regimeQuality = 0.3; // Low quality, prefer HOLD
    else if (regime === "CHOPPY" || regime === "HIGH_VOLATILITY") regimeQuality = 0.2; // Very low, avoid trading
    
    // More selective strategies - only trade in good conditions
    const shouldAvoidTrading = regime === "CHOPPY" || regime === "HIGH_VOLATILITY" || regime === "SIDEWAYS";
    
    // Strategy 1: RSI_EMA (40% weight) - only in trending markets
    if (!shouldAvoidTrading && ind.rsi_14 < 30 && ind.ema_20 > ind.ema_50 && regime.includes("TREND_UP")) {
      const outcome = simulateTradeOutcome(bars, i + 14, "BUY", ind.atr_14);
      trajectories.push({
        symbol,
        timeframe: "5m",
        timestamp: ind.timestamp,
        tactic_id: "RSI_EMA",
        action: 1, // BUY
        reward: outcome.reward,
        obs_features: { frame_stack: frameStack, current: ind },
        regime_tag: regime,
        entry_quality: outcome.entry_quality,
        rr_ratio: outcome.rr_ratio,
        delta_equity: outcome.delta_equity,
        fees: outcome.fees,
        slippage: outcome.slippage,
      });
    } else if (!shouldAvoidTrading && ind.rsi_14 > 70 && ind.ema_20 < ind.ema_50 && regime.includes("TREND_DOWN")) {
      const outcome = simulateTradeOutcome(bars, i + 14, "SELL", ind.atr_14);
      trajectories.push({
        symbol,
        timeframe: "5m",
        timestamp: ind.timestamp,
        tactic_id: "RSI_EMA",
        action: -1, // SELL
        reward: outcome.reward,
        obs_features: { frame_stack: frameStack, current: ind },
        regime_tag: regime,
        entry_quality: outcome.entry_quality,
        rr_ratio: outcome.rr_ratio,
        delta_equity: outcome.delta_equity,
        fees: outcome.fees,
        slippage: outcome.slippage,
      });
    } else if (shouldAvoidTrading && ind.rsi_14 < 30 && ind.ema_20 > ind.ema_50) {
      // Still generate bad trades in bad conditions to teach avoidance
      const outcome = simulateTradeOutcome(bars, i + 14, "BUY", ind.atr_14);
      trajectories.push({
        symbol,
        timeframe: "5m",
        timestamp: ind.timestamp,
        tactic_id: "RSI_EMA",
        action: 1, // BUY
        reward: outcome.reward, // Will likely be negative in choppy markets
        obs_features: { frame_stack: frameStack, current: ind },
        regime_tag: regime,
        entry_quality: outcome.entry_quality,
        rr_ratio: outcome.rr_ratio,
        delta_equity: outcome.delta_equity,
        fees: outcome.fees,
        slippage: outcome.slippage,
      });
    }
    
    // Strategy 2: VWAP_REVERSION (30% weight) - works in all regimes except choppy
    if (!shouldAvoidTrading && ind.vwap_distance_pct < -1.5 && ind.volume_zscore > 1.5) {
      const outcome = simulateTradeOutcome(bars, i + 14, "BUY", ind.atr_14, 15); // Shorter hold for mean reversion
      trajectories.push({
        symbol,
        timeframe: "5m",
        timestamp: ind.timestamp,
        tactic_id: "VWAP_REVERSION",
        action: 1, // BUY
        reward: outcome.reward,
        obs_features: { frame_stack: frameStack, current: ind },
        regime_tag: regime,
        entry_quality: outcome.entry_quality,
        rr_ratio: outcome.rr_ratio,
        delta_equity: outcome.delta_equity,
        fees: outcome.fees,
        slippage: outcome.slippage,
      });
    } else if (!shouldAvoidTrading && ind.vwap_distance_pct > 1.5 && ind.volume_zscore > 1.5) {
      const outcome = simulateTradeOutcome(bars, i + 14, "SELL", ind.atr_14, 15);
      trajectories.push({
        symbol,
        timeframe: "5m",
        timestamp: ind.timestamp,
        tactic_id: "VWAP_REVERSION",
        action: -1, // SELL
        reward: outcome.reward,
        obs_features: { frame_stack: frameStack, current: ind },
        regime_tag: regime,
        entry_quality: outcome.entry_quality,
        rr_ratio: outcome.rr_ratio,
        delta_equity: outcome.delta_equity,
        fees: outcome.fees,
        slippage: outcome.slippage,
      });
    } else if (shouldAvoidTrading && (ind.vwap_distance_pct < -1.5 || ind.vwap_distance_pct > 1.5) && ind.volume_zscore > 1.5) {
      // Simulate bad trades in choppy conditions
      const side = ind.vwap_distance_pct < 0 ? "BUY" : "SELL";
      const outcome = simulateTradeOutcome(bars, i + 14, side, ind.atr_14, 15);
      trajectories.push({
        symbol,
        timeframe: "5m",
        timestamp: ind.timestamp,
        tactic_id: "VWAP_REVERSION",
        action: ind.vwap_distance_pct < 0 ? 1 : -1,
        reward: outcome.reward,
        obs_features: { frame_stack: frameStack, current: ind },
        regime_tag: regime,
        entry_quality: outcome.entry_quality,
        rr_ratio: outcome.rr_ratio,
        delta_equity: outcome.delta_equity,
        fees: outcome.fees,
        slippage: outcome.slippage,
      });
    }
    
    // Strategy 3: TREND_PULLBACK (10% weight) - only in strong trends
    if (regime.includes("STRONG_TREND") && ind.ema_20 > ind.ema_50 && ind.rsi_14 < 45) {
      const outcome = simulateTradeOutcome(bars, i + 14, "BUY", ind.atr_14, 25); // Longer hold for trends
      trajectories.push({
        symbol,
        timeframe: "5m",
        timestamp: ind.timestamp,
        tactic_id: "TREND_PULLBACK",
        action: 1, // BUY
        reward: outcome.reward,
        obs_features: { frame_stack: frameStack, current: ind },
        regime_tag: regime,
        entry_quality: outcome.entry_quality,
        rr_ratio: outcome.rr_ratio,
        delta_equity: outcome.delta_equity,
        fees: outcome.fees,
        slippage: outcome.slippage,
      });
    }
    
    // HOLD actions - much more frequent in bad conditions (CRITICAL for learning to stay out)
    const holdProbability = shouldAvoidTrading ? 0.6 : 0.15; // 60% HOLD in choppy/volatile, 15% otherwise
    if (Math.random() < holdProbability) {
      // HOLD reward structure:
      // - In BAD conditions: +0.2 (better than losing money on bad trades at -0.3 to -0.8)
      // - In GOOD conditions: -0.1 (worse than winning trades at +0.6 to +2.5)
      // This teaches: trade in good conditions (earn +0.6 to +2.5), stay out in bad (earn +0.2 vs lose -0.3 to -0.8)
      const holdReward = shouldAvoidTrading ? 0.2 : -0.1;
      trajectories.push({
        symbol,
        timeframe: "5m",
        timestamp: ind.timestamp,
        tactic_id: "HOLD_BASELINE",
        action: 0, // HOLD
        reward: holdReward,
        obs_features: { frame_stack: frameStack, current: ind },
        regime_tag: regime,
        entry_quality: shouldAvoidTrading ? 0.85 : 0.2,
        rr_ratio: 1.0,
        delta_equity: 0,
        fees: 0,
        slippage: 0,
      });
    }
  }
  
  return trajectories;
}

async function generateDataFromRealBars(symbol: string, limitBars?: number) {
  // Fetch real historical bars from database
  let query = supabase
    .from("historical_bars")
    .select("*")
    .eq("symbol", symbol)
    .order("timestamp", { ascending: true });

  if (limitBars) {
    query = query.limit(limitBars);
  }

  const { data: realBars, error: fetchError } = await query;

  if (fetchError) {
    await log("ERROR", `Failed to fetch real bars for ${symbol}`, { error: fetchError.message });
    throw fetchError;
  }

  if (!realBars || realBars.length === 0) {
    await log("WARN", `No real data found for ${symbol}. Fetch market data first.`, { symbol });
    return { bars: 0, indicators: 0, trajectories: 0 };
  }

  await log("INFO", `Found ${realBars.length} real bars for ${symbol}`, { symbol, count: realBars.length });

  // Transform to match expected format
  const formattedBars = realBars.map(bar => ({
    open: Number(bar.open),
    high: Number(bar.high),
    low: Number(bar.low),
    close: Number(bar.close),
    volume: Number(bar.volume),
    timestamp: new Date(bar.timestamp),
    regime: "REAL_MARKET" // Mark as real data
  }));

  // Calculate indicators on real data
  const indicators = calculateTechnicalIndicators(formattedBars);

  // Generate expert trajectories from real bars + indicators
  const trajectories = generateExpertTrajectories(symbol, formattedBars, indicators);

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
    await supabase.from("technical_indicators").upsert(batch, { onConflict: "symbol,timeframe,timestamp" });
  }

  // Insert trajectories
  for (let i = 0; i < trajectories.length; i += 100) {
    const batch = trajectories.slice(i, i + 100);
    await supabase.from("expert_trajectories").insert(batch);
  }

  // Update symbol metadata
  await supabase.from("symbols").upsert({
    symbol,
    is_active: true,
    last_fetched: new Date().toISOString(),
  });

  await log("INFO", `Generated ${indicators.length} indicators and ${trajectories.length} trajectories from real data`, {
    symbol,
    indicators: indicators.length,
    trajectories: trajectories.length,
  });

  return {
    bars: realBars.length,
    indicators: indicators.length,
    trajectories: trajectories.length,
  };
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
  
  // Generate realistic bars with different market regimes
  const startTime = new Date();
  startTime.setDate(startTime.getDate() - 7); // Start 7 days ago
  
  const bars = generateRealisticBars(basePrice, numBars, startTime);
  
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
    const { symbols = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN"], bars_per_symbol = 500, use_real_data = false } = await req.json().catch(() => ({}));
    
    await log("INFO", `🚀 Auto data generator started for ${symbols.length} symbols`);
    
    const results = [];
    
    for (const symbol of symbols) {
      if (use_real_data) {
        await log("INFO", `Using real market data for ${symbol}`, { symbol, use_real_data });
        const result = await generateDataFromRealBars(symbol, bars_per_symbol);
        results.push({ symbol, ...result });
      } else {
        const result = await generateDataForSymbol(symbol, bars_per_symbol);
        results.push({ symbol, ...result });
      }
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
