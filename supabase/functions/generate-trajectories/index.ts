import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.78.0";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

// Expert Strategy implementations
class ExpertStrategy {
  name: string;
  
  constructor(name: string) {
    this.name = name;
  }

  // Returns action: -1 (sell), 0 (hold), 1 (buy), and entry_quality: 0-1
  evaluate(features: any, newsFeatures: any[], previousFeatures: any[] = []): { action: number; entry_quality: number; reason: string } {
    return { action: 0, entry_quality: 0, reason: "Not implemented" };
  }
}

class VWAPReversionStrategy extends ExpertStrategy {
  constructor() {
    super("VWAP_REVERSION");
  }

  override evaluate(features: any, newsFeatures: any[] = [], previousFeatures: any[] = []): { action: number; entry_quality: number; reason: string } {
    const { vwap_distance_pct, atr_14, close, rsi_14 } = features;
    
    if (!vwap_distance_pct || !atr_14 || !rsi_14) {
      return { action: 0, entry_quality: 0, reason: "Insufficient data" };
    }

    const k = 1.5; // ATR multiplier
    const atr_pct = (atr_14 / close) * 100;
    const threshold = k * atr_pct;

    // Long when price <= VWAP - k*ATR and RSI < 40 (oversold)
    if (vwap_distance_pct <= -threshold && rsi_14 < 40) {
      const quality = Math.min(1, Math.abs(vwap_distance_pct) / (threshold * 2));
      return { action: 1, entry_quality: quality, reason: "Oversold below VWAP" };
    }

    // Short when price >= VWAP + k*ATR and RSI > 60 (overbought)
    if (vwap_distance_pct >= threshold && rsi_14 > 60) {
      const quality = Math.min(1, vwap_distance_pct / (threshold * 2));
      return { action: -1, entry_quality: quality, reason: "Overbought above VWAP" };
    }

    return { action: 0, entry_quality: 0, reason: "No setup" };
  }
}

class NewsMomentumStrategy extends ExpertStrategy {
  constructor() {
    super("NEWS_MOMENTUM");
  }

  override evaluate(features: any, newsFeatures: any[]): { action: number; entry_quality: number; reason: string } {
    if (!newsFeatures || newsFeatures.length === 0) {
      return { action: 0, entry_quality: 0, reason: "No news" };
    }

    // Get most recent relevant news (within last 60 minutes)
    const recentNews = newsFeatures.filter(n => 
      n.freshness_minutes < 60 && n.relevance_score > 0.7
    );

    if (recentNews.length === 0) {
      return { action: 0, entry_quality: 0, reason: "No recent relevant news" };
    }

    const news = recentNews[0];
    const { volume_zscore } = features;

    // Positive sentiment + volume spike → long
    if (news.sentiment > 0.3 && volume_zscore > 2) {
      const quality = Math.min(1, (news.sentiment + news.surprise_score) / 2);
      return { action: 1, entry_quality: quality, reason: "Positive news momentum" };
    }

    // Negative sentiment + volume spike → short
    if (news.sentiment < -0.3 && volume_zscore > 2) {
      const quality = Math.min(1, (Math.abs(news.sentiment) + news.surprise_score) / 2);
      return { action: -1, entry_quality: quality, reason: "Negative news momentum" };
    }

    return { action: 0, entry_quality: 0, reason: "No strong news signal" };
  }
}

class TrendPullbackStrategy extends ExpertStrategy {
  constructor() {
    super("TREND_PULLBACK");
  }

  override evaluate(features: any, newsFeatures: any[] = [], previousFeatures: any[] = []): { action: number; entry_quality: number; reason: string } {
    const { close, ema_20, ema_50 } = features;
    
    if (!ema_20 || !ema_50) {
      return { action: 0, entry_quality: 0, reason: "Insufficient data" };
    }

    // Uptrend: price > EMA50, pullback to EMA20 → long
    if (close > ema_50 && close <= ema_20 * 1.01 && close >= ema_20 * 0.99) {
      const trendStrength = ((close - ema_50) / ema_50) * 100;
      const quality = Math.min(1, trendStrength / 5); // 5% move = full quality
      return { action: 1, entry_quality: quality, reason: "Uptrend pullback to EMA20" };
    }

    // Downtrend: price < EMA50, bounce to EMA20 → short
    if (close < ema_50 && close >= ema_20 * 0.99 && close <= ema_20 * 1.01) {
      const trendStrength = ((ema_50 - close) / ema_50) * 100;
      const quality = Math.min(1, trendStrength / 5);
      return { action: -1, entry_quality: quality, reason: "Downtrend bounce to EMA20" };
    }

    return { action: 0, entry_quality: 0, reason: "No pullback setup" };
  }
}

class RSIDivergenceStrategy extends ExpertStrategy {
  constructor() {
    super("RSI_DIVERGENCE");
  }

  override evaluate(features: any, newsFeatures: any[], previousFeatures: any[]): { action: number; entry_quality: number; reason: string } {
    if (previousFeatures.length < 5) {
      return { action: 0, entry_quality: 0, reason: "Insufficient history" };
    }

    const { close, rsi_14, volume_zscore } = features;
    const prev = previousFeatures[previousFeatures.length - 1];
    
    if (!rsi_14 || !prev.rsi_14) {
      return { action: 0, entry_quality: 0, reason: "Missing RSI data" };
    }

    // Bullish divergence: price lower low, RSI higher low
    if (close < prev.close && rsi_14 > prev.rsi_14 && rsi_14 < 40 && volume_zscore > 1) {
      const quality = Math.min(1, (40 - rsi_14) / 20);
      return { action: 1, entry_quality: quality, reason: "Bullish RSI divergence" };
    }

    // Bearish divergence: price higher high, RSI lower high
    if (close > prev.close && rsi_14 < prev.rsi_14 && rsi_14 > 60 && volume_zscore > 1) {
      const quality = Math.min(1, (rsi_14 - 60) / 20);
      return { action: -1, entry_quality: quality, reason: "Bearish RSI divergence" };
    }

    return { action: 0, entry_quality: 0, reason: "No divergence" };
  }
}

class LiquiditySweepStrategy extends ExpertStrategy {
  constructor() {
    super("LIQUIDITY_SWEEP");
  }

  override evaluate(features: any, newsFeatures: any[], previousFeatures: any[]): { action: number; entry_quality: number; reason: string } {
    if (previousFeatures.length < 3) {
      return { action: 0, entry_quality: 0, reason: "Insufficient history" };
    }

    const { high, low, close, volume_zscore, range_pct } = features;
    const prev = previousFeatures[previousFeatures.length - 1];
    
    // Detect sweep: break previous high/low with volume spike then immediate reversal
    const brokeHigh = high > prev.high && close < (high + prev.high) / 2;
    const brokeLow = low < prev.low && close > (low + prev.low) / 2;
    
    // High sweep (stop hunt above) → fade short
    if (brokeHigh && volume_zscore > 2 && range_pct > 1) {
      const quality = Math.min(1, volume_zscore / 4);
      return { action: -1, entry_quality: quality, reason: "High sweep fade" };
    }

    // Low sweep (stop hunt below) → fade long
    if (brokeLow && volume_zscore > 2 && range_pct > 1) {
      const quality = Math.min(1, volume_zscore / 4);
      return { action: 1, entry_quality: quality, reason: "Low sweep fade" };
    }

    return { action: 0, entry_quality: 0, reason: "No sweep detected" };
  }
}

class VolumeSpikereversalStrategy extends ExpertStrategy {
  constructor() {
    super("VOLUME_SPIKE_REVERSAL");
  }

  override evaluate(features: any, newsFeatures: any[], previousFeatures: any[]): { action: number; entry_quality: number; reason: string } {
    const { volume_zscore, close, rsi_14, range_pct } = features;
    
    if (volume_zscore < 3) {
      return { action: 0, entry_quality: 0, reason: "No volume spike" };
    }

    if (previousFeatures.length < 2) {
      return { action: 0, entry_quality: 0, reason: "Insufficient history" };
    }

    const prev = previousFeatures[previousFeatures.length - 1];
    const direction = close > prev.close ? 1 : -1;

    // Volume exhaustion reversal
    if (direction === 1 && rsi_14 > 70 && range_pct > 2) {
      const quality = Math.min(1, (volume_zscore - 3) / 3);
      return { action: -1, entry_quality: quality, reason: "Bullish exhaustion reversal" };
    }

    if (direction === -1 && rsi_14 < 30 && range_pct > 2) {
      const quality = Math.min(1, (volume_zscore - 3) / 3);
      return { action: 1, entry_quality: quality, reason: "Bearish exhaustion reversal" };
    }

    return { action: 0, entry_quality: 0, reason: "No exhaustion signal" };
  }
}

class VWAPDeltaStrategy extends ExpertStrategy {
  constructor() {
    super("VWAP_DELTA_CONFLUENCE");
  }

  override evaluate(features: any, newsFeatures: any[] = [], previousFeatures: any[] = []): { action: number; entry_quality: number; reason: string } {
    const { close, vwap, vwap_distance_pct, volume_zscore } = features;
    
    if (!vwap || !vwap_distance_pct) {
      return { action: 0, entry_quality: 0, reason: "Missing VWAP data" };
    }

    // Long: price above VWAP + positive volume (delta proxy)
    if (close > vwap && vwap_distance_pct > 0.1 && volume_zscore > 1) {
      const quality = Math.min(1, (vwap_distance_pct + volume_zscore) / 4);
      return { action: 1, entry_quality: quality, reason: "VWAP + positive delta" };
    }

    // Short: price below VWAP + negative volume (delta proxy)
    if (close < vwap && vwap_distance_pct < -0.1 && volume_zscore > 1) {
      const quality = Math.min(1, (Math.abs(vwap_distance_pct) + volume_zscore) / 4);
      return { action: -1, entry_quality: quality, reason: "VWAP + negative delta" };
    }

    return { action: 0, entry_quality: 0, reason: "No VWAP confluence" };
  }
}

class AfternoonFadeStrategy extends ExpertStrategy {
  constructor() {
    super("AFTERNOON_FADE");
  }

  override evaluate(features: any, newsFeatures: any[], previousFeatures: any[]): { action: number; entry_quality: number; reason: string } {
    const { timestamp, close, intraday_position, rsi_14 } = features;
    
    // Check if late session (14:00-16:00 ET would be around hour 19-21 UTC for US markets)
    const hour = new Date(timestamp).getUTCHours();
    if (hour < 19 || hour > 21) {
      return { action: 0, entry_quality: 0, reason: "Not afternoon session" };
    }

    if (previousFeatures.length < 5) {
      return { action: 0, entry_quality: 0, reason: "Insufficient history" };
    }

    // Fade from extremes when momentum weakens
    if (intraday_position > 0.8 && rsi_14 > 65) {
      const quality = Math.min(1, (intraday_position - 0.8) * 5);
      return { action: -1, entry_quality: quality, reason: "Afternoon fade from high" };
    }

    if (intraday_position < 0.2 && rsi_14 < 35) {
      const quality = Math.min(1, (0.2 - intraday_position) * 5);
      return { action: 1, entry_quality: quality, reason: "Afternoon fade from low" };
    }

    return { action: 0, entry_quality: 0, reason: "No fade setup" };
  }
}

class GapFillStrategy extends ExpertStrategy {
  constructor() {
    super("GAP_FILL");
  }

  override evaluate(features: any, newsFeatures: any[], previousFeatures: any[]): { action: number; entry_quality: number; reason: string } {
    if (previousFeatures.length < 1) {
      return { action: 0, entry_quality: 0, reason: "No previous bar" };
    }

    const { open, close, volume_zscore } = features;
    const prevClose = previousFeatures[previousFeatures.length - 1].close;
    
    const gapPct = ((open - prevClose) / prevClose) * 100;
    
    // Gap up > 1% → fade towards previous close
    if (gapPct > 1 && volume_zscore < 2) {
      const quality = Math.min(1, Math.abs(gapPct) / 3);
      return { action: -1, entry_quality: quality, reason: "Gap up fill" };
    }

    // Gap down > 1% → long towards previous close
    if (gapPct < -1 && volume_zscore < 2) {
      const quality = Math.min(1, Math.abs(gapPct) / 3);
      return { action: 1, entry_quality: quality, reason: "Gap down fill" };
    }

    return { action: 0, entry_quality: 0, reason: "No significant gap" };
  }
}

class OpeningRangeBreakoutStrategy extends ExpertStrategy {
  constructor() {
    super("OPENING_RANGE_BREAKOUT");
  }

  override evaluate(features: any, newsFeatures: any[], previousFeatures: any[]): { action: number; entry_quality: number; reason: string } {
    const { timestamp, high, low, close, volume_zscore } = features;
    
    // Check if within first 30 minutes of session (9:30-10:00 ET → ~14:30-15:00 UTC)
    const hour = new Date(timestamp).getUTCHours();
    const minute = new Date(timestamp).getUTCMinutes();
    
    if (hour !== 14 || minute > 30) {
      return { action: 0, entry_quality: 0, reason: "Not opening range period" };
    }

    if (previousFeatures.length < 6) {
      return { action: 0, entry_quality: 0, reason: "Insufficient opening bars" };
    }

    // Calculate opening range from first bars
    const openingBars = previousFeatures.slice(-6);
    const orHigh = Math.max(...openingBars.map((b: any) => b.high));
    const orLow = Math.min(...openingBars.map((b: any) => b.low));
    
    // Breakout above range
    if (close > orHigh && volume_zscore > 1.5) {
      const quality = Math.min(1, volume_zscore / 3);
      return { action: 1, entry_quality: quality, reason: "ORB breakout high" };
    }

    // Breakdown below range
    if (close < orLow && volume_zscore > 1.5) {
      const quality = Math.min(1, volume_zscore / 3);
      return { action: -1, entry_quality: quality, reason: "ORB breakdown low" };
    }

    return { action: 0, entry_quality: 0, reason: "No ORB signal" };
  }
}

const STRATEGIES = [
  new VWAPReversionStrategy(),
  new NewsMomentumStrategy(),
  new TrendPullbackStrategy(),
  new RSIDivergenceStrategy(),
  new LiquiditySweepStrategy(),
  new VolumeSpikereversalStrategy(),
  new VWAPDeltaStrategy(),
  new AfternoonFadeStrategy(),
  new GapFillStrategy(),
  new OpeningRangeBreakoutStrategy(),
];

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { symbol, timeframe = "5m", start_date, end_date } = await req.json();

    if (!symbol) {
      return new Response(JSON.stringify({ error: "Symbol is required" }), {
        status: 400,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    console.log(`Generating trajectories for ${symbol} ${timeframe}`);

    const supabaseAdmin = createClient(
      Deno.env.get("SUPABASE_URL") ?? "",
      Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "",
    );

    // Fetch technical indicators
    let query = supabaseAdmin
      .from("technical_indicators")
      .select("*")
      .eq("symbol", symbol)
      .eq("timeframe", timeframe)
      .order("timestamp", { ascending: true });

    if (start_date) query = query.gte("timestamp", start_date);
    if (end_date) query = query.lte("timestamp", end_date);

    const { data: indicators, error: indError } = await query;

    if (indError || !indicators || indicators.length === 0) {
      console.error("Error fetching indicators:", indError);
      return new Response(
        JSON.stringify({ error: "No indicators available. Run compute-indicators first." }),
        { status: 404, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    // Fetch news features
    const { data: newsData } = await supabaseAdmin
      .from("news_features")
      .select("*")
      .eq("symbol", symbol)
      .order("timestamp", { ascending: false });

    const trajectories = [];
    const fees_per_trade = 1.0; // $1 per trade
    const slippage_bps = 5; // 5 basis points

    // Frame stacking config
    const FRAME_STACK_SIZE = 32;
    
    // Detect regime for each bar
    const detectRegime = (features: any, prevBars: any[]): string => {
      if (prevBars.length < 20) return "unknown";
      
      const { ema_20, ema_50, atr_14, close } = features;
      const vol_recent = prevBars.slice(-10).reduce((sum, b) => sum + (b.atr_14 || 0), 0) / 10;
      const vol_older = prevBars.slice(-20, -10).reduce((sum, b) => sum + (b.atr_14 || 0), 0) / 10;
      
      // High volatility regime
      if (vol_recent > vol_older * 1.5) return "high_vol";
      
      // Trending regime
      if (ema_20 && ema_50) {
        const trend_strength = Math.abs((ema_20 - ema_50) / ema_50) * 100;
        if (trend_strength > 2) return "trend";
      }
      
      // Sideways regime
      return "sideways";
    };

    // Calculate drawdown penalty for risk adjustment
    const calculateDrawdown = (prevBars: any[], currentEquity: number): number => {
      if (prevBars.length < 5) return 0;
      const recentPrices = prevBars.slice(-20).map(b => b.close);
      const peak = Math.max(...recentPrices);
      const drawdown = ((peak - currentEquity) / peak) * 100;
      return Math.max(0, drawdown);
    };

    // Generate trajectories for each timestamp
    const lambda_risk = 0.2; // Risk penalty coefficient
    
    for (let i = FRAME_STACK_SIZE; i < indicators.length; i++) {
      const features = indicators[i];
      const timestamp = new Date(features.timestamp);
      
      // Get frame stack (last 32 bars including current)
      const frameStack = indicators.slice(i - FRAME_STACK_SIZE, i + 1);
      
      // Get relevant news (within 2 hours before this timestamp)
      const relevantNews = newsData?.filter(n => {
        const newsTime = new Date(n.timestamp);
        const diffMinutes = (timestamp.getTime() - newsTime.getTime()) / 60000;
        return diffMinutes >= 0 && diffMinutes <= 120;
      }) || [];

      // Get previous features for strategies that need history
      const previousFeatures = indicators.slice(Math.max(0, i - 10), i);
      
      // Detect regime
      const regime = detectRegime(features, previousFeatures);

      // Build observation with frame stack
      const obs_features = {
        current: {
          ...features,
          news_sentiment: relevantNews[0]?.sentiment || 0,
          news_surprise: relevantNews[0]?.surprise_score || 0,
          news_relevance: relevantNews[0]?.relevance_score || 0,
          news_freshness: relevantNews[0]?.freshness_minutes || 999,
          time_sin: Math.sin(2 * Math.PI * new Date(features.timestamp).getHours() / 24),
          time_cos: Math.cos(2 * Math.PI * new Date(features.timestamp).getHours() / 24),
        },
        frame_stack: frameStack.map(f => ({
          close: f.close,
          volume: f.volume,
          rsi_14: f.rsi_14,
          atr_14: f.atr_14,
          vwap_distance_pct: f.vwap_distance_pct,
          volume_zscore: f.volume_zscore,
        })),
      };

      // Run all strategies
      const strategySignals: Array<{strategy: string; action: number; quality: number; reason: string}> = [];
      
      for (const strategy of STRATEGIES) {
        const result = strategy.evaluate(features, relevantNews, previousFeatures);
        
        if (result.action !== 0 && result.entry_quality > 0.3) {
          strategySignals.push({
            strategy: strategy.name,
            action: result.action,
            quality: result.entry_quality,
            reason: result.reason,
          });
        }
      }

      // Generate trajectories for signals AND counterfactual HOLDs
      if (strategySignals.length > 0) {
        // Process signals
        for (const signal of strategySignals) {
          const close = Number(features.close);
          const atr = Number(features.atr_14);
          
          // Skip if invalid data
          if (!isFinite(close) || !isFinite(atr) || close <= 0 || atr <= 0) {
            console.warn(`Skipping trajectory due to invalid data: close=${close}, atr=${atr}`);
            continue;
          }
          
          const target_r = 1.5; // 1.5R target
          
          // Simulate trade outcome based on next K bars (simplified: use ATR)
          const delta_equity = signal.action * atr * target_r;
          const slippage = close * (slippage_bps / 10000);
          
          // Calculate drawdown penalty
          const drawdown = calculateDrawdown(previousFeatures, close);
          const drawdown_penalty = lambda_risk * drawdown;
          
          // Final reward with risk adjustment
          const reward = delta_equity - fees_per_trade - slippage - drawdown_penalty;
          
          // Validate reward is finite
          if (!isFinite(reward)) {
            console.warn(`Skipping trajectory due to invalid reward: ${reward}`);
            continue;
          }
          
          // Get next observation (if available)
          const next_obs = i + 1 < indicators.length ? {
            ...indicators[i + 1],
            news_sentiment: relevantNews[0]?.sentiment || 0,
            news_surprise: relevantNews[0]?.surprise_score || 0,
            news_relevance: relevantNews[0]?.relevance_score || 0,
            news_freshness: relevantNews[0]?.freshness_minutes || 999,
          } : null;

          trajectories.push({
            symbol,
            timeframe,
            tactic_id: signal.strategy,
            timestamp: features.timestamp,
            obs_features,
            action: signal.action,
            reward,
            delta_equity,
            fees: fees_per_trade,
            slippage,
            entry_quality: signal.quality,
            rr_ratio: target_r,
            regime_tag: regime,
          });
        }
      } else {
        // Counterfactual HOLD: Check if holding was a good decision
        // Look ahead 5 bars to see if any significant move occurred
        const lookAhead = 5;
        if (i + lookAhead < indicators.length) {
          const futurePrices = indicators.slice(i + 1, i + 1 + lookAhead).map(b => b.close);
          const currentPrice = Number(features.close);
          const atr = Number(features.atr_14);
          
          const maxMove = Math.max(...futurePrices.map(p => Math.abs(p - currentPrice) / atr));
          
          // Good skip: no significant move (< 1R)
          // Missed winner: significant move (> 1.5R)
          const isGoodSkip = maxMove < 1.0;
          const isMissedWinner = maxMove > 1.5;
          
          // Small shaping reward for good skips, small penalty for missed winners
          let hold_reward = 0;
          if (isGoodSkip) {
            hold_reward = 0.1 * atr; // Small positive for avoiding bad trade
          } else if (isMissedWinner) {
            hold_reward = -0.05 * atr; // Small penalty (much less than actual loss)
          }
          
          trajectories.push({
            symbol,
            timeframe,
            tactic_id: "HOLD",
            timestamp: features.timestamp,
            obs_features,
            action: 0,
            reward: hold_reward,
            delta_equity: 0,
            fees: 0,
            slippage: 0,
            entry_quality: isGoodSkip ? 0.8 : (isMissedWinner ? 0.2 : 0.5),
            rr_ratio: 0,
            regime_tag: regime,
          });
        }
      }
    }

    // Store trajectories
    if (trajectories.length > 0) {
      const { error: insertError } = await supabaseAdmin
        .from("expert_trajectories")
        .insert(trajectories);

      if (insertError) {
        console.error("Error storing trajectories:", insertError);
        throw new Error("Failed to store trajectories");
      }
    }

    console.log(`Generated ${trajectories.length} trajectories`);

    return new Response(
      JSON.stringify({
        success: true,
        symbol,
        timeframe,
        trajectories_count: trajectories.length,
        strategies_used: STRATEGIES.length,
      }),
      { headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  } catch (error) {
    console.error("Error in generate-trajectories:", error);
    return new Response(
      JSON.stringify({ error: error instanceof Error ? error.message : "Unknown error" }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});
