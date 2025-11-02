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
  evaluate(features: any, newsFeatures: any[]): { action: number; entry_quality: number; reason: string } {
    return { action: 0, entry_quality: 0, reason: "Not implemented" };
  }
}

class VWAPReversionStrategy extends ExpertStrategy {
  constructor() {
    super("VWAP_REVERSION");
  }

  override evaluate(features: any): { action: number; entry_quality: number; reason: string } {
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

  override evaluate(features: any): { action: number; entry_quality: number; reason: string } {
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

// Add more strategies here following the same pattern...

const STRATEGIES = [
  new VWAPReversionStrategy(),
  new NewsMomentumStrategy(),
  new TrendPullbackStrategy(),
  // TODO: Implement remaining 7 strategies
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

    // Generate trajectories for each timestamp
    for (let i = 0; i < indicators.length; i++) {
      const features = indicators[i];
      const timestamp = new Date(features.timestamp);
      
      // Get relevant news (within 2 hours before this timestamp)
      const relevantNews = newsData?.filter(n => {
        const newsTime = new Date(n.timestamp);
        const diffMinutes = (timestamp.getTime() - newsTime.getTime()) / 60000;
        return diffMinutes >= 0 && diffMinutes <= 120;
      }) || [];

      // Run all strategies
      for (const strategy of STRATEGIES) {
        const result = strategy.evaluate(features, relevantNews);
        
        if (result.action !== 0 && result.entry_quality > 0.3) {
          // Calculate reward (simplified - would need actual exit logic)
          const close = Number(features.close);
          const atr = Number(features.atr_14);
          const target_r = 1.5; // 1.5R target
          
          // Simulate trade outcome based on ATR
          const delta_equity = result.action * atr * target_r;
          const slippage = close * (slippage_bps / 10000);
          const reward = delta_equity - fees_per_trade - slippage;

          trajectories.push({
            symbol,
            timeframe,
            tactic_id: strategy.name,
            timestamp: features.timestamp,
            obs_features: {
              ...features,
              news_sentiment: relevantNews[0]?.sentiment || 0,
              news_surprise: relevantNews[0]?.surprise_score || 0,
              news_relevance: relevantNews[0]?.relevance_score || 0,
              news_freshness: relevantNews[0]?.freshness_minutes || 999,
            },
            action: result.action,
            reward,
            delta_equity,
            fees: fees_per_trade,
            slippage,
            entry_quality: result.entry_quality,
            rr_ratio: target_r,
            regime_tag: "unknown", // Would need regime detection
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
