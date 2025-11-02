import "https://deno.land/x/xhr@0.1.0/mod.ts";
import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const LOVABLE_API_KEY = Deno.env.get("LOVABLE_API_KEY");
const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

// Create Supabase client with service role for full access
const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);

interface MarketData {
  price: number;
  rsi: number;
  atr: number;
  vwap: number;
  vwapDeviation: number;
  volumeZScore: number;
  sentiment?: number;
  recentVolatility?: number;
}

interface TradingSignal {
  symbol: string;
  action: "buy" | "sell" | "hold";
  confidence: number;
  proposed_size: number;
  source: string;
  market_data: MarketData;
}

async function log(level: "INFO" | "WARN" | "ERROR", message: string, metadata?: any) {
  console.log(`[${level}] ${message}`, metadata || "");
  await supabase.from("system_logs").insert({
    level,
    source: "autonomous-trader",
    message,
    metadata: metadata || {},
  });
}

async function fetchMarketData(symbol: string): Promise<MarketData | null> {
  try {
    // Fetch från Yahoo Finance via din fetch-market-data function
    const { data, error } = await supabase.functions.invoke("fetch-market-data", {
      body: { symbol, timeframe: "5m", limit: 50 },
    });

    if (error || !data?.bars || data.bars.length === 0) {
      await log("WARN", `No market data for ${symbol}`, { error });
      return null;
    }

    const bars = data.bars;
    const latest = bars[bars.length - 1];
    
    // Calculate mock indicators (du kan ersätta med compute-indicators function)
    const prices = bars.map((b: any) => b.close);
    const avgPrice = prices.reduce((a: number, b: number) => a + b, 0) / prices.length;
    const volatility = Math.sqrt(
      prices.reduce((sum: number, p: number) => sum + Math.pow(p - avgPrice, 2), 0) / prices.length
    );

    return {
      price: latest.close,
      rsi: 50 + (Math.random() - 0.5) * 40, // Mock RSI - ersätt med riktig
      atr: volatility,
      vwap: avgPrice,
      vwapDeviation: ((latest.close - avgPrice) / avgPrice) * 100,
      volumeZScore: (latest.volume - avgPrice) / (volatility || 1),
      recentVolatility: (volatility / latest.close) * 100,
    };
  } catch (error) {
    await log("ERROR", `Failed to fetch market data for ${symbol}`, { error });
    return null;
  }
}

async function generateSignals(): Promise<TradingSignal[]> {
  const signals: TradingSignal[] = [];
  
  // Hämta watchlist symbols
  const watchlist = ["AAPL", "TSLA", "MSFT"]; // Kan hämtas från settings/watchlist

  for (const symbol of watchlist) {
    const marketData = await fetchMarketData(symbol);
    if (!marketData) continue;

    // Enkel momentum strategy som exempel
    // Du kan ersätta detta med RL-agent eller expertstrategier
    let action: "buy" | "sell" | "hold" = "hold";
    let confidence = 0.5;

    // VWAP Mean Reversion Strategy
    if (marketData.vwapDeviation < -1.5) {
      action = "buy";
      confidence = Math.min(0.9, 0.6 + Math.abs(marketData.vwapDeviation) * 0.1);
    } else if (marketData.vwapDeviation > 1.5) {
      action = "sell";
      confidence = Math.min(0.9, 0.6 + Math.abs(marketData.vwapDeviation) * 0.1);
    }

    // RSI Divergence
    if (marketData.rsi < 30 && action !== "sell") {
      action = "buy";
      confidence = Math.max(confidence, 0.7);
    } else if (marketData.rsi > 70 && action !== "buy") {
      action = "sell";
      confidence = Math.max(confidence, 0.7);
    }

    if (action !== "hold") {
      signals.push({
        symbol,
        action,
        confidence,
        proposed_size: 50, // Start med 50% av kapital
        source: "vwap_rsi_strategy",
        market_data: marketData,
      });
    }
  }

  return signals;
}

async function assessRisk(signal: TradingSignal): Promise<any> {
  const systemPrompt = `You are a risk analyst. Assess trading risk and return ONLY valid JSON:
{
  "riskScore": 0.65,
  "reason": "Brief risk explanation",
  "factors": {
    "volatilityRisk": 0.7,
    "sentimentRisk": 0.4,
    "technicalRisk": 0.8,
    "timingRisk": 0.6,
    "positionRisk": 0.5
  }
}`;

  const userPrompt = `Assess risk for ${signal.action.toUpperCase()} ${signal.symbol}:
Price: $${signal.market_data.price.toFixed(2)}
RSI: ${signal.market_data.rsi.toFixed(1)}
ATR: ${signal.market_data.atr.toFixed(2)}
VWAP Dev: ${signal.market_data.vwapDeviation.toFixed(2)}%
Volatility: ${signal.market_data.recentVolatility?.toFixed(2)}%`;

  try {
    const response = await fetch("https://ai.gateway.lovable.dev/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${LOVABLE_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "google/gemini-2.5-flash",
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: userPrompt },
        ],
        temperature: 0.3,
      }),
    });

    if (!response.ok) throw new Error(`AI error: ${response.status}`);

    const data = await response.json();
    const aiResponse = data.choices[0].message.content;
    const jsonMatch = aiResponse.match(/\{[\s\S]*\}/);
    
    if (!jsonMatch) throw new Error("No JSON in AI response");
    
    const assessment = JSON.parse(jsonMatch[0]);
    const riskScore = Math.max(0, Math.min(1, assessment.riskScore));

    // Apply 50% threshold rule
    let adjustedSize = signal.proposed_size;
    let shouldExecute = true;
    let reason = assessment.reason;

    if (riskScore > 0.7) {
      adjustedSize = 0;
      shouldExecute = false;
      reason = `🛑 VETO: ${assessment.reason}`;
    } else if (riskScore > 0.5) {
      const reductionFactor = (0.7 - riskScore) / 0.2;
      adjustedSize = signal.proposed_size * reductionFactor;
      reason = `⚠️ REDUCED: ${assessment.reason}`;
    } else {
      reason = `✅ APPROVED: ${assessment.reason}`;
    }

    return {
      riskScore,
      adjustedSize,
      shouldExecute,
      reason,
      factors: assessment.factors,
    };
  } catch (error) {
    await log("ERROR", "Risk assessment failed", { error, signal });
    return {
      riskScore: 1.0,
      adjustedSize: 0,
      shouldExecute: false,
      reason: "Risk assessment error - blocked for safety",
      factors: {
        volatilityRisk: 1,
        sentimentRisk: 1,
        technicalRisk: 1,
        timingRisk: 1,
        positionRisk: 1,
      },
    };
  }
}

async function executeTrade(signal: TradingSignal, riskAssessment: any) {
  try {
    // Insert signal
    const { data: signalData, error: signalError } = await supabase
      .from("trading_signals")
      .insert({
        symbol: signal.symbol,
        action: signal.action,
        confidence: signal.confidence,
        proposed_size: signal.proposed_size,
        source: signal.source,
        market_data: signal.market_data,
      })
      .select()
      .single();

    if (signalError) throw signalError;

    // Insert risk assessment
    const { data: assessmentData, error: assessmentError } = await supabase
      .from("risk_assessments")
      .insert({
        signal_id: signalData.id,
        risk_score: riskAssessment.riskScore,
        adjusted_size: riskAssessment.adjustedSize,
        should_execute: riskAssessment.shouldExecute,
        reason: riskAssessment.reason,
        factors: riskAssessment.factors,
      })
      .select()
      .single();

    if (assessmentError) throw assessmentError;

    // If approved, execute trade
    if (riskAssessment.shouldExecute) {
      const { error: tradeError } = await supabase.from("trades").insert({
        signal_id: signalData.id,
        risk_assessment_id: assessmentData.id,
        symbol: signal.symbol,
        action: signal.action,
        size: riskAssessment.adjustedSize,
        entry_price: signal.market_data.price,
        status: "open",
      });

      if (tradeError) throw tradeError;

      // Update or create position
      const { data: existingPosition } = await supabase
        .from("positions")
        .select()
        .eq("symbol", signal.symbol)
        .single();

      if (existingPosition) {
        // Update existing position
        await supabase
          .from("positions")
          .update({
            size: riskAssessment.adjustedSize,
            entry_price: signal.market_data.price,
            current_price: signal.market_data.price,
            side: signal.action === "buy" ? "long" : "short",
          })
          .eq("symbol", signal.symbol);
      } else {
        // Create new position
        await supabase.from("positions").insert({
          symbol: signal.symbol,
          side: signal.action === "buy" ? "long" : "short",
          size: riskAssessment.adjustedSize,
          entry_price: signal.market_data.price,
          current_price: signal.market_data.price,
        });
      }

      await log("INFO", `✅ Trade executed: ${signal.action.toUpperCase()} ${signal.symbol}`, {
        size: riskAssessment.adjustedSize,
        price: signal.market_data.price,
        risk: riskAssessment.riskScore,
      });
    } else {
      await log("WARN", `🛑 Trade blocked: ${signal.symbol}`, {
        reason: riskAssessment.reason,
        risk: riskAssessment.riskScore,
      });
    }
  } catch (error) {
    await log("ERROR", `Failed to execute trade for ${signal.symbol}`, { error });
  }
}

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    await log("INFO", "🤖 Autonomous trader started");

    // 1. Generate trading signals
    const signals = await generateSignals();
    await log("INFO", `Generated ${signals.length} signals`);

    // 2. Process each signal
    for (const signal of signals) {
      await log("INFO", `Processing signal: ${signal.action} ${signal.symbol}`, {
        confidence: signal.confidence,
        size: signal.proposed_size,
      });

      // 3. Assess risk
      const riskAssessment = await assessRisk(signal);
      
      // 4. Execute or veto
      await executeTrade(signal, riskAssessment);
    }

    await log("INFO", "✅ Autonomous trader completed");

    return new Response(
      JSON.stringify({
        success: true,
        signals_processed: signals.length,
        timestamp: new Date().toISOString(),
      }),
      { headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  } catch (error) {
    await log("ERROR", "Autonomous trader failed", { error });
    return new Response(
      JSON.stringify({ error: error instanceof Error ? error.message : "Unknown error" }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});
