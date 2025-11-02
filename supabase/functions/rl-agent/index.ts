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

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { signal, current_positions, account_state } = await req.json();

    // Fetch strategy performance for this signal's strategies
    const { data: perfData } = await supabase
      .from("strategy_performance")
      .select("*")
      .in("strategy_name", signal.strategies || []);

    const avgWinRate =
      perfData && perfData.length > 0
        ? perfData.reduce((sum, p) => sum + (p.win_rate || 0), 0) / perfData.length
        : 0.5;

    // Build state features
    const stateFeatures = {
      signal_confidence: signal.confidence,
      rsi: signal.market_data.rsi,
      vwap_deviation: signal.market_data.vwapDeviation,
      volatility: signal.market_data.recentVolatility,
      volume_zscore: signal.market_data.volumeZScore,
      num_open_positions: current_positions.length,
      strategy_win_rate: avgWinRate,
      account_drawdown: account_state?.drawdown || 0,
      time_of_day: new Date().getHours(),
    };

    // RL Decision via AI (mock RL until real model deployed)
    const systemPrompt = `You are a reinforcement learning trading agent. 
Analyze the state and return a decision in JSON format:
{
  "action": "buy" | "sell" | "hold",
  "confidence": 0.0-1.0,
  "q_value": -1.0 to 1.0,
  "reasoning": "Brief explanation"
}`;

    const userPrompt = `State:
Signal: ${signal.action} ${signal.symbol} (confidence: ${signal.confidence.toFixed(2)})
Market: RSI ${stateFeatures.rsi.toFixed(1)}, VWAP dev ${stateFeatures.vwap_deviation.toFixed(2)}%
Volatility: ${stateFeatures.volatility.toFixed(2)}%
Open Positions: ${stateFeatures.num_open_positions}
Strategy Win Rate: ${(avgWinRate * 100).toFixed(0)}%
Drawdown: ${stateFeatures.account_drawdown.toFixed(2)}%

Decide action:`;

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
        temperature: 0.5,
      }),
    });

    if (!response.ok) {
      throw new Error(`AI error: ${response.status}`);
    }

    const data = await response.json();
    const aiResponse = data.choices[0].message.content;
    const jsonMatch = aiResponse.match(/\{[\s\S]*\}/);

    if (!jsonMatch) {
      throw new Error("No JSON in AI response");
    }

    const decision = JSON.parse(jsonMatch[0]);

    // Log decision to database
    await supabase.from("rl_decisions").insert({
      signal_id: signal.id,
      action: decision.action,
      confidence: decision.confidence,
      q_value: decision.q_value,
      reasoning: decision.reasoning,
      state_features: stateFeatures,
    });

    return new Response(
      JSON.stringify({
        ...decision,
        state_features: stateFeatures,
      }),
      { headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  } catch (error) {
    console.error("RL agent error:", error);
    // Fallback to signal's original action on error
    return new Response(
      JSON.stringify({
        action: "hold",
        confidence: 0.0,
        q_value: 0,
        reasoning: "RL agent error - defaulting to hold",
      }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});
