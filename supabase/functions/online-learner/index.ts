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

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    // Fetch recent closed trades for learning
    const oneDayAgo = new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString();

    const { data: recentTrades } = await supabase
      .from("trades")
      .select("*, trading_signals(*), rl_decisions(*)")
      .eq("status", "closed")
      .gte("closed_at", oneDayAgo)
      .order("closed_at", { ascending: false })
      .limit(100);

    if (!recentTrades || recentTrades.length === 0) {
      return new Response(
        JSON.stringify({ message: "No recent trades to learn from" }),
        { headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    // Calculate learning metrics
    const totalTrades = recentTrades.length;
    const winningTrades = recentTrades.filter((t) => t.pnl && t.pnl > 0).length;
    const losingTrades = totalTrades - winningTrades;
    const winRate = winningTrades / totalTrades;

    const avgWinSize = recentTrades
      .filter((t) => t.pnl && t.pnl > 0)
      .reduce((sum, t) => sum + t.pnl, 0) / (winningTrades || 1);

    const avgLossSize = Math.abs(
      recentTrades
        .filter((t) => t.pnl && t.pnl < 0)
        .reduce((sum, t) => sum + t.pnl, 0) / (losingTrades || 1)
    );

    const profitFactor = avgWinSize / (avgLossSize || 1);

    // Analyze RL decisions accuracy
    const correctRLDecisions = recentTrades.filter((t) => {
      if (!t.rl_decisions || t.rl_decisions.length === 0) return false;
      const rlDecision = t.rl_decisions[0];
      const wasProfit = t.pnl && t.pnl > 0;
      return (
        (rlDecision.action === "buy" && wasProfit) ||
        (rlDecision.action === "sell" && !wasProfit)
      );
    }).length;

    const rlAccuracy = correctRLDecisions / totalTrades;

    // Mock learning update (in real system, would update RL model weights)
    const learningRate = 0.001;
    const loss = 1 - winRate; // Simple loss function

    // Log learning results
    await supabase.from("online_learning").insert({
      model_type: "rl_policy",
      learning_rate: learningRate,
      samples_processed: totalTrades,
      loss,
      metrics: {
        win_rate: winRate,
        profit_factor: profitFactor,
        rl_accuracy: rlAccuracy,
        avg_win_size: avgWinSize,
        avg_loss_size: avgLossSize,
      },
    });

    await supabase.from("system_logs").insert({
      level: "INFO",
      source: "online-learner",
      message: `Online learning completed: ${totalTrades} samples, WR: ${(winRate * 100).toFixed(1)}%`,
      metadata: {
        win_rate: winRate,
        profit_factor: profitFactor,
        rl_accuracy: rlAccuracy,
      },
    });

    return new Response(
      JSON.stringify({
        samples_processed: totalTrades,
        win_rate: winRate,
        profit_factor: profitFactor,
        rl_accuracy: rlAccuracy,
        loss,
      }),
      { headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  } catch (error) {
    console.error("Online learner error:", error);
    return new Response(
      JSON.stringify({ error: error instanceof Error ? error.message : "Unknown error" }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});
