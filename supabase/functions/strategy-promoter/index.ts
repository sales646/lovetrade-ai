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
    // Fetch all strategy performance
    const { data: strategies } = await supabase
      .from("strategy_performance")
      .select("*")
      .order("win_rate", { ascending: false });

    if (!strategies || strategies.length === 0) {
      return new Response(
        JSON.stringify({ message: "No strategies to evaluate" }),
        { headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    // Calculate composite score for each strategy
    const scoredStrategies = strategies.map((strategy) => {
      const minTrades = 10;
      const tradeReliability = Math.min(1, strategy.total_trades / minTrades);

      // Composite score: win_rate * profit_factor * trade_reliability
      const score =
        (strategy.win_rate || 0) *
        (strategy.profit_factor || 1) *
        tradeReliability;

      return {
        ...strategy,
        score,
        trade_reliability: tradeReliability,
      };
    }).sort((a, b) => b.score - a.score);

    // Promote top 5, demote bottom performers
    const topPerformers = scoredStrategies.slice(0, 5);
    const poorPerformers = scoredStrategies.filter(
      (s) => s.win_rate < 0.4 && s.total_trades >= 10
    );

    // Update active status
    for (const strategy of topPerformers) {
      if (!strategy.is_active) {
        await supabase
          .from("strategy_performance")
          .update({ is_active: true })
          .eq("id", strategy.id);

        await supabase.from("system_logs").insert({
          level: "INFO",
          source: "strategy-promoter",
          message: `✅ Strategy promoted: ${strategy.strategy_name}`,
          metadata: {
            win_rate: strategy.win_rate,
            score: strategy.score,
          },
        });
      }
    }

    for (const strategy of poorPerformers) {
      if (strategy.is_active) {
        await supabase
          .from("strategy_performance")
          .update({ is_active: false })
          .eq("id", strategy.id);

        await supabase.from("system_logs").insert({
          level: "WARN",
          source: "strategy-promoter",
          message: `❌ Strategy demoted: ${strategy.strategy_name}`,
          metadata: {
            win_rate: strategy.win_rate,
            total_trades: strategy.total_trades,
          },
        });
      }
    }

    return new Response(
      JSON.stringify({
        total_strategies: strategies.length,
        promoted: topPerformers.length,
        demoted: poorPerformers.length,
        top_performers: topPerformers.slice(0, 3).map((s) => ({
          name: s.strategy_name,
          win_rate: s.win_rate,
          score: s.score,
        })),
      }),
      { headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  } catch (error) {
    console.error("Strategy promoter error:", error);
    return new Response(
      JSON.stringify({ error: error instanceof Error ? error.message : "Unknown error" }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});
