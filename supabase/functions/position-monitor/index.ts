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

async function fetchCurrentPrice(symbol: string) {
  try {
    const { data } = await supabase.functions.invoke("fetch-market-data", {
      body: { symbol, timeframe: "5m", limit: 1 },
    });

    if (data?.bars && data.bars.length > 0) {
      return data.bars[0].close;
    }
    return null;
  } catch {
    return null;
  }
}

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    // Fetch all open positions
    const { data: positions } = await supabase
      .from("positions")
      .select("*");

    if (!positions || positions.length === 0) {
      return new Response(
        JSON.stringify({ closed: 0, message: "No positions to monitor" }),
        { headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    const closedPositions = [];

    for (const position of positions) {
      const currentPrice = await fetchCurrentPrice(position.symbol);
      if (!currentPrice) continue;

      // Update current price and PnL
      const pnl =
        position.side === "long"
          ? (currentPrice - position.entry_price) * position.size
          : (position.entry_price - currentPrice) * position.size;

      const pnlPercent = ((currentPrice - position.entry_price) / position.entry_price) * 100;

      // Get position metadata for ATR-based trailing stop
      const metadata = position.metadata || {};
      const atr = metadata.atr || 0;
      const initialStopDistance = position.stop_loss 
        ? Math.abs(position.entry_price - position.stop_loss)
        : 0;
      
      // Calculate ATR-based trailing stop
      let newStopLoss = position.stop_loss;
      if (atr > 0 && initialStopDistance > 0) {
        const trailingDistance = atr * 2; // 2x ATR trailing
        
        if (position.side === "long" && pnl > initialStopDistance) {
          // Move stop up as price increases (only if in profit)
          const potentialStop = currentPrice - trailingDistance;
          if (!newStopLoss || potentialStop > newStopLoss) {
            newStopLoss = potentialStop;
            console.log(`📈 Trailing stop for ${position.symbol}: ${newStopLoss.toFixed(2)}`);
          }
        } else if (position.side === "short" && pnl > initialStopDistance) {
          // Move stop down as price decreases (only if in profit)
          const potentialStop = currentPrice + trailingDistance;
          if (!newStopLoss || potentialStop < newStopLoss) {
            newStopLoss = potentialStop;
            console.log(`📉 Trailing stop for ${position.symbol}: ${newStopLoss.toFixed(2)}`);
          }
        }
      }

      await supabase
        .from("positions")
        .update({
          current_price: currentPrice,
          unrealized_pnl: pnl,
          stop_loss: newStopLoss,
          updated_at: new Date().toISOString(),
        })
        .eq("id", position.id);

      // Check exit conditions
      let shouldClose = false;
      let exitReason = "";

      // Stop Loss (including trailing stop)
      if (newStopLoss) {
        if (
          (position.side === "long" && currentPrice <= newStopLoss) ||
          (position.side === "short" && currentPrice >= newStopLoss)
        ) {
          shouldClose = true;
          exitReason = "Stop loss triggered";
        }
      }

      // Take Profit
      if (position.take_profit && !shouldClose) {
        if (
          (position.side === "long" && currentPrice >= position.take_profit) ||
          (position.side === "short" && currentPrice <= position.take_profit)
        ) {
          shouldClose = true;
          exitReason = "Take profit triggered";
        }
      }

      // Time-based stop (if position open > 4 hours and losing)
      const openedAt = new Date(position.opened_at);
      const hoursOpen = (Date.now() - openedAt.getTime()) / (1000 * 60 * 60);
      if (hoursOpen > 4 && pnl < 0 && !shouldClose) {
        shouldClose = true;
        exitReason = "Time decay - losing position";
      }

      // Time-based exit (end of day)
      const now = new Date();
      const marketClose = new Date(now);
      marketClose.setHours(15, 50, 0, 0); // 3:50 PM - close before market close

      if (now >= marketClose && !shouldClose) {
        shouldClose = true;
        exitReason = "End of day exit";
      }

      // Execute close if triggered
      if (shouldClose) {
        // Update trade record
        await supabase
          .from("trades")
          .update({
            status: "closed",
            exit_price: currentPrice,
            pnl,
            closed_at: new Date().toISOString(),
          })
          .eq("symbol", position.symbol)
          .eq("status", "open");

        // Delete position
        await supabase.from("positions").delete().eq("id", position.id);

        // Update strategy performance
        const isWinner = pnl > 0;
        const { data: trade } = await supabase
          .from("trades")
          .select("*, trading_signals(*)")
          .eq("symbol", position.symbol)
          .order("executed_at", { ascending: false })
          .limit(1)
          .single();

        if (trade?.trading_signals) {
          const strategies = JSON.parse(trade.trading_signals.source || "[]");
          for (const strategy of strategies) {
            const { data: perf } = await supabase
              .from("strategy_performance")
              .select("*")
              .eq("strategy_name", strategy)
              .single();

            if (perf) {
              const newTotal = perf.total_trades + 1;
              const newWinning = perf.winning_trades + (isWinner ? 1 : 0);
              const newLosing = perf.losing_trades + (isWinner ? 0 : 1);
              const newWinRate = newWinning / newTotal;

              await supabase
                .from("strategy_performance")
                .update({
                  total_trades: newTotal,
                  winning_trades: newWinning,
                  losing_trades: newLosing,
                  win_rate: newWinRate,
                  last_trade_at: new Date().toISOString(),
                })
                .eq("strategy_name", strategy);
            } else {
              // Insert new strategy performance
              await supabase.from("strategy_performance").insert({
                strategy_name: strategy,
                total_trades: 1,
                winning_trades: isWinner ? 1 : 0,
                losing_trades: isWinner ? 0 : 1,
                win_rate: isWinner ? 1.0 : 0.0,
                last_trade_at: new Date().toISOString(),
              });
            }
          }
        }

        closedPositions.push({
          symbol: position.symbol,
          pnl,
          pnl_percent: pnlPercent,
          reason: exitReason,
        });

        await supabase.from("system_logs").insert({
          level: "INFO",
          source: "position-monitor",
          message: `Position closed: ${position.symbol} - ${exitReason}`,
          metadata: { pnl, pnl_percent: pnlPercent },
        });
      }
    }

    return new Response(
      JSON.stringify({
        closed: closedPositions.length,
        positions: closedPositions,
      }),
      { headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  } catch (error) {
    console.error("Position monitor error:", error);
    return new Response(
      JSON.stringify({ error: error instanceof Error ? error.message : "Unknown error" }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});
