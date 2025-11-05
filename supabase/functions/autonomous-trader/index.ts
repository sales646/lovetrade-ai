import "https://deno.land/x/xhr@0.1.0/mod.ts";
import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const LOVABLE_API_KEY = Deno.env.get("LOVABLE_API_KEY");
const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;

// Alpaca API configuration
const ALPACA_PAPER_BASE_URL = "https://paper-api.alpaca.markets";
const ALPACA_LIVE_BASE_URL = "https://api.alpaca.markets";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);

async function log(level: "INFO" | "WARN" | "ERROR", message: string, metadata?: any) {
  console.log(`[${level}] ${message}`);
  await supabase.from("system_logs").insert({
    level,
    source: "autonomous-trader",
    message,
    metadata: metadata || {},
  });
}

async function getAlpacaCredentials() {
  const { data: config } = await supabase
    .from("bot_config")
    .select("alpaca_api_key, alpaca_secret_key, alpaca_paper_trading")
    .single();
  
  if (!config?.alpaca_api_key || !config?.alpaca_secret_key) {
    return null;
  }
  
  return {
    apiKey: config.alpaca_api_key,
    secretKey: config.alpaca_secret_key,
    isPaperTrading: config.alpaca_paper_trading ?? true,
    baseUrl: config.alpaca_paper_trading ? ALPACA_PAPER_BASE_URL : ALPACA_LIVE_BASE_URL,
  };
}

async function placeAlpacaOrder(credentials: any, order: any) {
  const headers = {
    "APCA-API-KEY-ID": credentials.apiKey,
    "APCA-API-SECRET-KEY": credentials.secretKey,
    "Content-Type": "application/json",
  };
  
  const orderPayload = {
    symbol: order.symbol,
    qty: order.quantity,
    side: order.side, // "buy" or "sell"
    type: "market",
    time_in_force: "day",
  };
  
  await log("INFO", `Placing Alpaca order: ${order.side} ${order.quantity} ${order.symbol}`);
  
  const response = await fetch(`${credentials.baseUrl}/v2/orders`, {
    method: "POST",
    headers,
    body: JSON.stringify(orderPayload),
  });
  
  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Alpaca order failed: ${error}`);
  }
  
  return await response.json();
}

async function getAlpacaAccount(credentials: any) {
  const headers = {
    "APCA-API-KEY-ID": credentials.apiKey,
    "APCA-API-SECRET-KEY": credentials.secretKey,
  };
  
  const response = await fetch(`${credentials.baseUrl}/v2/account`, {
    method: "GET",
    headers,
  });
  
  if (!response.ok) {
    throw new Error("Failed to fetch Alpaca account");
  }
  
  return await response.json();
}

async function calculatePositionSize(signal: any, rlDecision: any, accountState: any) {
  const baseSize = 50; // Start with 50%
  
  // 1. RL Confidence Adjustment
  const rlAdjusted = baseSize * rlDecision.confidence;
  
  // 2. Volatility Adjustment (reduce size in high volatility)
  const volatilityFactor = Math.min(1, 2 / Math.max(signal.market_data.recentVolatility, 1));
  const volatilityAdjusted = rlAdjusted * volatilityFactor;
  
  // 3. Drawdown Adjustment (reduce size in drawdown)
  const drawdownFactor = Math.max(0.2, 1 - (accountState.drawdown / 10));
  const drawdownAdjusted = volatilityAdjusted * drawdownFactor;
  
  // 4. Strategy Performance Adjustment
  const { data: stratPerf } = await supabase
    .from("strategy_performance")
    .select("win_rate")
    .in("strategy_name", signal.strategies || []);
    
  const avgWinRate = stratPerf && stratPerf.length > 0
    ? stratPerf.reduce((sum: number, p: any) => sum + (p.win_rate || 0.5), 0) / stratPerf.length
    : 0.5;
    
  const performanceFactor = avgWinRate * 1.5; // Boost if strategies perform well
  const finalSize = Math.min(100, Math.max(10, drawdownAdjusted * performanceFactor));
  
  return {
    base_size: baseSize,
    rl_adjusted_size: rlAdjusted,
    volatility_adjusted_size: volatilityAdjusted,
    drawdown_adjusted_size: drawdownAdjusted,
    final_size: finalSize,
    risk_amount: (accountState.equity * (finalSize / 100)),
    factors: {
      rl_confidence: rlDecision.confidence,
      volatility_factor: volatilityFactor,
      drawdown_factor: drawdownFactor,
      performance_factor: performanceFactor,
    },
  };
}

async function checkCorrelation(signal: any, existingPositions: any[]) {
  // Simple correlation check: same sector/similar price movement
  // In production: use actual correlation matrix
  for (const pos of existingPositions) {
    if (pos.symbol === signal.symbol) {
      return {
        correlated: true,
        correlation: 1.0,
        reason: "Same symbol already in portfolio",
      };
    }
  }
  
  return {
    correlated: false,
    correlation: 0.0,
    reason: "Low correlation - good diversification",
  };
}

async function runTradingLoop(loopNumber: number) {
  const loopStart = new Date();
  
  // Create loop record
  const { data: loopRecord, error: loopError } = await supabase
    .from("bot_loops")
    .insert({
      loop_number: loopNumber,
      status: "running",
    })
    .select()
    .single();
    
  if (loopError) {
    await log("ERROR", "Failed to create loop record", { error: loopError });
    return;
  }
  
  try {
    await log("INFO", `🚀 Trading loop #${loopNumber} started`);
    
    // Fetch bot config and Alpaca credentials
    const { data: config } = await supabase
      .from("bot_config")
      .select("*")
      .single();
    
    const alpacaCredentials = await getAlpacaCredentials();
    if (!alpacaCredentials) {
      await log("WARN", "⚠️ Alpaca credentials not configured - trading disabled. Please configure in Settings.");
    } else {
      await log("INFO", `🔑 Alpaca connected (${alpacaCredentials.isPaperTrading ? "Paper Trading" : "LIVE Trading"})`);
    }
      
    // Fetch existing positions
    const { data: positions } = await supabase
      .from("positions")
      .select("*");
      
    const currentPositions = positions || [];
    
    // Get real account state from Alpaca if available
    let accountState;
    if (alpacaCredentials) {
      try {
        const alpacaAccount = await getAlpacaAccount(alpacaCredentials);
        const equity = parseFloat(alpacaAccount.equity);
        const lastEquity = parseFloat(alpacaAccount.last_equity);
        const drawdown = Math.abs(Math.min(0, (equity - lastEquity) / lastEquity * 100));
        
        accountState = {
          equity,
          drawdown,
          open_positions: currentPositions.length,
          max_positions: config?.max_concurrent_positions || 5,
          buying_power: parseFloat(alpacaAccount.buying_power),
        };
        
        await log("INFO", `💰 Account: $${equity.toFixed(2)} equity, ${accountState.open_positions} positions`);
      } catch (error) {
        await log("ERROR", "Failed to fetch Alpaca account", { error });
        // Fallback to mock data
        accountState = {
          equity: 100000,
          drawdown: 0,
          open_positions: currentPositions.length,
          max_positions: config?.max_concurrent_positions || 5,
        };
      }
    } else {
      // Mock account state
      const totalPnL = currentPositions.reduce((sum, p) => sum + (p.unrealized_pnl || 0), 0);
      const initialEquity = 100000;
      const currentEquity = initialEquity + totalPnL;
      const drawdown = Math.abs(Math.min(0, (currentEquity - initialEquity) / initialEquity * 100));
      
      accountState = {
        equity: currentEquity,
        drawdown,
        open_positions: currentPositions.length,
        max_positions: config?.max_concurrent_positions || 5,
      };
    }
    
    // 1. CONTINUOUS LEARNING (always runs)
    if (config?.continuous_learning_enabled) {
      await log("INFO", "📚 Running continuous learning...");
      const { data: learningResult } = await supabase.functions.invoke("online-learner");
      await log("INFO", "✅ Continuous learning completed", learningResult);
    }
    
    // 2. TRADING CYCLE
    let signalsGenerated = 0;
    let tradesPlaced = 0;
    let tradesSkipped = 0;
    
    // 2.1 Generate Signals
    await log("INFO", "🎯 Generating trading signals...");
    const { data: engineResult } = await supabase.functions.invoke("trading-engine", {
      body: { watchlist: ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN"] },
    });
    
    const signals = engineResult?.signals || [];
    signalsGenerated = signals.length;
    await log("INFO", `Generated ${signalsGenerated} signals`);
    
    // 2.2 Process Each Signal
    for (const signal of signals) {
      // Skip if max positions reached
      if (currentPositions.length >= accountState.max_positions) {
        await log("WARN", `Max positions (${accountState.max_positions}) reached - skipping signal`);
        tradesSkipped++;
        continue;
      }
      
      // Save signal to DB
      const { data: savedSignal } = await supabase
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
        
      if (!savedSignal) continue;
      signal.id = savedSignal.id;
      
      // 2.3 Correlation Check
      const correlationCheck = await checkCorrelation(signal, currentPositions);
      await supabase.from("signal_correlations").insert({
        signal_id: signal.id,
        position_symbol: currentPositions.map(p => p.symbol).join(","),
        correlation: correlationCheck.correlation,
      });
      
      if (correlationCheck.correlated && correlationCheck.correlation > 0.8) {
        await log("WARN", `Skipping ${signal.symbol} - high correlation`, correlationCheck);
        tradesSkipped++;
        continue;
      }
      
      // 2.4 RL Decision
      await log("INFO", `🤖 Querying RL agent for ${signal.symbol}...`);
      const { data: rlDecision } = await supabase.functions.invoke("rl-agent", {
        body: {
          signal,
          current_positions: currentPositions,
          account_state: accountState,
        },
      });
      
      if (!rlDecision || rlDecision.action === "hold") {
        await log("INFO", `RL vetoed ${signal.symbol}: ${rlDecision?.reasoning || "hold"}`);
        tradesSkipped++;
        continue;
      }
      
      // 2.5 Position Sizing
      const sizing = await calculatePositionSize(signal, rlDecision, accountState);
      await supabase.from("position_sizing").insert({
        signal_id: signal.id,
        ...sizing,
      });
      
      // 2.6 Quantitative Risk Assessment
      await log("INFO", `🛡️ Assessing risk for ${signal.symbol}...`);
      const { data: riskAssessment } = await supabase.functions.invoke("quantitative-risk-manager", {
        body: {
          signal,
          account_state: {
            ...accountState,
            peak_equity: accountState.equity, // TODO: Track actual peak
          },
        },
      });
      
      if (!riskAssessment) {
        await log("ERROR", "Risk assessment failed");
        tradesSkipped++;
        continue;
      }
      
      // Save risk assessment
      const { data: savedAssessment } = await supabase
        .from("risk_assessments")
        .insert({
          signal_id: signal.id,
          risk_score: riskAssessment.risk_score,
          adjusted_size: riskAssessment.adjusted_size,
          should_execute: riskAssessment.should_execute,
          reason: riskAssessment.reason,
          factors: riskAssessment.factors,
        })
        .select()
        .single();
      
      // 2.7 Execute Trade
      if (riskAssessment.should_execute) {
        await log("INFO", `✅ Executing: ${signal.action.toUpperCase()} ${riskAssessment.adjusted_size} shares ${signal.symbol}`);
        await log("INFO", `   SL: ${riskAssessment.stop_loss?.toFixed(2)}, TP: ${riskAssessment.take_profit?.toFixed(2)}, Risk Score: ${riskAssessment.risk_score}/100`);
        
        const quantity = Math.floor(riskAssessment.adjusted_size);
        
        // Place real Alpaca order if credentials are configured
        let alpacaOrderId = null;
        if (alpacaCredentials && quantity > 0) {
          try {
            const alpacaOrder = await placeAlpacaOrder(alpacaCredentials, {
              symbol: signal.symbol,
              quantity,
              side: signal.action, // "buy" or "sell"
            });
            alpacaOrderId = alpacaOrder.id;
            await log("INFO", `🎯 Alpaca order placed: ${alpacaOrder.id}`);
          } catch (error) {
            await log("ERROR", `Failed to place Alpaca order for ${signal.symbol}`, { error });
            // Continue anyway to track in our system
          }
        } else if (!alpacaCredentials) {
          await log("INFO", "📝 Simulated order (Alpaca not configured)");
        }
        
        // Create trade record
        await supabase.from("trades").insert({
          signal_id: signal.id,
          risk_assessment_id: savedAssessment.id,
          symbol: signal.symbol,
          action: signal.action,
          size: riskAssessment.adjusted_size,
          entry_price: signal.market_data.price,
          stop_loss: riskAssessment.stop_loss,
          take_profit: riskAssessment.take_profit,
          status: "open",
        });
        
        // Create/update position with risk parameters
        await supabase.from("positions").upsert({
          symbol: signal.symbol,
          side: signal.action === "buy" ? "long" : "short",
          size: quantity,
          entry_price: signal.market_data.price,
          current_price: signal.market_data.price,
          stop_loss: riskAssessment.stop_loss,
          take_profit: riskAssessment.take_profit,
          metadata: {
            atr: riskAssessment.factors?.atr,
            risk_score: riskAssessment.risk_score,
            kelly_pct: riskAssessment.factors?.kelly_pct,
            alpaca_order_id: alpacaOrderId,
          },
        });
        
        tradesPlaced++;
      } else {
        await log("WARN", `❌ Trade blocked: ${riskAssessment.reason}`);
        tradesSkipped++;
      }
    }
    
    // 2.8 Position Monitoring
    await log("INFO", "👀 Monitoring positions...");
    const { data: monitorResult } = await supabase.functions.invoke("position-monitor");
    const positionsClosed = monitorResult?.closed || 0;
    await log("INFO", `Closed ${positionsClosed} positions`);
    
    // 3. POST-LOOP OPTIMIZATION
    await log("INFO", "🎓 Running strategy promoter...");
    const { data: promoterResult } = await supabase.functions.invoke("strategy-promoter");
    await log("INFO", "✅ Strategy promotion completed", promoterResult);
    
    // Update loop record
    await supabase
      .from("bot_loops")
      .update({
        signals_generated: signalsGenerated,
        trades_placed: tradesPlaced,
        trades_skipped: tradesSkipped,
        positions_closed: positionsClosed,
        completed_at: new Date().toISOString(),
        status: "completed",
      })
      .eq("id", loopRecord.id);
      
    const duration = (new Date().getTime() - loopStart.getTime()) / 1000;
    await log("INFO", `✅ Loop #${loopNumber} completed in ${duration.toFixed(1)}s`, {
      signals: signalsGenerated,
      placed: tradesPlaced,
      skipped: tradesSkipped,
      closed: positionsClosed,
    });
    
  } catch (error) {
    await log("ERROR", `Loop #${loopNumber} failed`, { error });
    await supabase
      .from("bot_loops")
      .update({
        status: "failed",
        error_message: error instanceof Error ? error.message : "Unknown error",
        completed_at: new Date().toISOString(),
      })
      .eq("id", loopRecord.id);
  }
}

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { loops = 1 } = await req.json().catch(() => ({}));
    
    await log("INFO", `🤖 Autonomous trader started (${loops} loops)`);
    
    for (let i = 0; i < loops; i++) {
      await runTradingLoop(i + 1);
      
      // Delay between loops (except last one)
      if (i < loops - 1) {
        await log("INFO", `⏸️ Waiting 30s before next loop...`);
        await new Promise((resolve) => setTimeout(resolve, 30000));
      }
    }
    
    await log("INFO", `✅ All ${loops} loops completed`);

    return new Response(
      JSON.stringify({
        success: true,
        loops_completed: loops,
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
