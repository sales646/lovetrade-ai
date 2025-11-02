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

// Expert weights for imitation learning
const EXPERT_WEIGHTS: Record<string, number> = {
  "RSI_EMA": 0.40,
  "VWAP_REVERSION": 0.30,
  "TREND_PULLBACK": 0.10,
  "VWAP_DELTA_CONFLUENCE": 0.10,
  "AFTERNOON_FADE": 0.05,
  "LIQUIDITY_SWEEP": 0.05,
  "HOLD_BASELINE": 0.00,
};

// Q-Learning parameters with imitation
interface QState {
  q_table: Record<string, number[]>; // State -> [Q(s,sell), Q(s,hold), Q(s,buy)]
  epsilon: number; // Exploration rate
  alpha: number; // Learning rate (RL)
  gamma: number; // Discount factor
  episode_count: number;
  alpha_imitation: number; // Current imitation mixing weight
}

interface ExpertTrajectory {
  id: string;
  tactic_id: string;
  action: number; // -1=sell, 0=hold, 1=buy
  obs_features: any;
  reward: number;
  symbol: string;
  timestamp: string;
}

async function log(level: "INFO" | "WARN" | "ERROR", message: string, metadata?: any) {
  console.log(`[${level}] ${message}`);
  await supabase.from("system_logs").insert({
    level,
    source: "autonomous-rl-trainer",
    message,
    metadata: metadata || {},
  });
}

async function loadQState(): Promise<QState> {
  const { data } = await supabase
    .from("rl_q_state")
    .select("*")
    .single();

  if (data) {
    const episodeCount = data.episode_count || 0;
    return {
      q_table: data.q_table || {},
      epsilon: data.epsilon || 0.2,
      alpha: data.alpha || 0.01,
      gamma: data.gamma || 0.99,
      episode_count: episodeCount,
      alpha_imitation: calculateAlphaMix(episodeCount),
    };
  }

  // Initialize new Q-state with imitation
  return {
    q_table: {},
    epsilon: 0.20, // Start with 20% exploration
    alpha: 0.01, // Learning rate (RL)
    gamma: 0.99, // Discount factor
    episode_count: 0,
    alpha_imitation: 0.6, // Start with 60% imitation weight
  };
}

function calculateAlphaMix(episodeCount: number): number {
  // Linear decay: α_start=0.6 → α_end=0.1 over first 60% of 1000 episodes (600 episodes)
  const decaySteps = 600;
  const alphaStart = 0.6;
  const alphaEnd = 0.1;
  
  if (episodeCount >= decaySteps) {
    return alphaEnd;
  }
  
  return alphaStart - (alphaStart - alphaEnd) * (episodeCount / decaySteps);
}

async function saveQState(state: QState) {
  await supabase.from("rl_q_state").upsert({
    id: 1,
    q_table: state.q_table,
    epsilon: state.epsilon,
    alpha: state.alpha,
    gamma: state.gamma,
    episode_count: state.episode_count,
    updated_at: new Date().toISOString(),
  });
}

async function loadExpertTrajectories(symbol: string, limit: number = 50): Promise<ExpertTrajectory[]> {
  const { data } = await supabase
    .from("expert_trajectories")
    .select("*")
    .eq("symbol", symbol)
    .order("timestamp", { ascending: false })
    .limit(limit);

  return data || [];
}

function calculateImitationLoss(
  stateKey: string,
  qValues: number[],
  expertAction: number, // -1=sell, 0=hold, 1=buy
  expertWeight: number
): number {
  // Convert expert action to index: -1→0 (sell), 0→1 (hold), 1→2 (buy)
  const expertActionIdx = expertAction + 1;
  
  // Softmax probabilities from Q-values
  const maxQ = Math.max(...qValues);
  const expQ = qValues.map(q => Math.exp(q - maxQ));
  const sumExpQ = expQ.reduce((a, b) => a + b, 0);
  const probs = expQ.map(e => e / sumExpQ);
  
  // Cross-entropy loss weighted by expert weight
  const loss = -Math.log(probs[expertActionIdx] + 1e-8);
  return expertWeight * loss;
}

function getStateKey(marketData: any): string {
  // Create state representation from market features
  const rsi = Math.floor(marketData.rsi_14 / 10) * 10; // Discretize RSI to bins of 10
  const vwapDist = Math.floor(marketData.vwap_distance_pct / 0.5) * 0.5; // Bins of 0.5%
  const volumeZ = Math.floor(marketData.volume_zscore);
  
  return `rsi_${rsi}_vwap_${vwapDist}_vol_${volumeZ}`;
}

function selectAction(stateKey: string, qState: QState): number {
  // Epsilon-greedy action selection
  if (Math.random() < qState.epsilon) {
    // Explore: random action
    return Math.floor(Math.random() * 3); // 0=sell, 1=hold, 2=buy
  } else {
    // Exploit: choose best Q-value
    const qValues = qState.q_table[stateKey] || [0, 0, 0];
    return qValues.indexOf(Math.max(...qValues));
  }
}

// Simulate a realistic trade with stops, targets, fees, and slippage
function simulateTradeOutcome(
  bars: any[],
  entryIndex: number,
  side: "BUY" | "SELL",
  atr: number,
  maxHoldBars: number = 12
) {
  const entryBar = bars[entryIndex];
  const entryPrice = Number(entryBar.close);
  
  // REALISTIC risk management parameters
  const stopLossDistance = atr * 1.5; // Tight 1.5x ATR stop
  const takeProfitPct = 3.5; // 3.5% target (realistic with leverage)
  const slippagePct = 0.08; // Slippage from entries
  const feesPct = 0.12; // Trading fees
  
  const stopLossPrice = side === "BUY" 
    ? entryPrice - stopLossDistance 
    : entryPrice + stopLossDistance;
  
  const takeProfitPrice = side === "BUY" 
    ? entryPrice * (1 + takeProfitPct / 100)
    : entryPrice * (1 - takeProfitPct / 100);
  
  // Walk forward through bars to find exit
  let exitPrice = entryPrice;
  let exitReason = "TIME"; // TIME, STOP_LOSS, TAKE_PROFIT
  let barsHeld = 0;
  
  for (let i = entryIndex + 1; i < Math.min(entryIndex + maxHoldBars + 1, bars.length); i++) {
    const bar = bars[i];
    barsHeld++;
    
    if (side === "BUY") {
      // Check if stop loss hit
      if (Number(bar.low) <= stopLossPrice) {
        exitPrice = stopLossPrice;
        exitReason = "STOP_LOSS";
        break;
      }
      // Check if take profit hit
      if (Number(bar.high) >= takeProfitPrice) {
        exitPrice = takeProfitPrice;
        exitReason = "TAKE_PROFIT";
        break;
      }
    } else { // SELL
      // Check if stop loss hit
      if (Number(bar.high) >= stopLossPrice) {
        exitPrice = stopLossPrice;
        exitReason = "STOP_LOSS";
        break;
      }
      // Check if take profit hit
      if (Number(bar.low) <= takeProfitPrice) {
        exitPrice = takeProfitPrice;
        exitReason = "TAKE_PROFIT";
        break;
      }
    }
    
    // Time-based exit
    if (barsHeld >= maxHoldBars) {
      exitPrice = Number(bar.close);
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
  
  // Reward scaling: +3.5% = +1.0 reward, -1.5% = -0.5 reward
  const reward = netPnlPct / 3.5;
  
  return {
    reward,
    pnl_pct: netPnlPct,
    exit_reason: exitReason,
    bars_held: barsHeld,
  };
}

async function runSimulationEpisode(qState: QState, symbol: string) {
  await log("INFO", `🎮 Running simulation episode for ${symbol}...`);
  
  // Fetch recent market data for simulation (RL experience)
  const { data: bars } = await supabase
    .from("historical_bars")
    .select("*")
    .eq("symbol", symbol)
    .eq("timeframe", "5m")
    .order("timestamp", { ascending: true }) // IMPORTANT: ascending for forward simulation
    .limit(100);
  
  if (!bars || bars.length < 20) {
    await log("WARN", "Not enough historical data for simulation");
    return { 
      reward: 0, 
      steps: 0, 
      lossRL: 0, 
      lossImitation: 0, 
      lossTotal: 0,
      actionCounts: { buy: 0, sell: 0, hold: 0 },
      expertAccuracies: {},
    };
  }
  
  // Load expert trajectories for imitation
  const expertTrajectories = await loadExpertTrajectories(symbol, 50);
  
  let totalReward = 0;
  let steps = 0;
  let lossRL = 0;
  let lossImitation = 0;
  let actionCounts = { buy: 0, sell: 0, hold: 0 };
  let expertCorrectCounts: Record<string, { correct: number; total: number }> = {};
  let totalTrades = 0;
  let winningTrades = 0;
  
  // Simulate through historical bars with REAL trade outcomes
  for (let i = 0; i < bars.length - 15; i++) { // Leave room for trade to complete
    const currentBar = bars[i];
    
    // Get technical indicators for state
    const { data: indicators } = await supabase
      .from("technical_indicators")
      .select("*")
      .eq("symbol", symbol)
      .eq("timeframe", "5m")
      .lte("timestamp", currentBar.timestamp)
      .order("timestamp", { ascending: false })
      .limit(1)
      .maybeSingle();
    
    if (!indicators || !indicators.atr_14) continue;
    
    const stateKey = getStateKey(indicators);
    
    // Initialize Q-values if not exists
    if (!qState.q_table[stateKey]) {
      qState.q_table[stateKey] = [0, 0, 0]; // [sell, hold, buy]
    }
    
    // Select action using epsilon-greedy
    const actionIdx = selectAction(stateKey, qState);
    
    let reward = 0;
    let nextStateKey = stateKey;
    
    // Track action distribution
    if (actionIdx === 0) {
      actionCounts.sell++;
      // SELL trade: simulate short position
      const tradeResult = simulateTradeOutcome(
        bars,
        i,
        "SELL",
        Number(indicators.atr_14),
        12 // max hold bars
      );
      reward = tradeResult.reward;
      totalReward += reward;
      totalTrades++;
      if (reward > 0) winningTrades++;
      
      // Get state after trade completes
      const exitIdx = Math.min(i + tradeResult.bars_held, bars.length - 1);
      const { data: nextIndicators } = await supabase
        .from("technical_indicators")
        .select("*")
        .eq("symbol", symbol)
        .eq("timeframe", "5m")
        .lte("timestamp", bars[exitIdx].timestamp)
        .order("timestamp", { ascending: false })
        .limit(1)
        .maybeSingle();
      
      if (nextIndicators) {
        nextStateKey = getStateKey(nextIndicators);
      }
      
      // Skip ahead past this trade
      i += tradeResult.bars_held;
      
    } else if (actionIdx === 2) {
      actionCounts.buy++;
      // BUY trade: simulate long position
      const tradeResult = simulateTradeOutcome(
        bars,
        i,
        "BUY",
        Number(indicators.atr_14),
        12 // max hold bars
      );
      reward = tradeResult.reward;
      totalReward += reward;
      totalTrades++;
      if (reward > 0) winningTrades++;
      
      // Get state after trade completes
      const exitIdx = Math.min(i + tradeResult.bars_held, bars.length - 1);
      const { data: nextIndicators } = await supabase
        .from("technical_indicators")
        .select("*")
        .eq("symbol", symbol)
        .eq("timeframe", "5m")
        .lte("timestamp", bars[exitIdx].timestamp)
        .order("timestamp", { ascending: false })
        .limit(1)
        .maybeSingle();
      
      if (nextIndicators) {
        nextStateKey = getStateKey(nextIndicators);
      }
      
      // Skip ahead past this trade
      i += tradeResult.bars_held;
      
    } else {
      actionCounts.hold++;
      // HOLD: small penalty, move to next bar
      reward = -0.05;
      totalReward += reward;
      
      if (i + 1 < bars.length) {
        const { data: nextIndicators } = await supabase
          .from("technical_indicators")
          .select("*")
          .eq("symbol", symbol)
          .eq("timeframe", "5m")
          .lte("timestamp", bars[i + 1].timestamp)
          .order("timestamp", { ascending: false })
          .limit(1)
          .maybeSingle();
        
        if (nextIndicators) {
          nextStateKey = getStateKey(nextIndicators);
        }
      }
    }
    
    // Initialize next state Q-values if needed
    if (!qState.q_table[nextStateKey]) {
      qState.q_table[nextStateKey] = [0, 0, 0];
    }
    
    // Standard Q-Learning update (RL loss)
    const currentQ = qState.q_table[stateKey][actionIdx];
    const maxNextQ = Math.max(...qState.q_table[nextStateKey]);
    const tdError = reward + qState.gamma * maxNextQ - currentQ;
    const newQ = currentQ + qState.alpha * tdError;
    
    // RL loss (TD error squared)
    lossRL += tdError * tdError;
    
    qState.q_table[stateKey][actionIdx] = newQ;
    
    steps++;
  }
  
  // Process expert trajectories for imitation learning
  for (const expertTraj of expertTrajectories) {
    const expertWeight = EXPERT_WEIGHTS[expertTraj.tactic_id] || 0;
    if (expertWeight === 0) continue; // Skip HOLD_BASELINE
    
    const stateKey = getStateKey(expertTraj.obs_features.current || {});
    
    // Initialize Q-values if not exists
    if (!qState.q_table[stateKey]) {
      qState.q_table[stateKey] = [0, 0, 0];
    }
    
    // Calculate imitation loss
    const imitLoss = calculateImitationLoss(
      stateKey,
      qState.q_table[stateKey],
      expertTraj.action,
      expertWeight
    );
    
    lossImitation += imitLoss;
    
    // Track expert accuracy
    if (!expertCorrectCounts[expertTraj.tactic_id]) {
      expertCorrectCounts[expertTraj.tactic_id] = { correct: 0, total: 0 };
    }
    
    const predictedAction = selectAction(stateKey, { ...qState, epsilon: 0 }); // Greedy
    const expertActionIdx = expertTraj.action + 1;
    
    expertCorrectCounts[expertTraj.tactic_id].total++;
    if (predictedAction === expertActionIdx) {
      expertCorrectCounts[expertTraj.tactic_id].correct++;
    }
    
    // Update Q-values using imitation gradient
    // Gradient of cross-entropy pushes Q-values toward expert action
    // expertActionIdx already declared above
    const softmaxGrad = [...qState.q_table[stateKey]];
    
    // Simple gradient update: increase Q for expert action, decrease others
    const imitLR = qState.alpha * qState.alpha_imitation;
    softmaxGrad[expertActionIdx] += imitLR * expertWeight;
    for (let i = 0; i < 3; i++) {
      if (i !== expertActionIdx) {
        softmaxGrad[i] -= imitLR * expertWeight / 2;
      }
    }
    
    qState.q_table[stateKey] = softmaxGrad;
  }
  
  // Calculate expert accuracies
  const expertAccuracies: Record<string, number> = {};
  for (const [expertName, counts] of Object.entries(expertCorrectCounts)) {
    expertAccuracies[expertName] = counts.total > 0 ? counts.correct / counts.total : 0;
  }
  
  // Combine losses: L_total = α * L_imitation + (1-α) * L_RL
  const lossTotal = qState.alpha_imitation * lossImitation + (1 - qState.alpha_imitation) * lossRL;
  
  const winRate = totalTrades > 0 ? (winningTrades / totalTrades) * 100 : 0;
  
  await log("INFO", `Episode complete: ${steps} steps, reward: ${totalReward.toFixed(2)}, Win Rate: ${winRate.toFixed(1)}%, L_RL: ${lossRL.toFixed(4)}, L_imit: ${lossImitation.toFixed(4)}, L_total: ${lossTotal.toFixed(4)}`);
  
  return { 
    reward: totalReward, 
    steps,
    lossRL,
    lossImitation,
    lossTotal,
    actionCounts,
    expertAccuracies,
    totalTrades,
    winningTrades,
    winRate,
  };
}

async function runTrainingLoop(iterations: number = 10) {
  const loopStart = new Date();
  
  await log("INFO", `🚀 Starting autonomous RL training (${iterations} episodes)`);
  
  // Load Q-state
  const qState = await loadQState();
  
  const symbols = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN"];
  let totalReward = 0;
  let totalSteps = 0;
  let totalLossRL = 0;
  let totalLossImitation = 0;
  let totalLossTotal = 0;
  let aggregateActionCounts = { buy: 0, sell: 0, hold: 0 };
  const aggregateExpertAccuracies: Record<string, number[]> = {};
  let totalTrades = 0;
  let totalWinningTrades = 0;
  
  for (let i = 0; i < iterations; i++) {
    // Pick random symbol for each episode
    const symbol = symbols[Math.floor(Math.random() * symbols.length)];
    
    const result = await runSimulationEpisode(qState, symbol);
    totalReward += result.reward;
    totalSteps += result.steps;
    totalLossRL += result.lossRL;
    totalLossImitation += result.lossImitation;
    totalLossTotal += result.lossTotal;
    
    // Aggregate action counts
    aggregateActionCounts.buy += result.actionCounts.buy;
    aggregateActionCounts.sell += result.actionCounts.sell;
    aggregateActionCounts.hold += result.actionCounts.hold;
    
    // Aggregate trade outcomes
    totalTrades += result.totalTrades || 0;
    totalWinningTrades += result.winningTrades || 0;
    
    // Aggregate expert accuracies
    for (const [expertName, accuracy] of Object.entries(result.expertAccuracies)) {
      if (!aggregateExpertAccuracies[expertName]) {
        aggregateExpertAccuracies[expertName] = [];
      }
      aggregateExpertAccuracies[expertName].push(accuracy);
    }
    
    // Update episode count
    qState.episode_count++;
    
    // Decay epsilon: ε_start=0.20 → ε_end=0.02 linearly over 1000 episodes
    const epsilonStart = 0.20;
    const epsilonEnd = 0.02;
    const epsilonDecaySteps = 1000;
    
    if (qState.episode_count < epsilonDecaySteps) {
      qState.epsilon = epsilonStart - (epsilonStart - epsilonEnd) * (qState.episode_count / epsilonDecaySteps);
    } else {
      qState.epsilon = epsilonEnd;
    }
    
    // Update alpha_imitation
    qState.alpha_imitation = calculateAlphaMix(qState.episode_count);
    
    // Log progress every 5 episodes
    if ((i + 1) % 5 === 0) {
      await log("INFO", `Progress: ${i + 1}/${iterations} episodes, avg reward: ${(totalReward / (i + 1)).toFixed(2)}, ε=${qState.epsilon.toFixed(3)}, α_imit=${qState.alpha_imitation.toFixed(3)}`);
    }
  }
  
  // Calculate average expert accuracies
  const avgExpertAccuracies: Record<string, number> = {};
  for (const [expertName, accuracies] of Object.entries(aggregateExpertAccuracies)) {
    avgExpertAccuracies[expertName] = accuracies.reduce((a, b) => a + b, 0) / accuracies.length;
  }
  
  // Save updated Q-state
  await saveQState(qState);
  
  // Calculate action percentages
  const totalActions = aggregateActionCounts.buy + aggregateActionCounts.sell + aggregateActionCounts.hold;
  const actionBuyPct = totalActions > 0 ? (aggregateActionCounts.buy / totalActions) * 100 : 0;
  const actionSellPct = totalActions > 0 ? (aggregateActionCounts.sell / totalActions) * 100 : 0;
  const actionHoldPct = totalActions > 0 ? (aggregateActionCounts.hold / totalActions) * 100 : 0;
  
  // Log metrics to database
  const avgReward = totalReward / iterations;
  const avgSteps = totalSteps / iterations;
  const avgLossRL = totalLossRL / iterations;
  const avgLossImitation = totalLossImitation / iterations;
  const avgLossTotal = totalLossTotal / iterations;
  const batchWinRate = totalTrades > 0 ? (totalWinningTrades / totalTrades) * 100 : 0;
  
  const { data: metricData } = await supabase.from("rl_training_metrics").insert({
    episodes: iterations,
    avg_reward: avgReward,
    avg_steps: avgSteps,
    epsilon: qState.epsilon,
    total_episodes: qState.episode_count,
    q_table_size: Object.keys(qState.q_table).length,
    duration_seconds: (new Date().getTime() - loopStart.getTime()) / 1000,
    l_imitation: avgLossImitation,
    l_rl: avgLossRL,
    l_total: avgLossTotal,
    alpha_mix: qState.alpha_imitation,
    action_buy_pct: actionBuyPct,
    action_sell_pct: actionSellPct,
    action_hold_pct: actionHoldPct,
    expert_accuracies: avgExpertAccuracies,
    win_rate_pct: batchWinRate,
    total_trades: totalTrades,
    winning_trades: totalWinningTrades,
  }).select().single();
  
  // Log per-expert contributions
  if (metricData) {
    const expertContributions = Object.entries(avgExpertAccuracies).map(([expertName, accuracy]) => ({
      training_metric_id: metricData.id,
      expert_name: expertName,
      weight: EXPERT_WEIGHTS[expertName] || 0,
      loss_contribution: (EXPERT_WEIGHTS[expertName] || 0) * avgLossImitation,
      accuracy: accuracy,
      sample_count: aggregateExpertAccuracies[expertName]?.length || 0,
    }));
    
    if (expertContributions.length > 0) {
      await supabase.from("expert_contributions").insert(expertContributions);
    }
  }
  
  const duration = (new Date().getTime() - loopStart.getTime()) / 1000;
  
  await log("INFO", `✅ Training loop completed in ${duration.toFixed(1)}s`, {
    iterations,
    totalReward: totalReward.toFixed(2),
    avgReward: avgReward.toFixed(2),
    avgLossRL: avgLossRL.toFixed(4),
    avgLossImitation: avgLossImitation.toFixed(4),
    avgLossTotal: avgLossTotal.toFixed(4),
    epsilon: qState.epsilon.toFixed(3),
    alphaImitation: qState.alpha_imitation.toFixed(3),
    totalEpisodes: qState.episode_count,
    qTableSize: Object.keys(qState.q_table).length,
    actionDistribution: `BUY:${actionBuyPct.toFixed(1)}% SELL:${actionSellPct.toFixed(1)}% HOLD:${actionHoldPct.toFixed(1)}%`,
    winRate: `${batchWinRate.toFixed(1)}% (${totalWinningTrades}/${totalTrades})`,
    expertAccuracies: avgExpertAccuracies,
  });
  
  return {
    success: true,
    iterations,
    avgReward,
    avgSteps,
    lossRL: avgLossRL,
    lossImitation: avgLossImitation,
    lossTotal: avgLossTotal,
    epsilon: qState.epsilon,
    alphaImitation: qState.alpha_imitation,
    totalEpisodes: qState.episode_count,
    qTableSize: Object.keys(qState.q_table).length,
    actionDistribution: { buy: actionBuyPct, sell: actionSellPct, hold: actionHoldPct },
    expertAccuracies: avgExpertAccuracies,
    winRate: batchWinRate,
    totalTrades,
    winningTrades: totalWinningTrades,
  };
}

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { iterations = 10 } = await req.json().catch(() => ({}));
    
    const result = await runTrainingLoop(iterations);

    return new Response(
      JSON.stringify(result),
      { headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  } catch (error) {
    await log("ERROR", "Autonomous RL trainer failed", { error });
    return new Response(
      JSON.stringify({ error: error instanceof Error ? error.message : "Unknown error" }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});
