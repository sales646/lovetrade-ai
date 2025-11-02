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

// Q-Learning parameters
interface QState {
  q_table: Record<string, number[]>; // State -> [Q(s,sell), Q(s,hold), Q(s,buy)]
  epsilon: number; // Exploration rate
  alpha: number; // Learning rate
  gamma: number; // Discount factor
  episode_count: number;
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
    return {
      q_table: data.q_table || {},
      epsilon: data.epsilon || 0.1,
      alpha: data.alpha || 0.01,
      gamma: data.gamma || 0.99,
      episode_count: data.episode_count || 0,
    };
  }

  // Initialize new Q-state
  return {
    q_table: {},
    epsilon: 0.3, // Start with 30% exploration
    alpha: 0.01, // Learning rate
    gamma: 0.99, // Discount factor
    episode_count: 0,
  };
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

function calculateReward(action: number, marketMove: number, holdTime: number): number {
  // Reward function that encourages trading with small incentives
  
  // Base reward from market move
  let reward = 0;
  
  if (action === 0) {
    // SELL action: profit from downward moves
    reward = -marketMove * 100; // Profit when price drops
  } else if (action === 2) {
    // BUY action: profit from upward moves  
    reward = marketMove * 100; // Profit when price rises
  } else {
    // HOLD action: small penalty to encourage trading
    reward = -0.05; // Small penalty for inaction
  }
  
  // Additional small reward for taking action (not holding)
  if (action !== 1) {
    reward += 0.1; // Small bonus for trading
  }
  
  // Penalty for holding too long
  if (action === 1 && holdTime > 10) {
    reward -= 0.1 * (holdTime - 10); // Increasing penalty
  }
  
  return reward;
}

async function runSimulationEpisode(qState: QState, symbol: string) {
  await log("INFO", `🎮 Running simulation episode for ${symbol}...`);
  
  // Fetch recent market data for simulation
  const { data: bars } = await supabase
    .from("historical_bars")
    .select("*")
    .eq("symbol", symbol)
    .eq("timeframe", "5m")
    .order("timestamp", { ascending: false })
    .limit(100);
  
  if (!bars || bars.length < 2) {
    await log("WARN", "Not enough historical data for simulation");
    return { reward: 0, steps: 0 };
  }
  
  let totalReward = 0;
  let steps = 0;
  let position = 1; // Start with HOLD
  let holdTime = 0;
  
  // Simulate through historical bars
  for (let i = 0; i < bars.length - 1; i++) {
    const currentBar = bars[i];
    const nextBar = bars[i + 1];
    
    // Get technical indicators for state
    const { data: indicators } = await supabase
      .from("technical_indicators")
      .select("*")
      .eq("symbol", symbol)
      .eq("timeframe", "5m")
      .gte("timestamp", currentBar.timestamp)
      .limit(1)
      .single();
    
    if (!indicators) continue;
    
    const stateKey = getStateKey(indicators);
    
    // Initialize Q-values if not exists
    if (!qState.q_table[stateKey]) {
      qState.q_table[stateKey] = [0, 0, 0]; // [sell, hold, buy]
    }
    
    // Select action using epsilon-greedy
    const action = selectAction(stateKey, qState);
    
    // Calculate market move
    const marketMove = (Number(nextBar.close) - Number(currentBar.close)) / Number(currentBar.close);
    
    // Track hold time
    if (action === 1) {
      holdTime++;
    } else {
      holdTime = 0;
    }
    
    // Calculate reward
    const reward = calculateReward(action, marketMove, holdTime);
    totalReward += reward;
    
    // Get next state
    const { data: nextIndicators } = await supabase
      .from("technical_indicators")
      .select("*")
      .eq("symbol", symbol)
      .eq("timeframe", "5m")
      .gte("timestamp", nextBar.timestamp)
      .limit(1)
      .single();
    
    if (!nextIndicators) continue;
    
    const nextStateKey = getStateKey(nextIndicators);
    if (!qState.q_table[nextStateKey]) {
      qState.q_table[nextStateKey] = [0, 0, 0];
    }
    
    // Q-Learning update: Q(s,a) = Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
    const currentQ = qState.q_table[stateKey][action];
    const maxNextQ = Math.max(...qState.q_table[nextStateKey]);
    const newQ = currentQ + qState.alpha * (reward + qState.gamma * maxNextQ - currentQ);
    
    qState.q_table[stateKey][action] = newQ;
    
    steps++;
  }
  
  await log("INFO", `Episode complete: ${steps} steps, total reward: ${totalReward.toFixed(2)}`);
  
  return { reward: totalReward, steps };
}

async function runTrainingLoop(iterations: number = 10) {
  const loopStart = new Date();
  
  await log("INFO", `🚀 Starting autonomous RL training (${iterations} episodes)`);
  
  // Load Q-state
  const qState = await loadQState();
  
  const symbols = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN"];
  let totalReward = 0;
  let totalSteps = 0;
  
  for (let i = 0; i < iterations; i++) {
    // Pick random symbol for each episode
    const symbol = symbols[Math.floor(Math.random() * symbols.length)];
    
    const result = await runSimulationEpisode(qState, symbol);
    totalReward += result.reward;
    totalSteps += result.steps;
    
    // Update episode count
    qState.episode_count++;
    
    // Decay epsilon (exploration rate) over time
    // Start at 0.3, decay to 0.05 over 1000 episodes
    if (qState.episode_count < 1000) {
      qState.epsilon = 0.3 - (0.25 * qState.episode_count / 1000);
    } else {
      qState.epsilon = 0.05; // Minimum exploration
    }
    
    // Log progress every 5 episodes
    if ((i + 1) % 5 === 0) {
      await log("INFO", `Progress: ${i + 1}/${iterations} episodes, avg reward: ${(totalReward / (i + 1)).toFixed(2)}, ε=${qState.epsilon.toFixed(3)}`);
    }
  }
  
  // Save updated Q-state
  await saveQState(qState);
  
  // Log metrics to database
  const avgReward = totalReward / iterations;
  const avgSteps = totalSteps / iterations;
  
  await supabase.from("rl_training_metrics").insert({
    episodes: iterations,
    avg_reward: avgReward,
    avg_steps: avgSteps,
    epsilon: qState.epsilon,
    total_episodes: qState.episode_count,
    q_table_size: Object.keys(qState.q_table).length,
    duration_seconds: (new Date().getTime() - loopStart.getTime()) / 1000,
  });
  
  const duration = (new Date().getTime() - loopStart.getTime()) / 1000;
  
  await log("INFO", `✅ Training loop completed in ${duration.toFixed(1)}s`, {
    iterations,
    totalReward: totalReward.toFixed(2),
    avgReward: avgReward.toFixed(2),
    epsilon: qState.epsilon.toFixed(3),
    totalEpisodes: qState.episode_count,
    qTableSize: Object.keys(qState.q_table).length,
  });
  
  return {
    success: true,
    iterations,
    avgReward,
    avgSteps,
    epsilon: qState.epsilon,
    totalEpisodes: qState.episode_count,
    qTableSize: Object.keys(qState.q_table).length,
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
