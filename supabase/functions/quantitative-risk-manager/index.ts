import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

interface RiskConfig {
  max_position_size_pct: number;
  risk_per_trade_pct: number;
  max_drawdown_pct: number;
  max_concurrent_positions: number;
  max_correlation: number;
  max_sector_exposure_pct: number;
  kelly_fraction: number;
  max_daily_loss_pct: number;
  max_leverage: number;
}

interface Signal {
  symbol: string;
  action: string;
  confidence: number;
  proposed_size: number;
  market_data: any;
}

interface StrategyStats {
  win_rate: number;
  avg_win: number;
  avg_loss: number;
  profit_factor: number;
}

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const supabaseUrl = Deno.env.get('SUPABASE_URL')!;
    const supabaseKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!;
    const supabase = createClient(supabaseUrl, supabaseKey);

    const { signal, account_state } = await req.json();
    
    console.log(`[Risk Manager] Analyzing signal for ${signal.symbol}`);

    // 1. Get risk configuration
    const { data: config } = await supabase
      .from('bot_config')
      .select('*')
      .single();

    const riskConfig: RiskConfig = {
      max_position_size_pct: config?.max_position_size_pct || 20,
      risk_per_trade_pct: config?.risk_per_trade_pct || 1,
      max_drawdown_pct: config?.max_drawdown_pct || 10,
      max_concurrent_positions: config?.max_concurrent_positions || 5,
      max_correlation: 0.7,
      max_sector_exposure_pct: 40,
      kelly_fraction: 0.25, // Quarter-Kelly for safety
      max_daily_loss_pct: 5,
      max_leverage: 10,
    };

    // 2. Get strategy statistics
    const { data: strategyPerf } = await supabase
      .from('strategy_performance')
      .select('*')
      .eq('strategy_name', signal.source || 'default')
      .single();

    const stats: StrategyStats = {
      win_rate: strategyPerf?.win_rate || 0.5,
      avg_win: calculateAvgWin(strategyPerf),
      avg_loss: calculateAvgLoss(strategyPerf),
      profit_factor: strategyPerf?.profit_factor || 1.0,
    };

    // 3. Check circuit breakers
    const circuitBreakers = await checkCircuitBreakers(
      supabase,
      account_state,
      riskConfig
    );

    if (circuitBreakers.triggered) {
      console.log(`[Risk Manager] Circuit breaker triggered: ${circuitBreakers.reason}`);
      return new Response(
        JSON.stringify({
          should_execute: false,
          reason: circuitBreakers.reason,
          risk_score: 100,
          adjusted_size: 0,
          factors: circuitBreakers,
        }),
        { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    // 4. Calculate ATR-based stop loss
    const atr = signal.market_data?.atr_14 || calculateATR(signal.market_data);
    const currentPrice = signal.market_data?.close || signal.market_data?.price;
    const stopLossDistance = atr * 2; // 2x ATR stop
    const stopLoss = signal.action === 'buy' 
      ? currentPrice - stopLossDistance 
      : currentPrice + stopLossDistance;
    
    const riskPerShare = Math.abs(currentPrice - stopLoss);
    
    // 5. Kelly Criterion Position Sizing
    const kellyPct = calculateKellyCriterion(stats, riskConfig.kelly_fraction);
    
    // 6. Get current positions for correlation check
    const { data: positions } = await supabase
      .from('positions')
      .select('*');

    // 7. Calculate correlation adjustment
    const correlationAdjustment = await calculateCorrelationAdjustment(
      signal.symbol,
      positions || [],
      riskConfig.max_correlation
    );

    // 8. Calculate drawdown adjustment
    const drawdownAdjustment = calculateDrawdownAdjustment(
      account_state.equity,
      account_state.peak_equity || account_state.equity,
      riskConfig.max_drawdown_pct
    );

    // 9. Calculate volatility adjustment (ATR-based)
    const volatilityAdjustment = calculateVolatilityAdjustment(atr, currentPrice);

    // 10. Calculate final position size
    const baseRiskAmount = account_state.equity * (riskConfig.risk_per_trade_pct / 100);
    const kellyRiskAmount = account_state.equity * (kellyPct / 100);
    
    // Use the more conservative of the two
    const riskAmount = Math.min(baseRiskAmount, kellyRiskAmount);
    
    // Base size from risk amount
    let positionSize = riskAmount / riskPerShare;
    
    // Apply all adjustments
    positionSize *= drawdownAdjustment;
    positionSize *= volatilityAdjustment;
    positionSize *= correlationAdjustment;
    positionSize *= (signal.confidence || 1.0);
    
    // Apply maximum position size limit
    const maxPositionValue = account_state.equity * (riskConfig.max_position_size_pct / 100);
    const maxShares = maxPositionValue / currentPrice;
    positionSize = Math.min(positionSize, maxShares);
    
    // Apply maximum concurrent positions adjustment
    const positionCount = positions?.length || 0;
    if (positionCount >= riskConfig.max_concurrent_positions - 1) {
      positionSize *= 0.5; // Reduce size by 50% when near limit
    }
    
    // Round down to whole shares
    positionSize = Math.floor(positionSize);

    // 11. Calculate risk score (0-100)
    const riskScore = calculateRiskScore({
      correlationAdjustment,
      drawdownAdjustment,
      volatilityAdjustment,
      positionCount,
      maxPositions: riskConfig.max_concurrent_positions,
      atrPct: (atr / currentPrice) * 100,
    });

    // 12. Determine if should execute
    const should_execute = positionSize > 0 && riskScore < 80;

    // 13. Calculate take profit (3x ATR for 1.5:1 R:R minimum)
    const takeProfitDistance = atr * 3;
    const takeProfit = signal.action === 'buy'
      ? currentPrice + takeProfitDistance
      : currentPrice - takeProfitDistance;

    const response = {
      should_execute,
      adjusted_size: positionSize,
      risk_score: riskScore,
      stop_loss: stopLoss,
      take_profit: takeProfit,
      reason: should_execute 
        ? `Kelly: ${kellyPct.toFixed(1)}%, Risk/Trade: $${riskAmount.toFixed(0)}, ATR: ${atr.toFixed(2)}`
        : `Position size too small or risk too high (score: ${riskScore})`,
      factors: {
        kelly_pct: kellyPct,
        base_risk_amount: riskAmount,
        risk_per_share: riskPerShare,
        atr: atr,
        stop_loss: stopLoss,
        take_profit: takeProfit,
        correlation_adjustment: correlationAdjustment,
        drawdown_adjustment: drawdownAdjustment,
        volatility_adjustment: volatilityAdjustment,
        position_count: positionCount,
        max_positions: riskConfig.max_concurrent_positions,
        strategy_stats: stats,
      },
    };

    // Log the decision
    await supabase.from('system_logs').insert({
      source: 'quantitative-risk-manager',
      level: 'info',
      message: `Risk assessment for ${signal.symbol}: ${should_execute ? 'APPROVED' : 'REJECTED'}`,
      metadata: response,
    });

    console.log(`[Risk Manager] ${should_execute ? 'APPROVED' : 'REJECTED'} - Size: ${positionSize}, Risk Score: ${riskScore}`);

    return new Response(
      JSON.stringify(response),
      { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );

  } catch (error) {
    console.error('[Risk Manager] Error:', error);
    return new Response(
      JSON.stringify({ 
        error: error instanceof Error ? error.message : 'Unknown error',
        should_execute: false,
        reason: 'Risk manager error - rejecting for safety',
        adjusted_size: 0,
        risk_score: 100,
      }),
      { 
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
      }
    );
  }
});

function calculateKellyCriterion(stats: StrategyStats, fraction: number): number {
  // Kelly formula: (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
  const kellyPct = ((stats.win_rate * stats.avg_win - (1 - stats.win_rate) * stats.avg_loss) / stats.avg_win) * 100;
  
  // Apply fractional Kelly (quarter-Kelly for safety)
  const fractionalKelly = kellyPct * fraction;
  
  // Clamp between 0.5% and 5%
  return Math.max(0.5, Math.min(5, fractionalKelly));
}

function calculateAvgWin(perf: any): number {
  if (!perf || perf.winning_trades === 0) return 1.0;
  const totalWin = (perf.profit_factor || 1) * perf.losing_trades * 1.0;
  return totalWin / perf.winning_trades;
}

function calculateAvgLoss(perf: any): number {
  if (!perf || perf.losing_trades === 0) return 1.0;
  return 1.0; // Simplified
}

async function checkCircuitBreakers(
  supabase: any,
  accountState: any,
  config: RiskConfig
): Promise<{ triggered: boolean; reason: string }> {
  // Check 1: Daily loss limit
  const startOfDay = new Date();
  startOfDay.setHours(0, 0, 0, 0);
  
  const { data: todayTrades } = await supabase
    .from('trades')
    .select('pnl')
    .gte('executed_at', startOfDay.toISOString())
    .not('pnl', 'is', null);

  const todayPnL = todayTrades?.reduce((sum: number, t: any) => sum + (t.pnl || 0), 0) || 0;
  const dailyLossPct = Math.abs(todayPnL / accountState.equity) * 100;

  if (todayPnL < 0 && dailyLossPct > config.max_daily_loss_pct) {
    return { 
      triggered: true, 
      reason: `Daily loss limit exceeded: ${dailyLossPct.toFixed(1)}% > ${config.max_daily_loss_pct}%` 
    };
  }

  // Check 2: Max drawdown
  const peakEquity = accountState.peak_equity || accountState.equity;
  const currentDrawdown = ((peakEquity - accountState.equity) / peakEquity) * 100;

  if (currentDrawdown > config.max_drawdown_pct) {
    return { 
      triggered: true, 
      reason: `Max drawdown exceeded: ${currentDrawdown.toFixed(1)}% > ${config.max_drawdown_pct}%` 
    };
  }

  // Check 3: Consecutive losses
  const { data: recentTrades } = await supabase
    .from('trades')
    .select('pnl')
    .order('executed_at', { ascending: false })
    .limit(5)
    .not('pnl', 'is', null);

  const consecutiveLosses = recentTrades?.findIndex((t: any) => t.pnl >= 0) || 0;
  
  if (consecutiveLosses >= 3) {
    return { 
      triggered: true, 
      reason: `3+ consecutive losses detected - pausing for review` 
    };
  }

  return { triggered: false, reason: '' };
}

async function calculateCorrelationAdjustment(
  symbol: string,
  positions: any[],
  maxCorrelation: number
): Promise<number> {
  if (positions.length === 0) return 1.0;

  // Simplified correlation: check if symbol already exists
  const existingPosition = positions.find(p => p.symbol === symbol);
  if (existingPosition) return 0.5; // 50% reduction if already have position

  // Check sector concentration (simplified)
  const symbolCount = positions.length;
  if (symbolCount >= 5) return 0.7; // 30% reduction with many positions

  return 1.0;
}

function calculateDrawdownAdjustment(
  currentEquity: number,
  peakEquity: number,
  maxDrawdownPct: number
): number {
  const drawdownPct = ((peakEquity - currentEquity) / peakEquity) * 100;
  
  if (drawdownPct < maxDrawdownPct * 0.3) return 1.0; // < 30% of max: full size
  if (drawdownPct < maxDrawdownPct * 0.6) return 0.7; // 30-60%: reduce 30%
  if (drawdownPct < maxDrawdownPct * 0.9) return 0.4; // 60-90%: reduce 60%
  return 0.2; // >90% of max drawdown: minimal size
}

function calculateVolatilityAdjustment(atr: number, price: number): number {
  const atrPct = (atr / price) * 100;
  
  if (atrPct < 2) return 1.2; // Low volatility: increase 20%
  if (atrPct < 5) return 1.0; // Normal volatility: no adjustment
  if (atrPct < 8) return 0.7; // High volatility: reduce 30%
  return 0.5; // Extreme volatility: reduce 50%
}

function calculateATR(marketData: any): number {
  // Fallback ATR calculation from OHLC if not provided
  if (marketData?.atr_14) return marketData.atr_14;
  
  const high = marketData?.high || marketData?.price || 100;
  const low = marketData?.low || marketData?.price || 100;
  const close = marketData?.close || marketData?.price || 100;
  
  // Simplified: use 2% of price as default ATR
  return close * 0.02;
}

function calculateRiskScore(factors: {
  correlationAdjustment: number;
  drawdownAdjustment: number;
  volatilityAdjustment: number;
  positionCount: number;
  maxPositions: number;
  atrPct: number;
}): number {
  let score = 0;

  // Correlation risk (0-25 points)
  score += (1 - factors.correlationAdjustment) * 25;

  // Drawdown risk (0-30 points)
  score += (1 - factors.drawdownAdjustment) * 30;

  // Volatility risk (0-25 points)
  score += (1 - factors.volatilityAdjustment) * 25;

  // Position concentration risk (0-20 points)
  const concentrationRisk = factors.positionCount / factors.maxPositions;
  score += concentrationRisk * 20;

  return Math.min(100, Math.max(0, score));
}
