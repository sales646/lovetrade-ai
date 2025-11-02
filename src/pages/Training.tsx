import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Brain, Zap, TrendingUp, Activity, Cpu, Target, Play, Pause, Database, RefreshCw } from "lucide-react";
import { useStartAutonomousTraining, useRLMetrics, useQState } from "@/lib/api/autonomous-training";
import { useAlpacaAccount } from "@/lib/api/alpaca";
import { useGenerateTrainingData } from "@/lib/api/data-generation";
import { useLocalGPUTrainingStatus } from "@/lib/api/training";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, ResponsiveContainer } from "recharts";
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";
import { toast } from "sonner";

export default function Training() {
  const [iterations, setIterations] = useState(10);
  const [isAutoRunning, setIsAutoRunning] = useState(false);
  const [autoIntervalId, setAutoIntervalId] = useState<number | null>(null);
  const [isAutoGenerating, setIsAutoGenerating] = useState(false);
  const [autoGenIntervalId, setAutoGenIntervalId] = useState<number | null>(null);
  const [useRealData, setUseRealData] = useState(false);
  
  const startTraining = useStartAutonomousTraining();
  const generateData = useGenerateTrainingData();
  const { data: metrics } = useRLMetrics();
  const { data: qState } = useQState();
  const { data: alpacaAccount, isLoading: alpacaLoading, error: alpacaError } = useAlpacaAccount();
  const { data: localGPUStatus } = useLocalGPUTrainingStatus();

  const handleStartTraining = () => {
    startTraining.mutate(iterations);
  };

  const handleGenerateData = () => {
    generateData.mutate({
      symbols: ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL"],
      barsPerSymbol: useRealData ? 1000 : 500,
      useRealData,
    });
  };

  const handleToggleAutoGeneration = () => {
    if (isAutoGenerating) {
      // Stop auto generation
      if (autoGenIntervalId) {
        clearInterval(autoGenIntervalId);
        setAutoGenIntervalId(null);
      }
      setIsAutoGenerating(false);
      toast.info("🛑 Autonomous data generation stopped");
    } else {
      // Start auto generation
      toast.success("🚀 Autonomous data generation started - running every 30 seconds");
      setIsAutoGenerating(true);
      
      // Run immediately
      handleGenerateData();
      
      // Then run every 30 seconds
      const id = window.setInterval(() => {
        generateData.mutate({
          symbols: ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL"],
          barsPerSymbol: useRealData ? 1000 : 500,
          useRealData,
        });
      }, 30000); // 30 seconds
      
      setAutoGenIntervalId(id);
    }
  };

  const handleToggleAutoTraining = () => {
    if (isAutoRunning) {
      // Stop auto training
      if (autoIntervalId) {
        clearInterval(autoIntervalId);
        setAutoIntervalId(null);
      }
      setIsAutoRunning(false);
      toast.info("🛑 Autonomous training stopped");
    } else {
      // Start auto training
      toast.success("🚀 Autonomous training started - running every 30 seconds");
      setIsAutoRunning(true);
      
      // Run immediately
      startTraining.mutate(iterations);
      
      // Then run every 30 seconds
      const id = window.setInterval(() => {
        startTraining.mutate(iterations);
      }, 30000);
      
      setAutoIntervalId(id);
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (autoIntervalId) {
        clearInterval(autoIntervalId);
      }
      if (autoGenIntervalId) {
        clearInterval(autoGenIntervalId);
      }
    };
  }, [autoIntervalId, autoGenIntervalId]);

  const latestMetric = metrics?.[0];
  const totalEpisodes = qState?.episode_count || latestMetric?.total_episodes || 0;
  const epsilon = qState?.epsilon || latestMetric?.epsilon || 0;
  const qTableSize = qState?.q_table ? Object.keys(qState.q_table).length : 0;
  
  // Calculate actual trading performance (% of profitable batches)
  const calculatePerformanceScore = () => {
    if (!metrics || metrics.length === 0) return 0;
    const recent = metrics.slice(0, Math.min(20, metrics.length));
    const profitableBatches = recent.filter(m => Number(m.avg_reward) > 0).length;
    return (profitableBatches / recent.length) * 100;
  };
  
  // Calculate average reward trend (last 20 batches)
  const calculateAvgRewardTrend = () => {
    if (!metrics || metrics.length < 2) return 0;
    const recent = metrics.slice(0, Math.min(20, metrics.length));
    const avgRecent = recent.reduce((sum, m) => sum + Number(m.avg_reward), 0) / recent.length;
    return avgRecent;
  };
  
  const performanceScore = calculatePerformanceScore();
  const avgRewardTrend = calculateAvgRewardTrend();

  // Prepare chart data
  const chartData = metrics?.slice(0, 20).reverse().map((m, i) => ({
    episode: totalEpisodes - metrics.length + i + 1,
    reward: Number(m.avg_reward),
    epsilon: Number(m.epsilon),
  })) || [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="relative overflow-hidden rounded-2xl border border-border bg-gradient-to-br from-card via-card to-card/50 p-8">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-primary/20 via-transparent to-transparent opacity-50" />
        <div className="relative flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10">
              <Brain className="h-8 w-8 text-primary" />
            </div>
            <div>
              <h2 className="mb-2 text-3xl font-bold">Autonomous RL Trading</h2>
              <p className="text-muted-foreground">
                Position-sized trades using your Alpaca account balance with outcome-based rewards from real P&L
              </p>
            </div>
          </div>
          {localGPUStatus?.isActive && (
            <Badge variant="default" className="animate-pulse flex items-center gap-2">
              <Cpu className="h-4 w-4" />
              Local GPU Active ({localGPUStatus.activeCount} worker{localGPUStatus.activeCount !== 1 ? 's' : ''})
            </Badge>
          )}
        </div>
      </div>

      {/* Alpaca Account Info */}
      {alpacaAccount && (
        <Card className="border-blue-500/30 bg-blue-500/5">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Database className="h-5 w-5 text-blue-500" />
              Alpaca Account
            </CardTitle>
            <CardDescription>
              Live trading account data {alpacaError && "(Failed to fetch)"}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <div className="text-sm text-muted-foreground mb-1">Equity</div>
                <div className="text-2xl font-bold text-green-600">
                  ${alpacaAccount.equity.toLocaleString()}
                </div>
              </div>
              <div>
                <div className="text-sm text-muted-foreground mb-1">Buying Power</div>
                <div className="text-2xl font-bold text-blue-600">
                  ${alpacaAccount.buyingPower.toLocaleString()}
                </div>
              </div>
              <div>
                <div className="text-sm text-muted-foreground mb-1">Cash</div>
                <div className="text-2xl font-bold">
                  ${alpacaAccount.cash.toLocaleString()}
                </div>
              </div>
              <div>
                <div className="text-sm text-muted-foreground mb-1">Status</div>
                <div className="flex flex-col gap-1">
                  <Badge variant={alpacaAccount.accountBlocked ? "destructive" : "default"}>
                    {alpacaAccount.status}
                  </Badge>
                  {alpacaAccount.tradingBlocked && (
                    <Badge variant="destructive" className="text-xs">Trading Blocked</Badge>
                  )}
                  <div className="text-xs text-muted-foreground mt-1">
                    Day trades: {alpacaAccount.daytradeCount}
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {!alpacaAccount && !alpacaLoading && (
        <Card className="border-yellow-500/30 bg-yellow-500/5">
          <CardContent className="pt-6">
            <p className="text-sm text-muted-foreground">
              ⚠️ Alpaca account not configured. Using default $100k for simulations.
              {alpacaError && ` Error: ${alpacaError.message}`}
            </p>
          </CardContent>
        </Card>
      )}

      {/* System Flow Visualization */}
      <Card className="border-primary/20">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5 text-primary" />
            System Architecture
          </CardTitle>
          <CardDescription>How the aggressive trading strategy learns from real outcomes</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="bg-muted/30 rounded-lg p-4 overflow-x-auto">
              <pre className="text-xs font-mono whitespace-pre">
{`graph TB
    A[Market Data] -->|Yahoo Finance / Synthetic| B[OHLCV Bars]
    B --> C[Technical Indicators]
    C -->|RSI ATR VWAP EMAs| D{Signal Detection}
    
    D -->|RSI 30 Trend Up| E1[BUY Signal]
    D -->|VWAP Distance -1.5%| E2[VWAP Reversion]
    D -->|Strong Trend Pullback| E3[Trend Entry]
    
    E1 --> F[Trade Simulation]
    E2 --> F
    E3 --> F
    
    F -->|Entry at Signal| G[Walk Through Bars]
    G --> H{Exit Condition?}
    
    H -->|Low Stop 1.5xATR| I1[Stop Loss -2.5%]
    H -->|High Target 6xATR| I2[Take Profit +10%]
    H -->|Max 10-15 Bars| I3[Time Exit 0%]
    
    I1 --> J[Calculate P&L]
    I2 --> J
    I3 --> J
    
    J -->|Net - Fees Slip| K[Real Reward]
    K -->|+5% = +1.5 reward| L[Q-Learning Update]
    K -->|-2.5% = -0.75 reward| L
    
    L --> M[Improved Policy]
    M -->|Every 60s| A`}
              </pre>
            </div>
            
            <div className="grid grid-cols-3 gap-3 text-xs">
              <div className="rounded-lg border border-border bg-muted/30 p-3">
                <div className="font-semibold mb-2">⚡ Realistic Settings</div>
                <div className="space-y-1 text-muted-foreground">
                  <div>• Stop: <span className="text-red-500 font-mono">1.5×ATR</span></div>
                  <div>• Target: <span className="text-green-500 font-mono">3.5%</span> fixed</div>
                  <div>• Max Hold: <span className="font-mono">12 bars</span> (1hr)</div>
                  <div>• Fees: <span className="font-mono">0.12%</span> | Slip: <span className="font-mono">0.08%</span></div>
                </div>
              </div>
              
              <div className="rounded-lg border border-border bg-muted/30 p-3">
                <div className="font-semibold mb-2">💰 Position Sizing</div>
                <div className="space-y-1 text-muted-foreground">
                  <div>• Risk: <span className="text-yellow-500 font-mono">100% buying power</span></div>
                  <div>• Full account on each trade</div>
                  <div>• Shares = BuyingPower / StopDistance</div>
                  <div>• Rewards scaled by <span className="font-mono">$P&L</span></div>
                </div>
              </div>

              <div className="rounded-lg border border-border bg-muted/30 p-3">
                <div className="font-semibold mb-2">🎯 Real Outcomes</div>
                <div className="space-y-1 text-muted-foreground">
                  <div>• Backtested through actual bars</div>
                  <div>• Stop/target checked each bar</div>
                  <div>• Real P&L → Reward mapping</div>
                  <div>• No fake "setup quality" scores</div>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Stats */}
      <div className="grid gap-4 md:grid-cols-3 lg:grid-cols-6">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Episodes</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{totalEpisodes.toLocaleString()}</div>
            <p className="text-xs text-muted-foreground">Training iterations</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Reward</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {latestMetric ? Number(latestMetric.avg_reward).toFixed(2) : "—"}
            </div>
            <p className="text-xs text-muted-foreground">Latest episode</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Exploration (ε)</CardTitle>
            <Target className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{(epsilon * 100).toFixed(1)}%</div>
            <Progress value={epsilon * 100} className="mt-2" />
            <p className="text-xs text-muted-foreground mt-1">
              {epsilon > 0.2 ? "High exploration" : epsilon > 0.1 ? "Balanced" : "Exploitation focused"}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Q-Table Size</CardTitle>
            <Cpu className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{qTableSize.toLocaleString()}</div>
            <p className="text-xs text-muted-foreground">Unique states</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Win Rate</CardTitle>
            <Target className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {latestMetric?.win_rate_pct ? `${Number(latestMetric.win_rate_pct).toFixed(1)}%` : "—"}
            </div>
            <Progress 
              value={latestMetric?.win_rate_pct ? Number(latestMetric.win_rate_pct) : 0} 
              className="mt-2" 
            />
            <p className="text-xs text-muted-foreground mt-1">
              {latestMetric?.winning_trades && latestMetric?.total_trades 
                ? `${latestMetric.winning_trades}/${latestMetric.total_trades} profitable trades`
                : "Latest batch"}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Return %</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${latestMetric?.avg_return_pct && Number(latestMetric.avg_return_pct) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              {latestMetric?.avg_return_pct 
                ? `${Number(latestMetric.avg_return_pct) >= 0 ? '+' : ''}${Number(latestMetric.avg_return_pct).toFixed(2)}%`
                : "—"}
            </div>
            <p className="text-xs text-muted-foreground mt-1">Per trade (net of fees)</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Dollar P&L</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${latestMetric?.avg_dollar_pnl && Number(latestMetric.avg_dollar_pnl) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              {latestMetric?.avg_dollar_pnl 
                ? `$${Number(latestMetric.avg_dollar_pnl) >= 0 ? '+' : ''}${Number(latestMetric.avg_dollar_pnl).toFixed(2)}`
                : "—"}
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              {latestMetric?.account_equity 
                ? `Account: $${Number(latestMetric.account_equity).toLocaleString()}`
                : "Per trade in dollars"}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Data Generation */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="h-5 w-5 text-primary" />
            Autonomous Trading System
          </CardTitle>
          <CardDescription>
            One-click control for the entire training pipeline
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Master Control */}
          <div className="rounded-lg border-2 border-primary/30 bg-gradient-to-br from-primary/10 to-primary/5 p-6">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="text-xl font-bold mb-2">Master Control</h3>
                <p className="text-sm text-muted-foreground mb-1">
                  {isAutoGenerating && isAutoRunning 
                    ? "🟢 System fully autonomous - generating data & training every 30s"
                    : isAutoGenerating 
                    ? "🟡 Data generation active - start training to complete the loop"
                    : isAutoRunning
                    ? "🟡 Training active - start data generation to complete the loop"
                    : "⚪ System idle - click Start to begin"}
                </p>
                <div className="flex gap-2 mt-2">
                  <Badge variant={useRealData ? "default" : "secondary"} className="text-xs">
                    {useRealData ? "Real Market Data" : "Synthetic Data"}
                  </Badge>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setUseRealData(!useRealData)}
                    className="h-6 px-2 text-xs"
                  >
                    Switch
                  </Button>
                </div>
              </div>
              <div className="flex flex-col gap-2">
                <Button
                  onClick={() => {
                    if (!isAutoGenerating) handleToggleAutoGeneration();
                    if (!isAutoRunning) handleToggleAutoTraining();
                  }}
                  disabled={(isAutoGenerating && isAutoRunning) || generateData.isPending || startTraining.isPending}
                  size="lg"
                  className="min-w-[140px]"
                >
                  {(isAutoGenerating && isAutoRunning) ? (
                    <>
                      <Activity className="mr-2 h-5 w-5 animate-pulse" />
                      Running
                    </>
                  ) : (
                    <>
                      <Play className="mr-2 h-5 w-5" />
                      Start System
                    </>
                  )}
                </Button>
                {(isAutoGenerating || isAutoRunning) && (
                  <Button
                    onClick={() => {
                      if (isAutoGenerating) handleToggleAutoGeneration();
                      if (isAutoRunning) handleToggleAutoTraining();
                    }}
                    variant="destructive"
                    size="lg"
                    className="min-w-[140px]"
                  >
                    <Pause className="mr-2 h-5 w-5" />
                    Stop All
                  </Button>
                )}
              </div>
            </div>
          </div>

          {/* System Visualization */}
          <div className="rounded-lg border border-border bg-muted/30 p-4">
            <h4 className="font-semibold mb-3 flex items-center gap-2">
              <Brain className="h-4 w-4 text-primary" />
              Aggressive Trading System Flow
            </h4>
            <div className="bg-card rounded-lg p-4 mb-4">
              <div className="text-xs font-mono space-y-2">
                <div className="flex items-center gap-2">
                  <span className="text-green-500">1.</span>
                  <span>Market Data → Real/Synthetic OHLCV bars</span>
                </div>
                <div className="flex items-center gap-2 ml-4">
                  <span className="text-blue-500">↓</span>
                  <span>Calculate RSI, ATR, VWAP, EMAs</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-green-500">2.</span>
                  <span>Signal Detection → RSI_EMA, VWAP_REVERSION, TREND_PULLBACK</span>
                </div>
                <div className="flex items-center gap-2 ml-4">
                  <span className="text-blue-500">↓</span>
                  <span>Trade Simulation → Entry at signal, track through bars</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-green-500">3.</span>
                  <span>Exit Logic → Stop: 1.5×ATR | Target: 3.5% | Max: 12 bars (1hr)</span>
                </div>
                <div className="flex items-center gap-2 ml-4">
                  <span className="text-blue-500">↓</span>
                  <span>P&L Calculation → (Exit - Entry) - Fees (0.12%) - Slippage (0.08%)</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-green-500">4.</span>
                  <span>Reward Assignment → +5% = +1.5 reward | -2.5% = -0.75 reward</span>
                </div>
                <div className="flex items-center gap-2 ml-4">
                  <span className="text-blue-500">↓</span>
                  <span>Q-Learning → Update action values based on real outcomes</span>
                </div>
              </div>
            </div>
            
            <div className="grid grid-cols-3 gap-2 text-xs">
              <div className="rounded border border-green-500/30 bg-green-500/10 p-2">
                <div className="font-semibold text-green-600">Realistic Profile</div>
                <div className="text-muted-foreground mt-1">
                  • Tight 1.5×ATR stops<br/>
                  • Fixed 3.5% targets<br/>
                  • Fast 12 bar exits (1hr)
                </div>
              </div>
              <div className="rounded border border-blue-500/30 bg-blue-500/10 p-2">
                <div className="font-semibold text-blue-600">Real Outcomes</div>
                <div className="text-muted-foreground mt-1">
                  • Backtested P&L<br/>
                  • Not rule-based scores<br/>
                  • Actual win/loss data
                </div>
              </div>
              <div className="rounded border border-purple-500/30 bg-purple-500/10 p-2">
                <div className="font-semibold text-purple-600">Learning Loop</div>
                <div className="text-muted-foreground mt-1">
                  • Every 30 seconds<br/>
                  • 5 symbols tested<br/>
                  • Q-table updated
                </div>
              </div>
            </div>
          </div>

          {/* Manual Override */}
          <details className="rounded-lg border border-border">
            <summary className="cursor-pointer p-4 font-semibold hover:bg-muted/50">
              Advanced: Manual Controls
            </summary>
            <div className="p-4 pt-0 space-y-4 border-t">
              <div className="flex items-center justify-between">
                <div>
                  <h4 className="font-semibold">Data Generation</h4>
                  <p className="text-xs text-muted-foreground">Generate {useRealData ? "1000 real" : "500 synthetic"} bars for 5 symbols</p>
                </div>
                <Button onClick={handleGenerateData} disabled={generateData.isPending} size="sm">
                  {generateData.isPending ? "Generating..." : "Generate Once"}
                </Button>
              </div>
              <div className="flex items-center justify-between">
                <div>
                  <h4 className="font-semibold">Training Session</h4>
                  <p className="text-xs text-muted-foreground">Run {iterations} training episodes</p>
                </div>
                <div className="flex gap-2 items-center">
                  <Input
                    type="number"
                    min="1"
                    max="100"
                    value={iterations}
                    onChange={(e) => setIterations(parseInt(e.target.value) || 10)}
                    className="w-20 h-9"
                  />
                  <Button onClick={handleStartTraining} disabled={startTraining.isPending} size="sm">
                    {startTraining.isPending ? "Training..." : "Train Once"}
                  </Button>
                </div>
              </div>
            </div>
          </details>
        </CardContent>
      </Card>

      {/* Local GPU Training */}
      <Card className="border-2 border-primary/20 bg-primary/5">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Cpu className="h-5 w-5 text-primary" />
            Local GPU Training (Optional)
          </CardTitle>
          <CardDescription>
            Supercharge training with your CUDA GPU - 5-10x faster than cloud
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* GPU Status & Stats */}
          <div className="rounded-lg border-2 border-primary/30 bg-gradient-to-br from-primary/10 to-primary/5 p-6">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="text-xl font-bold mb-2">GPU Status</h3>
                {localGPUStatus?.isActive ? (
                  <div className="space-y-1">
                    <p className="text-sm text-muted-foreground">
                      🟢 {localGPUStatus.activeCount} worker{localGPUStatus.activeCount !== 1 ? 's' : ''} active
                    </p>
                    <p className="text-xs text-muted-foreground">
                      Total scenarios trained: <span className="font-mono font-bold text-primary">{localGPUStatus.totalEpisodes || 0}</span>
                    </p>
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground">⚪ No GPU workers detected</p>
                )}
              </div>
              {localGPUStatus?.isActive && (
                <Badge variant="default" className="animate-pulse text-lg px-4 py-2">
                  <Activity className="mr-2 h-5 w-5" />
                  Training
                </Badge>
              )}
            </div>
            
            {/* Quick Stats */}
            {localGPUStatus?.recentRuns && localGPUStatus.recentRuns.length > 0 && (
              <div className="grid grid-cols-3 gap-3 mt-4">
                <div className="rounded-lg border border-border bg-card p-3">
                  <div className="text-xs text-muted-foreground mb-1">Recent Runs</div>
                  <div className="text-2xl font-bold">{localGPUStatus.recentRuns.length}</div>
                </div>
                <div className="rounded-lg border border-border bg-card p-3">
                  <div className="text-xs text-muted-foreground mb-1">Active Workers</div>
                  <div className="text-2xl font-bold">{localGPUStatus.activeCount}</div>
                </div>
                <div className="rounded-lg border border-border bg-card p-3">
                  <div className="text-xs text-muted-foreground mb-1">Total Episodes</div>
                  <div className="text-2xl font-bold">{localGPUStatus.totalEpisodes || 0}</div>
                </div>
              </div>
            )}
          </div>

          {/* One-Click Setup */}
          <div className="space-y-4">
            <div>
              <h4 className="font-semibold mb-3 flex items-center gap-2">
                <Zap className="h-4 w-4 text-primary" />
                One-Click GPU Training
              </h4>
              <div className="grid gap-3 md:grid-cols-2">
                <div className="rounded-lg border-2 border-green-500/30 bg-green-500/10 p-4 hover:border-green-500/50 transition-colors cursor-pointer"
                     onClick={() => window.open('file:///'+window.location.pathname.split('/').slice(0, -1).join('/')+'/python_training/START_GPU_TRAINING.bat', '_self')}>
                  <div className="flex items-start justify-between mb-2">
                    <div>
                      <h5 className="font-bold text-green-600 mb-1">Single Worker</h5>
                      <p className="text-xs text-muted-foreground">One BC + PPO training run</p>
                    </div>
                    <Play className="h-5 w-5 text-green-600" />
                  </div>
                  <code className="text-xs bg-card px-2 py-1 rounded block mt-2">START_GPU_TRAINING.bat</code>
                  <div className="text-xs text-muted-foreground mt-2">⏱️ ~2-5 minutes per run</div>
                </div>
                
                <div className="rounded-lg border-2 border-blue-500/30 bg-blue-500/10 p-4 hover:border-blue-500/50 transition-colors cursor-pointer"
                     onClick={() => window.open('file:///'+window.location.pathname.split('/').slice(0, -1).join('/')+'/python_training/START_PARALLEL_GPU_TRAINING.bat', '_self')}>
                  <div className="flex items-start justify-between mb-2">
                    <div>
                      <h5 className="font-bold text-blue-600 mb-1">Parallel (5 Workers)</h5>
                      <p className="text-xs text-muted-foreground">Continuous training, 5x throughput</p>
                    </div>
                    <Activity className="h-5 w-5 text-blue-600 animate-pulse" />
                  </div>
                  <code className="text-xs bg-card px-2 py-1 rounded block mt-2">START_PARALLEL_GPU_TRAINING.bat</code>
                  <div className="text-xs text-muted-foreground mt-2">🔥 Recommended for aggressive training</div>
                </div>
              </div>
            </div>

            <div className="rounded-lg border border-primary/20 bg-primary/5 p-4">
              <h5 className="font-semibold text-sm mb-2">How It Works:</h5>
              <ol className="space-y-1 text-xs text-muted-foreground">
                <li className="flex items-start gap-2">
                  <span className="font-bold text-primary">1.</span>
                  <span>Cloud generates new data every 60s → Stored in Supabase</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="font-bold text-primary">2.</span>
                  <span>GPU workers fetch latest data → Train BC + PPO policies</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="font-bold text-primary">3.</span>
                  <span>Results sync back to dashboard → Real-time metrics visible here</span>
                </li>
              </ol>
            </div>

            {/* Recent Runs Table */}
            {localGPUStatus?.recentRuns && localGPUStatus.recentRuns.length > 0 && (
              <details className="rounded-lg border border-border">
                <summary className="cursor-pointer p-4 font-semibold hover:bg-muted/50">
                  Recent GPU Training Runs ({localGPUStatus.recentRuns.length})
                </summary>
                <div className="p-4 pt-0 space-y-2 border-t">
                  {localGPUStatus.recentRuns.slice(0, 5).map((run) => (
                    <div key={run.id} className="flex items-center justify-between p-2 rounded bg-muted/30">
                      <div className="flex-1">
                        <div className="font-medium text-sm">{run.run_name}</div>
                        <div className="text-xs text-muted-foreground">
                          Phase: {run.phase} | {run.current_epoch}/{run.total_epochs} epochs
                        </div>
                      </div>
                      <Badge variant={run.status === "running" ? "default" : run.status === "completed" ? "secondary" : "destructive"}>
                        {run.status}
                      </Badge>
                    </div>
                  ))}
                </div>
              </details>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Expert Imitation Stats */}
      {latestMetric && latestMetric.expert_accuracies && (
        <Card>
          <CardHeader>
            <CardTitle>Expert Imitation Performance</CardTitle>
            <CardDescription>Weighted accuracy vs expert demonstrations</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
              {Object.entries(latestMetric.expert_accuracies as Record<string, number>)
                .sort((a, b) => b[1] - a[1])
                .map(([expert, accuracy]) => {
                  const weight = expert === "RSI_EMA" ? 0.40 
                    : expert === "VWAP_REVERSION" ? 0.30
                    : expert === "TREND_PULLBACK" ? 0.10
                    : expert === "VWAP_DELTA_CONFLUENCE" ? 0.10
                    : expert === "AFTERNOON_FADE" ? 0.05
                    : expert === "LIQUIDITY_SWEEP" ? 0.05 : 0;
                  
                  return (
                    <div key={expert} className="rounded-lg border border-border bg-muted/50 p-4">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium">{expert.replace(/_/g, " ")}</span>
                        <Badge variant="outline" className="text-xs">
                          w={weight.toFixed(2)}
                        </Badge>
                      </div>
                      <div className="text-2xl font-bold">{(accuracy * 100).toFixed(1)}%</div>
                      <Progress value={accuracy * 100} className="mt-2" />
                    </div>
                  );
                })}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Loss & Action Distribution */}
      {latestMetric && (
        <div className="grid gap-4 md:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle>Loss Components</CardTitle>
              <CardDescription>Imitation vs RL loss contributions</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-sm text-muted-foreground">L_imitation (α={latestMetric.alpha_mix?.toFixed(2) || "—"})</span>
                  <span className="text-sm font-medium">{Number(latestMetric.l_imitation || 0).toFixed(4)}</span>
                </div>
                <Progress value={(latestMetric.alpha_mix || 0) * 100} className="h-2" />
              </div>
              <div>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-sm text-muted-foreground">L_rl (1-α={(1 - (latestMetric.alpha_mix || 0)).toFixed(2)})</span>
                  <span className="text-sm font-medium">{Number(latestMetric.l_rl || 0).toFixed(4)}</span>
                </div>
                <Progress value={(1 - (latestMetric.alpha_mix || 0)) * 100} className="h-2" />
              </div>
              <div className="pt-2 border-t">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-semibold">L_total</span>
                  <span className="text-lg font-bold">{Number(latestMetric.l_total || 0).toFixed(4)}</span>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Action Distribution</CardTitle>
              <CardDescription>Policy action preferences</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-sm text-muted-foreground">BUY</span>
                  <span className="text-sm font-medium">{Number(latestMetric.action_buy_pct || 0).toFixed(1)}%</span>
                </div>
                <Progress value={Number(latestMetric.action_buy_pct || 0)} className="h-2 [&>div]:bg-green-500" />
              </div>
              <div>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-sm text-muted-foreground">HOLD</span>
                  <span className="text-sm font-medium">{Number(latestMetric.action_hold_pct || 0).toFixed(1)}%</span>
                </div>
                <Progress value={Number(latestMetric.action_hold_pct || 0)} className="h-2 [&>div]:bg-yellow-500" />
              </div>
              <div>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-sm text-muted-foreground">SELL</span>
                  <span className="text-sm font-medium">{Number(latestMetric.action_sell_pct || 0).toFixed(1)}%</span>
                </div>
                <Progress value={Number(latestMetric.action_sell_pct || 0)} className="h-2 [&>div]:bg-red-500" />
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Training Progress Chart */}
      {chartData.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Training Progress</CardTitle>
            <CardDescription>Average reward and exploration rate over recent episodes</CardDescription>
          </CardHeader>
          <CardContent>
            <ChartContainer
              config={{
                reward: { 
                  label: "Avg Reward", 
                  color: "hsl(var(--primary))" 
                },
                epsilon: { 
                  label: "Epsilon (ε)", 
                  color: "hsl(var(--chart-2))" 
                },
              }}
              className="h-[300px]"
            >
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                  <XAxis 
                    dataKey="episode" 
                    className="text-xs"
                    label={{ value: "Episode", position: "insideBottom", offset: -5 }}
                  />
                  <YAxis 
                    className="text-xs"
                    label={{ value: "Reward", angle: -90, position: "insideLeft" }}
                  />
                  <ChartTooltip content={<ChartTooltipContent />} />
                  <Line
                    type="monotone"
                    dataKey="reward"
                    stroke="hsl(var(--primary))"
                    strokeWidth={2}
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="epsilon"
                    stroke="hsl(var(--chart-2))"
                    strokeWidth={2}
                    dot={false}
                    strokeDasharray="5 5"
                  />
                </LineChart>
              </ResponsiveContainer>
            </ChartContainer>
          </CardContent>
        </Card>
      )}

      {/* Recent Training Runs */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Training Sessions</CardTitle>
          <CardDescription>Latest autonomous training episodes</CardDescription>
        </CardHeader>
        <CardContent>
          {metrics && metrics.length > 0 ? (
            <div className="space-y-2">
              {metrics.slice(0, 10).map((metric) => (
                <div
                  key={metric.id}
                  className="flex items-center justify-between rounded-lg border border-border bg-muted/50 p-3"
                >
                  <div>
                    <div className="text-sm font-medium">
                      {metric.episodes} episodes
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {new Date(metric.created_at).toLocaleString()}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-sm font-semibold">
                      Reward: {Number(metric.avg_reward).toFixed(2)}
                    </div>
                    <div className="text-xs text-muted-foreground">
                      L_total={Number(metric.l_total || 0).toFixed(4)} | ε={Number(metric.epsilon).toFixed(3)}
                    </div>
                    <div className="text-xs text-muted-foreground">
                      α_imit={Number(metric.alpha_mix || 0).toFixed(2)} | {metric.q_table_size} states
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="py-8 text-center text-muted-foreground">
              <Brain className="h-12 w-12 mx-auto mb-3 opacity-50" />
              <p>No training sessions yet</p>
              <p className="text-sm mt-1">Start training to see results here</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
