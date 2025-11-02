import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Brain, Zap, TrendingUp, Activity, Cpu, Target, Play, Pause, Database, RefreshCw } from "lucide-react";
import { useStartAutonomousTraining, useRLMetrics, useQState } from "@/lib/api/autonomous-training";
import { useGenerateTrainingData } from "@/lib/api/data-generation";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, ResponsiveContainer } from "recharts";
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";
import { toast } from "sonner";

export default function Training() {
  const [iterations, setIterations] = useState(10);
  const [isAutoRunning, setIsAutoRunning] = useState(false);
  const [autoIntervalId, setAutoIntervalId] = useState<number | null>(null);
  const [isAutoGenerating, setIsAutoGenerating] = useState(false);
  const [autoGenIntervalId, setAutoGenIntervalId] = useState<number | null>(null);
  
  const startTraining = useStartAutonomousTraining();
  const generateData = useGenerateTrainingData();
  const { data: metrics } = useRLMetrics();
  const { data: qState } = useQState();

  const handleStartTraining = () => {
    startTraining.mutate(iterations);
  };

  const handleGenerateData = () => {
    generateData.mutate({
      symbols: ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN"],
      barsPerSymbol: 500,
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
      toast.success("🚀 Autonomous data generation started - running every 5 minutes");
      setIsAutoGenerating(true);
      
      // Run immediately
      handleGenerateData();
      
      // Then run every 5 minutes
      const id = window.setInterval(() => {
        handleGenerateData();
      }, 300000); // 5 minutes
      
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
      toast.success("🚀 Autonomous training started - running every 60 seconds");
      setIsAutoRunning(true);
      
      // Run immediately
      startTraining.mutate(iterations);
      
      // Then run every 60 seconds
      const id = window.setInterval(() => {
        startTraining.mutate(iterations);
      }, 60000);
      
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
  const totalEpisodes = qState?.episode_count || 0;
  const epsilon = qState?.epsilon || 0;
  const qTableSize = qState?.q_table ? Object.keys(qState.q_table).length : 0;
  
  // Calculate weighted accuracy from expert accuracies
  const calculateWeightedAccuracy = () => {
    if (!latestMetric?.expert_accuracies) return 0;
    const expertAccs = latestMetric.expert_accuracies as Record<string, number>;
    const weights: Record<string, number> = {
      "RSI_EMA": 0.40,
      "VWAP_REVERSION": 0.30,
      "TREND_PULLBACK": 0.10,
      "VWAP_DELTA_CONFLUENCE": 0.10,
      "AFTERNOON_FADE": 0.05,
      "LIQUIDITY_SWEEP": 0.05
    };
    
    let totalWeighted = 0;
    let totalWeight = 0;
    
    Object.entries(expertAccs).forEach(([expert, accuracy]) => {
      const weight = weights[expert] || 0;
      totalWeighted += accuracy * weight;
      totalWeight += weight;
    });
    
    return totalWeight > 0 ? (totalWeighted / totalWeight) * 100 : 0;
  };
  
  // Calculate win rate based on positive rewards
  const calculateWinRate = () => {
    if (!metrics || metrics.length === 0) return 0;
    const recentMetrics = metrics.slice(0, 10);
    const positiveRewards = recentMetrics.filter(m => Number(m.avg_reward) > 0).length;
    return (positiveRewards / recentMetrics.length) * 100;
  };
  
  const weightedAccuracy = calculateWeightedAccuracy();
  const winRate = calculateWinRate();

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
        <div className="relative flex items-center gap-4">
          <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10">
            <Brain className="h-8 w-8 text-primary" />
          </div>
          <div>
            <h2 className="mb-2 text-3xl font-bold">Autonomous RL Training</h2>
            <p className="text-muted-foreground">
              Continuous Q-Learning with epsilon-greedy exploration and reward shaping
            </p>
          </div>
        </div>
      </div>

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
            <CardTitle className="text-sm font-medium">Accuracy Score</CardTitle>
            <Target className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{weightedAccuracy.toFixed(1)}%</div>
            <Progress value={weightedAccuracy} className="mt-2" />
            <p className="text-xs text-muted-foreground mt-1">Weighted expert match</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Win Rate</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{winRate.toFixed(1)}%</div>
            <Progress value={winRate} className="mt-2" />
            <p className="text-xs text-muted-foreground mt-1">Last 10 episodes</p>
          </CardContent>
        </Card>
      </div>

      {/* Data Generation */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Database className="h-5 w-5 text-primary" />
                Automatic Data Generation
              </CardTitle>
              <CardDescription>
                Generate synthetic training data for both cloud and local Python training
              </CardDescription>
            </div>
            {isAutoGenerating && (
              <Badge variant="default" className="animate-pulse">
                <RefreshCw className="mr-1 h-3 w-3" />
                Auto-Generating
              </Badge>
            )}
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="rounded-lg border border-primary/20 bg-primary/5 p-4">
              <div className="flex items-center justify-between mb-3">
                <div>
                  <h4 className="font-semibold mb-1">🎲 Manual Generation</h4>
                  <p className="text-sm text-muted-foreground">
                    Creates 500 bars + indicators + expert trajectories for 5 symbols
                  </p>
                </div>
                <Button
                  onClick={handleGenerateData}
                  disabled={generateData.isPending || isAutoGenerating}
                  size="lg"
                >
                  {generateData.isPending ? (
                    <>
                      <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                      Generating...
                    </>
                  ) : (
                    <>
                      <Database className="mr-2 h-4 w-4" />
                      Generate Once
                    </>
                  )}
                </Button>
              </div>
              
              <div className="pt-3 border-t border-primary/10 flex items-center justify-between">
                <div>
                  <h4 className="font-semibold mb-1">🤖 Autonomous Generation</h4>
                  <p className="text-sm text-muted-foreground">
                    Runs every 5 minutes continuously
                  </p>
                </div>
                <Button
                  onClick={handleToggleAutoGeneration}
                  variant={isAutoGenerating ? "destructive" : "default"}
                  disabled={generateData.isPending}
                >
                  {isAutoGenerating ? (
                    <>
                      <Pause className="mr-2 h-4 w-4" />
                      Stop Auto
                    </>
                  ) : (
                    <>
                      <Play className="mr-2 h-4 w-4" />
                      Start Auto
                    </>
                  )}
                </Button>
              </div>
            </div>

            <div className="rounded-lg border border-border bg-muted/50 p-4">
              <h4 className="font-semibold mb-3">📊 Realistic Multi-Market Scenario Generation:</h4>
              
              <div className="space-y-3">
                <div>
                  <h5 className="font-medium text-sm mb-1 flex items-center gap-2">
                    <span className="text-primary">1.</span> Market Regime Detection
                  </h5>
                  <p className="text-xs text-muted-foreground ml-5">
                    Each symbol cycles through 7 realistic market regimes: <strong>STRONG_TREND_UP/DOWN</strong> (clear trends), 
                    <strong>WEAK_TREND_UP/DOWN</strong> (uncertain trends), <strong>SIDEWAYS</strong> (ranging), 
                    <strong>CHOPPY</strong> (whipsaws), <strong>HIGH_VOLATILITY</strong> (erratic). Regimes last 30-100 bars.
                  </p>
                </div>
                
                <div>
                  <h5 className="font-medium text-sm mb-1 flex items-center gap-2">
                    <span className="text-primary">2.</span> Realistic Price Patterns
                  </h5>
                  <p className="text-xs text-muted-foreground ml-5">
                    Price drifts, volatility, and volume adapt to regime. Strong trends have 0.2% drift/bar with 1% vol. 
                    Choppy markets have random 0.1% swings with 2% vol. High vol has 3% swings.
                  </p>
                </div>
                
                <div>
                  <h5 className="font-medium text-sm mb-1 flex items-center gap-2">
                    <span className="text-primary">3.</span> Regime-Aware Expert Strategies
                  </h5>
                  <ul className="space-y-1 text-xs text-muted-foreground ml-5">
                    <li className="flex items-start gap-2">
                      <div className="h-1 w-1 rounded-full bg-primary mt-1.5" />
                      <span><strong>RSI_EMA:</strong> Only trades in TREND regimes (quality 0.9), skips choppy/sideways</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <div className="h-1 w-1 rounded-full bg-primary mt-1.5" />
                      <span><strong>VWAP_REVERSION:</strong> Avoids CHOPPY regime entirely (quality 0.2-0.8)</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <div className="h-1 w-1 rounded-full bg-primary mt-1.5" />
                      <span><strong>TREND_PULLBACK:</strong> Only fires in STRONG_TREND regimes (quality 0.9)</span>
                    </li>
                  </ul>
                </div>
                
                <div>
                  <h5 className="font-medium text-sm mb-1 flex items-center gap-2">
                    <span className="text-primary">4.</span> Stay-Out Logic (CRITICAL)
                  </h5>
                  <p className="text-xs text-muted-foreground ml-5">
                    <strong>70% HOLD probability</strong> in CHOPPY/HIGH_VOL/SIDEWAYS vs 20% in good conditions. 
                    HOLD gets <strong>+0.1 reward</strong> in bad regimes (teaches bot to stay out), -0.05 in good regimes. 
                    Entry quality: 0.8 for HOLD in bad conditions vs 0.3 in good conditions.
                  </p>
                </div>
                
                <div>
                  <h5 className="font-medium text-sm mb-1 flex items-center gap-2">
                    <span className="text-primary">5.</span> Quality & Reward Scaling
                  </h5>
                  <p className="text-xs text-muted-foreground ml-5">
                    All rewards and equity changes scale by regime quality (0.2-0.9). Strong trends = high rewards, 
                    choppy markets = low/negative rewards. Teaches bot to recognize and avoid unfavorable conditions.
                  </p>
                </div>
                
                <div className="pt-2 border-t border-primary/10">
                  <p className="text-xs font-medium text-primary">
                    🎯 Result: Bot learns to scan 5 symbols simultaneously, identify which are in good vs bad regimes, 
                    and only take high-quality trades while staying out of choppy/unclear markets.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Training Control */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Zap className="h-5 w-5 text-primary" />
                Training Controls
              </CardTitle>
              <CardDescription>
                Start manual training sessions or enable fully autonomous mode
              </CardDescription>
            </div>
            {isAutoRunning && (
              <Badge variant="default" className="animate-pulse">
                <Activity className="mr-1 h-3 w-3" />
                Auto-Running
              </Badge>
            )}
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            {/* Manual Training */}
            <div className="space-y-4">
              <div className="flex items-center gap-4">
                <div className="flex-1">
                  <Label htmlFor="iterations">Episodes per run</Label>
                  <Input
                    id="iterations"
                    type="number"
                    min="1"
                    max="100"
                    value={iterations}
                    onChange={(e) => setIterations(parseInt(e.target.value) || 10)}
                    disabled={isAutoRunning}
                  />
                </div>
                <div className="flex-1 flex items-end">
                  <Button 
                    onClick={handleStartTraining}
                    disabled={startTraining.isPending || isAutoRunning}
                    className="w-full"
                  >
                    <Play className="mr-2 h-4 w-4" />
                    {startTraining.isPending ? "Training..." : "Run Training"}
                  </Button>
                </div>
              </div>
            </div>

            <div className="relative">
              <div className="absolute inset-0 flex items-center">
                <span className="w-full border-t" />
              </div>
              <div className="relative flex justify-center text-xs uppercase">
                <span className="bg-card px-2 text-muted-foreground">Or</span>
              </div>
            </div>

            {/* Autonomous Mode */}
            <div className="rounded-lg border border-primary/20 bg-primary/5 p-4">
              <div className="flex items-center justify-between">
                <div>
                  <h4 className="font-semibold mb-1">🤖 Fully Autonomous Mode</h4>
                  <p className="text-sm text-muted-foreground">
                    Runs {iterations} episodes every 60 seconds continuously
                  </p>
                </div>
                <Button
                  onClick={handleToggleAutoTraining}
                  variant={isAutoRunning ? "destructive" : "default"}
                  disabled={startTraining.isPending}
                >
                  {isAutoRunning ? (
                    <>
                      <Pause className="mr-2 h-4 w-4" />
                      Stop Auto
                    </>
                  ) : (
                    <>
                      <Play className="mr-2 h-4 w-4" />
                      Start Auto
                    </>
                  )}
                </Button>
              </div>
            </div>

            {/* Info Box */}
            <div className="rounded-lg border border-border bg-muted/50 p-4">
              <h4 className="font-semibold mb-2">How it works:</h4>
              <ul className="space-y-1 text-sm text-muted-foreground">
                <li className="flex items-start gap-2">
                  <div className="h-1.5 w-1.5 rounded-full bg-primary mt-1.5" />
                  <span><strong>Q-Learning:</strong> Learns optimal actions for each market state</span>
                </li>
                <li className="flex items-start gap-2">
                  <div className="h-1.5 w-1.5 rounded-full bg-primary mt-1.5" />
                  <span><strong>Epsilon-greedy:</strong> Balances exploration (random) vs exploitation (best known)</span>
                </li>
                <li className="flex items-start gap-2">
                  <div className="h-1.5 w-1.5 rounded-full bg-primary mt-1.5" />
                  <span><strong>Reward shaping:</strong> +0.1 bonus for trading, -0.05 penalty for holding, extra penalty for long holds</span>
                </li>
                <li className="flex items-start gap-2">
                  <div className="h-1.5 w-1.5 rounded-full bg-primary mt-1.5" />
                  <span><strong>Continuous learning:</strong> Improves over time, epsilon decays from 30% to 5% over 1000 episodes</span>
                </li>
              </ul>
            </div>
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
