import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Brain, Zap, TrendingUp, Activity, Cpu, Target, Play, Pause } from "lucide-react";
import { useStartAutonomousTraining, useRLMetrics, useQState } from "@/lib/api/autonomous-training";
import { Line } from "recharts";
import { ChartContainer, ChartTooltip } from "@/components/ui/chart";
import { toast } from "sonner";

export default function Training() {
  const [iterations, setIterations] = useState(10);
  const [isAutoRunning, setIsAutoRunning] = useState(false);
  const [autoIntervalId, setAutoIntervalId] = useState<number | null>(null);
  
  const startTraining = useStartAutonomousTraining();
  const { data: metrics } = useRLMetrics();
  const { data: qState } = useQState();

  const handleStartTraining = () => {
    startTraining.mutate(iterations);
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
    };
  }, [autoIntervalId]);

  const latestMetric = metrics?.[0];
  const totalEpisodes = qState?.episode_count || 0;
  const epsilon = qState?.epsilon || 0;
  const qTableSize = qState?.q_table ? Object.keys(qState.q_table).length : 0;

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
      <div className="grid gap-4 md:grid-cols-4">
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
      </div>

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

      {/* Training Progress Chart */}
      {chartData.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Training Progress</CardTitle>
            <CardDescription>Average reward and exploration rate over recent episodes</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[300px]">
              <ChartContainer
                config={{
                  reward: { label: "Avg Reward", color: "hsl(var(--primary))" },
                  epsilon: { label: "Epsilon", color: "hsl(var(--destructive))" },
                }}
              >
                <Line
                  data={chartData}
                  dataKey="reward"
                  stroke="var(--color-reward)"
                  strokeWidth={2}
                />
                <ChartTooltip />
              </ChartContainer>
            </div>
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
                      ε={Number(metric.epsilon).toFixed(3)} | {metric.q_table_size} states
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
