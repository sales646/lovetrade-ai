import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Brain, Database, Download, TrendingUp, BarChart3, Activity, Zap, Target } from "lucide-react";
import { useFetchMarketData, useStoredSymbols } from "@/lib/api/historical";
import { useComputeIndicators, useGenerateTrajectories, useTrajectoryStats, useTrainingRuns } from "@/lib/api/training";
import { useWatchlistStore } from "@/store/watchlistStore";
import { toast } from "sonner";

export default function Training() {
  const [selectedSymbol, setSelectedSymbol] = useState("");
  const [period, setPeriod] = useState("1y");
  const [interval, setInterval] = useState("5m");
  
  const { symbols } = useWatchlistStore();
  const { data: storedSymbols } = useStoredSymbols();
  const { data: trajectoryStats } = useTrajectoryStats();
  const { data: trainingRuns } = useTrainingRuns();
  
  const fetchMarketData = useFetchMarketData();
  const computeIndicators = useComputeIndicators();
  const generateTrajectories = useGenerateTrajectories();

  const handlePrepareData = async () => {
    if (!selectedSymbol) {
      toast.error("Please select a symbol");
      return;
    }

    toast.info("Starting full pipeline...");
    
    // Step 1: Fetch OHLCV data
    await fetchMarketData.mutateAsync({
      symbol: selectedSymbol,
      period,
      interval,
    });

    // Step 2: Compute technical indicators
    await computeIndicators.mutateAsync({
      symbol: selectedSymbol,
      timeframe: interval,
    });

    // Step 3: Generate expert trajectories
    await generateTrajectories.mutateAsync({
      symbol: selectedSymbol,
      timeframe: interval,
    });

    toast.success("Data pipeline complete!");
  };

  const totalBars = storedSymbols?.reduce((sum, s) => sum + (s.last_fetched ? 1 : 0), 0) || 0;
  const totalTrajectories = trajectoryStats?.total || 0;
  const totalRuns = trainingRuns?.length || 0;

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
            <h2 className="mb-2 text-3xl font-bold">Model Training</h2>
            <p className="text-muted-foreground">
              Download and prepare historical market data for ML model training
            </p>
          </div>
        </div>
      </div>

      {/* Stats */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Stored Symbols</CardTitle>
            <Database className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{storedSymbols?.length || 0}</div>
            <p className="text-xs text-muted-foreground">With historical data</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Data Points</CardTitle>
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{totalBars > 0 ? "~" + (totalBars * 100).toLocaleString() : "—"}</div>
            <p className="text-xs text-muted-foreground">Historical bars</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Trajectories</CardTitle>
            <Target className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{totalTrajectories.toLocaleString()}</div>
            <p className="text-xs text-muted-foreground">Expert signals</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Training Runs</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{totalRuns}</div>
            <p className="text-xs text-muted-foreground">BC + PPO runs</p>
          </CardContent>
        </Card>
      </div>

      {/* Pipeline */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="h-5 w-5 text-primary" />
            RL Training Pipeline
          </CardTitle>
          <CardDescription>
            Full pipeline: Fetch data → Compute indicators → Generate expert trajectories → Train models
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="prepare">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="prepare">1. Prepare Data</TabsTrigger>
              <TabsTrigger value="train">2. Train Models</TabsTrigger>
            </TabsList>
            
            <TabsContent value="prepare" className="space-y-6">
              <div className="grid gap-4 md:grid-cols-4">
                <div className="space-y-2">
                  <Label>Symbol</Label>
                  <Select value={selectedSymbol} onValueChange={setSelectedSymbol}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select symbol" />
                    </SelectTrigger>
                    <SelectContent>
                      {symbols.map((symbol) => (
                        <SelectItem key={symbol} value={symbol}>
                          {symbol}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label>Period</Label>
                  <Select value={period} onValueChange={setPeriod}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="1mo">1 Month</SelectItem>
                      <SelectItem value="3mo">3 Months</SelectItem>
                      <SelectItem value="6mo">6 Months</SelectItem>
                      <SelectItem value="1y">1 Year</SelectItem>
                      <SelectItem value="2y">2 Years</SelectItem>
                      <SelectItem value="3y">3 Years</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label>Interval</Label>
                  <Select value={interval} onValueChange={setInterval}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="1m">1 Minute</SelectItem>
                      <SelectItem value="5m">5 Minutes</SelectItem>
                      <SelectItem value="15m">15 Minutes</SelectItem>
                      <SelectItem value="1h">1 Hour</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="flex items-end">
                  <Button 
                    onClick={handlePrepareData} 
                    disabled={!selectedSymbol || fetchMarketData.isPending || computeIndicators.isPending || generateTrajectories.isPending}
                    className="w-full"
                  >
                    {(fetchMarketData.isPending || computeIndicators.isPending || generateTrajectories.isPending) ? "Processing..." : "Run Pipeline"}
                  </Button>
                </div>
              </div>

              <div className="space-y-2 pt-4 border-t border-border">
                <h4 className="text-sm font-medium">Pipeline Steps:</h4>
                <div className="grid gap-2 text-sm">
                  <div className="flex items-center gap-2 text-muted-foreground">
                    <div className="h-2 w-2 rounded-full bg-primary" />
                    <span>1. Fetch OHLCV data from Yahoo Finance</span>
                  </div>
                  <div className="flex items-center gap-2 text-muted-foreground">
                    <div className="h-2 w-2 rounded-full bg-primary" />
                    <span>2. Compute technical indicators (RSI, ATR, EMA, VWAP, etc.)</span>
                  </div>
                  <div className="flex items-center gap-2 text-muted-foreground">
                    <div className="h-2 w-2 rounded-full bg-primary" />
                    <span>3. Generate expert trajectories from 10+ strategies</span>
                  </div>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="train" className="space-y-4">
              <div className="rounded-lg border border-border bg-muted/50 p-6 text-center">
                <Brain className="h-12 w-12 mx-auto mb-3 text-muted-foreground" />
                <h3 className="font-semibold mb-2">Model Training (Coming Soon)</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  Behavior Cloning (BC) warm-start followed by PPO fine-tuning with:
                </p>
                <div className="grid gap-2 text-sm text-left max-w-md mx-auto">
                  <div className="flex items-center gap-2">
                    <div className="h-1.5 w-1.5 rounded-full bg-primary" />
                    <span>Weighted cross-entropy loss (emphasize HOLD + high R:R)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="h-1.5 w-1.5 rounded-full bg-primary" />
                    <span>PPO with reward = Δequity - fees - λ·risk</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="h-1.5 w-1.5 rounded-full bg-primary" />
                    <span>Walk-forward validation with purged CV</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="h-1.5 w-1.5 rounded-full bg-primary" />
                    <span>Metrics: Sharpe, profit factor, win rate, max DD</span>
                  </div>
                </div>
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>

      {/* Expert Trajectories Stats */}
      {trajectoryStats && trajectoryStats.total > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Expert Trajectories</CardTitle>
            <CardDescription>Signals generated from rule-based strategies</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
              {Object.entries(trajectoryStats.byTactic).map(([tactic, stats]) => (
                <div
                  key={tactic}
                  className="rounded-lg border border-border bg-muted/50 p-4"
                >
                  <div className="font-semibold text-sm mb-2">{tactic.replace(/_/g, " ")}</div>
                  <div className="grid grid-cols-4 gap-2 text-xs">
                    <div>
                      <div className="text-muted-foreground">Total</div>
                      <div className="font-semibold">{stats.total}</div>
                    </div>
                    <div>
                      <div className="text-success">Buy</div>
                      <div className="font-semibold">{stats.buy}</div>
                    </div>
                    <div>
                      <div className="text-destructive">Sell</div>
                      <div className="font-semibold">{stats.sell}</div>
                    </div>
                    <div>
                      <div className="text-muted-foreground">Hold</div>
                      <div className="font-semibold">{stats.hold}</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Stored Data */}
      <Card>
        <CardHeader>
          <CardTitle>Downloaded Datasets</CardTitle>
          <CardDescription>Historical data ready for model training</CardDescription>
        </CardHeader>
        <CardContent>
          {storedSymbols && storedSymbols.length > 0 ? (
            <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
              {storedSymbols.map((symbol) => (
                <div
                  key={symbol.symbol}
                  className="flex items-center justify-between rounded-lg border border-border bg-muted/50 p-4"
                >
                  <div>
                    <div className="font-semibold">{symbol.symbol}</div>
                    <div className="text-xs text-muted-foreground">
                      {symbol.exchange || "Unknown Exchange"}
                    </div>
                    {symbol.last_fetched && (
                      <div className="text-xs text-muted-foreground mt-1">
                        Updated: {new Date(symbol.last_fetched).toLocaleDateString()}
                      </div>
                    )}
                  </div>
                  <Badge variant="secondary">
                    <Database className="mr-1 h-3 w-3" />
                    Ready
                  </Badge>
                </div>
              ))}
            </div>
          ) : (
            <div className="py-8 text-center text-muted-foreground">
              <Database className="h-12 w-12 mx-auto mb-3 opacity-50" />
              <p>No historical data downloaded yet</p>
              <p className="text-sm mt-1">Use the fetch tool above to download market data</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Training Placeholder */}
      <Card>
        <CardHeader>
          <CardTitle>Training Runs</CardTitle>
          <CardDescription>View and manage BC/PPO training jobs</CardDescription>
        </CardHeader>
        <CardContent>
          {trainingRuns && trainingRuns.length > 0 ? (
            <div className="space-y-3">
              {trainingRuns.slice(0, 5).map((run) => (
                <div key={run.id} className="flex items-center justify-between rounded-lg border border-border bg-muted/50 p-4">
                  <div>
                    <div className="font-semibold">{run.run_name}</div>
                    <div className="text-xs text-muted-foreground">
                      {run.phase} | Epoch {run.current_epoch}/{run.total_epochs}
                    </div>
                  </div>
                  <Badge variant={run.status === "completed" ? "default" : "secondary"}>
                    {run.status}
                  </Badge>
                </div>
              ))}
            </div>
          ) : (
            <div className="py-8 text-center text-muted-foreground">
              <Brain className="h-12 w-12 mx-auto mb-3 opacity-50" />
              <p>No training runs yet</p>
              <p className="text-sm mt-1">
                Prepare data first, then start training in the Pipeline tab
              </p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
