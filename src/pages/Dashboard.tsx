import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { TrendingUp, TrendingDown, Activity, DollarSign, BarChart3, Zap, Eye } from "lucide-react";
import { Value } from "@/components/Guard/Value";
import { useSettingsStore } from "@/store/settingsStore";

export default function Dashboard() {
  const { dataMode } = useSettingsStore();
  
  // In real implementation, these would come from stores/APIs
  // For now, we show "—" to indicate no live data
  const stats = [
    {
      title: "Portfolio Value",
      value: undefined, // No real data yet
      change: undefined,
      trend: "neutral",
      icon: DollarSign,
    },
    {
      title: "Active Positions",
      value: undefined,
      change: undefined,
      trend: "neutral",
      icon: Activity,
    },
    {
      title: "Total P&L",
      value: undefined,
      change: undefined,
      trend: "neutral",
      icon: TrendingUp,
    },
    {
      title: "Win Rate",
      value: undefined,
      change: undefined,
      trend: "neutral",
      icon: BarChart3,
    },
  ];

  // Recent signals - empty until strategy system is active
  const recentSignals: any[] = [];

  return (
    <div className="space-y-6">
      {/* Hero Section */}
      <div className="relative overflow-hidden rounded-2xl border border-border bg-gradient-to-br from-card via-card to-card/50 p-8">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-primary/20 via-transparent to-transparent opacity-50" />
        <div className="relative">
          <h2 className="mb-2 text-3xl font-bold">Welcome to TradePilot</h2>
          <p className="text-muted-foreground max-w-2xl">
            Your AI-powered algorithmic trading command center. Monitor markets, execute strategies, and train
            models in real-time.
          </p>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {stats.map((stat) => (
          <Card key={stat.title} className="relative overflow-hidden">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">{stat.title}</CardTitle>
              <stat.icon className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                <Value
                  value={stat.value}
                  source="none"
                  tooltip="Connect to a data source to see live portfolio data"
                  showBadge={false}
                />
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                {stat.change !== undefined ? (
                  <>
                    {stat.trend === "up" && <TrendingUp className="h-3 w-3 inline mr-1" />}
                    <Value
                      value={stat.change}
                      source="none"
                      showBadge={false}
                    />
                  </>
                ) : (
                  "—"
                )}
              </p>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        {/* Recent Signals */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Zap className="h-5 w-5 text-primary" />
              Recent Signals
            </CardTitle>
            <CardDescription>Latest strategy alerts from your ensemble</CardDescription>
          </CardHeader>
          <CardContent>
            {recentSignals.length === 0 ? (
              <div className="py-8 text-center text-muted-foreground">
                <Zap className="h-12 w-12 mx-auto mb-3 opacity-50" />
                <p>No signals yet</p>
                <p className="text-sm mt-1">Configure strategies to start receiving trading signals</p>
              </div>
            ) : (
              <div className="space-y-4">
                {recentSignals.map((signal, i) => (
                  <div key={i} className="flex items-center justify-between rounded-lg border border-border bg-muted/50 p-3">
                    <div className="flex items-center gap-3">
                      <div className={cn("h-8 w-8 rounded-lg flex items-center justify-center", signal.side === "buy" ? "bg-success/10 text-success" : "bg-destructive/10 text-destructive")}>
                        {signal.side === "buy" ? <TrendingUp className="h-4 w-4" /> : <TrendingDown className="h-4 w-4" />}
                      </div>
                      <div>
                        <div className="font-semibold">{signal.symbol}</div>
                        <div className="text-xs text-muted-foreground">{signal.strategy}</div>
                      </div>
                    </div>
                    <div className="text-right">
                      <Badge variant="outline" className="mb-1">
                        {(signal.confidence * 100).toFixed(0)}%
                      </Badge>
                      <div className="text-xs text-muted-foreground">{signal.time}</div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Quick Actions */}
        <Card>
          <CardHeader>
            <CardTitle>Quick Actions</CardTitle>
            <CardDescription>Common tasks and shortcuts</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-3">
              <button className="flex items-center gap-3 rounded-lg border border-border bg-card p-4 text-left transition-colors hover:bg-accent">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 text-primary">
                  <Eye className="h-5 w-5" />
                </div>
                <div>
                  <div className="font-medium">Add to Watchlist</div>
                  <div className="text-xs text-muted-foreground">Track new symbols</div>
                </div>
              </button>
              <button className="flex items-center gap-3 rounded-lg border border-border bg-card p-4 text-left transition-colors hover:bg-accent">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 text-primary">
                  <Zap className="h-5 w-5" />
                </div>
                <div>
                  <div className="font-medium">Configure Strategies</div>
                  <div className="text-xs text-muted-foreground">Adjust parameters</div>
                </div>
              </button>
              <button className="flex items-center gap-3 rounded-lg border border-border bg-card p-4 text-left transition-colors hover:bg-accent">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 text-primary">
                  <Activity className="h-5 w-5" />
                </div>
                <div>
                  <div className="font-medium">Start Training</div>
                  <div className="text-xs text-muted-foreground">Train new models</div>
                </div>
              </button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

function cn(...args: any[]) {
  return args.filter(Boolean).join(" ");
}
