import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { supabase } from "@/integrations/supabase/client";
import { Play, Pause, RefreshCw, TrendingUp, Shield, Activity } from "lucide-react";
import { toast } from "sonner";
import { format } from "date-fns";

interface SystemLog {
  id: string;
  level: "INFO" | "WARN" | "ERROR";
  source: string;
  message: string;
  metadata: any;
  created_at: string;
}

interface Position {
  id: string;
  symbol: string;
  side: "long" | "short";
  size: number;
  entry_price: number;
  current_price: number;
  unrealized_pnl: number;
  opened_at: string;
}

export default function AutoTrading() {
  const [isRunning, setIsRunning] = useState(false);
  const [logs, setLogs] = useState<SystemLog[]>([]);
  const [positions, setPositions] = useState<Position[]>([]);
  const [stats, setStats] = useState({
    signals_today: 0,
    trades_executed: 0,
    trades_blocked: 0,
    avg_risk_score: 0,
  });

  useEffect(() => {
    fetchLogs();
    fetchPositions();
    fetchStats();

    // Real-time subscriptions
    const logsChannel = supabase
      .channel("system_logs")
      .on(
        "postgres_changes",
        { event: "INSERT", schema: "public", table: "system_logs" },
        (payload) => {
          setLogs((prev) => [payload.new as SystemLog, ...prev].slice(0, 50));
        }
      )
      .subscribe();

    const positionsChannel = supabase
      .channel("positions")
      .on(
        "postgres_changes",
        { event: "*", schema: "public", table: "positions" },
        () => {
          fetchPositions();
        }
      )
      .subscribe();

    return () => {
      supabase.removeChannel(logsChannel);
      supabase.removeChannel(positionsChannel);
    };
  }, []);

  const fetchLogs = async () => {
    const { data } = await supabase
      .from("system_logs")
      .select("*")
      .order("created_at", { ascending: false })
      .limit(50);
    if (data) setLogs(data as SystemLog[]);
  };

  const fetchPositions = async () => {
    const { data } = await supabase.from("positions").select("*");
    if (data) setPositions(data as Position[]);
  };

  const fetchStats = async () => {
    const today = new Date();
    today.setHours(0, 0, 0, 0);

    const { data: signals } = await supabase
      .from("trading_signals")
      .select("*")
      .gte("created_at", today.toISOString());

    const { data: trades } = await supabase
      .from("trades")
      .select("*")
      .gte("executed_at", today.toISOString());

    const { data: assessments } = await supabase
      .from("risk_assessments")
      .select("*")
      .gte("assessed_at", today.toISOString());

    const executed = assessments?.filter((a) => a.should_execute).length || 0;
    const blocked = assessments?.filter((a) => !a.should_execute).length || 0;
    const avgRisk =
      assessments?.reduce((sum, a) => sum + a.risk_score, 0) / (assessments?.length || 1) || 0;

    setStats({
      signals_today: signals?.length || 0,
      trades_executed: executed,
      trades_blocked: blocked,
      avg_risk_score: avgRisk,
    });
  };

  const handleRunTrader = async () => {
    setIsRunning(true);
    toast.info("Running autonomous trader...");

    try {
      const { data, error } = await supabase.functions.invoke("autonomous-trader");

      if (error) throw error;

      toast.success(`Trader completed: ${data.signals_processed} signals processed`);
      fetchStats();
    } catch (error) {
      toast.error("Trader failed: " + (error as Error).message);
    } finally {
      setIsRunning(false);
    }
  };

  const getLevelIcon = (level: string) => {
    switch (level) {
      case "ERROR":
        return "🔴";
      case "WARN":
        return "🟡";
      default:
        return "🟢";
    }
  };

  return (
    <div className="space-y-6">
      {/* Control Panel */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Autonomous Trading System</CardTitle>
              <CardDescription>
                AI-driven trading med automatisk riskbedömning
              </CardDescription>
            </div>
            <Badge variant={isRunning ? "default" : "secondary"} className="text-sm">
              {isRunning ? "🟢 Running" : "⚪ Idle"}
            </Badge>
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex gap-4">
            <Button onClick={handleRunTrader} disabled={isRunning}>
              {isRunning ? (
                <>
                  <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                  Running...
                </>
              ) : (
                <>
                  <Play className="mr-2 h-4 w-4" />
                  Run Trader
                </>
              )}
            </Button>
            <Button variant="outline" onClick={fetchStats}>
              <RefreshCw className="mr-2 h-4 w-4" />
              Refresh Stats
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Stats Cards */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Signals Today
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              <Activity className="h-4 w-4 text-primary" />
              <span className="text-2xl font-bold">{stats.signals_today}</span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Trades Executed
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              <TrendingUp className="h-4 w-4 text-green-500" />
              <span className="text-2xl font-bold text-green-500">
                {stats.trades_executed}
              </span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Trades Blocked
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              <Shield className="h-4 w-4 text-red-500" />
              <span className="text-2xl font-bold text-red-500">
                {stats.trades_blocked}
              </span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Avg Risk Score
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              <Shield className="h-4 w-4" />
              <span className="text-2xl font-bold">
                {(stats.avg_risk_score * 100).toFixed(0)}%
              </span>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Active Positions */}
      <Card>
        <CardHeader>
          <CardTitle>Active Positions</CardTitle>
        </CardHeader>
        <CardContent>
          {positions.length === 0 ? (
            <p className="text-muted-foreground text-center py-4">
              No active positions
            </p>
          ) : (
            <div className="space-y-2">
              {positions.map((pos) => (
                <div
                  key={pos.id}
                  className="flex items-center justify-between p-3 border border-border rounded-lg"
                >
                  <div className="flex items-center gap-4">
                    <Badge variant={pos.side === "long" ? "default" : "secondary"}>
                      {pos.side.toUpperCase()}
                    </Badge>
                    <span className="font-semibold">{pos.symbol}</span>
                    <span className="text-sm text-muted-foreground">
                      Size: {pos.size}
                    </span>
                  </div>
                  <div className="text-right">
                    <div className="font-medium">
                      ${pos.current_price?.toFixed(2) || pos.entry_price.toFixed(2)}
                    </div>
                    <div className="text-xs text-muted-foreground">
                      Entry: ${pos.entry_price.toFixed(2)}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* System Logs */}
      <Card>
        <CardHeader>
          <CardTitle>System Logs</CardTitle>
          <CardDescription>Real-time trading activity</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-2 max-h-[400px] overflow-y-auto">
            {logs.map((log) => (
              <div
                key={log.id}
                className="flex items-start gap-3 p-2 text-sm border-b border-border last:border-0"
              >
                <span>{getLevelIcon(log.level)}</span>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <Badge variant="outline" className="text-xs">
                      {log.level}
                    </Badge>
                    <span className="text-xs text-muted-foreground">
                      {format(new Date(log.created_at), "HH:mm:ss")}
                    </span>
                  </div>
                  <p className="mt-1 break-words">{log.message}</p>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
