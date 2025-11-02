import { useMemo } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from "recharts";
import { format } from "date-fns";
import { useHistoricalBars } from "@/lib/api/historical";
import { TrendingUp, TrendingDown } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";

interface HistoricalChartProps {
  symbol: string;
  timeframe?: string;
  limit?: number;
}

export function HistoricalChart({ symbol, timeframe = "1d", limit = 100 }: HistoricalChartProps) {
  const { data: barsResult, isLoading } = useHistoricalBars(symbol, timeframe, limit);
  
  // Extract data and source
  const bars = barsResult?.data ?? null;
  const source = barsResult?.source ?? "none";

  const chartData = useMemo(() => {
    if (!bars || bars.length === 0) return [];

    return bars.map((bar) => ({
      time: format(bar.timestamp, timeframe === "1d" ? "MMM dd" : "HH:mm"),
      price: bar.close,
      volume: bar.volume,
    }));
  }, [bars, timeframe]);

  const stats = useMemo(() => {
    if (!bars || bars.length === 0) return null;

    const first = bars[0];
    const last = bars[bars.length - 1];
    const change = last.close - first.close;
    const changePercent = (change / first.close) * 100;
    const high = Math.max(...bars.map((b) => b.high));
    const low = Math.min(...bars.map((b) => b.low));

    return {
      current: last.close,
      change,
      changePercent,
      high,
      low,
      volume: bars.reduce((sum, b) => sum + b.volume, 0),
    };
  }, [bars]);

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <Skeleton className="h-6 w-32" />
          <Skeleton className="h-4 w-48 mt-2" />
        </CardHeader>
        <CardContent>
          <Skeleton className="h-[300px] w-full" />
        </CardContent>
      </Card>
    );
  }

  if (!bars || bars.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>{symbol} Historical Chart</CardTitle>
          <CardDescription>No historical data available</CardDescription>
        </CardHeader>
        <CardContent className="py-8 text-center text-muted-foreground">
          <p>Fetch historical data from the Training page to view charts</p>
        </CardContent>
      </Card>
    );
  }

  const isPositive = stats && stats.change >= 0;

  return (
    <Card>
      <CardHeader>
        <div className="flex items-start justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              {symbol}
              <Badge variant="outline" className="text-xs">
                {bars.length} bars
              </Badge>
              {source === "mock" && (
                <Badge variant="secondary" className="text-xs">
                  Mock
                </Badge>
              )}
            </CardTitle>
            <CardDescription>
              Historical price data from {source === "store" ? "database" : "Yahoo Finance"}
            </CardDescription>
          </div>
          {stats && (
            <div className="text-right">
              <div className="text-2xl font-bold data-cell">${stats.current.toFixed(2)}</div>
              <div className={`flex items-center gap-1 text-sm ${isPositive ? "profit" : "loss"}`}>
                {isPositive ? <TrendingUp className="h-3 w-3" /> : <TrendingDown className="h-3 w-3" />}
                {isPositive ? "+" : ""}
                {stats.change.toFixed(2)} ({isPositive ? "+" : ""}
                {stats.changePercent.toFixed(2)}%)
              </div>
            </div>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Price Chart */}
        <div>
          <h4 className="text-sm font-medium mb-4">Price History</h4>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--chart-grid))" />
              <XAxis dataKey="time" stroke="hsl(var(--muted-foreground))" fontSize={12} />
              <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "hsl(var(--popover))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: "var(--radius)",
                }}
                labelStyle={{ color: "hsl(var(--foreground))" }}
              />
              <Line
                type="monotone"
                dataKey="price"
                stroke="hsl(var(--primary))"
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Volume Chart */}
        <div>
          <h4 className="text-sm font-medium mb-4">Volume</h4>
          <ResponsiveContainer width="100%" height={150}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--chart-grid))" />
              <XAxis dataKey="time" stroke="hsl(var(--muted-foreground))" fontSize={12} />
              <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "hsl(var(--popover))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: "var(--radius)",
                }}
                labelStyle={{ color: "hsl(var(--foreground))" }}
              />
              <Bar dataKey="volume" fill="hsl(var(--primary))" opacity={0.6} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Stats */}
        {stats && (
          <div className="grid grid-cols-3 gap-4 pt-4 border-t border-border">
            <div>
              <div className="text-xs text-muted-foreground">High</div>
              <div className="text-lg font-semibold data-cell">${stats.high.toFixed(2)}</div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground">Low</div>
              <div className="text-lg font-semibold data-cell">${stats.low.toFixed(2)}</div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground">Total Volume</div>
              <div className="text-lg font-semibold data-cell">
                {(stats.volume / 1000000).toFixed(1)}M
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
