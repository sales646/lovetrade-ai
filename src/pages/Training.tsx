import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Brain, Database, Download, TrendingUp, BarChart3, Activity } from "lucide-react";
import { useFetchMarketData, useStoredSymbols } from "@/lib/api/historical";
import { useWatchlistStore } from "@/store/watchlistStore";
import { toast } from "sonner";

export default function Training() {
  const [selectedSymbol, setSelectedSymbol] = useState("");
  const [period, setPeriod] = useState("1y");
  const [interval, setInterval] = useState("1d");
  
  const { symbols } = useWatchlistStore();
  const { data: storedSymbols } = useStoredSymbols();
  const fetchMarketData = useFetchMarketData();

  const handleFetchData = async () => {
    if (!selectedSymbol) {
      toast.error("Please select a symbol");
      return;
    }

    await fetchMarketData.mutateAsync({
      symbol: selectedSymbol,
      period,
      interval,
    });
  };

  const handleBulkFetch = async () => {
    if (symbols.length === 0) {
      toast.error("Add symbols to watchlist first");
      return;
    }

    toast.info(`Fetching historical data for ${symbols.length} symbols...`);
    
    for (const symbol of symbols) {
      try {
        await fetchMarketData.mutateAsync({
          symbol,
          period,
          interval,
        });
      } catch (error) {
        console.error(`Failed to fetch ${symbol}:`, error);
      }
    }
    
    toast.success("Bulk fetch completed");
  };

  const totalBars = storedSymbols?.reduce((sum, s) => sum + (s.last_fetched ? 1 : 0), 0) || 0;

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
      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Stored Symbols</CardTitle>
            <Database className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{storedSymbols?.length || 0}</div>
            <p className="text-xs text-muted-foreground">Available for training</p>
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
            <CardTitle className="text-sm font-medium">Training Status</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">Ready</div>
            <p className="text-xs text-muted-foreground">No active training</p>
          </CardContent>
        </Card>
      </div>

      {/* Data Fetching */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Download className="h-5 w-5 text-primary" />
            Fetch Historical Data
          </CardTitle>
          <CardDescription>
            Download market data from Yahoo Finance for training. Free, no API key required.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
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
                  <SelectItem value="5y">5 Years</SelectItem>
                  <SelectItem value="max">Max</SelectItem>
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
                  <SelectItem value="1d">1 Day</SelectItem>
                  <SelectItem value="1wk">1 Week</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="flex items-end space-y-2">
              <Button 
                onClick={handleFetchData} 
                disabled={fetchMarketData.isPending || !selectedSymbol}
                className="w-full"
              >
                {fetchMarketData.isPending ? "Fetching..." : "Fetch Data"}
              </Button>
            </div>
          </div>

          <div className="flex items-center gap-3 pt-4 border-t border-border">
            <Button 
              onClick={handleBulkFetch}
              disabled={fetchMarketData.isPending || symbols.length === 0}
              variant="secondary"
            >
              <Download className="mr-2 h-4 w-4" />
              Bulk Fetch All Watchlist ({symbols.length})
            </Button>
            <span className="text-sm text-muted-foreground">
              Downloads historical data for all symbols in your watchlist
            </span>
          </div>
        </CardContent>
      </Card>

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
          <CardTitle>Model Training</CardTitle>
          <CardDescription>Configure and train trading models on historical data</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="py-8 text-center text-muted-foreground">
            <Brain className="h-12 w-12 mx-auto mb-3 opacity-50" />
            <p>Training interface coming soon</p>
            <p className="text-sm mt-1">
              Once data is downloaded, you'll be able to train RL/ML models here
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
