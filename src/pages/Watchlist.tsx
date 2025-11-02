import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Plus, X } from "lucide-react";
import { useWatchlistStore } from "@/store/watchlistStore";
import { useUIStore } from "@/store/uiStore";
import { useLatestQuote } from "@/lib/api/market";
import { Value } from "@/components/Guard/Value";
import { percentOrDash, currencyOrDash } from "@/lib/format";
import { useSettingsStore } from "@/store/settingsStore";
import { HistoricalChart } from "@/components/HistoricalChart";

interface WatchlistRowProps {
  symbol: string;
  onSelect: () => void;
  onRemove: () => void;
}

function WatchlistRow({ symbol, onSelect, onRemove }: WatchlistRowProps) {
  const { data: quoteResult, isLoading } = useLatestQuote(symbol);
  
  // Extract data and source from DataWithSource wrapper
  const quote = quoteResult?.data ?? null;
  const source = quoteResult?.source ?? "none";

  if (isLoading) {
    return (
      <tr className="animate-pulse">
        <td className="whitespace-nowrap px-6 py-4">
          <div className="h-4 w-16 bg-muted rounded"></div>
        </td>
        <td className="whitespace-nowrap px-6 py-4">
          <div className="h-4 w-20 bg-muted rounded"></div>
        </td>
        <td className="whitespace-nowrap px-6 py-4">
          <div className="h-4 w-24 bg-muted rounded"></div>
        </td>
        <td className="whitespace-nowrap px-6 py-4 text-right">
          <div className="h-8 w-8 bg-muted rounded"></div>
        </td>
      </tr>
    );
  }

  const isPositive = quote && quote.changePercent >= 0;

  return (
    <tr
      className="cursor-pointer transition-colors hover:bg-muted/50"
      onClick={onSelect}
    >
      <td className="whitespace-nowrap px-6 py-4">
        <div className="font-semibold">{symbol}</div>
      </td>
      <td className="whitespace-nowrap px-6 py-4">
        <Value
          value={quote?.price}
          source={source}
          formatter={(n) => currencyOrDash(n)}
          className="data-cell"
        />
      </td>
      <td className="whitespace-nowrap px-6 py-4">
        <div
          className={`data-cell ${
            isPositive ? "profit" : "loss"
          }`}
        >
          <Value
            value={quote?.changePercent}
            source={source}
            formatter={(n) => percentOrDash(n)}
          />
        </div>
      </td>
      <td className="whitespace-nowrap px-6 py-4 text-right">
        <Button
          variant="ghost"
          size="icon"
          onClick={(e) => {
            e.stopPropagation();
            onRemove();
          }}
        >
          <X className="h-4 w-4" />
        </Button>
      </td>
    </tr>
  );
}

export default function Watchlist() {
  const [newSymbol, setNewSymbol] = useState("");
  const { symbols, addSymbol, removeSymbol } = useWatchlistStore();
  const { activeSymbol, setActiveSymbol } = useUIStore();

  const handleAddSymbol = () => {
    if (newSymbol.trim()) {
      addSymbol(newSymbol.toUpperCase());
      setNewSymbol("");
    }
  };

  return (
    <div className="space-y-6">
      {/* Add Symbol */}
      <Card>
        <CardHeader>
          <CardTitle>Watchlist</CardTitle>
          <CardDescription>Monitor your favorite symbols in real-time</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex gap-2">
            <Input
              placeholder="Enter symbol (e.g., AAPL)"
              value={newSymbol}
              onChange={(e) => setNewSymbol(e.target.value)}
              onKeyPress={(e) => e.key === "Enter" && handleAddSymbol()}
              className="uppercase"
            />
            <Button onClick={handleAddSymbol}>
              <Plus className="mr-2 h-4 w-4" />
              Add
            </Button>
          </div>
        </CardContent>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Watchlist Table */}
        <Card>
          <CardHeader>
            <CardTitle>Active Symbols</CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-border">
                    <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-muted-foreground">
                      Symbol
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-muted-foreground">
                      Price
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-muted-foreground">
                      Change
                    </th>
                    <th className="px-6 py-3 text-right text-xs font-medium uppercase tracking-wider text-muted-foreground">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {symbols.map((symbol) => (
                    <WatchlistRow
                      key={symbol}
                      symbol={symbol}
                      onSelect={() => setActiveSymbol(symbol)}
                      onRemove={() => removeSymbol(symbol)}
                    />
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>

        {/* Historical Chart */}
        {activeSymbol ? (
          <HistoricalChart symbol={activeSymbol} timeframe="1d" limit={100} />
        ) : (
          <Card>
            <CardHeader>
              <CardTitle>Chart</CardTitle>
              <CardDescription>Select a symbol to view historical data</CardDescription>
            </CardHeader>
            <CardContent className="py-12 text-center text-muted-foreground">
              <p>Click on a symbol to see its chart</p>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
