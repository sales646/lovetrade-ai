import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Plus, X, TrendingUp, TrendingDown } from "lucide-react";
import { useWatchlistStore } from "@/store/watchlistStore";
import { useUIStore } from "@/store/uiStore";

export default function Watchlist() {
  const [newSymbol, setNewSymbol] = useState("");
  const { symbols, addSymbol, removeSymbol } = useWatchlistStore();
  const { setActiveSymbol } = useUIStore();

  const handleAddSymbol = () => {
    if (newSymbol.trim()) {
      addSymbol(newSymbol.toUpperCase());
      setNewSymbol("");
    }
  };

  // Mock data for demonstration
  const watchlistData = symbols.map((symbol) => ({
    symbol,
    price: (Math.random() * 500 + 100).toFixed(2),
    change: (Math.random() * 10 - 5).toFixed(2),
    changePercent: (Math.random() * 5 - 2.5).toFixed(2),
    volume: (Math.random() * 10000000).toFixed(0),
    status: Math.random() > 0.5 ? "active" : "idle",
  }));

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

      {/* Watchlist Table */}
      <Card>
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
                  <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-muted-foreground">
                    Volume
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-muted-foreground">
                    Status
                  </th>
                  <th className="px-6 py-3 text-right text-xs font-medium uppercase tracking-wider text-muted-foreground">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {watchlistData.map((item) => {
                  const isPositive = parseFloat(item.change) >= 0;
                  return (
                    <tr
                      key={item.symbol}
                      className="cursor-pointer transition-colors hover:bg-muted/50"
                      onClick={() => setActiveSymbol(item.symbol)}
                    >
                      <td className="whitespace-nowrap px-6 py-4">
                        <div className="font-semibold">{item.symbol}</div>
                      </td>
                      <td className="whitespace-nowrap px-6 py-4">
                        <div className="data-cell">${item.price}</div>
                      </td>
                      <td className="whitespace-nowrap px-6 py-4">
                        <div className={`flex items-center gap-1 data-cell ${isPositive ? "profit" : "loss"}`}>
                          {isPositive ? (
                            <TrendingUp className="h-3 w-3" />
                          ) : (
                            <TrendingDown className="h-3 w-3" />
                          )}
                          {isPositive ? "+" : ""}
                          {item.change} ({isPositive ? "+" : ""}
                          {item.changePercent}%)
                        </div>
                      </td>
                      <td className="whitespace-nowrap px-6 py-4">
                        <div className="data-cell text-muted-foreground">
                          {parseFloat(item.volume).toLocaleString()}
                        </div>
                      </td>
                      <td className="whitespace-nowrap px-6 py-4">
                        <Badge variant={item.status === "active" ? "default" : "secondary"}>
                          {item.status}
                        </Badge>
                      </td>
                      <td className="whitespace-nowrap px-6 py-4 text-right">
                        <Button
                          variant="ghost"
                          size="icon"
                          onClick={(e) => {
                            e.stopPropagation();
                            removeSymbol(item.symbol);
                          }}
                        >
                          <X className="h-4 w-4" />
                        </Button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
