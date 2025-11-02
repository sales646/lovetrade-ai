import { useState, useEffect, useRef } from "react";
import { useSearchParams } from "react-router-dom";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Plus, X, GripVertical } from "lucide-react";
import { useWatchlistStore } from "@/store/watchlistStore";
import { useUIStore } from "@/store/uiStore";
import { useLatestQuote, useBars } from "@/lib/api/market";
import { Value } from "@/components/Guard/Value";
import { percentOrDash, currencyOrDash } from "@/lib/format";
import { HistoricalChart } from "@/components/HistoricalChart";
import {
  DndContext,
  closestCenter,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
  DragEndEvent,
} from "@dnd-kit/core";
import {
  arrayMove,
  SortableContext,
  sortableKeyboardCoordinates,
  useSortable,
  verticalListSortingStrategy,
} from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import { useVirtualizer } from "@tanstack/react-virtual";
import {
  AreaChart,
  Area,
  BarChart as RechartsBar,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ComposedChart,
  Line,
} from "recharts";
import { format } from "date-fns";

interface WatchlistRowProps {
  symbol: string;
  onSelect: () => void;
  onRemove: () => void;
}

function SortableWatchlistRow({ symbol, onSelect, onRemove }: WatchlistRowProps) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({
    id: symbol,
  });
  const { data: quoteResult, isLoading } = useLatestQuote(symbol);

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.5 : 1,
  };

  const quote = quoteResult?.data ?? null;
  const source = quoteResult?.source ?? "none";

  if (isLoading) {
    return (
      <tr ref={setNodeRef} style={style} className="animate-pulse">
        <td className="whitespace-nowrap px-6 py-4">
          <div className="h-4 w-6 bg-muted rounded"></div>
        </td>
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
      ref={setNodeRef}
      style={style}
      className="cursor-pointer transition-colors hover:bg-muted/50"
      onClick={onSelect}
    >
      <td className="whitespace-nowrap px-2 py-4">
        <button
          {...attributes}
          {...listeners}
          className="cursor-grab active:cursor-grabbing text-muted-foreground hover:text-foreground"
          onClick={(e) => e.stopPropagation()}
        >
          <GripVertical className="h-4 w-4" />
        </button>
      </td>
      <td className="whitespace-nowrap px-6 py-4">
        <div className="flex items-center gap-2">
          <span className="font-semibold">{symbol}</span>
          {source === "mock" && (
            <Badge variant="outline" className="text-xs">
              Mock
            </Badge>
          )}
        </div>
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
        <div className={`data-cell ${isPositive ? "profit" : "loss"}`}>
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

function AdvancedChart({ symbol }: { symbol: string }) {
  const { data: barsResult, isLoading } = useBars(symbol, "5m", 100);
  const bars = barsResult?.data ?? [];

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Chart - {symbol}</CardTitle>
        </CardHeader>
        <CardContent className="h-[400px] flex items-center justify-center">
          <div className="text-muted-foreground">Loading chart...</div>
        </CardContent>
      </Card>
    );
  }

  if (!bars || bars.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Chart - {symbol}</CardTitle>
        </CardHeader>
        <CardContent className="h-[400px] flex items-center justify-center">
          <div className="text-muted-foreground">No data available</div>
        </CardContent>
      </Card>
    );
  }

  // Calculate SMA (20 period)
  const calculateSMA = (data: any[], period: number) => {
    return data.map((item, idx) => {
      if (idx < period - 1) return { ...item, sma: null };
      const sum = data
        .slice(idx - period + 1, idx + 1)
        .reduce((acc, d) => acc + d.close, 0);
      return { ...item, sma: sum / period };
    });
  };

  // Calculate VWAP
  const calculateVWAP = (data: any[]) => {
    let cumulativeTPV = 0;
    let cumulativeVolume = 0;
    return data.map((item) => {
      const typicalPrice = (item.high + item.low + item.close) / 3;
      cumulativeTPV += typicalPrice * item.volume;
      cumulativeVolume += item.volume;
      return {
        ...item,
        vwap: cumulativeVolume > 0 ? cumulativeTPV / cumulativeVolume : null,
      };
    });
  };

  const chartData = calculateVWAP(calculateSMA(bars, 20)).map((bar) => ({
    time: format(new Date(bar.timestamp), "HH:mm"),
    open: bar.open,
    high: bar.high,
    low: bar.low,
    close: bar.close,
    volume: bar.volume,
    sma: bar.sma,
    vwap: bar.vwap,
  }));

  return (
    <Card>
      <CardHeader>
        <CardTitle>Advanced Chart - {symbol}</CardTitle>
        <CardDescription>Candlesticks, Volume, SMA(20), VWAP</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Price Chart with Candlesticks */}
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                <XAxis dataKey="time" className="text-xs" />
                <YAxis domain={["auto", "auto"]} className="text-xs" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "hsl(var(--card))",
                    border: "1px solid hsl(var(--border))",
                  }}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="close"
                  stroke="hsl(var(--primary))"
                  strokeWidth={2}
                  dot={false}
                  name="Close"
                />
                <Line
                  type="monotone"
                  dataKey="sma"
                  stroke="hsl(var(--chart-2))"
                  strokeWidth={2}
                  dot={false}
                  name="SMA(20)"
                  strokeDasharray="5 5"
                />
                <Line
                  type="monotone"
                  dataKey="vwap"
                  stroke="hsl(var(--chart-3))"
                  strokeWidth={2}
                  dot={false}
                  name="VWAP"
                  strokeDasharray="3 3"
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>

          {/* Volume Chart */}
          <div className="h-[150px]">
            <ResponsiveContainer width="100%" height="100%">
              <RechartsBar data={chartData}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                <XAxis dataKey="time" className="text-xs" />
                <YAxis className="text-xs" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "hsl(var(--card))",
                    border: "1px solid hsl(var(--border))",
                  }}
                />
                <Bar dataKey="volume" fill="hsl(var(--chart-4))" name="Volume" />
              </RechartsBar>
            </ResponsiveContainer>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export default function Watchlist() {
  const [newSymbol, setNewSymbol] = useState("");
  const [searchParams, setSearchParams] = useSearchParams();
  const { symbols, addSymbol, removeSymbol, reorderSymbols } = useWatchlistStore();
  const { activeSymbol, setActiveSymbol } = useUIStore();

  const sensors = useSensors(
    useSensor(PointerSensor),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    })
  );

  // Handle ?add=SYMBOL query parameter
  useEffect(() => {
    const addParam = searchParams.get("add");
    if (addParam) {
      const upperSymbol = addParam.toUpperCase();
      addSymbol(upperSymbol);
      setActiveSymbol(upperSymbol);
      setSearchParams({}); // Clear the query param
    }
  }, [searchParams, addSymbol, setActiveSymbol, setSearchParams]);

  const handleAddSymbol = () => {
    if (newSymbol.trim()) {
      addSymbol(newSymbol.toUpperCase());
      setNewSymbol("");
    }
  };

  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event;

    if (over && active.id !== over.id) {
      const oldIndex = symbols.indexOf(active.id as string);
      const newIndex = symbols.indexOf(over.id as string);
      const newOrder = arrayMove(symbols, oldIndex, newIndex);
      reorderSymbols(newOrder);
    }
  };

  return (
    <div className="space-y-6">
      {/* Add Symbol */}
      <Card>
        <CardHeader>
          <CardTitle>Watchlist</CardTitle>
          <CardDescription>
            Monitor your favorite symbols in real-time. Drag to reorder.
          </CardDescription>
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
              <DndContext
                sensors={sensors}
                collisionDetection={closestCenter}
                onDragEnd={handleDragEnd}
              >
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-border">
                      <th className="px-2 py-3"></th>
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
                    <SortableContext items={symbols} strategy={verticalListSortingStrategy}>
                      {symbols.map((symbol) => (
                        <SortableWatchlistRow
                          key={symbol}
                          symbol={symbol}
                          onSelect={() => setActiveSymbol(symbol)}
                          onRemove={() => removeSymbol(symbol)}
                        />
                      ))}
                    </SortableContext>
                  </tbody>
                </table>
              </DndContext>
            </div>
          </CardContent>
        </Card>

        {/* Advanced Chart */}
        {activeSymbol ? (
          <AdvancedChart symbol={activeSymbol} />
        ) : (
          <Card>
            <CardHeader>
              <CardTitle>Chart</CardTitle>
              <CardDescription>Select a symbol to view advanced charts</CardDescription>
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
