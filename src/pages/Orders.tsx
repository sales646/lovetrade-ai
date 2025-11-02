import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useSettingsStore } from "@/store/settingsStore";
import { useLatestQuote } from "@/lib/api/market";
import { assessTradeRisk, RiskAssessment } from "@/lib/api/risk";
import { Value } from "@/components/Guard/Value";
import { currencyOrDash, percentOrDash } from "@/lib/format";
import { TrendingUp, TrendingDown, Shield, AlertTriangle, CheckCircle } from "lucide-react";
import { toast } from "sonner";

// Mock positions for demo
const mockPositions = [
  {
    id: "1",
    symbol: "AAPL",
    side: "long" as const,
    quantity: 10,
    entryPrice: 150.25,
    currentPrice: 155.50,
    unrealizedPnL: 52.5,
    unrealizedPnLPercent: 3.49,
    stopLoss: 145.0,
    takeProfit: 160.0,
    openedAt: new Date("2025-01-15T10:30:00"),
  },
  {
    id: "2",
    symbol: "TSLA",
    side: "short" as const,
    quantity: 5,
    entryPrice: 250.0,
    currentPrice: 248.75,
    unrealizedPnL: 6.25,
    unrealizedPnLPercent: 0.5,
    stopLoss: 255.0,
    takeProfit: 240.0,
    openedAt: new Date("2025-01-16T14:20:00"),
  },
];

export default function Orders() {
  const { riskDefaults } = useSettingsStore();
  const [symbol, setSymbol] = useState("");
  const [side, setSide] = useState<"buy" | "sell">("buy");
  const [orderType, setOrderType] = useState<"market" | "limit">("market");
  const [quantity, setQuantity] = useState("");
  const [limitPrice, setLimitPrice] = useState("");
  const [riskPercent, setRiskPercent] = useState(riskDefaults.riskPerTrade.toString());
  const [stopLoss, setStopLoss] = useState("");
  const [takeProfit, setTakeProfit] = useState("");
  const [riskAssessment, setRiskAssessment] = useState<RiskAssessment | null>(null);
  const [isAssessing, setIsAssessing] = useState(false);

  const { data: quoteResult } = useLatestQuote(symbol || null);
  const quote = quoteResult?.data;
  const source = quoteResult?.source ?? "none";

  // Calculate position size based on risk %
  const accountSize = 10000; // Mock account size
  const riskAmount = (parseFloat(riskPercent) / 100) * accountSize;
  const currentPrice = quote?.price || parseFloat(limitPrice) || 0;
  const slPrice = parseFloat(stopLoss) || 0;
  const tpPrice = parseFloat(takeProfit) || 0;
  const riskPerShare =
    side === "buy" ? currentPrice - slPrice : slPrice - currentPrice;
  const calculatedQty =
    riskPerShare > 0 ? Math.floor(riskAmount / riskPerShare) : 0;

  // Calculate Risk:Reward ratio
  const potentialProfit =
    side === "buy" ? tpPrice - currentPrice : currentPrice - tpPrice;
  const riskReward =
    riskPerShare > 0 && potentialProfit > 0
      ? (potentialProfit / riskPerShare).toFixed(2)
      : "—";

  const handleAssessRisk = async () => {
    if (!symbol || !quote) {
      toast.error("Please enter a valid symbol first");
      return;
    }

    setIsAssessing(true);
    try {
      // Get current time context (mock for now)
      const now = new Date();
      const marketOpen = new Date(now);
      marketOpen.setHours(9, 30, 0, 0);
      const marketClose = new Date(now);
      marketClose.setHours(16, 0, 0, 0);
      
      const minutesSinceOpen = Math.max(0, Math.floor((now.getTime() - marketOpen.getTime()) / 60000));
      const minutesUntilClose = Math.max(0, Math.floor((marketClose.getTime() - now.getTime()) / 60000));

      const assessment = await assessTradeRisk({
        symbol,
        action: side,
        proposedSize: parseFloat(quantity || "0"),
        marketData: {
          price: quote.price,
          rsi: 50 + (Math.random() - 0.5) * 40, // Mock RSI
          atr: quote.price * 0.02, // Mock ATR
          vwap: quote.price * (1 + (Math.random() - 0.5) * 0.01), // Mock VWAP
          vwapDeviation: (Math.random() - 0.5) * 2,
          volumeZScore: (Math.random() - 0.5) * 3,
          sentiment: (Math.random() - 0.5) * 2,
          recentVolatility: Math.abs(quote.changePercent),
        },
        timeContext: {
          minutesSinceOpen,
          minutesUntilClose,
        },
      });

      setRiskAssessment(assessment);
      
      if (!assessment.shouldExecute) {
        toast.error(assessment.reason);
      } else if (assessment.adjustedSize < parseFloat(quantity || "0")) {
        toast.warning(assessment.reason);
      } else {
        toast.success(assessment.reason);
      }
    } catch (error) {
      toast.error("Risk assessment failed");
      console.error(error);
    } finally {
      setIsAssessing(false);
    }
  };

  const handleSubmitOrder = () => {
    if (!symbol || !quantity) {
      toast.error("Please fill in all required fields");
      return;
    }

    if (!riskAssessment) {
      toast.error("Please assess risk before submitting order");
      return;
    }

    if (!riskAssessment.shouldExecute) {
      toast.error("Order blocked by risk assessment");
      return;
    }

    const finalSize = riskAssessment.adjustedSize;

    // Mock order submission
    toast.success(`${side.toUpperCase()} order for ${finalSize.toFixed(0)} ${symbol} submitted`, {
      description: `Type: ${orderType}, Risk: ${(riskAssessment.riskScore * 100).toFixed(0)}%`,
    });

    // Reset form
    setSymbol("");
    setQuantity("");
    setLimitPrice("");
    setStopLoss("");
    setTakeProfit("");
    setRiskAssessment(null);
  };

  const handleClosePosition = (positionId: string, symbol: string) => {
    toast.success(`Position ${symbol} closed`, {
      description: "Order sent to market",
    });
  };

  return (
    <div className="space-y-6">
      {/* Order Ticket */}
      <Card>
        <CardHeader>
          <CardTitle>Order Ticket</CardTitle>
          <CardDescription>Place new orders with risk management</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-6 md:grid-cols-2">
            {/* Left Column - Order Details */}
            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="symbol">Symbol</Label>
                <Input
                  id="symbol"
                  placeholder="e.g., AAPL"
                  value={symbol}
                  onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                  className="uppercase"
                />
                {quote && (
                  <div className="flex items-center gap-2 text-sm">
                    <span className="text-muted-foreground">Current Price:</span>
                    <Value
                      value={quote.price}
                      source={source}
                      formatter={currencyOrDash}
                    />
                    {source === "mock" && (
                      <Badge variant="outline" className="text-xs">
                        Mock
                      </Badge>
                    )}
                  </div>
                )}
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Side</Label>
                  <Select value={side} onValueChange={(v: any) => setSide(v)}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="buy">
                        <div className="flex items-center gap-2">
                          <TrendingUp className="h-4 w-4 text-green-500" />
                          Buy
                        </div>
                      </SelectItem>
                      <SelectItem value="sell">
                        <div className="flex items-center gap-2">
                          <TrendingDown className="h-4 w-4 text-red-500" />
                          Sell
                        </div>
                      </SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label>Type</Label>
                  <Select
                    value={orderType}
                    onValueChange={(v: any) => setOrderType(v)}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="market">Market</SelectItem>
                      <SelectItem value="limit">Limit</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="quantity">Quantity</Label>
                <Input
                  id="quantity"
                  type="number"
                  placeholder="0"
                  value={quantity}
                  onChange={(e) => setQuantity(e.target.value)}
                />
              </div>

              {orderType === "limit" && (
                <div className="space-y-2">
                  <Label htmlFor="limitPrice">Limit Price</Label>
                  <Input
                    id="limitPrice"
                    type="number"
                    step="0.01"
                    placeholder="0.00"
                    value={limitPrice}
                    onChange={(e) => setLimitPrice(e.target.value)}
                  />
                </div>
              )}
            </div>

            {/* Right Column - Risk Management */}
            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="riskPercent">Risk % of Account</Label>
                <Input
                  id="riskPercent"
                  type="number"
                  step="0.1"
                  value={riskPercent}
                  onChange={(e) => setRiskPercent(e.target.value)}
                />
                <p className="text-xs text-muted-foreground">
                  Risk Amount: ${riskAmount.toFixed(2)}
                </p>
              </div>

              <div className="space-y-2">
                <Label htmlFor="stopLoss">Stop Loss Price</Label>
                <Input
                  id="stopLoss"
                  type="number"
                  step="0.01"
                  placeholder="0.00"
                  value={stopLoss}
                  onChange={(e) => setStopLoss(e.target.value)}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="takeProfit">Take Profit Price</Label>
                <Input
                  id="takeProfit"
                  type="number"
                  step="0.01"
                  placeholder="0.00"
                  value={takeProfit}
                  onChange={(e) => setTakeProfit(e.target.value)}
                />
              </div>

              {/* Calculated Values */}
              <div className="rounded-lg border border-border bg-muted/50 p-4 space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Calculated Qty:</span>
                  <span className="font-medium">{calculatedQty}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Risk:Reward:</span>
                  <span className="font-medium">{riskReward}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Total Value:</span>
                  <span className="font-medium">
                    ${(currentPrice * parseFloat(quantity || "0")).toFixed(2)}
                  </span>
                </div>
              </div>

              {/* Risk Assessment */}
              {riskAssessment && (
                <div className="rounded-lg border border-border p-4 space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">Risk Assessment</span>
                    {riskAssessment.shouldExecute ? (
                      riskAssessment.riskScore < 0.5 ? (
                        <CheckCircle className="h-5 w-5 text-green-500" />
                      ) : (
                        <AlertTriangle className="h-5 w-5 text-yellow-500" />
                      )
                    ) : (
                      <Shield className="h-5 w-5 text-red-500" />
                    )}
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Overall Risk</span>
                      <span className={`font-medium ${
                        riskAssessment.riskScore > 0.7 ? "text-red-500" :
                        riskAssessment.riskScore > 0.5 ? "text-yellow-500" :
                        "text-green-500"
                      }`}>
                        {(riskAssessment.riskScore * 100).toFixed(0)}%
                      </span>
                    </div>
                    <Progress 
                      value={riskAssessment.riskScore * 100}
                      className="h-2"
                    />
                  </div>

                  <div className="text-xs text-muted-foreground">
                    {riskAssessment.reason}
                  </div>

                  {riskAssessment.adjustedSize !== parseFloat(quantity || "0") && (
                    <div className="text-sm">
                      <span className="text-muted-foreground">Adjusted Size: </span>
                      <span className="font-medium text-yellow-500">
                        {riskAssessment.adjustedSize.toFixed(0)}
                      </span>
                    </div>
                  )}
                </div>
              )}

              <Button 
                onClick={handleAssessRisk} 
                className="w-full" 
                variant="outline"
                disabled={!symbol || isAssessing}
              >
                <Shield className="mr-2 h-4 w-4" />
                {isAssessing ? "Assessing..." : "Assess Risk"}
              </Button>

              <Button 
                onClick={handleSubmitOrder} 
                className="w-full" 
                size="lg"
                disabled={!riskAssessment || !riskAssessment.shouldExecute}
              >
                Place {side.toUpperCase()} Order
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Positions Table */}
      <Card>
        <CardHeader>
          <CardTitle>Open Positions</CardTitle>
          <CardDescription>Manage your active positions</CardDescription>
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
                    Side
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-muted-foreground">
                    Qty
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-muted-foreground">
                    Entry
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-muted-foreground">
                    Current
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-muted-foreground">
                    P&L
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-muted-foreground">
                    SL/TP
                  </th>
                  <th className="px-6 py-3 text-right text-xs font-medium uppercase tracking-wider text-muted-foreground">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {mockPositions.map((position) => (
                  <tr
                    key={position.id}
                    className="hover:bg-muted/50 transition-colors"
                  >
                    <td className="whitespace-nowrap px-6 py-4 font-semibold">
                      {position.symbol}
                    </td>
                    <td className="whitespace-nowrap px-6 py-4">
                      <Badge
                        variant={position.side === "long" ? "default" : "secondary"}
                      >
                        {position.side.toUpperCase()}
                      </Badge>
                    </td>
                    <td className="whitespace-nowrap px-6 py-4">
                      {position.quantity}
                    </td>
                    <td className="whitespace-nowrap px-6 py-4">
                      ${position.entryPrice.toFixed(2)}
                    </td>
                    <td className="whitespace-nowrap px-6 py-4">
                      ${position.currentPrice.toFixed(2)}
                    </td>
                    <td className="whitespace-nowrap px-6 py-4">
                      <div
                        className={`font-medium ${
                          position.unrealizedPnL >= 0 ? "profit" : "loss"
                        }`}
                      >
                        ${position.unrealizedPnL.toFixed(2)} (
                        {position.unrealizedPnLPercent >= 0 ? "+" : ""}
                        {position.unrealizedPnLPercent.toFixed(2)}%)
                      </div>
                    </td>
                    <td className="whitespace-nowrap px-6 py-4 text-xs text-muted-foreground">
                      <div>SL: ${position.stopLoss?.toFixed(2) ?? "—"}</div>
                      <div>TP: ${position.takeProfit?.toFixed(2) ?? "—"}</div>
                    </td>
                    <td className="whitespace-nowrap px-6 py-4 text-right">
                      <Button
                        variant="destructive"
                        size="sm"
                        onClick={() =>
                          handleClosePosition(position.id, position.symbol)
                        }
                      >
                        Close
                      </Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
