import { z } from "zod";

// Market Data Types
export const QuoteSchema = z.object({
  symbol: z.string(),
  price: z.number(),
  change: z.number(),
  changePercent: z.number(),
  volume: z.number(),
  timestamp: z.date(),
});

export const BarSchema = z.object({
  timestamp: z.date(),
  open: z.number(),
  high: z.number(),
  low: z.number(),
  close: z.number(),
  volume: z.number(),
});

// Trading Types
export const PositionSchema = z.object({
  id: z.string(),
  symbol: z.string(),
  side: z.enum(["long", "short"]),
  quantity: z.number(),
  entryPrice: z.number(),
  currentPrice: z.number(),
  unrealizedPnL: z.number(),
  unrealizedPnLPercent: z.number(),
  stopLoss: z.number().optional(),
  takeProfit: z.number().optional(),
  openedAt: z.date(),
});

export const OrderSchema = z.object({
  id: z.string(),
  symbol: z.string(),
  side: z.enum(["buy", "sell"]),
  type: z.enum(["market", "limit"]),
  quantity: z.number(),
  price: z.number().optional(),
  status: z.enum(["pending", "filled", "cancelled"]),
  createdAt: z.date(),
  filledAt: z.date().optional(),
});

// Strategy Types
export const StrategyConfigSchema = z.object({
  id: z.string(),
  name: z.string(),
  enabled: z.boolean(),
  weight: z.number().min(0).max(1),
  params: z.record(z.any()),
});

export const TradeSignalSchema = z.object({
  id: z.string(),
  symbol: z.string(),
  strategyId: z.string(),
  side: z.enum(["buy", "sell"]),
  confidence: z.number().min(0).max(1),
  price: z.number(),
  timestamp: z.date(),
  reason: z.string(),
});

// Training Types
export const ModelMetricsSchema = z.object({
  accuracy: z.number(),
  precision: z.number(),
  recall: z.number(),
  f1Score: z.number(),
  sharpeRatio: z.number().optional(),
  maxDrawdown: z.number().optional(),
});

export const TrainingJobSchema = z.object({
  id: z.string(),
  status: z.enum(["running", "completed", "failed"]),
  progress: z.number().min(0).max(100),
  startedAt: z.date(),
  completedAt: z.date().optional(),
  metrics: ModelMetricsSchema.optional(),
});

// Logs
export const AppLogSchema = z.object({
  id: z.string(),
  level: z.enum(["INFO", "WARN", "ERROR"]),
  message: z.string(),
  timestamp: z.date(),
  source: z.string().optional(),
  metadata: z.record(z.any()).optional(),
});

// Export TypeScript types
export type Quote = z.infer<typeof QuoteSchema>;
export type Bar = z.infer<typeof BarSchema>;
export type Position = z.infer<typeof PositionSchema>;
export type Order = z.infer<typeof OrderSchema>;
export type StrategyConfig = z.infer<typeof StrategyConfigSchema>;
export type TradeSignal = z.infer<typeof TradeSignalSchema>;
export type ModelMetrics = z.infer<typeof ModelMetricsSchema>;
export type TrainingJob = z.infer<typeof TrainingJobSchema>;
export type AppLog = z.infer<typeof AppLogSchema>;

// Timeframe type
export type Timeframe = "1m" | "5m" | "15m" | "1h" | "1d";
