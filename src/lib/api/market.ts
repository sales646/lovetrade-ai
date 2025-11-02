import { useQuery } from "@tanstack/react-query";
import { Quote, Bar, QuoteSchema, BarSchema } from "@/lib/types";
import { useSettingsStore } from "@/store/settingsStore";
import { useHistoricalBars, type DataWithSource } from "./historical";
import { toast } from "sonner";

/**
 * Validate and parse data with Zod, showing toast on error
 */
function validateData<T>(
  schema: any,
  data: any,
  dataType: string
): DataWithSource<T> {
  try {
    const validated = schema.parse(data);
    return { data: validated, source: "api" };
  } catch (error) {
    console.error(`${dataType} validation failed:`, error);
    toast.error(`Invalid ${dataType} data received`, {
      description: "Data integrity check failed"
    });
    return { data: null, source: "none", error: String(error) };
  }
}

/**
 * Mock data generator with validation
 * Returns DataWithSource - never fabricates invalid data
 */
function generateMockQuote(symbol: string): DataWithSource<Quote> {
  try {
    const basePrice = 100 + Math.random() * 400;
    const change = (Math.random() - 0.5) * 20;
    
    const quote = QuoteSchema.parse({
      symbol,
      price: basePrice,
      change,
      changePercent: (change / basePrice) * 100,
      volume: Math.floor(Math.random() * 10000000),
      timestamp: new Date(),
    });
    
    return { data: quote, source: "mock" };
  } catch (error) {
    console.error("Failed to generate valid mock quote:", error);
    toast.error("Mock data generation failed", {
      description: "Unable to create valid test data"
    });
    return { data: null, source: "none", error: String(error) };
  }
}

function generateMockBars(symbol: string, limit: number): DataWithSource<Bar[]> {
  try {
    const bars: Bar[] = [];
    const now = Date.now();
    let price = 100 + Math.random() * 400;

    for (let i = limit - 1; i >= 0; i--) {
      const open = price;
      const change = (Math.random() - 0.5) * 5;
      const close = open + change;
      const high = Math.max(open, close) + Math.random() * 2;
      const low = Math.min(open, close) - Math.random() * 2;

      bars.push(
        BarSchema.parse({
          timestamp: new Date(now - i * 60000), // 1 minute bars
          open,
          high,
          low,
          close,
          volume: Math.floor(Math.random() * 100000),
        })
      );

      price = close;
    }

    return { data: bars, source: "mock" };
  } catch (error) {
    console.error("Failed to generate valid mock bars:", error);
    toast.error("Mock bars generation failed");
    return { data: null, source: "none", error: String(error) };
  }
}

/**
 * Hook to fetch latest quote with validation and source tracking
 */
export function useLatestQuote(symbol: string | null) {
  const { dataMode } = useSettingsStore();

  return useQuery({
    queryKey: ["quote", symbol, dataMode],
    queryFn: async (): Promise<DataWithSource<Quote>> => {
      if (!symbol) {
        return { data: null, source: "none" };
      }

      // In mock mode, generate fake data
      if (dataMode === "mock") {
        await new Promise((resolve) => setTimeout(resolve, 100)); // Simulate network delay
        return generateMockQuote(symbol);
      }

      // TODO: Implement real API calls when not in mock mode
      // Example:
      // const response = await fetch(`/api/quote/${symbol}`);
      // const json = await response.json();
      // return validateData(QuoteSchema, json, "quote");
      
      toast.error("Live/Polling mode not yet implemented");
      return { data: null, source: "none", error: "Not implemented" };
    },
    enabled: !!symbol,
    refetchInterval: dataMode === "polling" ? 5000 : false,
    staleTime: dataMode === "mock" ? 2000 : 1000,
  });
}

/**
 * Hook to fetch historical bars with validation
 * Uses real database data when available, falls back to mock
 */
export function useBars(
  symbol: string | null,
  timeframe: string,
  limit: number = 100
) {
  const { dataMode } = useSettingsStore();
  const { data: historicalResult, isLoading: historicalLoading } = useHistoricalBars(symbol, timeframe, limit);

  return useQuery({
    queryKey: ["bars", symbol, timeframe, limit, dataMode],
    queryFn: async (): Promise<DataWithSource<Bar[]>> => {
      if (!symbol) {
        return { data: null, source: "none" };
      }

      // If we have historical data from database, use it (validated by Supabase types)
      const historicalBars = historicalResult?.data;
      if (historicalBars && historicalBars.length > 0) {
        return { data: historicalBars, source: historicalResult.source };
      }

      // Otherwise, use mock mode
      if (dataMode === "mock") {
        await new Promise((resolve) => setTimeout(resolve, 200));
        return generateMockBars(symbol, limit);
      }

      // TODO: Implement real API calls for live mode
      toast.error("Live/Polling mode not yet implemented");
      return { data: null, source: "none", error: "Not implemented" };
    },
    enabled: !!symbol && !historicalLoading,
    staleTime: 30000, // 30 seconds
  });
}

/**
 * Hook to fetch news
 */
export function useNews(symbol: string | null, limit: number = 10) {
  const { dataMode } = useSettingsStore();

  return useQuery({
    queryKey: ["news", symbol, limit, dataMode],
    queryFn: async () => {
      if (!symbol) return [];

      // Mock news data
      if (dataMode === "mock") {
        await new Promise((resolve) => setTimeout(resolve, 150));
        return [
          {
            id: "1",
            title: `${symbol} announces quarterly earnings`,
            summary: "Company reports strong performance...",
            source: "Mock News",
            timestamp: new Date(),
            sentiment: 0.75,
          },
        ];
      }

      throw new Error("Live/Polling mode not yet implemented");
    },
    enabled: !!symbol,
    staleTime: 60000, // 1 minute
  });
}
