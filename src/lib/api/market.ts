import { useQuery } from "@tanstack/react-query";
import { Quote, Bar, QuoteSchema, BarSchema } from "@/lib/types";
import { useSettingsStore } from "@/store/settingsStore";
import { toast } from "sonner";

/**
 * Mock data generator with validation
 * Returns null on validation failure (UI will show "—")
 */
function generateMockQuote(symbol: string): Quote | null {
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
    
    return quote;
  } catch (error) {
    console.error("Failed to generate valid mock quote:", error);
    toast.error("Data validation failed");
    return null;
  }
}

function generateMockBars(symbol: string, limit: number): Bar[] | null {
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

    return bars;
  } catch (error) {
    console.error("Failed to generate valid mock bars:", error);
    toast.error("Data validation failed");
    return null;
  }
}

/**
 * Hook to fetch latest quote with validation
 */
export function useLatestQuote(symbol: string | null) {
  const { dataMode } = useSettingsStore();

  return useQuery({
    queryKey: ["quote", symbol, dataMode],
    queryFn: async () => {
      if (!symbol) return null;

      // In mock mode, generate fake data
      if (dataMode === "mock") {
        await new Promise((resolve) => setTimeout(resolve, 100)); // Simulate network delay
        return generateMockQuote(symbol);
      }

      // TODO: Implement real API calls when not in mock mode
      throw new Error("Live/Polling mode not yet implemented");
    },
    enabled: !!symbol,
    refetchInterval: dataMode === "polling" ? 5000 : false,
    staleTime: dataMode === "mock" ? 2000 : 1000,
  });
}

/**
 * Hook to fetch historical bars with validation
 */
export function useBars(
  symbol: string | null,
  timeframe: string,
  limit: number = 100
) {
  const { dataMode } = useSettingsStore();

  return useQuery({
    queryKey: ["bars", symbol, timeframe, limit, dataMode],
    queryFn: async () => {
      if (!symbol) return null;

      // In mock mode, generate fake data
      if (dataMode === "mock") {
        await new Promise((resolve) => setTimeout(resolve, 200));
        return generateMockBars(symbol, limit);
      }

      // TODO: Implement real API calls
      throw new Error("Live/Polling mode not yet implemented");
    },
    enabled: !!symbol,
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
