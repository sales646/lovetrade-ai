import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { supabase } from "@/integrations/supabase/client";
import { Bar } from "@/lib/types";
import { toast } from "sonner";

/**
 * Hook to fetch historical bars from database
 */
export function useHistoricalBars(
  symbol: string | null,
  timeframe: string = "1d",
  limit: number = 100
) {
  return useQuery({
    queryKey: ["historical-bars", symbol, timeframe, limit],
    queryFn: async () => {
      if (!symbol) return null;

      const { data, error } = await supabase
        .from("historical_bars")
        .select("*")
        .eq("symbol", symbol)
        .eq("timeframe", timeframe)
        .order("timestamp", { ascending: false })
        .limit(limit);

      if (error) {
        console.error("Error fetching historical bars:", error);
        toast.error("Failed to fetch historical data");
        return null;
      }

      // Transform to Bar format and reverse to ascending order
      return data
        .map((row) => ({
          timestamp: new Date(row.timestamp),
          open: Number(row.open),
          high: Number(row.high),
          low: Number(row.low),
          close: Number(row.close),
          volume: Number(row.volume),
        }))
        .reverse() as Bar[];
    },
    enabled: !!symbol,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}

/**
 * Hook to fetch market data from Yahoo Finance and store it
 */
export function useFetchMarketData() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({
      symbol,
      period = "1mo",
      interval = "1d",
    }: {
      symbol: string;
      period?: string;
      interval?: string;
    }) => {
      const { data, error } = await supabase.functions.invoke("fetch-market-data", {
        body: { symbol, period, interval },
      });

      if (error) throw error;
      if (!data.success) throw new Error(data.error || "Failed to fetch data");

      return data;
    },
    onSuccess: (data) => {
      toast.success(`Fetched ${data.barsCount} bars for ${data.symbol}`);
      // Invalidate historical bars query to refetch
      queryClient.invalidateQueries({ queryKey: ["historical-bars"] });
    },
    onError: (error) => {
      console.error("Error fetching market data:", error);
      toast.error(error instanceof Error ? error.message : "Failed to fetch market data");
    },
  });
}

/**
 * Hook to get all stored symbols
 */
export function useStoredSymbols() {
  return useQuery({
    queryKey: ["stored-symbols"],
    queryFn: async () => {
      const { data, error } = await supabase
        .from("symbols")
        .select("*")
        .eq("is_active", true)
        .order("symbol");

      if (error) {
        console.error("Error fetching symbols:", error);
        return [];
      }

      return data;
    },
    staleTime: 60000, // 1 minute
  });
}

/**
 * Get training dataset - fetches all historical data for specified symbols
 */
export async function getTrainingDataset(
  symbols: string[],
  timeframe: string = "1d"
): Promise<{ symbol: string; bars: Bar[] }[]> {
  const results = await Promise.all(
    symbols.map(async (symbol) => {
      const { data, error } = await supabase
        .from("historical_bars")
        .select("*")
        .eq("symbol", symbol)
        .eq("timeframe", timeframe)
        .order("timestamp", { ascending: true });

      if (error || !data) {
        console.error(`Error fetching training data for ${symbol}:`, error);
        return { symbol, bars: [] };
      }

      const bars: Bar[] = data.map((row) => ({
        timestamp: new Date(row.timestamp),
        open: Number(row.open),
        high: Number(row.high),
        low: Number(row.low),
        close: Number(row.close),
        volume: Number(row.volume),
      }));

      return { symbol, bars };
    })
  );

  return results;
}
