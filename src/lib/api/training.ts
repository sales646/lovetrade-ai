import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { supabase } from "@/integrations/supabase/client";
import { toast } from "sonner";

/**
 * Hook to compute technical indicators for a symbol
 */
export function useComputeIndicators() {
  return useMutation({
    mutationFn: async ({ symbol, timeframe = "5m" }: { symbol: string; timeframe?: string }) => {
      const { data, error } = await supabase.functions.invoke("compute-indicators", {
        body: { symbol, timeframe, lookback: 200 },
      });

      if (error) throw error;
      if (!data.success) throw new Error("Failed to compute indicators");

      return data;
    },
    onSuccess: (data) => {
      toast.success(`Computed ${data.indicatorsCount} indicators for ${data.symbol}`);
    },
    onError: (error) => {
      console.error("Error computing indicators:", error);
      toast.error(error instanceof Error ? error.message : "Failed to compute indicators");
    },
  });
}

/**
 * Hook to analyze news with AI
 */
export function useAnalyzeNews() {
  return useMutation({
    mutationFn: async ({
      symbol,
      headline,
      snippet,
      timestamp,
      source,
    }: {
      symbol: string;
      headline: string;
      snippet?: string;
      timestamp?: string;
      source?: string;
    }) => {
      const { data, error } = await supabase.functions.invoke("analyze-news", {
        body: { symbol, headline, snippet, timestamp, source },
      });

      if (error) throw error;
      if (!data.success) throw new Error("Failed to analyze news");

      return data;
    },
    onSuccess: (data) => {
      toast.success(`Analyzed news for ${data.symbol}`);
    },
    onError: (error) => {
      console.error("Error analyzing news:", error);
      toast.error(error instanceof Error ? error.message : "Failed to analyze news");
    },
  });
}

/**
 * Hook to generate expert trajectories
 */
export function useGenerateTrajectories() {
  return useMutation({
    mutationFn: async ({
      symbol,
      timeframe = "5m",
      start_date,
      end_date,
    }: {
      symbol: string;
      timeframe?: string;
      start_date?: string;
      end_date?: string;
    }) => {
      const { data, error } = await supabase.functions.invoke("generate-trajectories", {
        body: { symbol, timeframe, start_date, end_date },
      });

      if (error) throw error;
      if (!data.success) throw new Error("Failed to generate trajectories");

      return data;
    },
    onSuccess: (data) => {
      toast.success(`Generated ${data.trajectories_count} trajectories for ${data.symbol}`);
    },
    onError: (error) => {
      console.error("Error generating trajectories:", error);
      toast.error(error instanceof Error ? error.message : "Failed to generate trajectories");
    },
  });
}

/**
 * Hook to get training runs
 */
export function useTrainingRuns() {
  return useQuery({
    queryKey: ["training-runs"],
    queryFn: async () => {
      const { data, error } = await supabase
        .from("training_runs")
        .select("*")
        .order("created_at", { ascending: false });

      if (error) throw error;
      return data;
    },
  });
}

/**
 * Hook to get metrics for a training run
 */
export function useTrainingMetrics(runId: string | null) {
  return useQuery({
    queryKey: ["training-metrics", runId],
    queryFn: async () => {
      if (!runId) return null;

      const { data, error } = await supabase
        .from("training_metrics")
        .select("*")
        .eq("run_id", runId)
        .order("epoch", { ascending: true });

      if (error) throw error;
      return data;
    },
    enabled: !!runId,
  });
}

/**
 * Hook to get expert trajectories stats
 */
export function useTrajectoryStats() {
  return useQuery({
    queryKey: ["trajectory-stats"],
    queryFn: async () => {
      const { data, error } = await supabase
        .from("expert_trajectories")
        .select("tactic_id, symbol, action", { count: "exact" });

      if (error) throw error;

      // Group by tactic
      const stats: Record<string, { total: number; buy: number; sell: number; hold: number }> = {};
      
      data?.forEach((row) => {
        const tactic = row.tactic_id;
        if (!stats[tactic]) {
          stats[tactic] = { total: 0, buy: 0, sell: 0, hold: 0 };
        }
        stats[tactic].total++;
        if (row.action === 1) stats[tactic].buy++;
        else if (row.action === -1) stats[tactic].sell++;
        else stats[tactic].hold++;
      });

      return { total: data?.length || 0, byTactic: stats };
    },
  });
}

/**
 * Hook to check for active local GPU training
 */
export function useLocalGPUTrainingStatus() {
  return useQuery({
    queryKey: ["local-gpu-status"],
    queryFn: async () => {
      // Check for training runs created in last 5 minutes
      const fiveMinutesAgo = new Date(Date.now() - 5 * 60 * 1000).toISOString();
      
      const { data, error } = await supabase
        .from("training_runs")
        .select("*")
        .gte("started_at", fiveMinutesAgo)
        .order("started_at", { ascending: false })
        .limit(10);

      if (error) throw error;

      // Get total episodes from training_metrics
      const { count: totalEpisodes, error: metricsError } = await supabase
        .from("training_metrics")
        .select("*", { count: "exact", head: true })
        .in("run_id", (data || []).map(r => r.id));

      if (metricsError) console.error("Error fetching metrics count:", metricsError);
      
      const activeRuns = data?.filter(run => run.status === "running") || [];
      const recentCompletedRuns = data?.filter(run => 
        run.status === "completed" && 
        new Date(run.started_at).getTime() > Date.now() - 2 * 60 * 1000 // Last 2 min
      ) || [];
      
      return {
        isActive: activeRuns.length > 0 || recentCompletedRuns.length > 0,
        activeCount: activeRuns.length,
        recentRuns: data || [],
        totalEpisodes: totalEpisodes || 0,
      };
    },
    refetchInterval: 10000, // Check every 10 seconds
  });
}
