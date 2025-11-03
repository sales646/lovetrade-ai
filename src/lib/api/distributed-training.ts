import { useQuery } from "@tanstack/react-query";
import { supabase } from "@/integrations/supabase/client";

export interface DistributedTrainingStatus {
  is_active: boolean;
  active_count: number;
  total_gpus: number;
  total_environments: number;
  performance: {
    avg_reward: number;
    avg_loss: number;
  };
  pbt: {
    generation: number;
    best_performance: number;
    mean_performance: number;
    population_size: number;
  } | null;
  recent_runs: Array<{
    id: string;
    status: string;
    started_at: string;
    config: {
      world_size: number;
      envs_per_gpu: number;
      model_type: string;
      pbt_enabled: boolean;
    };
  }>;
  system_info: {
    distributed_available: boolean;
    max_gpus: number;
    bf16_supported: boolean;
  };
}

export function useDistributedTrainingStatus() {
  return useQuery({
    queryKey: ["distributed-training-status"],
    queryFn: async () => {
      const { data, error } = await supabase.functions.invoke(
        "distributed-training-status"
      );

      if (error) {
        console.error("Failed to fetch distributed training status:", error);
        throw error;
      }

      return data as DistributedTrainingStatus;
    },
    refetchInterval: 5000, // Poll every 5 seconds
  });
}

export function useGPUMetrics(runId?: string) {
  return useQuery({
    queryKey: ["gpu-metrics", runId],
    queryFn: async () => {
      if (!runId) return null;

      const { data, error } = await supabase
        .from("training_metrics")
        .select("*")
        .eq("run_id", runId)
        .order("epoch", { ascending: false })
        .limit(100);

      if (error) throw error;
      return data;
    },
    enabled: !!runId,
    refetchInterval: 10000,
  });
}

export function usePBTHistory(limit: number = 50) {
  return useQuery({
    queryKey: ["pbt-history", limit],
    queryFn: async () => {
      const { data, error } = await supabase
        .from("pbt_populations")
        .select("*")
        .order("generation", { ascending: false })
        .limit(limit);

      if (error) throw error;
      return data;
    },
    refetchInterval: 15000,
  });
}
