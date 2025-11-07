import { useQuery } from "@tanstack/react-query";
import { supabase } from "@/integrations/supabase/client";

export interface TrainedModel {
  id: string;
  run_id: string;
  model_name: string;
  model_type: string;
  storage_path: string;
  file_size_bytes: number;
  performance_metrics: {
    sharpe_ratio: number;
    win_rate: number;
    mean_return: number;
    max_return: number;
    min_return: number;
  };
  hyperparameters: Record<string, any>;
  trained_on_symbols: string[];
  training_duration_seconds: number;
  final_sharpe_ratio: number;
  final_win_rate: number;
  is_best: boolean;
  created_at: string;
  updated_at: string;
}

/**
 * Hook to get all trained models
 */
export function useTrainedModels() {
  return useQuery({
    queryKey: ["trained-models"],
    queryFn: async () => {
      const { data, error } = await supabase
        .from("trained_models")
        .select("*")
        .order("created_at", { ascending: false });

      if (error) throw error;
      return data as TrainedModel[];
    },
    refetchInterval: 10000, // Refresh every 10 seconds
  });
}

/**
 * Hook to get best performing models
 */
export function useBestModels(limit: number = 10) {
  return useQuery({
    queryKey: ["best-models", limit],
    queryFn: async () => {
      const { data, error } = await supabase
        .from("trained_models")
        .select("*")
        .order("final_sharpe_ratio", { ascending: false })
        .limit(limit);

      if (error) throw error;
      return data as TrainedModel[];
    },
  });
}

/**
 * Hook to get models for a specific training run
 */
export function useRunModels(runId: string | null) {
  return useQuery({
    queryKey: ["run-models", runId],
    queryFn: async () => {
      if (!runId) return null;

      const { data, error } = await supabase
        .from("trained_models")
        .select("*")
        .eq("run_id", runId)
        .order("created_at", { ascending: false });

      if (error) throw error;
      return data as TrainedModel[];
    },
    enabled: !!runId,
  });
}

/**
 * Get download URL for a trained model
 */
export async function getModelDownloadUrl(storagePath: string): Promise<string> {
  const { data, error } = await supabase.storage
    .from("trained-models")
    .createSignedUrl(storagePath, 3600); // 1 hour expiry

  if (error) throw error;
  if (!data?.signedUrl) throw new Error("Failed to generate download URL");

  return data.signedUrl;
}

/**
 * Format file size in human-readable format
 */
export function formatFileSize(bytes: number): string {
  if (bytes === 0) return "0 Bytes";
  const k = 1024;
  const sizes = ["Bytes", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return Math.round(bytes / Math.pow(k, i) * 100) / 100 + " " + sizes[i];
}

/**
 * Format training duration
 */
export function formatDuration(seconds: number): string {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = seconds % 60;
  
  if (hours > 0) {
    return `${hours}h ${minutes}m ${secs}s`;
  } else if (minutes > 0) {
    return `${minutes}m ${secs}s`;
  } else {
    return `${secs}s`;
  }
}
