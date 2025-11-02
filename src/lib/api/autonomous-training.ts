import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { supabase } from "@/integrations/supabase/client";
import { toast } from "sonner";

export function useStartAutonomousTraining() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (iterations: number) => {
      const { data, error } = await supabase.functions.invoke("autonomous-rl-trainer", {
        body: { iterations },
      });

      if (error) throw error;
      return data;
    },
    onSuccess: (data) => {
      toast.success(`Training complete! Avg reward: ${data.avgReward.toFixed(2)}, ε=${data.epsilon.toFixed(3)}`);
      queryClient.invalidateQueries({ queryKey: ["rl-metrics"] });
      queryClient.invalidateQueries({ queryKey: ["q-state"] });
    },
    onError: (error: Error) => {
      toast.error(`Training failed: ${error.message}`);
    },
  });
}

export function useRLMetrics() {
  return useQuery({
    queryKey: ["rl-metrics"],
    queryFn: async () => {
      const { data, error } = await supabase
        .from("rl_training_metrics")
        .select("*")
        .order("created_at", { ascending: false })
        .limit(50);

      if (error) throw error;
      return data;
    },
    refetchInterval: 10000, // Refetch every 10s
  });
}

export function useQState() {
  return useQuery({
    queryKey: ["q-state"],
    queryFn: async () => {
      const { data, error } = await supabase
        .from("rl_q_state")
        .select("*")
        .maybeSingle();

      if (error) throw error;
      return data;
    },
    refetchInterval: 5000, // Refetch every 5s
  });
}
