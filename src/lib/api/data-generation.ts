import { useMutation, useQueryClient } from "@tanstack/react-query";
import { supabase } from "@/integrations/supabase/client";
import { toast } from "sonner";

export function useGenerateTrainingData() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async ({ symbols, barsPerSymbol }: { symbols: string[]; barsPerSymbol: number }) => {
      const { data, error } = await supabase.functions.invoke("auto-data-generator", {
        body: { symbols, bars_per_symbol: barsPerSymbol },
      });

      if (error) throw error;
      return data;
    },
    onSuccess: (data) => {
      toast.success(`Generated ${data.totals.trajectories} expert trajectories for ${data.symbols.length} symbols!`);
      queryClient.invalidateQueries({ queryKey: ["rl-metrics"] });
      queryClient.invalidateQueries({ queryKey: ["q-state"] });
    },
    onError: (error: Error) => {
      toast.error(`Data generation failed: ${error.message}`);
    },
  });
}
