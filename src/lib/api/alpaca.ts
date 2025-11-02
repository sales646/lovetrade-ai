import { useQuery } from "@tanstack/react-query";
import { supabase } from "@/integrations/supabase/client";

export interface AlpacaAccount {
  equity: number;
  cash: number;
  buyingPower: number;
  portfolioValue: number;
  daytradeCount: number;
  accountBlocked: boolean;
  tradingBlocked: boolean;
  status: string;
}

export function useAlpacaAccount() {
  return useQuery({
    queryKey: ["alpaca-account"],
    queryFn: async () => {
      const { data, error } = await supabase.functions.invoke("fetch-alpaca-account");

      if (error) throw error;
      if (!data?.success) throw new Error(data?.error || "Failed to fetch account");
      
      return data.account as AlpacaAccount;
    },
    refetchInterval: 60000, // Refetch every 60 seconds
    retry: 1,
  });
}
