import "https://deno.land/x/xhr@0.1.0/mod.ts";
import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    // Check for active distributed training runs
    const { data: recentRuns, error: runsError } = await supabase
      .from("training_runs")
      .select("*")
      .order("created_at", { ascending: false })
      .limit(10);

    if (runsError) throw runsError;

    // Find active distributed runs
    const distributedRuns = recentRuns?.filter(run => 
      run.status === 'running' && 
      (run.config?.world_size > 1 || run.config?.distributed === true)
    ) || [];

    // Get GPU metrics for active runs
    const gpuMetrics = [];
    for (const run of distributedRuns) {
      const { data: metrics } = await supabase
        .from("training_metrics")
        .select("*")
        .eq("run_id", run.id)
        .order("epoch", { ascending: false })
        .limit(1);

      if (metrics && metrics.length > 0) {
        gpuMetrics.push({
          run_id: run.id,
          metrics: metrics[0]
        });
      }
    }

    // Calculate stats
    const isActive = distributedRuns.length > 0;
    const activeCount = distributedRuns.length;
    
    let totalGPUs = 0;
    let totalEnvs = 0;
    let avgReward = 0;
    let avgLoss = 0;

    if (distributedRuns.length > 0) {
      totalGPUs = distributedRuns.reduce((sum, run) => 
        sum + (run.config?.world_size || 0), 0
      );
      totalEnvs = distributedRuns.reduce((sum, run) => 
        sum + (run.config?.total_envs || 0), 0
      );

      if (gpuMetrics.length > 0) {
        avgReward = gpuMetrics.reduce((sum, m) => 
          sum + (m.metrics.mean_reward || 0), 0
        ) / gpuMetrics.length;
        avgLoss = gpuMetrics.reduce((sum, m) => 
          sum + (m.metrics.loss || 0), 0
        ) / gpuMetrics.length;
      }
    }

    // Get PBT status
    const pbtEnabled = distributedRuns.some(run => run.config?.pbt_enabled === true);
    let pbtStats = null;

    if (pbtEnabled) {
      // Get PBT population performance
      const { data: pbtData } = await supabase
        .from("pbt_populations")
        .select("*")
        .order("generation", { ascending: false })
        .limit(10);

      if (pbtData && pbtData.length > 0) {
        const latest = pbtData[0];
        pbtStats = {
          generation: latest.generation,
          best_performance: latest.best_performance,
          mean_performance: latest.mean_performance,
          population_size: latest.population_size
        };
      }
    }

    return new Response(
      JSON.stringify({
        is_active: isActive,
        active_count: activeCount,
        total_gpus: totalGPUs,
        total_environments: totalEnvs,
        performance: {
          avg_reward: avgReward,
          avg_loss: avgLoss
        },
        pbt: pbtEnabled ? pbtStats : null,
        recent_runs: distributedRuns.map(run => ({
          id: run.id,
          status: run.status,
          started_at: run.created_at,
          config: {
            world_size: run.config?.world_size || 1,
            envs_per_gpu: run.config?.envs_per_gpu || 0,
            model_type: run.config?.model_type || 'unknown',
            pbt_enabled: run.config?.pbt_enabled || false
          }
        })),
        system_info: {
          distributed_available: true,
          max_gpus: 8,
          bf16_supported: true
        }
      }),
      { headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  } catch (error) {
    console.error("Distributed training status error:", error);
    return new Response(
      JSON.stringify({ 
        error: error.message,
        is_active: false 
      }),
      { 
        status: 500, 
        headers: { ...corsHeaders, "Content-Type": "application/json" } 
      }
    );
  }
});
