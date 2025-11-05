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
    // Get latest training metrics from last 24 hours
    const oneDayAgo = new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString();
    const { data: metrics, error: metricsError } = await supabase
      .from('training_metrics')
      .select('*')
      .gte('created_at', oneDayAgo)
      .order('created_at', { ascending: false })
      .limit(1000);

    if (metricsError) throw metricsError;

    // Get PBT populations
    const { data: pbt, error: pbtError } = await supabase
      .from('pbt_populations')
      .select('*')
      .order('created_at', { ascending: false })
      .limit(10);

    if (pbtError) throw pbtError;

    // Calculate aggregate stats from recent metrics
    const recentMetrics = metrics?.slice(0, 20) || [];
    const avgReward = recentMetrics.reduce((sum, m) => sum + (m.mean_reward || 0), 0) / (recentMetrics.length || 1);
    const avgLoss = recentMetrics.reduce((sum, m) => sum + (m.policy_loss || 0), 0) / (recentMetrics.length || 1);

    // Check if training is active (metrics in last 10 minutes)
    const tenMinutesAgo = new Date(Date.now() - 10 * 60 * 1000).toISOString();
    const isActive = (metrics?.[0]?.created_at || '') > tenMinutesAgo;

    // Get unique run IDs
    const runIds = [...new Set(metrics?.map(m => m.run_id) || [])];

    // System info from latest metric
    const latestMetric = metrics?.[0];
    const totalGPUs = latestMetric?.metadata?.world_size || 2;
    const envsPerGPU = latestMetric?.metadata?.envs_per_gpu || 256;
    const totalEnvs = totalGPUs * envsPerGPU;

    const status = {
      is_active: isActive,
      total_gpus: totalGPUs,
      total_environments: totalEnvs,
      performance: {
        avg_reward: avgReward,
        avg_loss: avgLoss,
      },
      pbt: pbt && pbt.length > 0 ? {
        generation: pbt[0].generation,
        population_size: pbt[0].population_size,
        best_performance: pbt[0].best_performance,
        mean_performance: pbt[0].mean_performance,
      } : null,
      recent_runs: runIds.slice(0, 5).map(id => {
        const runMetrics = metrics?.filter(m => m.run_id === id) || [];
        const firstMetric = runMetrics[runMetrics.length - 1];
        const lastMetric = runMetrics[0];
        
        return {
          id,
          status: lastMetric?.created_at > tenMinutesAgo ? 'running' : 'completed',
          started_at: firstMetric?.created_at,
          config: {
            world_size: firstMetric?.metadata?.world_size || 2,
            envs_per_gpu: firstMetric?.metadata?.envs_per_gpu || 256,
            model_type: 'transformer',
            pbt_enabled: pbt && pbt.length > 0,
          }
        }
      }),
      system_info: {
        distributed_available: true,
        max_gpus: totalGPUs,
        bf16_supported: true,
      }
    };

    return new Response(JSON.stringify(status), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  } catch (error) {
    console.error("Distributed training status error:", error);
    return new Response(
      JSON.stringify({ 
        error: error instanceof Error ? error.message : "Unknown error",
        is_active: false 
      }),
      { 
        status: 500, 
        headers: { ...corsHeaders, "Content-Type": "application/json" } 
      }
    );
  }
});
