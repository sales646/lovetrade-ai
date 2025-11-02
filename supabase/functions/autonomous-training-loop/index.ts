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

async function log(level: "INFO" | "WARN" | "ERROR", message: string, metadata?: any) {
  console.log(`[${level}] ${message}`);
  await supabase.from("system_logs").insert({
    level,
    source: "autonomous-training-loop",
    message,
    metadata: metadata || {},
  });
}

// Check if training should be active
async function isTrainingActive(): Promise<boolean> {
  const { data } = await supabase
    .from("bot_config")
    .select("is_active")
    .single();
  
  return data?.is_active || false;
}

// Run single training iteration (called by cron)
async function runTrainingIteration() {
  await log("INFO", "🚀 Starting scheduled training iteration");
  
  try {
    // Check if training is active
    const active = await isTrainingActive();
    
    if (!active) {
      await log("INFO", "Training disabled in config, skipping iteration");
      return { success: true, message: "Training disabled" };
    }
    
    // 1. Generate data (smaller batch for faster iterations)
    await log("INFO", "📊 Generating training data from Yahoo Finance...");
    const dataResponse = await supabase.functions.invoke("auto-data-generator", {
      body: {
        symbols: ["AAPL", "TSLA", "NVDA"],
        barsPerSymbol: 500,
        useRealData: true,
      }
    });
    
    if (dataResponse.error) {
      await log("ERROR", "Data generation failed", { error: dataResponse.error });
      throw new Error(`Data generation failed: ${dataResponse.error.message}`);
    }
    
    await log("INFO", "✅ Data generation complete");
    
    // Wait 2 seconds for data to settle
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // 2. Run training (smaller batch for faster iterations)
    await log("INFO", "🧠 Starting RL training...");
    const trainingResponse = await supabase.functions.invoke("autonomous-rl-trainer", {
      body: { iterations: 3 }
    });
    
    if (trainingResponse.error) {
      await log("ERROR", "Training failed", { error: trainingResponse.error });
      throw new Error(`Training failed: ${trainingResponse.error.message}`);
    }
    
    const result = trainingResponse.data;
    await log("INFO", `✅ Training complete: ${result?.iterations || 0} episodes, avg reward: ${result?.avgReward?.toFixed(2) || 'N/A'}`, result);
    
    return { 
      success: true, 
      message: "Training iteration completed successfully",
      result 
    };
    
  } catch (error) {
    await log("ERROR", "Training iteration error", { 
      error: error instanceof Error ? error.message : String(error) 
    });
    throw error;
  }
}

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    // Run a single training iteration (called by cron job)
    const result = await runTrainingIteration();
    
    return new Response(
      JSON.stringify(result),
      { headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
    
  } catch (error) {
    await log("ERROR", "Failed to handle request", { 
      error: error instanceof Error ? error.message : String(error) 
    });
    
    return new Response(
      JSON.stringify({ 
        success: false,
        error: error instanceof Error ? error.message : "Unknown error" 
      }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});
