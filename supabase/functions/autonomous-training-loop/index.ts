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

// Run continuous training loop
async function runContinuousTraining() {
  await log("INFO", "🚀 Starting autonomous training loop");
  
  let isRunning = true;
  let loopCount = 0;
  
  // Listen for shutdown signals
  const shutdownHandler = () => {
    isRunning = false;
    log("INFO", "Shutdown signal received, stopping training loop");
  };
  
  addEventListener('beforeunload', shutdownHandler);
  
  while (isRunning) {
    try {
      // Check if training is still active
      const active = await isTrainingActive();
      
      if (!active) {
        await log("INFO", "Training disabled in config, pausing loop");
        await new Promise(resolve => setTimeout(resolve, 10000)); // Wait 10s before checking again
        continue;
      }
      
      loopCount++;
      await log("INFO", `🔄 Training loop iteration #${loopCount}`);
      
      // 1. Generate data
      await log("INFO", "📊 Generating training data...");
      const dataResponse = await supabase.functions.invoke("auto-data-generator", {
        body: {
          symbols: ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL"],
          barsPerSymbol: 500,
          useRealData: false,
        }
      });
      
      if (dataResponse.error) {
        await log("ERROR", "Data generation failed", { error: dataResponse.error });
      } else {
        await log("INFO", "✅ Data generation complete");
      }
      
      // Wait 2 seconds for data to settle
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // 2. Run training
      await log("INFO", "🧠 Starting RL training...");
      const trainingResponse = await supabase.functions.invoke("autonomous-rl-trainer", {
        body: { iterations: 10 }
      });
      
      if (trainingResponse.error) {
        await log("ERROR", "Training failed", { error: trainingResponse.error });
      } else {
        const result = trainingResponse.data;
        await log("INFO", `✅ Training complete: ${result?.iterations || 0} episodes, avg reward: ${result?.avgReward?.toFixed(2) || 'N/A'}`, result);
      }
      
      // Wait 30 seconds before next iteration
      await log("INFO", `⏳ Waiting 30 seconds before next iteration...`);
      await new Promise(resolve => setTimeout(resolve, 30000));
      
    } catch (error) {
      await log("ERROR", "Training loop error", { 
        error: error instanceof Error ? error.message : String(error) 
      });
      
      // Wait 60 seconds on error before retrying
      await new Promise(resolve => setTimeout(resolve, 60000));
    }
  }
  
  await log("INFO", "Training loop stopped");
}

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { action } = await req.json().catch(() => ({ action: "start" }));
    
    if (action === "start") {
      // Start the continuous training loop in background
      // Use Promise to avoid blocking the response
      runContinuousTraining().catch(error => 
        log("ERROR", "Training loop crashed", { error: error.message })
      );
      
      return new Response(
        JSON.stringify({ 
          success: true, 
          message: "Autonomous training loop started" 
        }),
        { headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }
    
    return new Response(
      JSON.stringify({ 
        success: false, 
        error: "Unknown action" 
      }),
      { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } }
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
