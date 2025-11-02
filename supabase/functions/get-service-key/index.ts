import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const serviceKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY");
    const url = Deno.env.get("SUPABASE_URL");
    
    // Log to edge function logs (you'll see this in Cloud -> Edge Functions -> Logs)
    console.log("=== SUPABASE CREDENTIALS ===");
    console.log("SUPABASE_URL:", url);
    console.log("SUPABASE_SERVICE_ROLE_KEY:", serviceKey);
    console.log("===========================");
    
    return new Response(
      JSON.stringify({
        message: "Check edge function logs for credentials",
        url_prefix: url?.substring(0, 30) + "...",
        key_prefix: serviceKey?.substring(0, 20) + "...",
      }),
      {
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      }
    );
  } catch (error) {
    console.error("Error:", error);
    return new Response(
      JSON.stringify({ error: error instanceof Error ? error.message : "Unknown error" }),
      {
        status: 500,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      }
    );
  }
});
