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
    source: "fetch-alpaca-account",
    message,
    metadata: metadata || {},
  });
}

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    // Fetch Alpaca API credentials from bot_config
    const { data: config, error: configError } = await supabase
      .from("bot_config")
      .select("alpaca_api_key, alpaca_secret_key, alpaca_paper_trading")
      .single();

    if (configError || !config) {
      throw new Error("Bot config not found or missing Alpaca credentials");
    }

    if (!config.alpaca_api_key || !config.alpaca_secret_key) {
      throw new Error("Alpaca API credentials not configured");
    }

    // Determine Alpaca API base URL (paper or live)
    const baseUrl = config.alpaca_paper_trading
      ? "https://paper-api.alpaca.markets"
      : "https://api.alpaca.markets";

    await log("INFO", `Fetching account info from Alpaca (${config.alpaca_paper_trading ? 'paper' : 'live'} trading)`);

    // Fetch account information
    const accountResponse = await fetch(`${baseUrl}/v2/account`, {
      headers: {
        "APCA-API-KEY-ID": config.alpaca_api_key,
        "APCA-API-SECRET-KEY": config.alpaca_secret_key,
      },
    });

    if (!accountResponse.ok) {
      const errorText = await accountResponse.text();
      throw new Error(`Alpaca API error: ${accountResponse.status} - ${errorText}`);
    }

    const accountData = await accountResponse.json();

    await log("INFO", "Successfully fetched Alpaca account data", {
      equity: accountData.equity,
      cash: accountData.cash,
      buyingPower: accountData.buying_power,
    });

    // Return relevant account information
    return new Response(
      JSON.stringify({
        success: true,
        account: {
          equity: parseFloat(accountData.equity),
          cash: parseFloat(accountData.cash),
          buyingPower: parseFloat(accountData.buying_power),
          portfolioValue: parseFloat(accountData.portfolio_value),
          daytradeCount: accountData.daytrade_count,
          accountBlocked: accountData.account_blocked,
          tradingBlocked: accountData.trading_blocked,
          status: accountData.status,
        },
      }),
      { headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  } catch (error) {
    await log("ERROR", "Failed to fetch Alpaca account", { error: error instanceof Error ? error.message : String(error) });
    return new Response(
      JSON.stringify({ 
        success: false,
        error: error instanceof Error ? error.message : "Unknown error" 
      }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});
