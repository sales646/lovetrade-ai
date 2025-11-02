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
    source: "macro-data-fetcher",
    message,
    metadata: metadata || {},
  });
}

// Fetch VIX (volatility index) data
async function fetchVIX(): Promise<number> {
  try {
    // Using Yahoo Finance API
    const response = await fetch(`https://query1.finance.yahoo.com/v8/finance/chart/^VIX?interval=1d&range=1d`);
    const data = await response.json();
    
    if (data.chart?.result?.[0]?.meta?.regularMarketPrice) {
      return data.chart.result[0].meta.regularMarketPrice;
    }
    
    return 15; // Default VIX value
  } catch (error) {
    await log("WARN", "Failed to fetch VIX", { error: error instanceof Error ? error.message : String(error) });
    return 15;
  }
}

// Fetch SPY (S&P 500) for market correlation
async function fetchSPY(): Promise<{ price: number; change_pct: number }> {
  try {
    const response = await fetch(`https://query1.finance.yahoo.com/v8/finance/chart/SPY?interval=1d&range=5d`);
    const data = await response.json();
    
    const quotes = data.chart?.result?.[0]?.indicators?.quote?.[0];
    const timestamps = data.chart?.result?.[0]?.timestamp;
    
    if (quotes && timestamps) {
      const closes = quotes.close.filter((c: number) => c !== null);
      const currentPrice = closes[closes.length - 1];
      const previousPrice = closes[closes.length - 2];
      const changePct = ((currentPrice - previousPrice) / previousPrice) * 100;
      
      return { price: currentPrice, change_pct: changePct };
    }
    
    return { price: 450, change_pct: 0 };
  } catch (error) {
    await log("WARN", "Failed to fetch SPY", { error: error instanceof Error ? error.message : String(error) });
    return { price: 450, change_pct: 0 };
  }
}

// Fetch DXY (US Dollar Index) for currency correlation
async function fetchDXY(): Promise<{ price: number; change_pct: number }> {
  try {
    const response = await fetch(`https://query1.finance.yahoo.com/v8/finance/chart/DX-Y.NYB?interval=1d&range=5d`);
    const data = await response.json();
    
    const quotes = data.chart?.result?.[0]?.indicators?.quote?.[0];
    const timestamps = data.chart?.result?.[0]?.timestamp;
    
    if (quotes && timestamps) {
      const closes = quotes.close.filter((c: number) => c !== null);
      const currentPrice = closes[closes.length - 1];
      const previousPrice = closes[closes.length - 2];
      const changePct = ((currentPrice - previousPrice) / previousPrice) * 100;
      
      return { price: currentPrice, change_pct: changePct };
    }
    
    return { price: 103, change_pct: 0 };
  } catch (error) {
    await log("WARN", "Failed to fetch DXY", { error: error instanceof Error ? error.message : String(error) });
    return { price: 103, change_pct: 0 };
  }
}

// Fetch 10-year Treasury yield
async function fetchTreasuryYield(): Promise<number> {
  try {
    const response = await fetch(`https://query1.finance.yahoo.com/v8/finance/chart/^TNX?interval=1d&range=1d`);
    const data = await response.json();
    
    if (data.chart?.result?.[0]?.meta?.regularMarketPrice) {
      return data.chart.result[0].meta.regularMarketPrice;
    }
    
    return 4.2; // Default yield
  } catch (error) {
    await log("WARN", "Failed to fetch Treasury yield", { error: error instanceof Error ? error.message : String(error) });
    return 4.2;
  }
}

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    await log("INFO", "📈 Fetching macro-economic data");
    
    // Fetch all macro indicators
    const [vix, spy, dxy, treasury] = await Promise.all([
      fetchVIX(),
      fetchSPY(),
      fetchDXY(),
      fetchTreasuryYield(),
    ]);
    
    const timestamp = new Date().toISOString();
    
    // Calculate market regime
    let market_regime = "NORMAL";
    if (vix > 25) market_regime = "HIGH_VOLATILITY";
    else if (vix < 12) market_regime = "LOW_VOLATILITY";
    else if (spy.change_pct > 1.5) market_regime = "STRONG_BULLISH";
    else if (spy.change_pct < -1.5) market_regime = "STRONG_BEARISH";
    
    const macroData = {
      timestamp,
      vix,
      spy_price: spy.price,
      spy_change_pct: spy.change_pct,
      dxy_price: dxy.price,
      dxy_change_pct: dxy.change_pct,
      treasury_yield_10y: treasury,
      market_regime,
      risk_on: vix < 18 && spy.change_pct > 0,
      risk_off: vix > 22 && spy.change_pct < 0,
    };
    
    // Store in database
    await supabase.from("macro_data").insert(macroData);
    
    await log("INFO", `✅ Macro data stored: VIX=${vix.toFixed(2)}, SPY=${spy.change_pct.toFixed(2)}%, regime=${market_regime}`);

    return new Response(
      JSON.stringify({
        success: true,
        data: macroData,
      }),
      { headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  } catch (error) {
    await log("ERROR", "Macro data fetcher failed", { error: error instanceof Error ? error.message : String(error) });
    return new Response(
      JSON.stringify({ error: error instanceof Error ? error.message : "Unknown error" }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});
