import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.78.0";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

interface FetchRequest {
  symbol: string;
  period?: string; // "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"
  interval?: string; // "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"
}

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { symbol, period = "1mo", interval = "1d" } = await req.json() as FetchRequest;

    if (!symbol) {
      return new Response(JSON.stringify({ error: "Symbol is required" }), {
        status: 400,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    console.log(`Fetching ${symbol} data: period=${period}, interval=${interval}`);

    // Yahoo Finance API v8 (free, no API key required)
    const url = `https://query1.finance.yahoo.com/v8/finance/chart/${symbol}?interval=${interval}&period1=0&period2=9999999999&range=${period}`;
    
    const response = await fetch(url, {
      headers: {
        "User-Agent": "Mozilla/5.0",
      },
    });

    if (!response.ok) {
      console.error(`Yahoo Finance API error: ${response.status}`);
      return new Response(
        JSON.stringify({ error: `Yahoo Finance API returned ${response.status}` }),
        { status: response.status, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    const data = await response.json();
    
    if (!data.chart?.result?.[0]) {
      return new Response(
        JSON.stringify({ error: "Invalid symbol or no data available" }),
        { status: 404, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    const result = data.chart.result[0];
    const meta = result.meta;
    const timestamps = result.timestamp || [];
    const quotes = result.indicators.quote[0];

    // Transform to our format
    const bars = timestamps.map((ts: number, i: number) => ({
      symbol,
      timeframe: interval,
      timestamp: new Date(ts * 1000).toISOString(),
      open: quotes.open[i],
      high: quotes.high[i],
      low: quotes.low[i],
      close: quotes.close[i],
      volume: quotes.volume[i] || 0,
    })).filter((bar: any) => 
      bar.open !== null && 
      bar.high !== null && 
      bar.low !== null && 
      bar.close !== null
    );

    // Store in database using service role
    const supabaseAdmin = createClient(
      Deno.env.get("SUPABASE_URL") ?? "",
      Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "",
    );

    // Upsert symbol metadata
    await supabaseAdmin.from("symbols").upsert({
      symbol,
      name: meta.longName || symbol,
      exchange: meta.exchangeName,
      last_fetched: new Date().toISOString(),
      is_active: true,
    });

    // Bulk insert bars (upsert to handle duplicates)
    if (bars.length > 0) {
      const { error: barsError } = await supabaseAdmin
        .from("historical_bars")
        .upsert(bars, { onConflict: "symbol,timeframe,timestamp" });

      if (barsError) {
        console.error("Error inserting bars:", barsError);
        return new Response(
          JSON.stringify({ error: "Failed to store historical data" }),
          { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
        );
      }
    }

    console.log(`Successfully fetched and stored ${bars.length} bars for ${symbol}`);

    return new Response(
      JSON.stringify({
        success: true,
        symbol,
        period,
        interval,
        barsCount: bars.length,
        meta: {
          currency: meta.currency,
          exchange: meta.exchangeName,
          regularMarketPrice: meta.regularMarketPrice,
        },
      }),
      { headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  } catch (error) {
    console.error("Error in fetch-market-data:", error);
    return new Response(
      JSON.stringify({ error: error instanceof Error ? error.message : "Unknown error" }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});
