import "https://deno.land/x/xhr@0.1.0/mod.ts";
import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const LOVABLE_API_KEY = Deno.env.get("LOVABLE_API_KEY")!;

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);

async function log(level: "INFO" | "WARN" | "ERROR", message: string, metadata?: any) {
  console.log(`[${level}] ${message}`);
  await supabase.from("system_logs").insert({
    level,
    source: "news-analyzer",
    message,
    metadata: metadata || {},
  });
}

// Fetch news from multiple sources
async function fetchNewsForSymbol(symbol: string): Promise<any[]> {
  try {
    // Fetch from Alpha Vantage News API (free tier)
    const newsUrl = `https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=${symbol}&apikey=demo&limit=50`;
    const response = await fetch(newsUrl);
    const data = await response.json();
    
    if (data.feed) {
      return data.feed.map((article: any) => ({
        title: article.title,
        summary: article.summary,
        source: article.source,
        url: article.url,
        time_published: article.time_published,
        sentiment_score: article.overall_sentiment_score,
        sentiment_label: article.overall_sentiment_label,
      }));
    }
    
    return [];
  } catch (error) {
    await log("ERROR", `Failed to fetch news for ${symbol}`, { error: error instanceof Error ? error.message : String(error) });
    return [];
  }
}

// Analyze news sentiment with AI
async function analyzeNewsSentiment(articles: any[]): Promise<any> {
  if (articles.length === 0) {
    return {
      overall_sentiment: 0,
      bullish_count: 0,
      bearish_count: 0,
      neutral_count: 0,
      confidence: 0,
      key_themes: [],
    };
  }

  // Prepare news summary for AI analysis
  const newsSummary = articles.slice(0, 10).map((a, i) => 
    `${i + 1}. ${a.title}\n${a.summary.substring(0, 200)}...`
  ).join("\n\n");

  const systemPrompt = `You are a financial sentiment analyst. Analyze the following news articles and provide:
1. Overall sentiment score (-1 to 1, where -1 is very bearish, 0 is neutral, 1 is very bullish)
2. Key themes affecting the stock
3. Market impact assessment

Be concise and data-driven in your analysis.`;

  const userPrompt = `Analyze these recent news articles:\n\n${newsSummary}\n\nProvide your analysis in JSON format with keys: sentiment_score, bullish_themes, bearish_themes, market_impact`;

  try {
    const response = await fetch("https://ai.gateway.lovable.dev/v1/chat/completions", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${LOVABLE_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "google/gemini-2.5-flash",
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: userPrompt }
        ],
        temperature: 0.3,
      }),
    });

    const data = await response.json();
    const aiAnalysis = JSON.parse(data.choices[0].message.content);

    // Count sentiment distribution
    let bullish = 0, bearish = 0, neutral = 0;
    articles.forEach(a => {
      if (a.sentiment_score > 0.15) bullish++;
      else if (a.sentiment_score < -0.15) bearish++;
      else neutral++;
    });

    return {
      overall_sentiment: aiAnalysis.sentiment_score || 0,
      bullish_count: bullish,
      bearish_count: bearish,
      neutral_count: neutral,
      confidence: Math.min(articles.length / 10, 1),
      key_themes: [
        ...(aiAnalysis.bullish_themes || []),
        ...(aiAnalysis.bearish_themes || [])
      ],
      market_impact: aiAnalysis.market_impact || "unknown",
    };
  } catch (error) {
    await log("ERROR", "AI sentiment analysis failed", { error: error instanceof Error ? error.message : String(error) });
    
    // Fallback to simple averaging
    const avgSentiment = articles.reduce((sum, a) => sum + (a.sentiment_score || 0), 0) / articles.length;
    return {
      overall_sentiment: avgSentiment,
      bullish_count: articles.filter(a => a.sentiment_score > 0.15).length,
      bearish_count: articles.filter(a => a.sentiment_score < -0.15).length,
      neutral_count: articles.filter(a => Math.abs(a.sentiment_score) <= 0.15).length,
      confidence: Math.min(articles.length / 10, 1),
      key_themes: [],
      market_impact: avgSentiment > 0.2 ? "positive" : avgSentiment < -0.2 ? "negative" : "neutral",
    };
  }
}

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { symbols = ["AAPL", "TSLA", "NVDA"] } = await req.json().catch(() => ({}));
    
    await log("INFO", `📰 Analyzing news for ${symbols.length} symbols`);
    
    const results = [];
    
    for (const symbol of symbols) {
      await log("INFO", `Fetching news for ${symbol}`);
      const articles = await fetchNewsForSymbol(symbol);
      
      await log("INFO", `Found ${articles.length} articles for ${symbol}`);
      const sentiment = await analyzeNewsSentiment(articles);
      
      // Store in database
      await supabase.from("news_sentiment").upsert({
        symbol,
        timestamp: new Date().toISOString(),
        overall_sentiment: sentiment.overall_sentiment,
        bullish_count: sentiment.bullish_count,
        bearish_count: sentiment.bearish_count,
        neutral_count: sentiment.neutral_count,
        confidence: sentiment.confidence,
        key_themes: sentiment.key_themes,
        market_impact: sentiment.market_impact,
        article_count: articles.length,
      }, { onConflict: "symbol,timestamp" });
      
      results.push({
        symbol,
        articles: articles.length,
        sentiment: sentiment.overall_sentiment,
        impact: sentiment.market_impact,
      });
      
      // Rate limiting
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
    
    await log("INFO", `✅ News analysis complete for ${symbols.length} symbols`);

    return new Response(
      JSON.stringify({
        success: true,
        results,
      }),
      { headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  } catch (error) {
    await log("ERROR", "News analyzer failed", { error: error instanceof Error ? error.message : String(error) });
    return new Response(
      JSON.stringify({ error: error instanceof Error ? error.message : "Unknown error" }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});
