import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.78.0";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { symbol, headline, snippet, timestamp, source } = await req.json();

    if (!symbol || !headline) {
      return new Response(JSON.stringify({ error: "Symbol and headline required" }), {
        status: 400,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    console.log(`Analyzing news for ${symbol}: "${headline}"`);

    const LOVABLE_API_KEY = Deno.env.get("LOVABLE_API_KEY");
    if (!LOVABLE_API_KEY) {
      throw new Error("LOVABLE_API_KEY not configured");
    }

    // Use Lovable AI to analyze news sentiment, surprise, and relevance
    const analysisPrompt = `Analyze this financial news for the stock ${symbol}:

Headline: ${headline}
${snippet ? `Content: ${snippet}` : ""}

Provide analysis in JSON format with these exact fields:
{
  "sentiment": <number between -1 (very negative) and 1 (very positive)>,
  "surprise_score": <number between 0 (expected) and 1 (very surprising/unexpected)>,
  "relevance_score": <number between 0 (irrelevant) and 1 (highly relevant to ${symbol})>
}

Consider:
- Sentiment: Overall tone (negative/neutral/positive) for stock performance
- Surprise: How unexpected or unusual is this news? (earnings beats, M&A, regulatory)
- Relevance: How directly does this impact ${symbol}? (not just sector news)

Return ONLY the JSON, no other text.`;

    const response = await fetch("https://ai.gateway.lovable.dev/v1/chat/completions", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${LOVABLE_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "google/gemini-2.5-flash",
        messages: [
          { role: "system", content: "You are a financial news analyst. Return only valid JSON." },
          { role: "user", content: analysisPrompt },
        ],
        temperature: 0.3, // Lower temperature for consistent analysis
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error("AI API error:", response.status, errorText);
      throw new Error(`AI API returned ${response.status}`);
    }

    const aiData = await response.json();
    const aiContent = aiData.choices?.[0]?.message?.content || "{}";
    
    console.log("AI response:", aiContent);

    // Parse AI response
    let analysis;
    try {
      // Try to extract JSON from markdown code blocks if present
      const jsonMatch = aiContent.match(/```json\n?([\s\S]*?)\n?```/) || aiContent.match(/\{[\s\S]*\}/);
      const jsonStr = jsonMatch ? (jsonMatch[1] || jsonMatch[0]) : aiContent;
      analysis = JSON.parse(jsonStr);
    } catch (e) {
      console.error("Failed to parse AI response:", e, aiContent);
      // Fallback to neutral values
      analysis = {
        sentiment: 0,
        surprise_score: 0,
        relevance_score: 0.5,
      };
    }

    // Calculate freshness in minutes
    const newsTime = timestamp ? new Date(timestamp) : new Date();
    const freshness_minutes = Math.floor((Date.now() - newsTime.getTime()) / 60000);

    // Store in database
    const supabaseAdmin = createClient(
      Deno.env.get("SUPABASE_URL") ?? "",
      Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "",
    );

    const newsFeature = {
      symbol,
      headline,
      timestamp: newsTime.toISOString(),
      source: source || "unknown",
      sentiment: Math.max(-1, Math.min(1, analysis.sentiment || 0)),
      surprise_score: Math.max(0, Math.min(1, analysis.surprise_score || 0)),
      relevance_score: Math.max(0, Math.min(1, analysis.relevance_score || 0)),
      freshness_minutes,
      article_snippet: snippet || null,
    };

    const { error: insertError } = await supabaseAdmin
      .from("news_features")
      .insert(newsFeature);

    if (insertError) {
      console.error("Error storing news feature:", insertError);
      throw new Error("Failed to store news analysis");
    }

    console.log(`Successfully analyzed and stored news for ${symbol}`);

    return new Response(
      JSON.stringify({
        success: true,
        ...newsFeature,
      }),
      { headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  } catch (error) {
    console.error("Error in analyze-news:", error);
    return new Response(
      JSON.stringify({ error: error instanceof Error ? error.message : "Unknown error" }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});
