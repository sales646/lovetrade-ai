import "https://deno.land/x/xhr@0.1.0/mod.ts";
import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const LOVABLE_API_KEY = Deno.env.get("LOVABLE_API_KEY");

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

interface RiskAssessmentRequest {
  symbol: string;
  action: "buy" | "sell";
  proposedSize: number;
  marketData: {
    price: number;
    rsi: number;
    atr: number;
    vwap: number;
    vwapDeviation: number;
    volumeZScore: number;
    sentiment?: number;
    recentVolatility?: number;
  };
  position?: {
    currentSize: number;
    entryPrice: number;
    unrealizedPnL: number;
  };
  timeContext: {
    minutesSinceOpen: number;
    minutesUntilClose: number;
  };
}

interface RiskAssessment {
  riskScore: number; // 0-1, där 1 = highest risk
  adjustedSize: number;
  shouldExecute: boolean;
  reason: string;
  factors: {
    volatilityRisk: number;
    sentimentRisk: number;
    technicalRisk: number;
    timingRisk: number;
    positionRisk: number;
  };
}

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const request: RiskAssessmentRequest = await req.json();

    // Build comprehensive prompt for AI risk assessment
    const systemPrompt = `You are an expert quantitative risk analyst specializing in day trading risk assessment.
Your task is to evaluate trade risk based on multiple factors and provide a risk score between 0 (lowest risk) and 1 (highest risk).

RISK FACTORS TO CONSIDER:
1. Volatility Risk (ATR, recent volatility)
2. Sentiment Risk (news sentiment alignment with trade direction)
3. Technical Risk (RSI extremes, VWAP deviation)
4. Timing Risk (time of day, proximity to market close)
5. Position Risk (existing position size, concentration)

CRITICAL THRESHOLDS:
- Risk Score < 0.5: APPROVE full position size
- Risk Score 0.5-0.7: REDUCE position size proportionally
- Risk Score > 0.7: VETO trade completely

Respond with ONLY a valid JSON object matching this structure:
{
  "riskScore": 0.65,
  "reason": "Brief explanation of main risk factors",
  "factors": {
    "volatilityRisk": 0.7,
    "sentimentRisk": 0.4,
    "technicalRisk": 0.8,
    "timingRisk": 0.6,
    "positionRisk": 0.5
  }
}`;

    const userPrompt = `Assess risk for this trade:

PROPOSED TRADE:
- Symbol: ${request.symbol}
- Action: ${request.action.toUpperCase()}
- Proposed Size: ${request.proposedSize}%

MARKET DATA:
- Current Price: $${request.marketData.price.toFixed(2)}
- RSI(14): ${request.marketData.rsi.toFixed(1)}
- ATR(14): ${request.marketData.atr.toFixed(2)}
- VWAP: $${request.marketData.vwap.toFixed(2)}
- VWAP Deviation: ${request.marketData.vwapDeviation.toFixed(2)}%
- Volume Z-Score: ${request.marketData.volumeZScore.toFixed(2)}
${request.marketData.sentiment !== undefined ? `- News Sentiment: ${request.marketData.sentiment.toFixed(2)} (-1 to +1)` : ""}
${request.marketData.recentVolatility !== undefined ? `- Recent Volatility: ${request.marketData.recentVolatility.toFixed(2)}%` : ""}

TIME CONTEXT:
- Minutes Since Open: ${request.timeContext.minutesSinceOpen}
- Minutes Until Close: ${request.timeContext.minutesUntilClose}

${request.position ? `EXISTING POSITION:
- Current Size: ${request.position.currentSize}%
- Entry Price: $${request.position.entryPrice.toFixed(2)}
- Unrealized P&L: $${request.position.unrealizedPnL.toFixed(2)}
` : "No existing position"}

Provide your risk assessment now.`;

    console.log("Sending risk assessment request to Lovable AI");

    const response = await fetch("https://ai.gateway.lovable.dev/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${LOVABLE_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "google/gemini-2.5-flash",
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: userPrompt },
        ],
        temperature: 0.3, // Lower temperature for more consistent risk assessment
      }),
    });

    if (!response.ok) {
      if (response.status === 429) {
        return new Response(
          JSON.stringify({ error: "Rate limit exceeded. Please try again later." }),
          { status: 429, headers: { ...corsHeaders, "Content-Type": "application/json" } }
        );
      }
      if (response.status === 402) {
        return new Response(
          JSON.stringify({ error: "Payment required. Please add credits to your Lovable AI workspace." }),
          { status: 402, headers: { ...corsHeaders, "Content-Type": "application/json" } }
        );
      }
      const errorText = await response.text();
      console.error("Lovable AI error:", response.status, errorText);
      throw new Error(`AI gateway error: ${response.status}`);
    }

    const data = await response.json();
    const aiResponse = data.choices[0].message.content;
    console.log("AI Risk Assessment:", aiResponse);

    // Parse AI response
    let aiAssessment;
    try {
      // Try to extract JSON from response
      const jsonMatch = aiResponse.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        aiAssessment = JSON.parse(jsonMatch[0]);
      } else {
        throw new Error("No JSON found in response");
      }
    } catch (parseError) {
      console.error("Failed to parse AI response:", parseError);
      // Fallback to conservative risk assessment
      aiAssessment = {
        riskScore: 0.75,
        reason: "AI parsing error - conservative assessment applied",
        factors: {
          volatilityRisk: 0.7,
          sentimentRisk: 0.7,
          technicalRisk: 0.7,
          timingRisk: 0.7,
          positionRisk: 0.7,
        },
      };
    }

    const riskScore = Math.max(0, Math.min(1, aiAssessment.riskScore));

    // Apply 50% threshold rule
    let adjustedSize = request.proposedSize;
    let shouldExecute = true;
    let finalReason = aiAssessment.reason;

    if (riskScore > 0.7) {
      // VETO: Risk too high
      adjustedSize = 0;
      shouldExecute = false;
      finalReason = `🛑 VETO: ${aiAssessment.reason} (Risk: ${(riskScore * 100).toFixed(0)}%)`;
    } else if (riskScore > 0.5) {
      // REDUCE: Scale down position
      const reductionFactor = (0.7 - riskScore) / 0.2; // Linear scale from 0.5 to 0.7
      adjustedSize = request.proposedSize * reductionFactor;
      finalReason = `⚠️ REDUCED: ${aiAssessment.reason} (Risk: ${(riskScore * 100).toFixed(0)}%, Size: ${adjustedSize.toFixed(1)}%)`;
    } else {
      // APPROVE: Full size
      finalReason = `✅ APPROVED: ${aiAssessment.reason} (Risk: ${(riskScore * 100).toFixed(0)}%)`;
    }

    const assessment: RiskAssessment = {
      riskScore,
      adjustedSize,
      shouldExecute,
      reason: finalReason,
      factors: aiAssessment.factors,
    };

    console.log("Final Risk Assessment:", assessment);

    return new Response(JSON.stringify(assessment), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  } catch (error) {
    console.error("Error in assess-risk function:", error);
    return new Response(
      JSON.stringify({
        error: error instanceof Error ? error.message : "Unknown error",
        riskScore: 1.0,
        adjustedSize: 0,
        shouldExecute: false,
        reason: "Error in risk assessment - trade blocked for safety",
      }),
      {
        status: 500,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      }
    );
  }
});
