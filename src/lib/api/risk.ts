import { supabase } from "@/integrations/supabase/client";

export interface RiskAssessmentRequest {
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

export interface RiskAssessment {
  riskScore: number;
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

export async function assessTradeRisk(
  request: RiskAssessmentRequest
): Promise<RiskAssessment> {
  try {
    const { data, error } = await supabase.functions.invoke("assess-risk", {
      body: request,
    });

    if (error) {
      console.error("Risk assessment error:", error);
      // Return conservative assessment on error
      return {
        riskScore: 1.0,
        adjustedSize: 0,
        shouldExecute: false,
        reason: `Error: ${error.message}`,
        factors: {
          volatilityRisk: 1.0,
          sentimentRisk: 1.0,
          technicalRisk: 1.0,
          timingRisk: 1.0,
          positionRisk: 1.0,
        },
      };
    }

    return data as RiskAssessment;
  } catch (error) {
    console.error("Failed to assess risk:", error);
    return {
      riskScore: 1.0,
      adjustedSize: 0,
      shouldExecute: false,
      reason: error instanceof Error ? error.message : "Unknown error",
      factors: {
        volatilityRisk: 1.0,
        sentimentRisk: 1.0,
        technicalRisk: 1.0,
        timingRisk: 1.0,
        positionRisk: 1.0,
      },
    };
  }
}
