/**
 * Safe number formatting utilities
 * Never fabricates values - returns "—" for invalid/missing data
 */

export interface NumberFormatOptions {
  decimals?: number;
  prefix?: string;
  suffix?: string;
  fallback?: string;
}

/**
 * Safely formats a number, returning "—" if value is invalid
 */
export function safeNumber(
  value: number | null | undefined,
  options: NumberFormatOptions = {}
): string {
  const { decimals = 2, prefix = "", suffix = "", fallback = "—" } = options;

  if (value === null || value === undefined || isNaN(value) || !isFinite(value)) {
    return fallback;
  }

  const formatted = value.toFixed(decimals);
  return `${prefix}${formatted}${suffix}`;
}

/**
 * Formats a percentage or returns "—"
 */
export function percentOrDash(
  value: number | null | undefined,
  decimals: number = 2
): string {
  if (value === null || value === undefined || isNaN(value) || !isFinite(value)) {
    return "—";
  }

  const sign = value >= 0 ? "+" : "";
  return `${sign}${value.toFixed(decimals)}%`;
}

/**
 * Formats currency or returns "—"
 */
export function currencyOrDash(
  value: number | null | undefined,
  currency: string = "$",
  decimals: number = 2
): string {
  return safeNumber(value, { prefix: currency, decimals });
}

/**
 * Formats large numbers with K/M/B suffix
 */
export function compactNumber(
  value: number | null | undefined,
  decimals: number = 1
): string {
  if (value === null || value === undefined || isNaN(value) || !isFinite(value)) {
    return "—";
  }

  const absValue = Math.abs(value);
  const sign = value < 0 ? "-" : "";

  if (absValue >= 1e9) {
    return `${sign}${(absValue / 1e9).toFixed(decimals)}B`;
  }
  if (absValue >= 1e6) {
    return `${sign}${(absValue / 1e6).toFixed(decimals)}M`;
  }
  if (absValue >= 1e3) {
    return `${sign}${(absValue / 1e3).toFixed(decimals)}K`;
  }
  return `${sign}${absValue.toFixed(decimals)}`;
}

/**
 * Type guard to check if a value is a valid number
 */
export function isValidNumber(value: unknown): value is number {
  return typeof value === "number" && !isNaN(value) && isFinite(value);
}
