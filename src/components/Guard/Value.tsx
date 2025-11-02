import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { AlertCircle } from "lucide-react";
import { safeNumber } from "@/lib/format";

export type DataSource = "api" | "store" | "computed" | "mock" | "none";

interface ValueProps {
  value: number | null | undefined;
  source: DataSource;
  formatter?: (n: number) => string;
  tooltip?: string;
  showBadge?: boolean;
  className?: string;
}

/**
 * Guarded value component that never shows fabricated numbers
 * - Renders "—" for invalid/missing data
 * - Shows source badges for mock data
 * - Provides tooltips for context
 */
export function Value({
  value,
  source,
  formatter,
  tooltip,
  showBadge = true,
  className = "",
}: ValueProps) {
  // Refuse to render if source is 'none' or value is invalid
  const isInvalid =
    source === "none" ||
    value === null ||
    value === undefined ||
    isNaN(value) ||
    !isFinite(value);

  const displayValue = isInvalid
    ? "—"
    : formatter
    ? formatter(value!)
    : safeNumber(value);

  const tooltipText = isInvalid
    ? tooltip || "No data available"
    : tooltip;

  const content = (
    <span className={`inline-flex items-center gap-1.5 ${className}`}>
      {tooltipText ? (
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <span className="inline-flex items-center gap-1">
                {displayValue}
                {isInvalid && (
                  <AlertCircle className="h-3 w-3 text-muted-foreground" />
                )}
              </span>
            </TooltipTrigger>
            <TooltipContent>
              <p>{tooltipText}</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      ) : (
        displayValue
      )}
      {showBadge && source === "mock" && !isInvalid && (
        <Badge variant="outline" className="text-[10px] px-1 py-0 h-4">
          Mock
        </Badge>
      )}
    </span>
  );

  return content;
}
