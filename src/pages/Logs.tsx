import { useState, useMemo, useRef } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Search, Copy, AlertCircle, Info, AlertTriangle } from "lucide-react";
import { useVirtualizer } from "@tanstack/react-virtual";
import { toast } from "sonner";
import { format } from "date-fns";

interface LogEntry {
  id: string;
  level: "INFO" | "WARN" | "ERROR";
  message: string;
  timestamp: Date;
  source?: string;
}

// Generate mock logs
const generateMockLogs = (count: number): LogEntry[] => {
  const levels: Array<"INFO" | "WARN" | "ERROR"> = ["INFO", "WARN", "ERROR"];
  const sources = ["API", "Strategy", "Trading", "Data", "System"];
  const messages = [
    "Successfully fetched market data",
    "Connection timeout retrying...",
    "Strategy signal generated",
    "Order executed successfully",
    "Failed to connect to data provider",
    "Risk limit exceeded",
    "Data validation passed",
    "Position opened",
    "Position closed with profit",
    "Stop loss triggered",
  ];

  return Array.from({ length: count }, (_, i) => ({
    id: `log-${i}`,
    level: levels[Math.floor(Math.random() * levels.length)],
    message: messages[Math.floor(Math.random() * messages.length)],
    timestamp: new Date(Date.now() - Math.random() * 86400000), // Random within last 24h
    source: sources[Math.floor(Math.random() * sources.length)],
  })).sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
};

const mockLogs = generateMockLogs(1000);

export default function Logs() {
  const [levelFilter, setLevelFilter] = useState<"ALL" | "INFO" | "WARN" | "ERROR">("ALL");
  const [searchQuery, setSearchQuery] = useState("");
  const parentRef = useRef<HTMLDivElement>(null);

  // Filter logs
  const filteredLogs = useMemo(() => {
    return mockLogs.filter((log) => {
      const matchesLevel = levelFilter === "ALL" || log.level === levelFilter;
      const matchesSearch =
        !searchQuery ||
        log.message.toLowerCase().includes(searchQuery.toLowerCase()) ||
        log.source?.toLowerCase().includes(searchQuery.toLowerCase());
      return matchesLevel && matchesSearch;
    });
  }, [levelFilter, searchQuery]);

  // Virtualize the list
  const rowVirtualizer = useVirtualizer({
    count: filteredLogs.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 60,
    overscan: 10,
  });

  const handleCopyLogs = () => {
    const last200 = filteredLogs.slice(0, 200);
    const text = last200
      .map(
        (log) =>
          `[${format(log.timestamp, "yyyy-MM-dd HH:mm:ss")}] ${log.level} [${
            log.source || "—"
          }] ${log.message}`
      )
      .join("\n");

    navigator.clipboard.writeText(text);
    toast.success("Copied last 200 logs to clipboard");
  };

  const getLevelIcon = (level: string) => {
    switch (level) {
      case "ERROR":
        return <AlertCircle className="h-4 w-4" />;
      case "WARN":
        return <AlertTriangle className="h-4 w-4" />;
      default:
        return <Info className="h-4 w-4" />;
    }
  };

  const getLevelColor = (level: string) => {
    switch (level) {
      case "ERROR":
        return "text-red-500";
      case "WARN":
        return "text-yellow-500";
      default:
        return "text-blue-500";
    }
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>System Logs</CardTitle>
              <CardDescription>
                Monitor system events and activities ({filteredLogs.length} entries)
              </CardDescription>
            </div>
            <Button onClick={handleCopyLogs} variant="outline">
              <Copy className="mr-2 h-4 w-4" />
              Copy Last 200
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {/* Filters */}
          <div className="flex gap-4 mb-6">
            <div className="flex-1">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <Input
                  placeholder="Search logs..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-9"
                />
              </div>
            </div>
            <Select
              value={levelFilter}
              onValueChange={(v: any) => setLevelFilter(v)}
            >
              <SelectTrigger className="w-[150px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="ALL">All Levels</SelectItem>
                <SelectItem value="INFO">INFO</SelectItem>
                <SelectItem value="WARN">WARN</SelectItem>
                <SelectItem value="ERROR">ERROR</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Virtual List */}
          <div
            ref={parentRef}
            className="h-[600px] overflow-auto border border-border rounded-lg"
          >
            <div
              style={{
                height: `${rowVirtualizer.getTotalSize()}px`,
                width: "100%",
                position: "relative",
              }}
            >
              {rowVirtualizer.getVirtualItems().map((virtualRow) => {
                const log = filteredLogs[virtualRow.index];
                return (
                  <div
                    key={virtualRow.key}
                    style={{
                      position: "absolute",
                      top: 0,
                      left: 0,
                      width: "100%",
                      height: `${virtualRow.size}px`,
                      transform: `translateY(${virtualRow.start}px)`,
                    }}
                    className="border-b border-border px-4 py-3 hover:bg-muted/50 transition-colors"
                  >
                    <div className="flex items-start gap-3">
                      <div className={`mt-0.5 ${getLevelColor(log.level)}`}>
                        {getLevelIcon(log.level)}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <Badge
                            variant={
                              log.level === "ERROR"
                                ? "destructive"
                                : log.level === "WARN"
                                ? "secondary"
                                : "default"
                            }
                            className="text-xs"
                          >
                            {log.level}
                          </Badge>
                          <span className="text-xs text-muted-foreground">
                            {format(log.timestamp, "yyyy-MM-dd HH:mm:ss")}
                          </span>
                          {log.source && (
                            <span className="text-xs text-muted-foreground">
                              [{log.source}]
                            </span>
                          )}
                        </div>
                        <p className="text-sm text-foreground break-words">
                          {log.message}
                        </p>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
