import { Badge } from "@/components/ui/badge";
import { Activity, Wifi, WifiOff } from "lucide-react";
import { useSettingsStore } from "@/store/settingsStore";
import { useConnectionStore } from "@/store/connectionStore";

/**
 * Global data mode indicator for the header
 * Shows: Live | Polling | Mock based on settings
 */
export function DataModeBadge() {
  const { dataMode } = useSettingsStore();
  const { wsStatus } = useConnectionStore();

  if (dataMode === "live") {
    const isConnected = wsStatus === "connected";
    return (
      <Badge
        variant={isConnected ? "default" : "destructive"}
        className={`text-xs ${isConnected ? "bg-success hover:bg-success/90" : ""}`}
      >
        <Activity className="mr-1.5 h-3 w-3" />
        Live
      </Badge>
    );
  }

  if (dataMode === "polling") {
    return (
      <Badge variant="secondary" className="text-xs">
        <Wifi className="mr-1.5 h-3 w-3" />
        Polling
      </Badge>
    );
  }

  return (
    <Badge variant="outline" className="text-xs border-amber-500 text-amber-500">
      <WifiOff className="mr-1.5 h-3 w-3" />
      Mock Data
    </Badge>
  );
}
