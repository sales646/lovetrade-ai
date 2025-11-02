import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Separator } from "@/components/ui/separator";
import { Settings as SettingsIcon, Key, Shield, Database, Download, Upload, Radio } from "lucide-react";
import { useSettingsStore } from "@/store/settingsStore";
import { toast } from "sonner";

export default function Settings() {
  const {
    dataMode,
    apiKeys,
    riskDefaults,
    externalWsUrl,
    setDataMode,
    updateAPIKey,
    updateRiskDefaults,
    setExternalWsUrl,
    exportSettings,
    importSettings,
  } = useSettingsStore();

  const [localKeys, setLocalKeys] = useState(apiKeys);
  const [localRisk, setLocalRisk] = useState(riskDefaults);
  const [localWsUrl, setLocalWsUrl] = useState(externalWsUrl);

  const handleSaveKeys = () => {
    updateAPIKey("alpaca", localKeys.alpaca);
    updateAPIKey("polygon", localKeys.polygon);
    updateAPIKey("finnhub", localKeys.finnhub);
    toast.success("API keys saved");
  };

  const handleSaveRisk = () => {
    updateRiskDefaults(localRisk);
    toast.success("Risk defaults saved");
  };

  const handleSaveWsUrl = () => {
    setExternalWsUrl(localWsUrl);
    toast.success("WebSocket URL saved");
  };

  const handleExport = () => {
    const json = exportSettings();
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `tradepilot-settings-${new Date().toISOString().split("T")[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success("Settings exported");
  };

  const handleImport = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const json = event.target?.result as string;
      if (importSettings(json)) {
        setLocalRisk(useSettingsStore.getState().riskDefaults);
        setLocalWsUrl(useSettingsStore.getState().externalWsUrl);
        toast.success("Settings imported");
      } else {
        toast.error("Invalid settings file");
      }
    };
    reader.readAsText(file);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="relative overflow-hidden rounded-2xl border border-border bg-gradient-to-br from-card via-card to-card/50 p-8">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-primary/20 via-transparent to-transparent opacity-50" />
        <div className="relative flex items-center gap-4">
          <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10">
            <SettingsIcon className="h-8 w-8 text-primary" />
          </div>
          <div>
            <h2 className="mb-2 text-3xl font-bold">Settings</h2>
            <p className="text-muted-foreground">
              Configure API keys, data mode, risk parameters, and app preferences
            </p>
          </div>
        </div>
      </div>

      <Tabs defaultValue="api-keys" className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="api-keys">API Keys</TabsTrigger>
          <TabsTrigger value="data-mode">Data Mode</TabsTrigger>
          <TabsTrigger value="risk">Risk</TabsTrigger>
          <TabsTrigger value="export">Import/Export</TabsTrigger>
        </TabsList>

        {/* API Keys Tab */}
        <TabsContent value="api-keys">
          <Card>
            <CardHeader>
              <div className="flex items-center gap-2">
                <Key className="h-5 w-5 text-primary" />
                <CardTitle>API Keys</CardTitle>
              </div>
              <CardDescription>
                Configure your market data provider API keys
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="alpaca-key">Alpaca API Key</Label>
                  <Input
                    id="alpaca-key"
                    type="password"
                    placeholder="Enter Alpaca API key"
                    value={localKeys.alpaca}
                    onChange={(e) => setLocalKeys({ ...localKeys, alpaca: e.target.value })}
                  />
                  <p className="text-sm text-muted-foreground">
                    For live trading and real-time market data
                  </p>
                </div>

                <Separator />

                <div className="space-y-2">
                  <Label htmlFor="polygon-key">Polygon.io API Key</Label>
                  <Input
                    id="polygon-key"
                    type="password"
                    placeholder="Enter Polygon API key"
                    value={localKeys.polygon}
                    onChange={(e) => setLocalKeys({ ...localKeys, polygon: e.target.value })}
                  />
                  <p className="text-sm text-muted-foreground">
                    For historical market data and backtesting
                  </p>
                </div>

                <Separator />

                <div className="space-y-2">
                  <Label htmlFor="finnhub-key">Finnhub API Key</Label>
                  <Input
                    id="finnhub-key"
                    type="password"
                    placeholder="Enter Finnhub API key"
                    value={localKeys.finnhub}
                    onChange={(e) => setLocalKeys({ ...localKeys, finnhub: e.target.value })}
                  />
                  <p className="text-sm text-muted-foreground">
                    For news, earnings, and fundamental data
                  </p>
                </div>
              </div>

              <Button onClick={handleSaveKeys}>Save API Keys</Button>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Data Mode Tab */}
        <TabsContent value="data-mode">
          <Card>
            <CardHeader>
              <div className="flex items-center gap-2">
                <Radio className="h-5 w-5 text-primary" />
                <CardTitle>Data Mode</CardTitle>
              </div>
              <CardDescription>
                Choose how market data is fetched and displayed
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-4">
                <div className="space-y-2">
                  <Label>Current Mode</Label>
                  <Select value={dataMode} onValueChange={(v) => setDataMode(v as any)}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="live">
                        <div className="flex items-center gap-2">
                          <div className="h-2 w-2 rounded-full bg-green-500" />
                          Live - Real-time WebSocket
                        </div>
                      </SelectItem>
                      <SelectItem value="polling">
                        <div className="flex items-center gap-2">
                          <div className="h-2 w-2 rounded-full bg-yellow-500" />
                          Polling - HTTP requests
                        </div>
                      </SelectItem>
                      <SelectItem value="mock">
                        <div className="flex items-center gap-2">
                          <div className="h-2 w-2 rounded-full bg-gray-500" />
                          Mock - Simulated data
                        </div>
                      </SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="rounded-lg border border-border bg-muted/50 p-4">
                  <h4 className="font-semibold mb-2">Mode Descriptions</h4>
                  <ul className="space-y-2 text-sm text-muted-foreground">
                    <li>
                      <strong className="text-foreground">Live:</strong> Real-time WebSocket connection.
                      Requires API keys and low latency.
                    </li>
                    <li>
                      <strong className="text-foreground">Polling:</strong> Periodic HTTP requests.
                      More reliable but with 1-5s delay.
                    </li>
                    <li>
                      <strong className="text-foreground">Mock:</strong> Simulated data for testing.
                      No API keys required.
                    </li>
                  </ul>
                </div>

                {dataMode === "live" && (
                  <>
                    <Separator />
                    <div className="space-y-2">
                      <Label htmlFor="ws-url">External WebSocket URL (Optional)</Label>
                      <Input
                        id="ws-url"
                        type="url"
                        placeholder="wss://your-websocket-server.com"
                        value={localWsUrl}
                        onChange={(e) => setLocalWsUrl(e.target.value)}
                      />
                      <p className="text-sm text-muted-foreground">
                        Leave blank to use default provider WebSocket
                      </p>
                    </div>
                    <Button onClick={handleSaveWsUrl}>Save WebSocket URL</Button>
                  </>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Risk Defaults Tab */}
        <TabsContent value="risk">
          <Card>
            <CardHeader>
              <div className="flex items-center gap-2">
                <Shield className="h-5 w-5 text-primary" />
                <CardTitle>Risk Management</CardTitle>
              </div>
              <CardDescription>
                Set default risk parameters for all strategies
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="risk-per-trade">Risk Per Trade (%)</Label>
                  <Input
                    id="risk-per-trade"
                    type="number"
                    min="0.1"
                    max="10"
                    step="0.1"
                    value={localRisk.riskPerTrade}
                    onChange={(e) =>
                      setLocalRisk({ ...localRisk, riskPerTrade: parseFloat(e.target.value) })
                    }
                  />
                  <p className="text-sm text-muted-foreground">
                    Maximum percentage of capital to risk on a single trade
                  </p>
                </div>

                <Separator />

                <div className="space-y-2">
                  <Label htmlFor="max-positions">Max Concurrent Positions</Label>
                  <Input
                    id="max-positions"
                    type="number"
                    min="1"
                    max="20"
                    value={localRisk.maxConcurrentPositions}
                    onChange={(e) =>
                      setLocalRisk({
                        ...localRisk,
                        maxConcurrentPositions: parseInt(e.target.value),
                      })
                    }
                  />
                  <p className="text-sm text-muted-foreground">
                    Maximum number of positions open at the same time
                  </p>
                </div>

                <Separator />

                <div className="space-y-2">
                  <Label htmlFor="max-leverage">Max Leverage</Label>
                  <Input
                    id="max-leverage"
                    type="number"
                    min="1"
                    max="10"
                    step="0.5"
                    value={localRisk.maxLeverage}
                    onChange={(e) =>
                      setLocalRisk({ ...localRisk, maxLeverage: parseFloat(e.target.value) })
                    }
                  />
                  <p className="text-sm text-muted-foreground">
                    Maximum leverage multiplier for margin trading
                  </p>
                </div>
              </div>

              <Button onClick={handleSaveRisk}>Save Risk Defaults</Button>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Import/Export Tab */}
        <TabsContent value="export">
          <Card>
            <CardHeader>
              <div className="flex items-center gap-2">
                <Database className="h-5 w-5 text-primary" />
                <CardTitle>Import/Export Settings</CardTitle>
              </div>
              <CardDescription>
                Backup or restore your application configuration
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-4">
                <div className="flex items-center gap-4">
                  <Button onClick={handleExport} className="flex items-center gap-2">
                    <Download className="h-4 w-4" />
                    Export Settings
                  </Button>
                  <p className="text-sm text-muted-foreground">
                    Download settings as JSON (API keys excluded)
                  </p>
                </div>

                <Separator />

                <div className="flex items-center gap-4">
                  <label htmlFor="import-file">
                    <Button
                      type="button"
                      variant="outline"
                      className="flex items-center gap-2"
                      onClick={() => document.getElementById("import-file")?.click()}
                    >
                      <Upload className="h-4 w-4" />
                      Import Settings
                    </Button>
                    <input
                      id="import-file"
                      type="file"
                      accept=".json"
                      className="hidden"
                      onChange={handleImport}
                    />
                  </label>
                  <p className="text-sm text-muted-foreground">
                    Restore settings from a JSON file
                  </p>
                </div>
              </div>

              <div className="rounded-lg border border-border bg-muted/50 p-4">
                <h4 className="font-semibold mb-2">What's Exported?</h4>
                <ul className="space-y-1 text-sm text-muted-foreground list-disc list-inside">
                  <li>Data mode preferences</li>
                  <li>Risk management defaults</li>
                  <li>WebSocket configuration</li>
                  <li>App state and preferences</li>
                </ul>
                <p className="text-sm text-muted-foreground mt-3">
                  <strong className="text-foreground">Note:</strong> API keys are never exported
                  for security reasons.
                </p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
