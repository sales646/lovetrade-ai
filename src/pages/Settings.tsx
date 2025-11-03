import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Separator } from "@/components/ui/separator";
import { Settings as SettingsIcon, Key, Shield, Database, Download, Upload, Radio } from "lucide-react";
import { useSettingsStore } from "@/store/settingsStore";
import { supabase } from "@/integrations/supabase/client";
import { toast } from "sonner";
import { triggerDataPopulation } from "@/lib/api/populate-data";

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
  const [isPopulatingData, setIsPopulatingData] = useState(false);
  
  // Alpaca trading credentials (stored in database)
  const [alpacaApiKey, setAlpacaApiKey] = useState("");
  const [alpacaSecretKey, setAlpacaSecretKey] = useState("");
  const [alpacaPaperTrading, setAlpacaPaperTrading] = useState(true);
  const [loadingAlpaca, setLoadingAlpaca] = useState(false);

  // Load Alpaca credentials from database
  useEffect(() => {
    const loadAlpacaCredentials = async () => {
      const { data, error } = await supabase
        .from("bot_config")
        .select("alpaca_api_key, alpaca_secret_key, alpaca_paper_trading")
        .single();
      
      if (data) {
        setAlpacaApiKey(data.alpaca_api_key || "");
        setAlpacaSecretKey(data.alpaca_secret_key || "");
        setAlpacaPaperTrading(data.alpaca_paper_trading ?? true);
      }
    };
    
    loadAlpacaCredentials();
  }, []);

  const handleSaveKeys = () => {
    updateAPIKey("alpaca", localKeys.alpaca);
    updateAPIKey("polygon", localKeys.polygon);
    updateAPIKey("finnhub", localKeys.finnhub);
    toast.success("API keys saved");
  };

  const handleSaveAlpacaCredentials = async () => {
    setLoadingAlpaca(true);
    try {
      // Get the current config to update
      const { data: currentConfig } = await supabase
        .from("bot_config")
        .select("id")
        .single();
      
      if (currentConfig) {
        const { error } = await supabase
          .from("bot_config")
          .update({
            alpaca_api_key: alpacaApiKey,
            alpaca_secret_key: alpacaSecretKey,
            alpaca_paper_trading: alpacaPaperTrading,
          })
          .eq("id", currentConfig.id);

        if (error) throw error;
      }
      
      toast.success(`Alpaca credentials saved (${alpacaPaperTrading ? "Paper Trading" : "Live Trading"})`);
    } catch (error) {
      toast.error("Failed to save Alpaca credentials");
      console.error(error);
    } finally {
      setLoadingAlpaca(false);
    }
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

  const handlePopulateData = async () => {
    setIsPopulatingData(true);
    try {
      await triggerDataPopulation();
      toast.success("Data population started", {
        description: "Fetching 5 years of historical data from Yahoo Finance. This will take several minutes.",
      });
    } catch (error) {
      toast.error("Failed to start data population", {
        description: error instanceof Error ? error.message : "Unknown error",
      });
    } finally {
      setIsPopulatingData(false);
    }
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

      <Tabs defaultValue="alpaca" className="space-y-6">
        <TabsList className="grid w-full grid-cols-6">
          <TabsTrigger value="alpaca">Alpaca Trading</TabsTrigger>
          <TabsTrigger value="database">Database</TabsTrigger>
          <TabsTrigger value="api-keys">Data APIs</TabsTrigger>
          <TabsTrigger value="data-mode">Data Mode</TabsTrigger>
          <TabsTrigger value="risk">Risk</TabsTrigger>
          <TabsTrigger value="export">Import/Export</TabsTrigger>
        </TabsList>

        {/* Alpaca Trading Tab */}
        <TabsContent value="alpaca">
          <Card>
            <CardHeader>
              <div className="flex items-center gap-2">
                <Key className="h-5 w-5 text-primary" />
                <CardTitle>Alpaca Trading Credentials</CardTitle>
              </div>
              <CardDescription>
                Configure your Alpaca API credentials for live or paper trading
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="rounded-lg border border-yellow-500/20 bg-yellow-500/10 p-4">
                <p className="text-sm text-yellow-600 dark:text-yellow-400">
                  <strong>⚠️ Security Notice:</strong> Your API keys are stored securely in the database and used by the autonomous trading bot. Never share these keys publicly.
                </p>
              </div>

              <div className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="alpaca-api-key">Alpaca API Key</Label>
                  <Input
                    id="alpaca-api-key"
                    type="password"
                    placeholder="PKXXXXXXXXXXXXXXXX"
                    value={alpacaApiKey}
                    onChange={(e) => setAlpacaApiKey(e.target.value)}
                  />
                  <p className="text-sm text-muted-foreground">
                    Your Alpaca API Key ID
                  </p>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="alpaca-secret-key">Alpaca Secret Key</Label>
                  <Input
                    id="alpaca-secret-key"
                    type="password"
                    placeholder="Enter your secret key"
                    value={alpacaSecretKey}
                    onChange={(e) => setAlpacaSecretKey(e.target.value)}
                  />
                  <p className="text-sm text-muted-foreground">
                    Your Alpaca Secret Key (keep this private!)
                  </p>
                </div>

                <Separator />

                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label htmlFor="paper-trading">Paper Trading Mode</Label>
                    <p className="text-sm text-muted-foreground">
                      {alpacaPaperTrading 
                        ? "Using paper trading (simulated, no real money)" 
                        : "⚠️ Using LIVE trading (real money at risk!)"}
                    </p>
                  </div>
                  <Switch
                    id="paper-trading"
                    checked={alpacaPaperTrading}
                    onCheckedChange={setAlpacaPaperTrading}
                  />
                </div>
              </div>

              <div className="rounded-lg border border-border bg-muted/50 p-4">
                <h4 className="font-semibold mb-2">How to get Alpaca API keys:</h4>
                <ol className="space-y-1 text-sm text-muted-foreground list-decimal list-inside">
                  <li>Create an account at <a href="https://alpaca.markets" target="_blank" rel="noopener noreferrer" className="text-primary hover:underline">alpaca.markets</a></li>
                  <li>Navigate to your account dashboard</li>
                  <li>Generate API keys for Paper Trading or Live Trading</li>
                  <li>Copy both the API Key ID and Secret Key</li>
                  <li>Paste them here and save</li>
                </ol>
              </div>

              <Button onClick={handleSaveAlpacaCredentials} disabled={loadingAlpaca}>
                {loadingAlpaca ? "Saving..." : "Save Alpaca Credentials"}
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Database Tab */}
        <TabsContent value="database">
          <Card>
            <CardHeader>
              <div className="flex items-center gap-2">
                <Database className="h-5 w-5 text-primary" />
                <CardTitle>Historical Market Data</CardTitle>
              </div>
              <CardDescription>
                Download historical data from Yahoo Finance for training your models
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="rounded-lg border border-blue-500/20 bg-blue-500/10 p-4">
                <p className="text-sm text-blue-600 dark:text-blue-400">
                  <strong>📊 Data Source:</strong> Downloads 5 years of historical bars from Yahoo Finance for multiple symbols and timeframes. This data is used to train reinforcement learning models.
                </p>
              </div>

              <div className="space-y-4">
                <div className="rounded-lg border border-border bg-muted/50 p-4">
                  <h4 className="font-semibold mb-2">What will be downloaded:</h4>
                  <ul className="space-y-1 text-sm text-muted-foreground list-disc list-inside">
                    <li><strong className="text-foreground">Symbols:</strong> AAPL, MSFT, GOOGL, TSLA, NVDA, AMZN, META, JPM, BAC, WMT</li>
                    <li><strong className="text-foreground">Timeframes:</strong> 1m, 5m, 1h, 1d</li>
                    <li><strong className="text-foreground">Period:</strong> Up to 5 years of historical data</li>
                    <li><strong className="text-foreground">Estimated time:</strong> 3-5 minutes</li>
                  </ul>
                </div>

                <Button 
                  onClick={handlePopulateData} 
                  disabled={isPopulatingData}
                  className="w-full"
                  size="lg"
                >
                  <Download className="mr-2 h-4 w-4" />
                  {isPopulatingData ? 'Downloading Data...' : 'Download Training Data'}
                </Button>

                <p className="text-xs text-muted-foreground">
                  Data is fetched from Yahoo Finance and stored in your database. The process runs in the background and you can continue using the app.
                </p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Data API Keys Tab */}
        <TabsContent value="api-keys">
          <Card>
            <CardHeader>
              <div className="flex items-center gap-2">
                <Database className="h-5 w-5 text-primary" />
                <CardTitle>Market Data API Keys</CardTitle>
              </div>
              <CardDescription>
                Configure your market data provider API keys (optional)
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-4">

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
