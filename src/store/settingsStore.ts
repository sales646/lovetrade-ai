import { create } from "zustand";
import { persist } from "zustand/middleware";

export type DataMode = "live" | "polling" | "mock";

interface APIKeys {
  alpaca: string;
  polygon: string;
  finnhub: string;
}

interface RiskDefaults {
  riskPerTrade: number; // percentage
  maxConcurrentPositions: number;
  maxLeverage: number;
}

interface SettingsState {
  dataMode: DataMode;
  apiKeys: APIKeys;
  riskDefaults: RiskDefaults;
  externalWsUrl: string;
  
  // Actions
  setDataMode: (mode: DataMode) => void;
  updateAPIKey: (provider: keyof APIKeys, key: string) => void;
  updateRiskDefaults: (defaults: Partial<RiskDefaults>) => void;
  setExternalWsUrl: (url: string) => void;
  exportSettings: () => string;
  importSettings: (json: string) => boolean;
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set, get) => ({
      dataMode: "mock",
      apiKeys: {
        alpaca: "",
        polygon: "",
        finnhub: "",
      },
      riskDefaults: {
        riskPerTrade: 1.0,
        maxConcurrentPositions: 5,
        maxLeverage: 2,
      },
      externalWsUrl: "",

      setDataMode: (mode) => set({ dataMode: mode }),
      
      updateAPIKey: (provider, key) =>
        set((state) => ({
          apiKeys: { ...state.apiKeys, [provider]: key },
        })),
      
      updateRiskDefaults: (defaults) =>
        set((state) => ({
          riskDefaults: { ...state.riskDefaults, ...defaults },
        })),
      
      setExternalWsUrl: (url) => set({ externalWsUrl: url }),
      
      exportSettings: () => {
        const state = get();
        return JSON.stringify({
          dataMode: state.dataMode,
          riskDefaults: state.riskDefaults,
          externalWsUrl: state.externalWsUrl,
          // Don't export API keys for security
        }, null, 2);
      },
      
      importSettings: (json) => {
        try {
          const settings = JSON.parse(json);
          set({
            dataMode: settings.dataMode || "mock",
            riskDefaults: settings.riskDefaults || get().riskDefaults,
            externalWsUrl: settings.externalWsUrl || "",
          });
          return true;
        } catch {
          return false;
        }
      },
    }),
    {
      name: "settings-storage",
    }
  )
);
