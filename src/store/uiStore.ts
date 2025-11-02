import { create } from "zustand";
import { persist } from "zustand/middleware";
import { Timeframe } from "@/lib/types";

interface UIState {
  sidebarCollapsed: boolean;
  activeSymbol: string | null;
  timeframe: Timeframe;
  toggleSidebar: () => void;
  setActiveSymbol: (symbol: string | null) => void;
  setTimeframe: (timeframe: Timeframe) => void;
}

export const useUIStore = create<UIState>()(
  persist(
    (set) => ({
      sidebarCollapsed: false,
      activeSymbol: null,
      timeframe: "5m",
      toggleSidebar: () => set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),
      setActiveSymbol: (symbol) => set({ activeSymbol: symbol }),
      setTimeframe: (timeframe) => set({ timeframe }),
    }),
    {
      name: "ui-storage",
    }
  )
);
