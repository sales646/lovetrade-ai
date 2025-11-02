import { create } from "zustand";
import { persist } from "zustand/middleware";

interface WatchlistState {
  symbols: string[];
  addSymbol: (symbol: string) => void;
  removeSymbol: (symbol: string) => void;
  reorderSymbols: (symbols: string[]) => void;
}

export const useWatchlistStore = create<WatchlistState>()(
  persist(
    (set) => ({
      symbols: ["AAPL", "TSLA", "MSFT"],
      addSymbol: (symbol) =>
        set((state) => ({
          symbols: state.symbols.includes(symbol) ? state.symbols : [...state.symbols, symbol],
        })),
      removeSymbol: (symbol) =>
        set((state) => ({
          symbols: state.symbols.filter((s) => s !== symbol),
        })),
      reorderSymbols: (symbols) => set({ symbols }),
    }),
    {
      name: "watchlist-storage",
    }
  )
);
