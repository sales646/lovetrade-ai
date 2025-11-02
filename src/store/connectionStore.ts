import { create } from "zustand";

interface ConnectionState {
  wsStatus: "connected" | "disconnected" | "connecting";
  lastHeartbeat: Date | null;
  setWsStatus: (status: "connected" | "disconnected" | "connecting") => void;
  updateHeartbeat: () => void;
}

export const useConnectionStore = create<ConnectionState>((set) => ({
  wsStatus: "disconnected",
  lastHeartbeat: null,
  setWsStatus: (status) => set({ wsStatus: status }),
  updateHeartbeat: () => set({ lastHeartbeat: new Date() }),
}));
