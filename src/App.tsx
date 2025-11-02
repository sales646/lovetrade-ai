import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import DashboardLayout from "./components/DashboardLayout";
import Dashboard from "./pages/Dashboard";
import Watchlist from "./pages/Watchlist";
import Orders from "./pages/Orders";
import Strategies from "./pages/Strategies";
import Training from "./pages/Training";
import Logs from "./pages/Logs";
import Settings from "./pages/Settings";
import AutoTrading from "./pages/AutoTrading";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <DashboardLayout>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/watchlist" element={<Watchlist />} />
            <Route path="/orders" element={<Orders />} />
            <Route path="/auto-trading" element={<AutoTrading />} />
            <Route path="/strategies" element={<Strategies />} />
            <Route path="/training" element={<Training />} />
            <Route path="/logs" element={<Logs />} />
            <Route path="/settings" element={<Settings />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </DashboardLayout>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
