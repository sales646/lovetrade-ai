import { ReactNode, useState, KeyboardEvent } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import {
  LayoutDashboard,
  Eye,
  ShoppingCart,
  Zap,
  Brain,
  ScrollText,
  Settings,
  Menu,
  Activity,
  Search,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { useUIStore } from "@/store/uiStore";
import { DataModeBadge } from "@/components/DataModeBadge";

const navigation = [
  { name: "Dashboard", href: "/", icon: LayoutDashboard },
  { name: "Watchlist", href: "/watchlist", icon: Eye },
  { name: "Orders", href: "/orders", icon: ShoppingCart },
  { name: "Auto Trading", href: "/auto-trading", icon: Activity },
  { name: "Strategies", href: "/strategies", icon: Zap },
  { name: "Training", href: "/training", icon: Brain },
  { name: "Logs", href: "/logs", icon: ScrollText },
  { name: "Settings", href: "/settings", icon: Settings },
];

interface DashboardLayoutProps {
  children: ReactNode;
}

export default function DashboardLayout({ children }: DashboardLayoutProps) {
  const location = useLocation();
  const navigate = useNavigate();
  const { sidebarCollapsed, toggleSidebar } = useUIStore();
  const [searchSymbol, setSearchSymbol] = useState("");

  const handleSearch = () => {
    if (searchSymbol.trim()) {
      navigate(`/watchlist?add=${searchSymbol.toUpperCase()}`);
      setSearchSymbol("");
    }
  };

  const handleKeyPress = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      handleSearch();
    }
  };

  return (
    <div className="flex h-screen w-full overflow-hidden bg-background">
      {/* Sidebar */}
      <aside
        className={cn(
          "flex flex-col border-r border-sidebar-border bg-sidebar transition-all duration-300",
          sidebarCollapsed ? "w-16" : "w-64"
        )}
      >
        {/* Logo */}
        <div className="flex h-16 items-center justify-between border-b border-sidebar-border px-4">
          {!sidebarCollapsed && (
            <div className="flex items-center gap-2">
              <Activity className="h-6 w-6 text-primary" />
              <span className="text-lg font-bold bg-gradient-to-r from-primary to-primary/60 bg-clip-text text-transparent">
                TradePilot
              </span>
            </div>
          )}
          <Button
            variant="ghost"
            size="icon"
            onClick={toggleSidebar}
            className="h-8 w-8 text-sidebar-foreground hover:bg-sidebar-accent"
          >
            <Menu className="h-4 w-4" />
          </Button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 space-y-1 p-2">
          {navigation.map((item) => {
            const isActive = location.pathname === item.href;
            return (
              <Link
                key={item.name}
                to={item.href}
                className={cn(
                  "flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                  isActive
                    ? "bg-sidebar-accent text-sidebar-primary"
                    : "text-sidebar-foreground hover:bg-sidebar-accent/50"
                )}
              >
                <item.icon className="h-4 w-4 shrink-0" />
                {!sidebarCollapsed && <span>{item.name}</span>}
              </Link>
            );
          })}
        </nav>
      </aside>

      {/* Main Content */}
      <div className="flex flex-1 flex-col overflow-hidden">
        {/* Header */}
        <header className="flex h-16 items-center justify-between border-b border-border bg-card px-6">
          <div className="flex items-center gap-4">
            <h1 className="text-xl font-semibold">
              {navigation.find((item) => item.href === location.pathname)?.name || "Dashboard"}
            </h1>
          </div>

          <div className="flex items-center gap-4">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                placeholder="Search symbol..."
                value={searchSymbol}
                onChange={(e) => setSearchSymbol(e.target.value)}
                onKeyPress={handleKeyPress}
                className="w-[200px] pl-9 uppercase"
              />
            </div>
            <DataModeBadge />
          </div>
        </header>

        {/* Page Content */}
        <main className="flex-1 overflow-auto p-6">{children}</main>
      </div>
    </div>
  );
}
