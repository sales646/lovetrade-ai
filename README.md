# TradePilot - AI-Powered Algorithmic Trading Platform

![TradePilot](https://img.shields.io/badge/TradePilot-Trading%20Platform-00a0dc?style=for-the-badge)

An advanced algorithmic trading platform with AI-powered strategies, real-time market monitoring, and automated execution capabilities.

## 🚀 Features

### Current (v0.2)
- ✅ **Modern Dashboard UI** - Dark theme optimized for trading
- ✅ **Responsive Layout** - Collapsible sidebar navigation
- ✅ **Watchlist Management** - Track multiple symbols with live charts
- ✅ **Type-Safe Architecture** - Full TypeScript with Zod validation
- ✅ **State Management** - Zustand with persistence
- ✅ **Routing** - Multi-page React Router setup
- ✅ **Data Integrity** - No fabricated numbers, validated data sources
- ✅ **Lovable Cloud Backend** - PostgreSQL database + Edge Functions
- ✅ **Yahoo Finance Integration** - Free historical market data API
- ✅ **Historical Data Storage** - OHLCV bars stored in database
- ✅ **Chart Visualization** - Price & volume charts with Recharts
- ✅ **Training Data Pipeline** - Download & prepare ML datasets

### Coming Soon
- 🔄 Real-time WebSocket data streaming
- 🔄 Order ticket & position management
- 🔄 Strategy configuration (News Momentum, VWAP Play, Exhaustion Reversal)
- 🔄 RL/ML model training interface
- 🔄 System logs viewer
- 🔄 Settings & API key management
- 🔄 Advanced technical indicators

## 🛠️ Tech Stack

- **Framework**: React 18 + TypeScript + Vite
- **UI**: Tailwind CSS + shadcn/ui
- **State**: Zustand (with persistence)
- **Data Fetching**: React Query
- **Charts**: Recharts
- **Validation**: Zod
- **Icons**: Lucide React
- **Date Utils**: date-fns
- **Backend**: Lovable Cloud (Supabase)
- **Database**: PostgreSQL
- **Edge Functions**: Deno runtime
- **Data Source**: Yahoo Finance API (free, no key required)

## 📁 Project Structure

```
src/
├── components/
│   ├── ui/              # shadcn UI components
│   └── DashboardLayout.tsx
├── pages/
│   ├── Dashboard.tsx    # Main dashboard
│   ├── Watchlist.tsx    # Symbol watchlist
│   ├── Orders.tsx       # Orders & positions
│   ├── Strategies.tsx   # Strategy config
│   ├── Training.tsx     # Model training
│   ├── Logs.tsx         # System logs
│   └── Settings.tsx     # App settings
├── store/
│   ├── uiStore.ts       # UI state
│   ├── watchlistStore.ts
│   └── connectionStore.ts
├── lib/
│   └── types.ts         # Type definitions & schemas
└── App.tsx
```

## 🎨 Design System

**Color Palette:**
- Primary: Cyan (#00a0dc) - Interactive elements
- Success: Emerald - Profitable trades
- Destructive: Red - Losses
- Background: Deep charcoal (#0d1117)

**Design Principles:**
- High contrast for data visibility
- Smooth animations for real-time updates
- Trading conventions (green = up, red = down)
- Professional, data-dense interface

## 🚦 Getting Started

### Prerequisites
- Node.js 18+ & npm

### Installation

```bash
# Clone the repository
git clone <your-repo-url>

# Navigate to directory
cd tradepilot

# Install dependencies
npm install

# Start development server
npm run dev
```

The app will be available at `http://localhost:8080`

## 📖 Usage

1. **Dashboard** - View portfolio overview and recent signals
2. **Watchlist** - Add symbols, view live prices and historical charts
3. **Training** - Download historical data from Yahoo Finance for ML training
4. **Orders** - (Coming soon) Place orders and manage positions
5. **Strategies** - (Coming soon) Configure trading strategies
6. **Logs** - (Coming soon) View system events
7. **Settings** - (Coming soon) Manage preferences

## 🔐 Data & Privacy

- **Historical Data**: Fetched from Yahoo Finance (free public API)
- **Database Storage**: Lovable Cloud (PostgreSQL) for training datasets
- **Mock Mode**: Available for testing without real data
- **Watchlist**: Persisted in browser localStorage
- **RLS Policies**: Database protected with Row Level Security

## 📝 Roadmap

### Phase 1 (✅ Completed)
- [x] Core UI/UX foundation
- [x] Routing & navigation
- [x] State management setup
- [x] Data integrity guardrails
- [x] Lovable Cloud backend
- [x] Yahoo Finance integration
- [x] Historical data storage
- [x] Chart visualization

### Phase 2 (Next)
- [ ] WebSocket integration
- [ ] Live market data
- [ ] Order placement UI
- [ ] Strategy configuration

### Phase 3
- [ ] RL/ML training interface
- [ ] Broker integrations (Alpaca, Polygon)
- [ ] Advanced charting
- [ ] Performance analytics

### Phase 4
- [ ] Multi-ticker concurrency
- [ ] Backtesting engine
- [ ] Alert system
- [ ] Mobile optimization

## 🤝 Contributing

This is a work in progress. Contributions welcome!

## 📄 License

MIT License - See LICENSE file for details

## ⚠️ Disclaimer

This is a demo/educational platform. Not financial advice. Trading carries risk. Always do your own research and consult with financial professionals.

---

**Built with** [Lovable](https://lovable.dev) 💜
