# TradePilot - AI-Powered Algorithmic Trading Platform

![TradePilot](https://img.shields.io/badge/TradePilot-Trading%20Platform-00a0dc?style=for-the-badge)

An advanced algorithmic trading platform with AI-powered strategies, real-time market monitoring, and automated execution capabilities.

## 🚀 Features

### Current (v0.1)
- ✅ **Modern Dashboard UI** - Dark theme optimized for trading
- ✅ **Responsive Layout** - Collapsible sidebar navigation
- ✅ **Watchlist Management** - Track multiple symbols
- ✅ **Type-Safe Architecture** - Full TypeScript with Zod validation
- ✅ **State Management** - Zustand with persistence
- ✅ **Routing** - Multi-page React Router setup

### Coming Soon
- 🔄 Real-time market data (WebSocket)
- 🔄 Order ticket & position management
- 🔄 Strategy configuration (News Momentum, VWAP Play, Exhaustion Reversal)
- 🔄 Model training interface
- 🔄 System logs viewer
- 🔄 Settings & API key management
- 🔄 Live charts with Recharts
- 🔄 Broker integration (Alpaca/Polygon)

## 🛠️ Tech Stack

- **Framework**: React 18 + TypeScript + Vite
- **UI**: Tailwind CSS + shadcn/ui
- **State**: Zustand (with persistence)
- **Data Fetching**: React Query
- **Charts**: Recharts
- **Validation**: Zod
- **Icons**: Lucide React
- **Date Utils**: date-fns

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
2. **Watchlist** - Add symbols to monitor (e.g., AAPL, TSLA)
3. **Orders** - (Coming soon) Place orders and manage positions
4. **Strategies** - (Coming soon) Configure trading strategies
5. **Training** - (Coming soon) Train and evaluate models
6. **Logs** - (Coming soon) View system events
7. **Settings** - (Coming soon) Manage API keys and preferences

## 🔐 Data & Privacy

- All data currently uses mock/demo values
- No real trading connections in this version
- Watchlist persisted in browser localStorage
- No external API calls yet

## 📝 Roadmap

### Phase 1 (Current)
- [x] Core UI/UX foundation
- [x] Routing & navigation
- [x] State management setup

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
