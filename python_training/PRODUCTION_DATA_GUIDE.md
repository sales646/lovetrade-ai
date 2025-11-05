# Production Data Guide - Real Market Data Only

## 🎯 Overview

Detta system använder **ENDAST RIKTIG MARKNADSDATA** från flera källor. Inga simuleringar, bara faktiska historiska utfall.

## 📊 Datakällor

### 1. Polygon S3 (via Massive) - Primary Stock Data
- **Källa**: US Stocks minutdata från Polygon.io
- **Access**: Via S3-kompatibel API hos Massive
- **Format**: Gzippade CSV-filer per dag
- **Path**: `us_stocks_sip/minute_aggs_v1/YYYY/MM/YYYY-MM-DD.csv.gz`
- **Data**: Open, High, Low, Close, Volume per minut
- **Coverage**: Alla US stocks, historiska data tillgänglig

**Konfiguration** (i `.env`):
```bash
POLYGON_S3_ACCESS_KEY=your_access_key
POLYGON_S3_SECRET_KEY=your_secret_key
POLYGON_S3_ENDPOINT=https://files.massive.com
POLYGON_S3_BUCKET=flatfiles
```

### 2. Binance - Crypto Data
- **Källa**: Binance Data Vision (officiell historisk data)
- **Format**: ZIP-arkiv per månad med CSV-data
- **URL**: `https://data.binance.vision/data/spot/monthly/klines/{SYMBOL}/1m/`
- **Data**: OHLCV per minut + antal transaktioner
- **Coverage**: Alla major crypto pairs, flera år tillbaka
- **Kostnad**: Gratis, ingen API-nyckel krävs

### 3. Yahoo Finance - Supplementary Stock Data
- **Källa**: Yahoo Finance API via yfinance library
- **Data**: OHLCV + supplementär data för stocks
- **Usage**: Backup för Polygon, fyll luckor
- **Limit**: Minutdata endast senaste ~60 dagarna

### 4. Yahoo Finance News - Sentiment Data
- **Källa**: Yahoo Finance News API
- **Data**: Nyhetsartiklar med timestamps
- **Sentiment**: Beräknas från positiva/negativa ord i rubriker
- **Alignment**: Matchas mot exakta tidsstämplar i OHLCV-data

## 🔧 Setup

### 1. Installera Dependencies
```bash
pip install -r requirements_production.txt
```

### 2. Konfigurera Miljövariabler
Skapa `.env` fil:
```bash
# Polygon S3 (Required för stocks)
POLYGON_S3_ACCESS_KEY=xxx
POLYGON_S3_SECRET_KEY=xxx
POLYGON_S3_ENDPOINT=https://files.massive.com
POLYGON_S3_BUCKET=flatfiles

# Supabase (Required för logging)
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_SERVICE_ROLE_KEY=xxx
```

### 3. Testa Datafetcher
```bash
python production_data_fetcher.py
```

## 🚀 Träning

### Quick Start
```bash
# Windows
START_PRODUCTION_TRAINING.bat

# Linux/Mac
chmod +x START_PRODUCTION_TRAINING.sh
./START_PRODUCTION_TRAINING.sh
```

### Manuellt
```bash
python production_train.py
```

## 📈 Dataflöde

```
1. FETCH DATA
   ├─ Polygon S3 → US Stocks (AAPL, MSFT, etc.)
   ├─ Binance → Crypto (BTCUSDT, ETHUSDT, etc.)
   ├─ Yahoo Finance → Supplementary stocks
   └─ Yahoo News → Sentiment per symbol

2. PREPROCESS
   ├─ Standardize timestamps
   ├─ Align news with price data
   ├─ Filter symbols med >= 1000 bars
   └─ Create environments per symbol

3. TRAIN
   ├─ Sample episodes from real data
   ├─ Agent interacts med ACTUAL historical prices
   ├─ Rewards baserade på REAL outcomes
   └─ Update policy using PPO

4. EVALUATE
   ├─ Test på UNSEEN real data
   ├─ Calculate metrics from actual trades
   └─ Log to Supabase
```

## 🎯 Key Features

### No Simulations
- All data kommer från faktiska marknader
- Prices är exakt vad som tradades
- Outcomes är historiskt verifierade
- Ingen synthetic data generation

### Timestamp Alignment
- News matchas mot exakt minutdata
- Sentiment påverkar decisions vid rätt tidpunkt
- Inget framtidsläckage (no look-ahead bias)

### Realistic Training
- Commission fees ingår (0.1% default)
- Slippage kan modelleras
- Market hours respekteras
- Weekends/holidays hanteras korrekt

### Multi-Asset
- Stocks: US equities från flera sektorer
- Crypto: Major pairs med 24/7 trading
- ETFs: SPY, QQQ för market sentiment
- News: Symbol-specific sentiment

## 📊 Data Quality

### Validation
Varje datakälla valideras:
- Minst 1000 bars per symbol
- Inga gaps > 5% av trading time
- Timestamps i kronologisk ordning
- OHLC relationships korrekt (H >= O,C >= L, etc.)

### Coverage
Aktuell coverage (exempel):
```
Polygon S3:    25+ US stocks, 1000+ bars each
Binance:       20+ crypto pairs, full historical
Yahoo Finance: Supplementary för 20+ stocks
Yahoo News:    100+ articles per symbol
```

## 🔍 Debugging

### Test Single Source
```python
from production_data_fetcher import ProductionDataFetcher

fetcher = ProductionDataFetcher("2024-01-01", "2024-01-31")

# Test Polygon only
stocks = fetcher.fetch_polygon_stocks()
print(f"Loaded {len(stocks)} stocks")

# Test Binance only
crypto = fetcher.fetch_binance_crypto()
print(f"Loaded {len(crypto)} crypto")

# Test Yahoo news
news = fetcher.fetch_yahoo_news()
print(f"Loaded {len(news)} articles")
```

### Check Data Quality
```python
# Check for missing data
for symbol, df in stocks.items():
    print(f"{symbol}: {len(df)} bars")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Gaps: {df['timestamp'].diff().dt.seconds.max() / 60} minutes")
```

## 📝 Configuration

Training config i `production_train.py`:
```python
config = {
    'data_days': 60,              # Fetch 60 days of data
    'min_bars_per_symbol': 1000,  # Require 1000+ bars
    'num_iterations': 100,         # 100 training iterations
    'episodes_per_iteration': 50,  # 50 episodes per iter
    'batch_size': 256,             # PPO batch size
}
```

## ⚠️ Viktiga Noteringar

1. **API Rate Limits**
   - Yahoo Finance: Max ~2000 requests/hour
   - Binance: Unlimited (static files)
   - Polygon S3: Beroende på Massive plan

2. **Storage**
   - Polygon data kan bli stort (100+ MB per dag)
   - Binance monthly files ~50-200 MB each
   - Cache lokalt för snabbare access

3. **Costs**
   - Polygon S3: Via Massive subscription
   - Binance: Free
   - Yahoo Finance: Free
   - Supabase: Free tier tillräckligt för logging

4. **Data Freshness**
   - Polygon: T+1 (data available next day)
   - Binance: T+0 (available same day för completed months)
   - Yahoo: Near real-time för news, delayed för prices

## 🎓 Next Steps

Efter lyckad träning:
1. Evaluera på out-of-sample data
2. Backtest med realistiska assumptions
3. Paper trading med live data
4. Gradvis scale upp till live trading

Använd alltid REAL DATA för träning → REAL RESULTS i production!
