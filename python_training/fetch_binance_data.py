#!/usr/bin/env python3
"""
Fetch historical crypto data from Binance and populate Supabase
Features:
- 1-minute timeframe (best for RL training)
- Diverse crypto markets (majors, DeFi, altcoins, Layer-1s)
- 3 years of historical data
- Adds to existing stock data (doesn't replace)
"""

import os
import requests
import zipfile
import io
import csv
from datetime import datetime, timedelta
from supabase import create_client
from dotenv import load_dotenv
import time

load_dotenv()

# Diverse crypto pairs for maximum market variety
CRYPTO_PAIRS = [
    # Major cryptocurrencies
    "BTCUSDT",   # Bitcoin - crypto king
    "ETHUSDT",   # Ethereum - smart contracts
    
    # Major altcoins
    "BNBUSDT",   # Binance Coin - exchange token
    "SOLUSDT",   # Solana - high-speed Layer-1
    "ADAUSDT",   # Cardano - academic blockchain
    "DOTUSDT",   # Polkadot - interoperability
    "AVAXUSDT",  # Avalanche - fast consensus
    
    # DeFi tokens
    "UNIUSDT",   # Uniswap - DEX leader
    "AAVEUSDT",  # Aave - lending protocol
    "LINKUSDT",  # Chainlink - oracle network
    
    # Layer-2 & scaling
    "MATICUSDT", # Polygon - Ethereum scaling
    "ARBUSDT",   # Arbitrum - Layer-2
    
    # Meme & community
    "DOGEUSDT",  # Dogecoin - meme king
    "SHIBUSDT",  # Shiba Inu - meme token
    
    # Stablecoins trading (volatility plays)
    "BTCBUSD",   # BTC vs stablecoin
    "ETHBUSD",   # ETH vs stablecoin
]

TIMEFRAME = "1m"
YEARS_BACK = 3


def get_binance_klines_api(symbol: str, start_date: datetime, end_date: datetime):
    """
    Fetch klines from Binance REST API (for recent data or small ranges)
    Note: Limited to 1000 candles per request
    """
    base_url = "https://api.binance.com/api/v3/klines"
    
    all_klines = []
    current_start = int(start_date.timestamp() * 1000)
    end_ts = int(end_date.timestamp() * 1000)
    
    while current_start < end_ts:
        params = {
            "symbol": symbol,
            "interval": TIMEFRAME,
            "startTime": current_start,
            "limit": 1000
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            klines = response.json()
            
            if not klines:
                break
            
            all_klines.extend(klines)
            current_start = klines[-1][0] + 60000  # +1 minute
            
            # Rate limiting
            time.sleep(0.1)
            
        except Exception as e:
            print(f"❌ Error fetching {symbol}: {e}")
            break
    
    return all_klines


def download_binance_monthly_data(symbol: str, year: int, month: int):
    """
    Download monthly ZIP archive from Binance data vision
    Format: https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-01.zip
    """
    url = f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/{TIMEFRAME}/{symbol}-{TIMEFRAME}-{year}-{month:02d}.zip"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Extract CSV from ZIP
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            csv_filename = z.namelist()[0]
            with z.open(csv_filename) as csv_file:
                csv_data = csv_file.read().decode('utf-8')
                reader = csv.reader(io.StringIO(csv_data))
                return list(reader)
                
    except Exception as e:
        print(f"⚠️  Could not download {symbol} {year}-{month:02d}: {e}")
        return []


def parse_binance_kline(kline_row):
    """
    Parse Binance kline format to our database format
    Binance format: [timestamp, open, high, low, close, volume, close_time, ...]
    """
    try:
        timestamp_ms = int(kline_row[0])
        timestamp = datetime.fromtimestamp(timestamp_ms / 1000)
        
        return {
            "timestamp": timestamp.isoformat(),
            "open": float(kline_row[1]),
            "high": float(kline_row[2]),
            "low": float(kline_row[3]),
            "close": float(kline_row[4]),
            "volume": int(float(kline_row[5]))
        }
    except Exception as e:
        print(f"⚠️  Parse error: {e}")
        return None


def calculate_indicators(bars, symbol):
    """Calculate technical indicators for crypto bars"""
    if len(bars) < 50:
        return []
    
    indicators = []
    
    for i in range(50, len(bars)):
        window = bars[i-50:i+1]
        closes = [b['close'] for b in window]
        highs = [b['high'] for b in window]
        lows = [b['low'] for b in window]
        volumes = [b['volume'] for b in window]
        
        # RSI-14
        gains = []
        losses = []
        for j in range(1, len(closes)):
            change = closes[j] - closes[j-1]
            gains.append(max(0, change))
            losses.append(max(0, -change))
        
        avg_gain = sum(gains[-14:]) / 14 if len(gains) >= 14 else 0
        avg_loss = sum(losses[-14:]) / 14 if len(losses) >= 14 else 0
        rs = avg_gain / avg_loss if avg_loss > 0 else 100
        rsi = 100 - (100 / (1 + rs))
        
        # ATR-14
        true_ranges = []
        for j in range(1, len(window)):
            hl = highs[j] - lows[j]
            hc = abs(highs[j] - closes[j-1])
            lc = abs(lows[j] - closes[j-1])
            true_ranges.append(max(hl, hc, lc))
        atr = sum(true_ranges[-14:]) / 14 if len(true_ranges) >= 14 else 0
        
        # EMAs
        ema_20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else closes[-1]
        ema_50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else closes[-1]
        
        # VWAP (simplified - daily VWAP)
        pv_sum = sum(closes[j] * volumes[j] for j in range(len(closes)))
        v_sum = sum(volumes)
        vwap = pv_sum / v_sum if v_sum > 0 else closes[-1]
        
        indicators.append({
            "symbol": symbol,
            "timeframe": TIMEFRAME,
            "timestamp": bars[i]["timestamp"],
            "rsi_14": rsi,
            "atr_14": atr,
            "ema_20": ema_20,
            "ema_50": ema_50,
            "vwap": vwap,
            "vwap_distance_pct": ((closes[-1] - vwap) / vwap * 100) if vwap > 0 else 0,
            "volume_zscore": (volumes[-1] - sum(volumes[-20:]) / 20) / (max(volumes[-20:]) - min(volumes[-20:])) if len(volumes) >= 20 else 0
        })
    
    return indicators


def fetch_and_populate():
    """Main function to fetch Binance data and populate Supabase"""
    
    print("=" * 70)
    print("🚀 BINANCE CRYPTO DATA FETCHER")
    print("=" * 70)
    print(f"\n📊 Configuration:")
    print(f"   Timeframe: {TIMEFRAME}")
    print(f"   Years back: {YEARS_BACK}")
    print(f"   Crypto pairs: {len(CRYPTO_PAIRS)}")
    print(f"   Mode: ADD to existing stock data\n")
    
    # Connect to Supabase
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    supabase = create_client(supabase_url, supabase_key)
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=YEARS_BACK * 365)
    
    print(f"📅 Date range: {start_date.date()} to {end_date.date()}\n")
    
    total_bars = 0
    total_indicators = 0
    
    for symbol in CRYPTO_PAIRS:
        print(f"\n{'='*70}")
        print(f"💰 Processing {symbol}")
        print(f"{'='*70}")
        
        all_bars = []
        
        # Download monthly archives
        current = start_date
        while current <= end_date:
            year = current.year
            month = current.month
            
            print(f"   📦 Downloading {year}-{month:02d}...", end=" ")
            
            klines = download_binance_monthly_data(symbol, year, month)
            
            if klines:
                parsed = [parse_binance_kline(k) for k in klines]
                parsed = [p for p in parsed if p is not None]
                all_bars.extend(parsed)
                print(f"✅ {len(parsed):,} bars")
            else:
                print("⚠️  No data")
            
            # Next month
            if month == 12:
                current = datetime(year + 1, 1, 1)
            else:
                current = datetime(year, month + 1, 1)
        
        if not all_bars:
            print(f"❌ No data for {symbol}, skipping")
            continue
        
        print(f"\n   ✅ Total bars collected: {len(all_bars):,}")
        
        # Add symbol to bars
        for bar in all_bars:
            bar["symbol"] = symbol
            bar["timeframe"] = TIMEFRAME
        
        # Calculate indicators
        print(f"   📈 Calculating indicators...")
        indicators = calculate_indicators(all_bars, symbol)
        print(f"   ✅ {len(indicators):,} indicators calculated")
        
        # Insert into Supabase (batch insert)
        print(f"   💾 Inserting into Supabase...")
        
        batch_size = 1000
        
        # Insert bars
        for i in range(0, len(all_bars), batch_size):
            batch = all_bars[i:i+batch_size]
            try:
                supabase.table("historical_bars").insert(batch).execute()
                print(f"      Bars: {i+len(batch):,}/{len(all_bars):,}", end="\r")
            except Exception as e:
                print(f"\n      ⚠️  Error inserting bars batch {i}: {e}")
        
        print(f"\n   ✅ Bars inserted: {len(all_bars):,}")
        
        # Insert indicators
        for i in range(0, len(indicators), batch_size):
            batch = indicators[i:i+batch_size]
            try:
                supabase.table("technical_indicators").insert(batch).execute()
                print(f"      Indicators: {i+len(batch):,}/{len(indicators):,}", end="\r")
            except Exception as e:
                print(f"\n      ⚠️  Error inserting indicators batch {i}: {e}")
        
        print(f"\n   ✅ Indicators inserted: {len(indicators):,}")
        
        total_bars += len(all_bars)
        total_indicators += len(indicators)
        
        # Add symbol to symbols table
        try:
            supabase.table("symbols").upsert({
                "symbol": symbol,
                "exchange": "Binance",
                "name": symbol.replace("USDT", "").replace("BUSD", ""),
                "sector": "Cryptocurrency",
                "is_active": True
            }).execute()
        except Exception as e:
            print(f"   ⚠️  Could not update symbols table: {e}")
    
    print(f"\n{'='*70}")
    print("🎉 COMPLETED!")
    print(f"{'='*70}")
    print(f"✅ Total bars inserted: {total_bars:,}")
    print(f"✅ Total indicators inserted: {total_indicators:,}")
    print(f"✅ Crypto pairs added: {len(CRYPTO_PAIRS)}")
    print(f"\n💡 Your training environment now has:")
    print(f"   - Stock data (AAPL, MSFT, etc.) ✅")
    print(f"   - Crypto data ({len(CRYPTO_PAIRS)} pairs) ✅")
    print(f"   - 1-minute timeframe for optimal RL training ✅")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    fetch_and_populate()
