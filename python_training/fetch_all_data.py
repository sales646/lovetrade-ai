#!/usr/bin/env python3
"""Comprehensive data fetcher: Polygon S3 + Binance + Yahoo Finance"""

import os
import sys
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path
import boto3
from io import BytesIO
from supabase import create_client

load_dotenv()

class MultiSourceDataFetcher:
    """Fetches data from Polygon S3, Binance, and Yahoo Finance"""
    
    def __init__(self):
        # Supabase
        self.supabase = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        )
        
        # Polygon S3
        self.s3_access_key = os.getenv("POLYGON_S3_ACCESS_KEY")
        self.s3_secret_key = os.getenv("POLYGON_S3_SECRET_KEY")
        self.s3_endpoint = os.getenv("POLYGON_S3_ENDPOINT", "https://files.massive.com")
        self.s3_bucket = os.getenv("POLYGON_S3_BUCKET", "flatfiles")
        
        if self.s3_access_key and self.s3_secret_key:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=self.s3_access_key,
                aws_secret_access_key=self.s3_secret_key,
                endpoint_url=self.s3_endpoint
            )
        else:
            self.s3_client = None
            print("⚠️  Polygon S3 credentials not found")
    
    def fetch_binance_data(self, symbol: str, days: int = 30):
        """Fetch crypto data from Binance"""
        print(f"📊 Fetching {symbol} from Binance...")
        
        # Convert symbol format (BTCUSD -> BTCUSDT)
        binance_symbol = symbol.replace("USD", "USDT")
        
        url = "https://api.binance.com/api/v3/klines"
        
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        
        params = {
            'symbol': binance_symbol,
            'interval': '1m',
            'startTime': start_time,
            'endTime': end_time,
            'limit': 1000
        }
        
        all_bars = []
        
        try:
            # Fetch in batches (Binance has 1000 limit per request)
            current_start = start_time
            
            while current_start < end_time:
                params['startTime'] = current_start
                response = requests.get(url, params=params, timeout=30)
                
                if response.status_code != 200:
                    print(f"   ❌ Error {response.status_code}")
                    break
                
                data = response.json()
                if not data:
                    break
                
                for bar in data:
                    all_bars.append({
                        'symbol': symbol,
                        'timestamp': datetime.fromtimestamp(bar[0] / 1000).isoformat(),
                        'timeframe': '1Min',
                        'open': float(bar[1]),
                        'high': float(bar[2]),
                        'low': float(bar[3]),
                        'close': float(bar[4]),
                        'volume': int(float(bar[5]))
                    })
                
                # Move to next batch
                current_start = data[-1][0] + 60000  # Add 1 minute
                
                if len(data) < 1000:
                    break
            
            print(f"   ✅ Fetched {len(all_bars)} bars")
            return all_bars
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return []
    
    def fetch_yahoo_data(self, symbol: str, days: int = 30):
        """Fetch stock data from Yahoo Finance"""
        print(f"📊 Fetching {symbol} from Yahoo Finance...")
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Fetch 1-minute data (Yahoo only provides 7 days of 1min data)
            df = ticker.history(period=f"{min(days, 7)}d", interval="1m")
            
            if df.empty:
                print(f"   ❌ No data found")
                return []
            
            bars = []
            for idx, row in df.iterrows():
                bars.append({
                    'symbol': symbol,
                    'timestamp': idx.isoformat(),
                    'timeframe': '1Min',
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': int(row['Volume'])
                })
            
            print(f"   ✅ Fetched {len(bars)} bars")
            return bars
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return []
    
    def fetch_polygon_s3_data(self, symbol: str, market: str = "stocks"):
        """Fetch data from Polygon S3 flat files"""
        if not self.s3_client:
            return []
        
        print(f"📊 Fetching {symbol} from Polygon S3...")
        
        try:
            # Construct S3 key path
            year = datetime.now().year
            month = datetime.now().month
            
            if market == "crypto":
                prefix = f"us_stocks_sip/minute_aggs_v1/{symbol}/{year}/{month:02d}/"
            else:
                prefix = f"us_stocks_sip/minute_aggs_v1/{symbol}/{year}/{month:02d}/"
            
            # List available files
            response = self.s3_client.list_objects_v2(
                Bucket=self.s3_bucket,
                Prefix=prefix
            )
            
            if 'Contents' not in response:
                print(f"   ❌ No S3 data found")
                return []
            
            all_bars = []
            
            # Download and parse parquet files
            for obj in response['Contents'][:10]:  # Limit to 10 files
                key = obj['Key']
                
                if not key.endswith('.parquet'):
                    continue
                
                file_obj = BytesIO()
                self.s3_client.download_fileobj(self.s3_bucket, key, file_obj)
                file_obj.seek(0)
                
                df = pd.read_parquet(file_obj)
                
                for _, row in df.iterrows():
                    all_bars.append({
                        'symbol': symbol,
                        'timestamp': pd.to_datetime(row['timestamp']).isoformat(),
                        'timeframe': '1Min',
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': int(row['volume'])
                    })
            
            print(f"   ✅ Fetched {len(all_bars)} bars from S3")
            return all_bars
            
        except Exception as e:
            print(f"   ❌ S3 Error: {e}")
            return []
    
    def insert_to_supabase(self, bars: list):
        """Batch insert bars into Supabase"""
        if not bars:
            return 0
        
        try:
            # Insert in batches of 500
            batch_size = 500
            total_inserted = 0
            
            for i in range(0, len(bars), batch_size):
                batch = bars[i:i + batch_size]
                self.supabase.table('historical_bars').insert(batch).execute()
                total_inserted += len(batch)
            
            return total_inserted
            
        except Exception as e:
            print(f"   ⚠️  Insert error: {e}")
            return 0
    
    def fetch_symbol(self, symbol: str, is_crypto: bool = False):
        """Fetch symbol from appropriate source"""
        all_bars = []
        
        if is_crypto:
            # Try Binance first for crypto
            bars = self.fetch_binance_data(symbol, days=30)
            all_bars.extend(bars)
            
            # Fallback to Polygon S3 if available
            if not bars and self.s3_client:
                bars = self.fetch_polygon_s3_data(symbol, market="crypto")
                all_bars.extend(bars)
        else:
            # Try Polygon S3 first for stocks
            if self.s3_client:
                bars = self.fetch_polygon_s3_data(symbol, market="stocks")
                all_bars.extend(bars)
            
            # Fallback to Yahoo Finance
            if not bars:
                bars = self.fetch_yahoo_data(symbol, days=7)
                all_bars.extend(bars)
        
        # Insert to Supabase
        if all_bars:
            inserted = self.insert_to_supabase(all_bars)
            print(f"   ✅ Inserted {inserted} bars to Supabase\n")
        
        return len(all_bars)

def main():
    print("=" * 70)
    print("🧠 PNU - Multi-Source Data Fetcher")
    print("   Sources: Polygon S3 + Binance + Yahoo Finance")
    print("=" * 70)
    
    fetcher = MultiSourceDataFetcher()
    
    # Stock symbols (use Polygon S3 or Yahoo)
    stocks = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"]
    
    # Crypto symbols (use Binance)
    crypto = ["BTCUSD", "ETHUSD"]
    
    print(f"\n📈 Fetching {len(stocks)} stocks...\n")
    stock_count = 0
    for symbol in stocks:
        count = fetcher.fetch_symbol(symbol, is_crypto=False)
        stock_count += count
    
    print(f"\n🪙 Fetching {len(crypto)} crypto...\n")
    crypto_count = 0
    for symbol in crypto:
        count = fetcher.fetch_symbol(symbol, is_crypto=True)
        crypto_count += count
    
    print("\n" + "=" * 70)
    print(f"✅ Complete!")
    print(f"   Stocks: {stock_count} bars")
    print(f"   Crypto: {crypto_count} bars")
    print(f"   Total: {stock_count + crypto_count} bars")
    print("=" * 70)
    print("\nNow run: python quick_train.py")

if __name__ == "__main__":
    main()
