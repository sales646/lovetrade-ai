#!/usr/bin/env python3
"""
Unified Multi-Source Data Fetcher
Combines data from: Polygon S3, Binance, Yahoo Finance
"""

import os
import boto3
import gzip
import io
import pandas as pd
import numpy as np
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from datetime import datetime, timedelta
import random
import requests
import zipfile
import time

load_dotenv()


class UnifiedDataFetcher:
    """Fetches data from all sources: Polygon S3, Binance, Yahoo Finance"""
    
    def __init__(self):
        # Polygon S3
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv("POLYGON_S3_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("POLYGON_S3_SECRET_KEY"),
            endpoint_url=os.getenv("POLYGON_S3_ENDPOINT")
        )
        self.s3_bucket = os.getenv("POLYGON_S3_BUCKET", "flatfiles")
        
        # Binance pairs
        self.binance_pairs = [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", 
            "DOTUSDT", "AVAXUSDT", "UNIUSDT", "AAVEUSDT", "LINKUSDT",
            "MATICUSDT", "ARBUSDT", "DOGEUSDT", "SHIBUSDT", "XRPUSDT",
            "LTCUSDT", "BCHUSDT", "ETCUSDT", "ATOMUSDT", "NEARUSDT"
        ]
        
        # Yahoo Finance stocks
        self.yahoo_stocks = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
            "JPM", "V", "WMT", "PG", "UNH", "MA", "HD", "DIS",
            "BAC", "CSCO", "NFLX", "ADBE", "CRM", "ORCL", "INTC"
        ]
    
    def fetch_polygon_s3(self, start_date: str, end_date: str, asset_type: str = "stocks") -> Dict[str, pd.DataFrame]:
        """Fetch from Polygon S3"""
        print(f"\n📊 Fetching {asset_type.upper()} from Polygon S3...")
        
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        dates = []
        current = start
        while current <= end:
            if current.weekday() < 5:  # Weekdays only
                dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
        
        prefix = "us_stocks_sip" if asset_type == "stocks" else "global_crypto"
        file_keys = []
        for date in dates:
            year, month, day = date.split("-")
            file_key = f"{prefix}/minute_aggs_v1/{year}/{month}/{date}.csv.gz"
            file_keys.append((date, file_key))
        
        all_dfs = []
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = {executor.submit(self._download_s3_file, fk): date 
                      for date, fk in file_keys}
            
            for future in as_completed(futures):
                date = futures[future]
                df = future.result()
                if not df.empty:
                    all_dfs.append(df)
        
        if not all_dfs:
            print(f"  ⚠️ No {asset_type} data from Polygon S3")
            return {}
        
        combined = pd.concat(all_dfs, ignore_index=True)
        combined = combined.sort_values(['ticker', 'timestamp'])
        
        symbol_data = {}
        for symbol in combined['ticker'].unique():
            symbol_df = combined[combined['ticker'] == symbol].copy()
            symbol_df = symbol_df.sort_values('timestamp').reset_index(drop=True)
            symbol_data[symbol] = symbol_df
        
        print(f"  ✅ Loaded {len(symbol_data)} symbols from Polygon S3")
        return symbol_data
    
    def _download_s3_file(self, file_key: str) -> pd.DataFrame:
        """Download single S3 file"""
        try:
            response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=file_key)
            compressed = response['Body'].read()
            decompressed = gzip.decompress(compressed)
            df = pd.read_csv(io.BytesIO(decompressed))
            df['timestamp'] = pd.to_datetime(df['window_start'], unit='ns')
            return df
        except:
            return pd.DataFrame()
    
    def fetch_binance(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Fetch from Binance"""
        print(f"\n🪙 Fetching CRYPTO from Binance...")
        
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        symbol_data = {}
        for symbol in self.binance_pairs:
            all_bars = []
            current = start
            while current <= end:
                year = current.year
                month = current.month
                
                klines = self._download_binance_month(symbol, year, month)
                if klines:
                    parsed = [self._parse_binance_kline(k) for k in klines]
                    parsed = [p for p in parsed if p is not None]
                    all_bars.extend(parsed)
                
                # Next month
                if month == 12:
                    current = datetime(year + 1, 1, 1)
                else:
                    current = datetime(year, month + 1, 1)
            
            if all_bars:
                df = pd.DataFrame(all_bars)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp').reset_index(drop=True)
                symbol_data[symbol] = df
        
        print(f"  ✅ Loaded {len(symbol_data)} crypto pairs from Binance")
        return symbol_data
    
    def _download_binance_month(self, symbol: str, year: int, month: int) -> List:
        """Download Binance monthly data"""
        url = f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/1m/{symbol}-1m-{year}-{month:02d}.zip"
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                csv_filename = z.namelist()[0]
                with z.open(csv_filename) as csv_file:
                    csv_data = csv_file.read().decode('utf-8')
                    import csv
                    reader = csv.reader(io.StringIO(csv_data))
                    return list(reader)
        except:
            return []
    
    def _parse_binance_kline(self, kline_row):
        """Parse Binance kline"""
        try:
            timestamp_ms = int(kline_row[0])
            timestamp = datetime.fromtimestamp(timestamp_ms / 1000)
            return {
                "timestamp": timestamp,
                "open": float(kline_row[1]),
                "high": float(kline_row[2]),
                "low": float(kline_row[3]),
                "close": float(kline_row[4]),
                "volume": float(kline_row[5]),
                "transactions": 1
            }
        except:
            return None
    
    def fetch_yahoo(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Fetch from Yahoo Finance"""
        print(f"\n📈 Fetching STOCKS from Yahoo Finance...")
        
        try:
            import yfinance as yf
        except ImportError:
            print("  ⚠️ yfinance not installed. Run: pip install yfinance")
            return {}
        
        symbol_data = {}
        for symbol in self.yahoo_stocks:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date, interval="1m")
                
                if not df.empty:
                    df = df.reset_index()
                    df = df.rename(columns={
                        'Datetime': 'timestamp',
                        'Open': 'open',
                        'High': 'high',
                        'Low': 'low',
                        'Close': 'close',
                        'Volume': 'volume'
                    })
                    df['transactions'] = 1
                    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'transactions']]
                    symbol_data[symbol] = df
                    
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                print(f"  ⚠️ Error fetching {symbol}: {e}")
        
        print(f"  ✅ Loaded {len(symbol_data)} stocks from Yahoo Finance")
        return symbol_data
    
    def fetch_all(self, start_date: str = "2023-01-01", end_date: str = "2024-01-31") -> Dict[str, pd.DataFrame]:
        """Fetch from all sources and combine"""
        print("\n" + "="*70)
        print("🌍 UNIFIED MULTI-SOURCE DATA FETCHER")
        print("="*70)
        print(f"📅 Date range: {start_date} to {end_date}")
        
        all_data = {}
        
        # Polygon S3 - Stocks
        polygon_stocks = self.fetch_polygon_s3(start_date, end_date, "stocks")
        all_data.update(polygon_stocks)
        
        # Polygon S3 - Crypto
        polygon_crypto = self.fetch_polygon_s3(start_date, end_date, "crypto")
        all_data.update(polygon_crypto)
        
        # Binance
        binance_data = self.fetch_binance(start_date, end_date)
        all_data.update(binance_data)
        
        # Yahoo Finance
        yahoo_data = self.fetch_yahoo(start_date, end_date)
        all_data.update(yahoo_data)
        
        print("\n" + "="*70)
        print(f"✅ TOTAL: {len(all_data)} symbols loaded")
        print(f"   Polygon Stocks: {len(polygon_stocks)}")
        print(f"   Polygon Crypto: {len(polygon_crypto)}")
        print(f"   Binance: {len(binance_data)}")
        print(f"   Yahoo Finance: {len(yahoo_data)}")
        print("="*70)
        
        return all_data
    
    def get_symbols_with_data(self, symbol_data: Dict[str, pd.DataFrame], min_bars: int = 2000) -> List[str]:
        """Get symbols with sufficient data"""
        valid = [s for s, df in symbol_data.items() if len(df) >= min_bars]
        print(f"\n✅ {len(valid)}/{len(symbol_data)} symbols have >= {min_bars} bars")
        return valid


if __name__ == "__main__":
    fetcher = UnifiedDataFetcher()
    data = fetcher.fetch_all("2023-01-01", "2024-01-31")
    symbols = fetcher.get_symbols_with_data(data, min_bars=1000)
    print(f"\n🎯 Ready for training with {len(symbols)} symbols")
