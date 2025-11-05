#!/usr/bin/env python3
"""
Production Data Fetcher - Real market data only, no simulations
Sources:
1. Polygon S3 (via Massive) - US Stocks minute data
2. Binance - Crypto minute data
3. Yahoo Finance - Stocks minute data (backup/supplementary)
4. Yahoo Finance News - Sentiment data aligned with timestamps
"""

import os
import boto3
import gzip
import io
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from datetime import datetime, timedelta
import requests
import zipfile
import time
import csv
from tqdm import tqdm

load_dotenv()


class ProductionDataFetcher:
    """Fetches real market data from multiple sources"""
    
    def __init__(self, start_date: str, end_date: str):
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Polygon S3 Setup
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv("POLYGON_S3_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("POLYGON_S3_SECRET_KEY"),
            endpoint_url=os.getenv("POLYGON_S3_ENDPOINT", "https://files.massive.com")
        )
        self.s3_bucket = os.getenv("POLYGON_S3_BUCKET", "flatfiles")
        
        # Stock symbols for focused training
        self.stocks = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD",
            "SPY", "QQQ", "IWM", "DIA",  # ETFs for market sentiment
            "JPM", "BAC", "GS", "WFC",  # Financials
            "XOM", "CVX", "COP",  # Energy
            "UNH", "JNJ", "PFE",  # Healthcare
        ]
        
        # Crypto pairs for diversification
        self.crypto = [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
            "ADAUSDT", "DOGEUSDT", "MATICUSDT", "DOTUSDT", "AVAXUSDT",
        ]
        
        print(f"📅 Configured for: {start_date} to {end_date}")
        print(f"📊 Stocks: {len(self.stocks)}")
        print(f"🪙 Crypto: {len(self.crypto)}")
    
    # ===== POLYGON S3 (MASSIVE) =====
    
    def fetch_polygon_stocks(self) -> Dict[str, pd.DataFrame]:
        """Fetch US stocks from Polygon S3 via Massive"""
        print("\n" + "="*70)
        print("📊 FETCHING STOCKS FROM POLYGON S3 (Massive)")
        print("="*70)
        
        # Generate date list (weekdays only for stocks)
        dates = []
        current = self.start_date
        while current <= self.end_date:
            if current.weekday() < 5:  # Monday-Friday
                dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
        
        print(f"🗓️  Trading days: {len(dates)}")
        
        # Build S3 file keys
        # Format: us_stocks_sip/minute_aggs_v1/2024/01/2024-01-15.csv.gz
        file_keys = []
        for date in dates:
            year, month, day = date.split("-")
            file_key = f"us_stocks_sip/minute_aggs_v1/{year}/{month}/{date}.csv.gz"
            file_keys.append((date, file_key))
        
        # Download all files in parallel
        all_dfs = []
        print(f"⬇️  Downloading {len(file_keys)} daily files...")
        
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = {executor.submit(self._download_polygon_file, fk): date 
                      for date, fk in file_keys}
            
            with tqdm(total=len(futures), desc="Polygon S3") as pbar:
                for future in as_completed(futures):
                    date = futures[future]
                    df = future.result()
                    if not df.empty:
                        # Filter for our stock symbols only
                        df = df[df['ticker'].isin(self.stocks)]
                        if not df.empty:
                            all_dfs.append(df)
                    pbar.update(1)
        
        if not all_dfs:
            print("❌ No Polygon stock data retrieved")
            return {}
        
        # Combine and organize by symbol
        combined = pd.concat(all_dfs, ignore_index=True)
        combined = combined.sort_values(['ticker', 'timestamp'])
        
        symbol_data = {}
        for symbol in self.stocks:
            symbol_df = combined[combined['ticker'] == symbol].copy()
            if len(symbol_df) > 0:
                symbol_df = symbol_df.sort_values('timestamp').reset_index(drop=True)
                symbol_data[symbol] = symbol_df
        
        print(f"✅ Loaded {len(symbol_data)} stocks from Polygon S3")
        for symbol, df in symbol_data.items():
            print(f"   {symbol}: {len(df):,} bars")
        
        return symbol_data
    
    def _download_polygon_file(self, file_key: str) -> pd.DataFrame:
        """Download and parse single Polygon S3 file"""
        try:
            response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=file_key)
            compressed = response['Body'].read()
            decompressed = gzip.decompress(compressed)
            df = pd.read_csv(io.BytesIO(decompressed))
            
            # Standardize column names
            if 'window_start' in df.columns:
                df['timestamp'] = pd.to_datetime(df['window_start'], unit='ns')
            elif 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            else:
                return pd.DataFrame()
            
            if 'symbol' in df.columns and 'ticker' not in df.columns:
                df['ticker'] = df['symbol']
            
            # Keep only required columns
            required = ['ticker', 'open', 'high', 'low', 'close', 'volume', 'timestamp']
            if all(col in df.columns for col in required):
                return df[required]
            
            return pd.DataFrame()
        except Exception as e:
            return pd.DataFrame()
    
    # ===== BINANCE CRYPTO =====
    
    def fetch_binance_crypto(self) -> Dict[str, pd.DataFrame]:
        """Fetch crypto from Binance historical data"""
        print("\n" + "="*70)
        print("🪙 FETCHING CRYPTO FROM BINANCE")
        print("="*70)
        
        symbol_data = {}
        
        for symbol in tqdm(self.crypto, desc="Binance Crypto"):
            all_bars = []
            current = self.start_date
            
            # Download month by month
            while current <= self.end_date:
                year = current.year
                month = current.month
                
                klines = self._download_binance_month(symbol, year, month)
                if klines:
                    parsed = [self._parse_binance_kline(k) for k in klines]
                    parsed = [p for p in parsed if p is not None and 
                             self.start_date <= p['timestamp'] <= self.end_date]
                    all_bars.extend(parsed)
                
                # Next month
                if month == 12:
                    current = datetime(year + 1, 1, 1)
                else:
                    current = datetime(year, month + 1, 1)
            
            if all_bars:
                df = pd.DataFrame(all_bars)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['ticker'] = symbol  # Add ticker column for consistency
                df = df.sort_values('timestamp').reset_index(drop=True)
                symbol_data[symbol] = df
        
        print(f"✅ Loaded {len(symbol_data)} crypto pairs from Binance")
        for symbol, df in symbol_data.items():
            print(f"   {symbol}: {len(df):,} bars")
        
        return symbol_data
    
    def _download_binance_month(self, symbol: str, year: int, month: int) -> List:
        """Download Binance monthly data archive"""
        url = f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/1m/{symbol}-1m-{year}-{month:02d}.zip"
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                csv_filename = z.namelist()[0]
                with z.open(csv_filename) as csv_file:
                    csv_data = csv_file.read().decode('utf-8')
                    reader = csv.reader(io.StringIO(csv_data))
                    return list(reader)
        except:
            return []
    
    def _parse_binance_kline(self, kline_row) -> Dict:
        """Parse Binance kline format"""
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
                "transactions": int(float(kline_row[8])) if len(kline_row) > 8 else 1
            }
        except:
            return None
    
    # ===== YAHOO FINANCE =====
    
    def fetch_yahoo_stocks(self) -> Dict[str, pd.DataFrame]:
        """Fetch stocks from Yahoo Finance - PRIMARY DATA SOURCE"""
        print("\n" + "="*70)
        print("📈 FETCHING STOCKS FROM YAHOO FINANCE (Primary Source)")
        print("="*70)
        
        try:
            import yfinance as yf
        except ImportError:
            print("⚠️  yfinance not installed. Skipping Yahoo data.")
            print("   Install: pip install yfinance")
            return {}
        
        symbol_data = {}
        start_str = self.start_date.strftime("%Y-%m-%d")
        end_str = (self.end_date + timedelta(days=1)).strftime("%Y-%m-%d")
        
        for symbol in tqdm(self.stocks, desc="Yahoo Finance"):
            try:
                ticker = yf.Ticker(symbol)
                
                # Get both minute and daily data
                # Minute data (last 7 days only - Yahoo Finance limit)
                recent_start = max(self.start_date, datetime.now() - timedelta(days=7))
                df_minute = pd.DataFrame()
                
                if recent_start <= self.end_date:
                    df_minute = ticker.history(
                        start=recent_start.strftime("%Y-%m-%d"), 
                        end=end_str, 
                        interval="1m"
                    )
                
                # Daily data (full historical range)
                df_daily = ticker.history(start=start_str, end=end_str, interval="1d")
                
                # Combine both (prefer minute data where available)
                df = df_minute if not df_minute.empty else df_daily
                
                if not df.empty:
                    df = df.reset_index()
                    
                    # Rename columns
                    rename_map = {
                        'Datetime': 'timestamp',
                        'Date': 'timestamp',
                        'Open': 'open',
                        'High': 'high',
                        'Low': 'low',
                        'Close': 'close',
                        'Volume': 'volume'
                    }
                    df = df.rename(columns=rename_map)
                    
                    df['ticker'] = symbol
                    df['transactions'] = 1
                    df = df[['ticker', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'transactions']]
                    symbol_data[symbol] = df
                
                time.sleep(0.3)  # Rate limiting
                
            except Exception as e:
                print(f"  ⚠️  Error fetching {symbol}: {e}")
        
        print(f"✅ Loaded {len(symbol_data)} stocks from Yahoo Finance")
        for symbol, df in symbol_data.items():
            print(f"   {symbol}: {len(df):,} bars")
        
        return symbol_data
    
    def fetch_yahoo_news(self) -> pd.DataFrame:
        """Fetch news sentiment from Yahoo Finance aligned with timestamps"""
        print("\n" + "="*70)
        print("📰 FETCHING NEWS FROM YAHOO FINANCE")
        print("="*70)
        
        try:
            import yfinance as yf
        except ImportError:
            print("⚠️  yfinance not installed. Skipping news data.")
            return pd.DataFrame()
        
        all_news = []
        
        for symbol in tqdm(self.stocks, desc="Yahoo News"):
            try:
                ticker = yf.Ticker(symbol)
                news = ticker.news
                
                for article in news:
                    try:
                        pub_time = datetime.fromtimestamp(article.get('providerPublishTime', 0))
                        
                        # Only include news within our date range
                        if self.start_date <= pub_time <= self.end_date:
                            # Basic sentiment from title (positive/negative word count)
                            title = article.get('title', '').lower()
                            positive_words = ['up', 'gain', 'rise', 'profit', 'beat', 'surge', 'rally', 'jump']
                            negative_words = ['down', 'loss', 'fall', 'miss', 'drop', 'plunge', 'crash', 'decline']
                            
                            pos_count = sum(1 for word in positive_words if word in title)
                            neg_count = sum(1 for word in negative_words if word in title)
                            sentiment = (pos_count - neg_count) / max(pos_count + neg_count, 1)
                            
                            all_news.append({
                                'timestamp': pub_time,
                                'symbol': symbol,
                                'title': article.get('title', ''),
                                'publisher': article.get('publisher', ''),
                                'sentiment': sentiment,
                                'link': article.get('link', '')
                            })
                    except:
                        continue
                
                time.sleep(0.2)  # Rate limiting
                
            except Exception as e:
                print(f"  ⚠️  Error fetching news for {symbol}: {e}")
        
        if not all_news:
            print("⚠️  No news data retrieved")
            return pd.DataFrame()
        
        news_df = pd.DataFrame(all_news)
        news_df = news_df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"✅ Loaded {len(news_df)} news articles")
        print(f"   Date range: {news_df['timestamp'].min()} to {news_df['timestamp'].max()}")
        print(f"   Symbols covered: {news_df['symbol'].nunique()}")
        
        return news_df
    
    # ===== UNIFIED FETCH =====
    
    def fetch_all(self) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
        """Fetch all data sources and return combined"""
        print("\n" + "="*70)
        print("🌍 PRODUCTION DATA FETCHER - REAL DATA ONLY")
        print("="*70)
        print(f"📅 {self.start_date.date()} → {self.end_date.date()}")
        print()
        
        # Priority: Yahoo Finance (Primary) > Polygon > Binance
        all_data = {}
        
        # 1. Yahoo Finance (Primary - most reliable, both minute + daily)
        yahoo_stocks = self.fetch_yahoo_stocks()
        all_data.update(yahoo_stocks)
        
        # 2. Polygon S3 (Supplement stocks - minute data only)
        polygon_stocks = self.fetch_polygon_stocks()
        for symbol, df in polygon_stocks.items():
            if symbol not in all_data:  # Only add if not from Yahoo
                all_data[symbol] = df
            else:
                # Merge Polygon minute data with Yahoo daily data
                print(f"   Merging {symbol}: Yahoo + Polygon")
        
        # 3. Binance (Crypto - minute data)
        binance_crypto = self.fetch_binance_crypto()
        all_data.update(binance_crypto)
        
        # 4. Yahoo News (Sentiment)
        news_df = self.fetch_yahoo_news()
        
        # Summary
        print("\n" + "="*70)
        print("📊 DATA SUMMARY")
        print("="*70)
        print(f"Total symbols: {len(all_data)}")
        print(f"  Yahoo stocks (primary): {len(yahoo_stocks)}")
        print(f"  Polygon stocks (supplementary): {len([s for s in polygon_stocks if s not in yahoo_stocks])}")
        print(f"  Binance crypto: {len(binance_crypto)}")
        print(f"  News articles: {len(news_df)}")
        print()
        
        # Validate data quality
        total_bars = sum(len(df) for df in all_data.values())
        print(f"Total bars: {total_bars:,}")
        
        symbols_with_data = [s for s, df in all_data.items() if len(df) >= 1000]
        print(f"Symbols with >=1000 bars: {len(symbols_with_data)}/{len(all_data)}")
        
        print("="*70)
        
        return all_data, news_df


def test_fetcher():
    """Test the production data fetcher"""
    # Test with maximum historical range
    end = datetime.now()
    start = datetime(2021, 1, 1)  # 3+ years of data
    
    fetcher = ProductionDataFetcher(
        start_date=start.strftime("%Y-%m-%d"),
        end_date=end.strftime("%Y-%m-%d")
    )
    
    market_data, news_data = fetcher.fetch_all()
    
    print("\n🎯 TEST RESULTS:")
    print(f"  Market data symbols: {len(market_data)}")
    print(f"  News articles: {len(news_data)}")
    
    if market_data:
        sample_symbol = list(market_data.keys())[0]
        print(f"\n📊 Sample data ({sample_symbol}):")
        print(market_data[sample_symbol].head())
    
    if not news_data.empty:
        print(f"\n📰 Sample news:")
        print(news_data.head())


if __name__ == "__main__":
    test_fetcher()
