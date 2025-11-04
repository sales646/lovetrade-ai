#!/usr/bin/env python3
"""High-performance S3 data loader for massive parallel training"""

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
import random

load_dotenv()

class S3MarketDataLoader:
    """Loads massive amounts of market data from Polygon S3 efficiently"""
    
    def __init__(self, cache_size_gb: float = 10.0):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv("POLYGON_S3_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("POLYGON_S3_SECRET_KEY"),
            endpoint_url=os.getenv("POLYGON_S3_ENDPOINT")
        )
        self.bucket = os.getenv("POLYGON_S3_BUCKET", "flatfiles")
        self.cache = {}  # {date_key: DataFrame}
        self.cache_size_bytes = int(cache_size_gb * 1e9)
        self.current_cache_size = 0
        
    def _download_and_parse(self, file_key: str) -> pd.DataFrame:
        """Download and parse a single S3 file"""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=file_key)
            compressed = response['Body'].read()
            decompressed = gzip.decompress(compressed)
            df = pd.read_csv(io.BytesIO(decompressed))
            
            # Convert window_start from nanoseconds to datetime
            df['timestamp'] = pd.to_datetime(df['window_start'], unit='ns')
            df = df.sort_values(['ticker', 'timestamp'])
            
            return df
        except Exception as e:
            print(f"⚠️ Error loading {file_key}: {e}")
            return pd.DataFrame()
    
    def load_date_range(self, 
                        start_date: str = "2024-01-02",
                        end_date: str = "2024-01-31",
                        asset_type: str = "stocks",
                        max_workers: int = 8) -> Dict[str, pd.DataFrame]:
        """
        Load multiple days of data in parallel
        
        Returns: {symbol: DataFrame} with OHLCV data
        """
        # Generate list of trading days
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        dates = []
        current = start
        while current <= end:
            # Skip weekends (basic filter)
            if current.weekday() < 5:
                dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
        
        print(f"📅 Loading {len(dates)} days from S3 ({start_date} to {end_date})...")
        
        # Build file keys
        prefix = "us_stocks_sip" if asset_type == "stocks" else "global_crypto"
        file_keys = []
        for date in dates:
            year, month, day = date.split("-")
            file_key = f"{prefix}/minute_aggs_v1/{year}/{month}/{date}.csv.gz"
            file_keys.append((date, file_key))
        
        # Download in parallel
        all_dfs = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._download_and_parse, fk): date 
                      for date, fk in file_keys}
            
            for future in as_completed(futures):
                date = futures[future]
                df = future.result()
                if not df.empty:
                    all_dfs.append(df)
                    print(f"  ✅ {date}: {len(df):,} rows, {df['ticker'].nunique()} symbols")
        
        # Combine all dates
        if not all_dfs:
            print("❌ No data loaded!")
            return {}
        
        print(f"📊 Combining {len(all_dfs)} dataframes...")
        combined = pd.concat(all_dfs, ignore_index=True)
        combined = combined.sort_values(['ticker', 'timestamp'])
        
        print(f"✅ Loaded {len(combined):,} total rows")
        print(f"✅ Unique symbols: {combined['ticker'].nunique()}")
        
        # Split by symbol
        symbol_data = {}
        for symbol in combined['ticker'].unique():
            symbol_df = combined[combined['ticker'] == symbol].copy()
            symbol_df = symbol_df.sort_values('timestamp').reset_index(drop=True)
            symbol_data[symbol] = symbol_df
        
        return symbol_data
    
    def get_random_symbols(self, 
                          symbol_data: Dict[str, pd.DataFrame],
                          n_symbols: int = 100,
                          min_bars: int = 1000) -> List[str]:
        """Get random symbols with sufficient data"""
        valid_symbols = [s for s, df in symbol_data.items() if len(df) >= min_bars]
        if len(valid_symbols) < n_symbols:
            print(f"⚠️ Only {len(valid_symbols)} symbols have >= {min_bars} bars")
            return valid_symbols
        return random.sample(valid_symbols, n_symbols)
    
    def prepare_training_data(self,
                            symbol_data: Dict[str, pd.DataFrame],
                            symbols: List[str] = None,
                            max_bars_per_symbol: int = 5000) -> Dict[str, np.ndarray]:
        """
        Prepare data for training environments
        
        Returns: {symbol: array of shape (T, 6)} where columns are [open, high, low, close, volume, transactions]
        """
        if symbols is None:
            symbols = list(symbol_data.keys())
        
        training_data = {}
        for symbol in symbols:
            if symbol not in symbol_data:
                continue
            
            df = symbol_data[symbol]
            
            # Limit to recent data if too long
            if len(df) > max_bars_per_symbol:
                df = df.tail(max_bars_per_symbol)
            
            # Extract OHLCV + transactions
            data = df[['open', 'high', 'low', 'close', 'volume', 'transactions']].values
            training_data[symbol] = data
        
        return training_data


if __name__ == "__main__":
    print("🧪 Testing S3 Data Loader\n")
    
    loader = S3MarketDataLoader(cache_size_gb=5.0)
    
    # Load January 2024 data
    symbol_data = loader.load_date_range(
        start_date="2024-01-02",
        end_date="2024-01-31",
        asset_type="stocks",
        max_workers=16
    )
    
    print(f"\n📊 Data Summary:")
    print(f"   Total symbols: {len(symbol_data)}")
    
    # Get top 10 symbols by data volume
    top_symbols = sorted(symbol_data.items(), key=lambda x: len(x[1]), reverse=True)[:10]
    print(f"\n📈 Top 10 symbols by bar count:")
    for symbol, df in top_symbols:
        print(f"   {symbol}: {len(df):,} bars")
    
    # Prepare training data
    train_symbols = loader.get_random_symbols(symbol_data, n_symbols=50, min_bars=1000)
    training_data = loader.prepare_training_data(symbol_data, train_symbols)
    
    print(f"\n✅ Training data ready:")
    print(f"   Symbols: {len(training_data)}")
    print(f"   Total data points: {sum(len(d) for d in training_data.values()):,}")
