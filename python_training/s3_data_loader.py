#!/usr/bin/env python3
"""High-performance S3 data loader for massive parallel training"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import boto3
import gzip
import io
import numpy as np
import pandas as pd

from botocore.exceptions import ClientError

try:  # Optional dependency for nicer CLI feedback
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback for minimal installs
    tqdm = None
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

from data_discovery import load_discovered_symbols

load_dotenv()


def _is_crypto_symbol(symbol: str) -> bool:
    return symbol.endswith(("USD", "USDT", "BUSD", "BTC", "ETH")) and len(symbol) > 4


class S3MarketDataLoader:
    """Loads massive amounts of market data from Polygon S3 efficiently"""

    def __init__(self, cache_size_gb: float = 10.0):
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("POLYGON_S3_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("POLYGON_S3_SECRET_KEY"),
            endpoint_url=os.getenv("POLYGON_S3_ENDPOINT"),
        )
        self.bucket = os.getenv("POLYGON_S3_BUCKET", "flatfiles")
        self.cache: Dict[str, pd.DataFrame] = {}  # {date_key: DataFrame}
        self.cache_size_bytes = int(cache_size_gb * 1e9)
        self.current_cache_size = 0
        
    def _download_and_parse(self, file_key: str) -> Tuple[pd.DataFrame, Optional[str], Optional[str]]:
        """Download and parse a single S3 file.

        Returns the dataframe along with an optional error code/message so that
        the caller can aggregate warnings instead of spamming the console for
        every missing day.
        """

        try:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=file_key)
        except ClientError as exc:  # pragma: no cover - requires network
            error = exc.response.get("Error", {})
            return pd.DataFrame(), error.get("Code"), error.get("Message")
        except Exception as exc:  # pragma: no cover - defensive
            return pd.DataFrame(), "Unknown", str(exc)

        try:
            compressed = response["Body"].read()
            decompressed = gzip.decompress(compressed)
            df = pd.read_csv(io.BytesIO(decompressed))
        except Exception as exc:  # pragma: no cover - malformed payloads
            return pd.DataFrame(), "ParseError", str(exc)

        # Convert window_start from nanoseconds to datetime
        df["timestamp"] = pd.to_datetime(df["window_start"], unit="ns")
        df = df.sort_values(["ticker", "timestamp"])

        return df, None, None
    
    def load_date_range(
        self,
        start_date: str = "2024-01-02",
        end_date: str = "2024-01-31",
        asset_type: str = "stocks",
        max_workers: int = 8,
    ) -> Dict[str, pd.DataFrame]:
        """Load multiple days of data in parallel.

        Returns a mapping of ``symbol -> DataFrame`` with OHLCV data.
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
        error_counts: Dict[str, int] = defaultdict(int)
        error_examples: Dict[str, Tuple[str, str]] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._download_and_parse, fk): (date, fk)
                      for date, fk in file_keys}

            for future in as_completed(futures):
                date, file_key = futures[future]
                df, error_code, error_msg = future.result()
                if not df.empty:
                    all_dfs.append(df)
                    print(f"  ✅ {date}: {len(df):,} rows, {df['ticker'].nunique()} symbols")
                elif error_code:
                    error_counts[error_code] += 1
                    if error_code not in error_examples:
                        error_examples[error_code] = (file_key, error_msg or "")
                elif error_msg:
                    error_counts["Unknown"] += 1
                    if "Unknown" not in error_examples:
                        error_examples["Unknown"] = (file_key, error_msg)

        if error_counts:
            for code, count in sorted(error_counts.items(), key=lambda item: item[0]):
                sample_key, sample_message = error_examples.get(code, ("", ""))
                if code in {"403", "AccessDenied"}:
                    print(
                        f"⚠️  Access denied for {count} {asset_type} file(s); "
                        f"example: {sample_key or 'unknown'} ({sample_message or 'permission denied'})"
                    )
                elif code in {"NoSuchKey", "404"}:
                    print(
                        f"⚠️  Missing {count} {asset_type} file(s); "
                        f"example: {sample_key or 'unknown'}"
                    )
                elif code == "ParseError":
                    print(
                        f"⚠️  Failed to parse {count} {asset_type} file(s); "
                        f"example: {sample_key or 'unknown'} ({sample_message})"
                    )
                else:
                    print(
                        f"⚠️  Error loading {count} {asset_type} file(s); "
                        f"example: {sample_key or 'unknown'} ({sample_message})"
                    )

        # Combine all dates
        if not all_dfs:
            print("❌ No data loaded!")
            return {}
        
        print(f"📊 Combining {len(all_dfs)} dataframes...")
        combined = pd.concat(all_dfs, ignore_index=True)
        combined = combined.sort_values(['ticker', 'timestamp'])
        
        print(f"✅ Loaded {len(combined):,} total rows")
        unique_symbols = combined['ticker'].nunique()
        print(f"✅ Unique symbols: {unique_symbols}")

        # Split by symbol with progress feedback
        print("   🔄 Preparing per-symbol data (this may take a minute)...")
        symbol_data = {}
        symbols = combined['ticker'].unique()

        if tqdm is None:
            for symbol, symbol_df in combined.groupby('ticker'):
                symbol_df = symbol_df.sort_values('timestamp').reset_index(drop=True)
                symbol_data[symbol] = symbol_df
        else:
            with tqdm(total=len(symbols), desc="   Symbols", unit="symbol", leave=False) as pbar:
                for symbol, symbol_df in combined.groupby('ticker'):
                    symbol_df = symbol_df.sort_values('timestamp').reset_index(drop=True)
                    symbol_data[symbol] = symbol_df
                    pbar.update(1)

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
    
    def prepare_training_data(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        symbols: Optional[List[str]] = None,
        max_bars_per_symbol: int = 5000,
    ) -> Dict[str, np.ndarray]:
        """Prepare data for training environments."""

        if symbols is None:
            symbols = list(symbol_data.keys())

        training_data: Dict[str, np.ndarray] = {}
        for symbol in symbols:
            if symbol not in symbol_data:
                continue

            df = symbol_data[symbol]

            # Limit to recent data if too long
            if len(df) > max_bars_per_symbol:
                df = df.tail(max_bars_per_symbol)

            # Extract OHLCV + transactions
            data = df[["open", "high", "low", "close", "volume", "transactions"]].values
            training_data[symbol] = data

        return training_data


@dataclass
class SyntheticSymbolConfig:
    symbol: str
    asset_type: str


class S3DataLoader:
    """Facade used by the training scripts.

    The original project expected a ``S3DataLoader`` class with
    ``discover_all_symbols`` and ``load_multi_day_data`` helpers.
    The refactor to ``S3MarketDataLoader`` removed that wrapper, which
    caused imports to fail and training to silently stop before any GPU
    work began.  This class restores the original interface while keeping
    the optimized S3 loading code and providing deterministic synthetic
    fallbacks when credentials are unavailable.
    """

    def __init__(
        self,
        cache_size_gb: float = 10.0,
        max_synthetic_symbols: int = 256,
    ) -> None:
        self.max_synthetic_symbols = max_synthetic_symbols
        self._symbol_cache: Optional[List[str]] = None

        try:
            self.market_loader = S3MarketDataLoader(cache_size_gb=cache_size_gb)
        except Exception as exc:
            print(f"⚠️  Could not initialise S3 market loader: {exc}")
            print("   Falling back to deterministic synthetic data.")
            self.market_loader = None

        self.synthetic_configs = self._build_synthetic_universe(max_synthetic_symbols)

    @staticmethod
    def _build_synthetic_universe(max_symbols: int) -> List[SyntheticSymbolConfig]:
        synthetic = []
        for i in range(max_symbols):
            asset_type = "crypto" if i % 5 == 0 else "stocks"
            symbol = f"SYN{i:04d}{'USD' if asset_type == 'crypto' else ''}"
            synthetic.append(SyntheticSymbolConfig(symbol=symbol, asset_type=asset_type))
        return synthetic

    def discover_all_symbols(self, max_symbols: int = 200) -> List[str]:
        if self._symbol_cache is not None:
            return self._symbol_cache[:max_symbols]

        discovered: List[str] = []
        try:
            cache = load_discovered_symbols()
            discovered.extend(cache.get("stocks", []))
            discovered.extend(cache.get("crypto", []))
        except Exception as exc:
            print(f"⚠️  Could not load discovered symbols: {exc}")

        if not discovered:
            # Fallback to synthetic symbols
            discovered = [cfg.symbol for cfg in self.synthetic_configs]

        random.shuffle(discovered)
        self._symbol_cache = discovered
        return discovered[:max_symbols]

    def _generate_synthetic_series(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        num_rows: int = 10_000,
    ) -> List[Dict[str, float]]:
        rng = np.random.default_rng(abs(hash((symbol, start_date, end_date))) % 2**32)
        prices = rng.lognormal(mean=0.0, sigma=0.01, size=num_rows).cumprod() * 100
        volumes = rng.integers(1_000, 100_000, size=num_rows)

        records = []
        for idx in range(num_rows):
            price = float(prices[idx])
            high = price * (1 + float(rng.normal(0, 0.002)))
            low = price * (1 - float(rng.normal(0, 0.002)))
            record = {
                "open": price,
                "high": high,
                "low": low,
                "close": price * (1 + float(rng.normal(0, 0.001))),
                "volume": float(volumes[idx]),
                "transactions": int(volumes[idx] * rng.uniform(0.01, 0.05)),
                "timestamp": datetime.utcnow().isoformat(),
            }
            records.append(record)
        return records

    def _load_from_s3(
        self,
        start_date: str,
        end_date: str,
        symbols: Iterable[str],
        max_workers: int = 16,
    ) -> Dict[str, List[Dict[str, float]]]:
        assert self.market_loader is not None

        # Separate symbols by asset type for efficient downloads
        stocks = [s for s in symbols if not _is_crypto_symbol(s)]
        crypto = [s for s in symbols if _is_crypto_symbol(s)]

        symbol_frames: Dict[str, pd.DataFrame] = {}

        if stocks:
            stock_frames = self.market_loader.load_date_range(
                start_date=start_date,
                end_date=end_date,
                asset_type="stocks",
                max_workers=max_workers,
            )
            for symbol in stocks:
                if symbol in stock_frames:
                    symbol_frames[symbol] = stock_frames[symbol]

        if crypto:
            crypto_frames = self.market_loader.load_date_range(
                start_date=start_date,
                end_date=end_date,
                asset_type="crypto",
                max_workers=max_workers,
            )
            for symbol in crypto:
                if symbol in crypto_frames:
                    symbol_frames[symbol] = crypto_frames[symbol]

        if not symbol_frames:
            print("⚠️  Requested symbols not found in S3 dump; falling back to synthetic data")

        prepared: Dict[str, List[Dict[str, float]]] = {}
        for symbol in symbols:
            frame = symbol_frames.get(symbol)
            if frame is None:
                prepared[symbol] = self._generate_synthetic_series(symbol, start_date, end_date)
                continue

            frame = frame.copy()
            frame.rename(columns={"ticker": "symbol"}, inplace=True, errors="ignore")
            frame = frame[["open", "high", "low", "close", "volume", "timestamp"]]
            frame["transactions"] = frame.get("transactions", 0)
            prepared[symbol] = frame.to_dict("records")

        return prepared

    def load_multi_day_data(
        self,
        start_date: str,
        end_date: str,
        symbols: Iterable[str],
        max_workers: int = 16,
    ) -> Dict[str, List[Dict[str, float]]]:
        symbols = list(symbols)
        if not symbols:
            return {}

        if self.market_loader is None:
            print("⚠️  Using synthetic market data (no S3 credentials available)")
            return {
                symbol: self._generate_synthetic_series(symbol, start_date, end_date)
                for symbol in symbols
            }

        return self._load_from_s3(
            start_date=start_date,
            end_date=end_date,
            symbols=symbols,
            max_workers=max_workers,
        )


if __name__ == "__main__":
    print("🧪 Testing S3 Data Loader\n")

    loader = S3DataLoader(cache_size_gb=5.0)
    symbols = loader.discover_all_symbols(max_symbols=25)
    print(f"Discovered {len(symbols)} symbols")

    data = loader.load_multi_day_data(
        start_date="2024-01-02",
        end_date="2024-01-31",
        symbols=symbols[:10],
        max_workers=8,
    )

    lengths = {symbol: len(rows) for symbol, rows in data.items()}
    print("Sample counts:")
    for symbol, count in list(lengths.items())[:5]:
        print(f"  {symbol}: {count:,} rows")
