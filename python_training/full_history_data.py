"""Full-history data orchestration with caching, features, and windowing."""
from __future__ import annotations

import io
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from .full_history_config import FullHistoryConfig

try:  # Optional dependency for NYSE calendar
    import pandas_market_calendars as mcal
except Exception:  # pragma: no cover - optional dependency
    mcal = None

try:  # Alternate optional dependency for NYSE calendar
    import exchange_calendars as xcal
except Exception:  # pragma: no cover - optional dependency
    xcal = None

try:  # Optional dependency for Yahoo finance
    import yfinance as yf
except Exception:  # pragma: no cover - optional dependency
    yf = None

try:  # Optional dependency for Polygon S3
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
except Exception:  # pragma: no cover - optional dependency
    boto3 = None
    BotoCoreError = ClientError = NoCredentialsError = Exception


@dataclass
class WindowSpec:
    symbol: str
    split: str
    start_idx: int
    end_idx: int
    start_ts: datetime
    end_ts: datetime
    regime: str
    data_path: str


class FullHistoryDataManager:
    """Handles fetching, caching, feature building, and manifest generation."""

    def __init__(
        self,
        config: Optional[FullHistoryConfig] = None,
        stocks: Optional[Sequence[str]] = None,
        crypto: Optional[Sequence[str]] = None,
    ) -> None:
        self.config = config or FullHistoryConfig()
        self.config.resolve_dates()
        self.config.ensure_cache_dir()

        self.cache_dir: Path = self.config.cache_path
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.stocks = list(stocks or [])
        self.crypto = list(crypto or [])
        self._sentinel = self.cache_dir / ".prep_done"
        self._manifest_path = self.cache_dir / "env_manifest.parquet"
        self._feature_path = self.cache_dir / "feature_columns.txt"
        self._source_path = self.cache_dir / "sources.json"
        self._symbol_sources: Dict[str, str] = {}

        self._start_dt = self.config.resolved_datetime(self.config.start)
        self._end_dt = self.config.resolved_datetime(self.config.end)
        self._trading_days: pd.DatetimeIndex = pd.DatetimeIndex([])
        self._files_processed: int = 0
        self._coverage_stats: Dict[str, Dict[str, object]] = {}
        self._s3_client = None
        self._s3_client_failed = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def prepare(self, force_full_prep: Optional[bool] = None) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
        """Prepare full-history data and return market data and manifest."""

        force = force_full_prep if force_full_prep is not None else self.config.force_full_prep
        if force:
            for path in [self._sentinel, self._manifest_path, self._feature_path, self._source_path]:
                if path.exists():
                    path.unlink()

        self._start_dt = self.config.resolved_datetime(self.config.start)
        self._end_dt = self.config.resolved_datetime(self.config.end)

        if self.config.sanity_mode:
            self.config.window_size = min(self.config.window_size, self.config.lookback_sanity)
            self.config.min_bars_for_windowing = min(
                self.config.min_bars_for_windowing,
                self.config.lookback_sanity,
            )
            self._end_dt = max(
                self._start_dt,
                self._start_dt + timedelta(minutes=self.config.lookback_sanity),
            )

        self._trading_days = self._build_trading_calendar()
        self._print_range_banner()

        if not force and self._can_short_circuit():
            manifest = pd.read_parquet(self._manifest_path)
            market_data = self._load_cached_symbols(manifest)
            feature_columns = self._load_feature_columns()
            self._symbol_sources = self._load_sources()
            if feature_columns:
                for df in market_data.values():
                    missing = [col for col in feature_columns if col not in df.columns]
                    for col in missing:
                        df[col] = np.nan
            return market_data, manifest

        tickers = sorted(set(self.stocks + self.crypto))
        if self.config.sanity_mode:
            tickers = tickers[:3]

        if not tickers:
            raise ValueError("No tickers configured for data preparation")

        market_data = self._fetch_all_symbols(tickers)
        market_data = self._build_features(market_data)
        manifest = self._build_manifest(market_data)

        manifest.to_parquet(self._manifest_path, index=False)
        self._persist_feature_columns(market_data)
        self._sentinel.touch()

        return market_data, manifest

    # ------------------------------------------------------------------
    # Fetch helpers
    # ------------------------------------------------------------------
    def _build_trading_calendar(self) -> pd.DatetimeIndex:
        start_date = self._start_dt.date()
        end_date = self._end_dt.date()

        if mcal is not None:
            cal = mcal.get_calendar("XNYS")
            schedule = cal.schedule(start_date=start_date, end_date=end_date, tz="UTC")
            trading_days = schedule.index
        elif xcal is not None:
            cal = xcal.get_calendar("XNYS")
            start_ts = pd.Timestamp(start_date, tz="UTC")
            end_ts = pd.Timestamp(end_date, tz="UTC")
            trading_days = cal.sessions_in_range(start_ts, end_ts)
        else:  # pragma: no cover - optional dependency missing
            raise RuntimeError(
                "pandas_market_calendars or exchange_calendars is required for NYSE trading days"
            )

        trading_days = pd.DatetimeIndex(trading_days)
        if getattr(trading_days, "tz", None) is not None:
            trading_days = trading_days.tz_convert("UTC").tz_localize(None)

        if not self.config.sanity_mode and len(trading_days) < 1000:
            raise RuntimeError("Range resolution bug: expected ≥1500 trading days.")

        return trading_days

    def _print_range_banner(self) -> None:
        start_str = self._start_dt.date().isoformat()
        end_str = self._end_dt.date().isoformat()
        day_strings = [day.strftime("%Y-%m-%d") for day in self._trading_days]
        first_days = ", ".join(day_strings[:3]) if day_strings else "n/a"
        last_days = ", ".join(day_strings[-3:]) if day_strings else "n/a"
        print(
            f"start_date={start_str}, end_date={end_str}, calendar=XNYS, "
            f"trading_days={len(self._trading_days)}, first_days=[{first_days}], "
            f"last_days=[{last_days}]"
        )

    def _fetch_all_symbols(self, tickers: Sequence[str]) -> Dict[str, pd.DataFrame]:
        """Fetch full history for every ticker with a progress bar."""

        total_bars = 0
        market_data: Dict[str, pd.DataFrame] = {}
        coverage_failures: List[Tuple[str, Dict[str, object]]] = []

        self._files_processed = 0
        self._coverage_stats = {}

        print()
        with tqdm(total=len(tickers), desc="FETCH FULL HISTORY", unit="symbol") as pbar:
            for symbol in tickers:
                df, source, files_used = self._fetch_symbol(symbol)
                if df is None or df.empty:
                    pbar.update(1)
                    continue

                cleaned = self._clean_dataframe(df)
                if cleaned.empty:
                    pbar.update(1)
                    continue

                self._files_processed += files_used
                total_bars += len(cleaned)
                market_data[symbol] = cleaned
                self._persist_full_symbol(symbol, cleaned)
                self._symbol_sources[symbol] = source or "unknown"

                stats = self._compute_coverage_stats(symbol, cleaned)
                self._coverage_stats[symbol] = stats
                if (
                    not self.config.sanity_mode
                    and (
                        stats["unique_days"] < 1000
                        or stats["coverage_pct"] < self.config.coverage_threshold * 100
                    )
                ):
                    coverage_failures.append((symbol, stats))

                pbar.update(1)

        if not market_data:
            raise RuntimeError("No market data available after fetching")

        self._print_coverage_summary(market_data)
        print(
            f"✅ FETCH DONE — {self._files_processed} files, {total_bars:,} bars, "
            f"days={len(self._trading_days)}"
        )
        self._persist_sources()

        if coverage_failures:
            failure_lines = []
            for symbol, stats in coverage_failures:
                failure_lines.append(
                    f"  {symbol}: {stats['coverage_pct']:.2f}% coverage, "
                    f"unique_days={int(stats['unique_days'])}"
                )
            failure_msg = "\n".join(["Coverage validation failed:"] + failure_lines + [
                "Hint: Check START/END, NYSE calendar, S3 day iteration, Polygon cursor, Binance startTime advancement."
            ])
            raise RuntimeError(failure_msg)

        return market_data

    def _fetch_symbol(self, symbol: str) -> Tuple[Optional[pd.DataFrame], Optional[str], int]:
        """Fetch a symbol from configured sources with fallback."""

        is_crypto = symbol in self.crypto or symbol.upper().endswith("USDT")

        preferred_sources: List[str] = []
        for source in self.config.sources:
            if is_crypto and source == "polygon":
                continue
            if not is_crypto and source == "binance":
                continue
            preferred_sources.append(source)

        if is_crypto:
            preferred_sources = ["binance", "yahoo", "polygon"] + preferred_sources
        else:
            preferred_sources = ["polygon", "yahoo", "binance"] + preferred_sources

        ordered_sources = []
        seen: set[str] = set()
        for src in preferred_sources:
            if src not in seen:
                ordered_sources.append(src)
                seen.add(src)

        for source in ordered_sources:
            if source == "polygon":
                df, files = self._fetch_polygon_symbol(symbol)
            elif source == "binance":
                df, files = self._fetch_binance_symbol(symbol)
            elif source == "yahoo":
                df, files = self._fetch_yahoo_symbol(symbol)
            else:
                continue

            if df is not None and not df.empty:
                df["source"] = source
                return df, source, files

        print(f"⚠️  No data retrieved for {symbol} from any source")
        return None, None, 0

    def _fetch_polygon_symbol(self, symbol: str) -> Tuple[Optional[pd.DataFrame], int]:
        df, files = self._fetch_polygon_s3(symbol)
        if df is not None and not df.empty:
            return df, files

        api_df, api_files = self._fetch_polygon_api(symbol)
        return api_df, api_files

    def _fetch_polygon_s3(self, symbol: str) -> Tuple[Optional[pd.DataFrame], int]:
        client = self._get_s3_client()
        if client is None or self._trading_days.empty:
            return None, 0

        bucket = os.getenv("POLYGON_S3_BUCKET", "polygon-data")
        prefix_base = os.getenv("POLYGON_S3_PREFIX", "stocks/aggregates_v3/1min")

        frames: List[pd.DataFrame] = []
        files = 0
        page = 0

        for day in self._trading_days:
            day_str = day.strftime("%Y-%m-%d")
            prefix = f"{prefix_base}/{day_str}/"
            try:
                keys = self._list_s3_keys(client, bucket, prefix)
            except Exception as exc:  # pragma: no cover - network
                print(f"⚠️  Polygon S3 list failed for {symbol} on {day_str}: {exc}")
                continue

            for key in sorted(keys):
                if not self._polygon_key_matches_symbol(key, symbol):
                    continue

                cache_path = self._page_cache_path(symbol, page)
                if self.config.use_cache and cache_path.exists():
                    df = pd.read_parquet(cache_path)
                else:
                    try:
                        obj = client.get_object(Bucket=bucket, Key=key)
                        body = obj["Body"].read()
                    except Exception as exc:  # pragma: no cover - network
                        print(f"⚠️  Polygon S3 download failed for {symbol}: {exc}")
                        continue

                    df = pd.read_parquet(io.BytesIO(body))
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    df.to_parquet(cache_path, index=False)

                if df is None or df.empty:
                    continue

                frames.append(df)
                files += 1
                page += 1

                if self.config.sanity_mode:
                    break

            if self.config.sanity_mode and frames:
                break

        if not frames:
            return None, files

        return pd.concat(frames, ignore_index=True), files

    def _fetch_polygon_api(self, symbol: str) -> Tuple[Optional[pd.DataFrame], int]:
        api_key = os.getenv("POLYGON_API_KEY")
        if not api_key:
            return None, 0

        base_url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/{self.config.start}/{self.config.end}"
        cursor: Optional[str] = None
        page = 0
        frames: List[pd.DataFrame] = []
        files = 0

        while True:
            cache_path = self._page_cache_path(symbol, page)
            if self.config.use_cache and cache_path.exists():
                df = pd.read_parquet(cache_path)
            else:
                params = {
                    "adjusted": "true",
                    "limit": min(self.config.chunk_size, 50_000),
                    "sort": "asc",
                    "apiKey": api_key,
                }
                if cursor:
                    params["cursor"] = cursor

                try:
                    response = self._request_with_backoff("get", base_url, params=params)
                except Exception as exc:
                    print(f"⚠️  Polygon API fetch failed for {symbol}: {exc}")
                    break

                payload = response.json()
                results = payload.get("results") or []
                if not results:
                    break

                df = pd.DataFrame(results)
                df = df.rename(
                    columns={
                        "t": "timestamp",
                        "o": "open",
                        "h": "high",
                        "l": "low",
                        "c": "close",
                        "v": "volume",
                    }
                )
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(None)

                cache_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(cache_path, index=False)

                cursor = payload.get("next_page_token") or payload.get("next_url")
                if cursor and "cursor=" in cursor:
                    cursor = cursor.split("cursor=")[-1]

            if df is None or df.empty:
                break

            frames.append(df)
            files += 1
            page += 1

            if not self.config.paginate or not cursor or self.config.sanity_mode:
                break

        if not frames:
            return None, files

        return pd.concat(frames, ignore_index=True), files

    def _get_s3_client(self):
        if boto3 is None or self._s3_client_failed:
            return None
        if self._s3_client is not None:
            return self._s3_client
        try:  # pragma: no cover - external dependency
            self._s3_client = boto3.client("s3")
        except (BotoCoreError, NoCredentialsError, ClientError) as exc:
            print(f"⚠️  Polygon S3 client unavailable: {exc}")
            self._s3_client_failed = True
            self._s3_client = None
        return self._s3_client

    @staticmethod
    def _polygon_key_matches_symbol(key: str, symbol: str) -> bool:
        filename = key.rsplit("/", 1)[-1]
        stem = filename.split(".")[0]
        return stem.upper().startswith(symbol.upper())

    @staticmethod
    def _list_s3_keys(client, bucket: str, prefix: str) -> List[str]:
        keys: List[str] = []
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):  # pragma: no cover - network
            for entry in page.get("Contents", []):
                keys.append(entry["Key"])
        return keys

    def _fetch_binance_symbol(self, symbol: str) -> Tuple[Optional[pd.DataFrame], int]:
        limit = min(self.config.chunk_size, 1000)
        start_ms = int(self._start_dt.timestamp() * 1000)
        end_ms = int(self._end_dt.timestamp() * 1000)

        frames: List[pd.DataFrame] = []
        files = 0
        page = 0

        while start_ms < end_ms:
            cache_path = self._page_cache_path(symbol, page)
            if self.config.use_cache and cache_path.exists():
                df = pd.read_parquet(cache_path)
                timestamps = pd.to_datetime(df["timestamp"])
                last_close_ms = int(timestamps.iloc[-1].timestamp() * 1000 + 59_999)
            else:
                params = {
                    "symbol": symbol,
                    "interval": "1m",
                    "limit": limit,
                    "startTime": start_ms,
                    "endTime": min(end_ms, start_ms + limit * 60_000),
                }
                try:
                    response = self._request_with_backoff(
                        "get",
                        "https://api.binance.com/api/v3/klines",
                        params=params,
                    )
                except Exception as exc:
                    print(f"⚠️  Binance fetch failed for {symbol}: {exc}")
                    break

                rows = response.json()
                if not rows:
                    break

                df = pd.DataFrame(
                    rows,
                    columns=[
                        "open_time",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "close_time",
                        "quote_asset_volume",
                        "number_of_trades",
                        "taker_buy_base",
                        "taker_buy_quote",
                        "ignore",
                    ],
                )
                last_close_ms = int(df["close_time"].iloc[-1])
                df = df[["open_time", "open", "high", "low", "close", "volume", "close_time"]]
                df = df.rename(columns={"open_time": "timestamp"})
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

                cache_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(cache_path, index=False)

            if df is None or df.empty:
                break

            frames.append(df)
            files += 1
            page += 1
            start_ms = last_close_ms + 1

            if not self.config.paginate or self.config.sanity_mode:
                break

        if not frames:
            return None, files

        return pd.concat(frames, ignore_index=True), files

    def _fetch_yahoo_symbol(self, symbol: str) -> Tuple[Optional[pd.DataFrame], int]:
        if yf is None:
            return None, 0

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(interval="1m", start=self.config.start, end=self.config.end)
        except Exception:
            return None, 0

        if df.empty:
            return None, 0

        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        df = df.reset_index().rename(columns={"Datetime": "timestamp"})
        return df, 1

    def _request_with_backoff(self, method: str, url: str, **kwargs) -> requests.Response:
        delay = self.config.initial_backoff
        attempt = 0
        while True:
            try:
                response = requests.request(method, url, timeout=kwargs.pop("timeout", 30), **kwargs)
                if response.status_code == 429:
                    raise requests.HTTPError("rate limited", response=response)
                response.raise_for_status()
                return response
            except Exception as exc:
                attempt += 1
                if attempt >= self.config.max_retries:
                    raise
                jitter = random.uniform(0, delay)
                time.sleep(delay + jitter)
                delay = min(delay * 2, self.config.max_backoff)

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
    def _build_features(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        feature_columns_set: Set[str] = set()
        enriched: Dict[str, pd.DataFrame] = {}

        with tqdm(total=len(market_data), desc="FEATURE BUILD", unit="symbol") as pbar:
            for symbol, df in market_data.items():
                bars = df.copy()
                bars = bars.sort_values("timestamp").reset_index(drop=True)

                bars["log_return"] = np.log(bars["close"]).diff().fillna(0.0)
                bars["pct_change"] = bars["close"].pct_change().fillna(0.0)
                bars["rolling_vol_64"] = bars["log_return"].rolling(64, min_periods=8).std().fillna(method="bfill").fillna(0.0)
                bars["rolling_vol_256"] = bars["log_return"].rolling(256, min_periods=32).std().fillna(method="bfill").fillna(0.0)
                bars["rolling_mean_32"] = (
                    bars["close"].rolling(32, min_periods=8).mean().fillna(method="bfill").fillna(method="ffill")
                )
                bars["rolling_mean_128"] = (
                    bars["close"].rolling(128, min_periods=32).mean().fillna(method="bfill").fillna(method="ffill")
                )
                rolling_vol = bars["volume"].rolling(128, min_periods=32)
                bars["volume_zscore"] = (bars["volume"] - rolling_vol.mean()) / rolling_vol.std().replace(0, np.nan)
                bars["volume_zscore"] = bars["volume_zscore"].fillna(0.0)
                bars["high_low_range"] = (bars["high"] - bars["low"]) / bars["close"].replace(0, np.nan)
                bars["high_low_range"] = bars["high_low_range"].fillna(0.0)

                # Clip extreme outliers to reduce noise
                bars = self._clip_outliers(bars)

                feature_columns = [
                    col for col in bars.columns if col not in {"timestamp", "open", "high", "low", "close", "volume"}
                ]
                feature_columns_set.update(feature_columns)
                enriched[symbol] = bars
                self._persist_full_symbol(symbol, bars)
                pbar.update(1)

        feature_count = len(feature_columns_set)
        print(f"✅ FEATURES DONE — {feature_count} features")
        return enriched

    # ------------------------------------------------------------------
    # Windowing
    # ------------------------------------------------------------------
    def _build_manifest(self, market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        specs: List[WindowSpec] = []

        train_start, train_end = self.config.resolved_range(self.config.train_range)
        val_start, val_end = self.config.resolved_range(self.config.val_range)
        test_start, test_end = self.config.resolved_range(self.config.test_range)

        with tqdm(total=len(market_data), desc="WINDOWING", unit="symbol") as pbar:
            for symbol, df in market_data.items():
                if len(df) < self.config.min_bars_for_windowing:
                    pbar.update(1)
                    continue

                timestamps = df["timestamp"].tolist()
                total = len(df)
                data_path = self._relative_data_path(symbol)

                for start_idx in range(0, total - self.config.window_size + 1, self.config.window_stride):
                    end_idx = start_idx + self.config.window_size
                    window_start = timestamps[start_idx]
                    window_end = timestamps[end_idx - 1]

                    split = self._determine_split(
                        window_start,
                        window_end,
                        train_range=(train_start, train_end),
                        val_range=(val_start, val_end),
                        test_range=(test_start, test_end),
                    )
                    if not split:
                        continue

                    window_df = df.iloc[start_idx:end_idx]
                    window_return = window_df["close"].iloc[-1] / window_df["close"].iloc[0] - 1
                    if window_return > 0.02:
                        regime = "bull"
                    elif window_return < -0.02:
                        regime = "bear"
                    else:
                        regime = "sideways"

                    specs.append(
                        WindowSpec(
                            symbol=symbol,
                            split=split,
                            start_idx=start_idx,
                            end_idx=end_idx,
                            start_ts=window_start,
                            end_ts=window_end,
                            regime=regime,
                            data_path=data_path,
                        )
                    )

                pbar.update(1)

        if not specs:
            raise RuntimeError("No window specs generated - check data availability")

        manifest = pd.DataFrame([spec.__dict__ for spec in specs])
        manifest = self._balance_manifest(manifest)
        split_counts = manifest["split"].value_counts()
        print(
            "✅ WINDOWING DONE — train/val/test: "
            f"{split_counts.get('TRAIN', 0)}/{split_counts.get('VAL', 0)}/{split_counts.get('TEST', 0)}"
        )
        return manifest

    def _balance_manifest(self, manifest: pd.DataFrame) -> pd.DataFrame:
        balanced_frames: List[pd.DataFrame] = []
        rng = np.random.default_rng(42)

        for split, split_df in manifest.groupby("split"):
            split_df = split_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
            counts = split_df["regime"].value_counts()
            if counts.empty:
                balanced_frames.append(split_df)
                continue

            target = counts.min()
            if target == 0:
                balanced_frames.append(split_df)
                continue

            sampled_frames = []
            for regime, regime_df in split_df.groupby("regime"):
                take = min(len(regime_df), target)
                sampled = regime_df.sample(n=take, random_state=rng.integers(1, 1_000_000))
                sampled_frames.append(sampled)

            merged = pd.concat(sampled_frames, ignore_index=True)
            merged = merged.sample(frac=1.0, random_state=42).reset_index(drop=True)
            balanced_frames.append(merged)

        balanced = pd.concat(balanced_frames, ignore_index=True)
        balanced = balanced.sample(frac=1.0, random_state=42).reset_index(drop=True)
        return balanced

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        rename_map = {col: col.lower() for col in df.columns}
        df = df.rename(columns=rename_map)

        if "t" in df.columns and "timestamp" not in df.columns:
            df["timestamp"] = pd.to_datetime(df["t"], unit="ms")
        if "time" in df.columns and "timestamp" not in df.columns:
            df["timestamp"] = pd.to_datetime(df["time"])

        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        df = df.sort_values("timestamp").drop_duplicates("timestamp")

        df = df[(df["timestamp"] >= self._start_dt) & (df["timestamp"] <= self._end_dt)]

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        keep_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        df = df.dropna(subset=keep_cols)
        df = df[keep_cols]
        df = df.reset_index(drop=True)
        return df

    def _clip_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = [col for col in df.columns if df[col].dtype.kind in "if"]
        for col in numeric_cols:
            series = df[col]
            if series.empty:
                continue
            q_low = series.quantile(0.001)
            q_high = series.quantile(0.999)
            df[col] = series.clip(lower=q_low, upper=q_high)
        return df

    def _print_coverage_summary(self, market_data: Dict[str, pd.DataFrame]) -> None:
        print("\nCoverage summary per symbol:")
        for symbol, df in market_data.items():
            stats = self._coverage_stats.get(symbol) or self._compute_coverage_stats(symbol, df)
            source = self._symbol_sources.get(symbol, "unknown")
            print(
                f"  {symbol}: {stats['first_ts']} → {stats['last_ts']} — {int(stats['bars']):,} bars "
                f"({stats['coverage_pct']:.2f}% coverage, unique_days={int(stats['unique_days'])}, "
                f"expected_days={int(stats['expected_days'])}) [{source}]"
            )

    def _page_cache_path(self, symbol: str, page: int) -> Path:
        directory = self._symbol_cache_dir(symbol)
        directory.mkdir(parents=True, exist_ok=True)
        new_name = (
            f"{symbol}_{self.config.timeframe}_{self.config.start}_{self.config.end}_p{page:04d}.parquet"
        )
        new_path = directory / new_name
        legacy_path = directory / f"p{page:04d}.parquet"
        if legacy_path.exists() and not new_path.exists():
            return legacy_path
        return new_path

    def _compute_coverage_stats(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        if df.empty:
            return {
                "first_ts": None,
                "last_ts": None,
                "unique_days": 0,
                "bars": 0,
                "coverage_pct": 0.0,
                "expected_days": len(self._trading_days),
                "is_crypto": self._is_crypto_symbol(symbol),
            }

        timestamps = pd.to_datetime(df["timestamp"])
        unique_days = int(timestamps.dt.normalize().nunique())
        bars = int(len(df))

        is_crypto = self._is_crypto_symbol(symbol)
        if is_crypto:
            total_minutes = max(1, int((self._end_dt - self._start_dt).total_seconds() / 60) + 1)
            coverage = min(1.0, bars / total_minutes)
            expected_days = (self._end_dt.date() - self._start_dt.date()).days + 1
        else:
            expected_days = len(self._trading_days)
            coverage = min(1.0, unique_days / expected_days) if expected_days else 0.0

        return {
            "first_ts": timestamps.iloc[0],
            "last_ts": timestamps.iloc[-1],
            "unique_days": unique_days,
            "bars": bars,
            "coverage_pct": coverage * 100,
            "expected_days": expected_days,
            "is_crypto": is_crypto,
        }

    def _is_crypto_symbol(self, symbol: str) -> bool:
        return symbol in self.crypto or symbol.upper().endswith("USDT")

    def _symbol_cache_dir(self, symbol: str) -> Path:
        return self.cache_dir / symbol / self.config.timeframe / f"{self.config.start}_{self.config.end}"

    def _full_symbol_path(self, symbol: str) -> Path:
        return self._symbol_cache_dir(symbol) / "full.parquet"

    def _relative_data_path(self, symbol: str) -> str:
        return str(self._full_symbol_path(symbol).relative_to(self.cache_dir))

    def _persist_full_symbol(self, symbol: str, df: pd.DataFrame) -> None:
        path = self._full_symbol_path(symbol)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)

    def _persist_feature_columns(self, market_data: Dict[str, pd.DataFrame]) -> None:
        if not market_data:
            return
        columns = [
            col
            for col in next(iter(market_data.values())).columns
            if col not in {"timestamp", "open", "high", "low", "close", "volume"}
        ]
        self._feature_path.write_text("\n".join(columns))

    def _persist_sources(self) -> None:
        if not self._symbol_sources:
            return
        import json

        self._source_path.write_text(json.dumps(self._symbol_sources, indent=2, sort_keys=True))

    def _load_sources(self) -> Dict[str, str]:
        if not self._source_path.exists():
            return {}
        import json

        try:
            return json.loads(self._source_path.read_text())
        except json.JSONDecodeError:
            return {}

    def _load_feature_columns(self) -> List[str]:
        if not self._feature_path.exists():
            return []
        return [line.strip() for line in self._feature_path.read_text().splitlines() if line.strip()]

    def _can_short_circuit(self) -> bool:
        return (
            self.config.use_cache
            and self._sentinel.exists()
            and self._manifest_path.exists()
        )

    def _load_cached_symbols(self, manifest: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        market_data: Dict[str, pd.DataFrame] = {}
        for symbol, group in manifest.groupby("symbol"):
            relative_path = group.iloc[0]["data_path"]
            data_path = (self.cache_dir / relative_path).resolve()
            if not data_path.exists():
                df = self._rebuild_from_pages(symbol)
                if df is not None:
                    df = self._clean_dataframe(df)
                    self._persist_full_symbol(symbol, df)
            else:
                df = pd.read_parquet(data_path)

            if df is None or df.empty:
                continue

            market_data[symbol] = df

        if not market_data:
            raise RuntimeError("Cached manifest present but no symbol data found")

        print("Loaded cached datasets from previous preparation")
        return market_data

    def _rebuild_from_pages(self, symbol: str) -> Optional[pd.DataFrame]:
        frames: List[pd.DataFrame] = []
        page = 0
        while True:
            cache_path = self._page_cache_path(symbol, page)
            if not cache_path.exists():
                break
            frames.append(pd.read_parquet(cache_path))
            page += 1

        if not frames:
            return None

        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _determine_split(
        window_start: datetime,
        window_end: datetime,
        train_range: Tuple[datetime, datetime],
        val_range: Tuple[datetime, datetime],
        test_range: Tuple[datetime, datetime],
    ) -> Optional[str]:
        def within(rng: Tuple[datetime, datetime]) -> bool:
            return window_start >= rng[0] and window_end <= rng[1]

        if within(train_range):
            return "TRAIN"
        if within(val_range):
            return "VAL"
        if within(test_range):
            return "TEST"
        return None


__all__ = ["FullHistoryDataManager", "WindowSpec"]
