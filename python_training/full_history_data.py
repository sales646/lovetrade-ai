"""Full-history data orchestration with caching, pagination and window manifest."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import requests

from .full_history_config import FullHistoryConfig


@dataclass
class WindowSpec:
    symbol: str
    split: str
    start_idx: int
    end_idx: int
    start_ts: datetime
    end_ts: datetime
    data_path: str


class FullHistoryDataManager:
    """Handles fetching, caching and manifest generation for market data."""

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
        self.stocks = list(stocks or [])
        self.crypto = list(crypto or [])

        self._sentinel = self.cache_dir / ".prep_done"
        self._manifest_path = self.cache_dir / "env_manifest.parquet"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def prepare(self) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
        """Prepare full-history data and return market data + manifest."""
        if self._can_short_circuit():
            manifest = pd.read_parquet(self._manifest_path)
            market_data = self._load_cached_symbols(manifest)
            return market_data, manifest

        tickers = sorted(set(self.stocks + self.crypto))
        if self.config.sanity_mode:
            tickers = tickers[:3]
            self.config.window_size = min(self.config.window_size, self.config.lookback_sanity)
            self.config.min_bars_for_windowing = min(
                self.config.min_bars_for_windowing,
                self.config.lookback_sanity,
            )

        if not tickers:
            raise ValueError("No tickers configured for data preparation")

        print(f"📥 Fetching data for {len(tickers)} tickers ({self.config.timeframe})")

        market_data: Dict[str, pd.DataFrame] = {}
        crypto_set = set(self.crypto)

        for symbol in tickers:
            if symbol in crypto_set or symbol.upper().endswith("USDT"):
                df = self._fetch_binance_symbol(symbol)
            else:
                df = self._fetch_polygon_symbol(symbol)

            if df is None or df.empty:
                continue

            cleaned = self._clean_dataframe(df)
            if cleaned.empty:
                continue

            if len(cleaned) < self.config.window_size:
                continue

            market_data[symbol] = cleaned
            self._persist_full_symbol(symbol, cleaned)

        if not market_data:
            raise RuntimeError("No market data available after fetching")

        manifest = self._build_manifest(market_data)
        manifest.to_parquet(self._manifest_path, index=False)
        self._sentinel.touch()

        return market_data, manifest

    # ------------------------------------------------------------------
    # Fetch helpers
    # ------------------------------------------------------------------
    def _fetch_polygon_symbol(self, symbol: str) -> Optional[pd.DataFrame]:
        api_key = os.getenv("POLYGON_API_KEY")
        if not api_key:
            print(f"⚠️  POLYGON_API_KEY missing, skipping {symbol}")
            return None

        start = self.config.start
        end = self.config.end
        base_url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/{start}/{end}"

        all_frames: List[pd.DataFrame] = []
        cursor: Optional[str] = None
        page = 0

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
                    response = requests.get(base_url, params=params, timeout=30)
                    response.raise_for_status()
                except Exception as exc:
                    print(f"⚠️  Polygon fetch failed for {symbol}: {exc}")
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

                cursor = payload.get("next_page_token")
                if not cursor:
                    next_url = payload.get("next_url")
                    if next_url and "cursor=" in next_url:
                        cursor = next_url.split("cursor=")[-1]

            if df is None or df.empty:
                break

            all_frames.append(df)
            page += 1

            if not self.config.paginate or not cursor:
                break

            if self.config.sanity_mode:
                break

        if not all_frames:
            return None

        combined = pd.concat(all_frames, ignore_index=True)
        return combined

    def _fetch_binance_symbol(self, symbol: str) -> Optional[pd.DataFrame]:
        start_dt = self.config.resolved_datetime(self.config.start)
        end_dt = self.config.resolved_datetime(self.config.end)

        interval_minutes = 1
        limit = min(self.config.chunk_size, 1000)
        all_frames: List[pd.DataFrame] = []
        page = 0
        current = start_dt

        while current < end_dt:
            cache_path = self._page_cache_path(symbol, page)
            if self.config.use_cache and cache_path.exists():
                df = pd.read_parquet(cache_path)
            else:
                next_time = current + timedelta(minutes=limit * interval_minutes)
                params = {
                    "symbol": symbol,
                    "interval": "1m",
                    "limit": limit,
                    "startTime": int(current.timestamp() * 1000),
                    "endTime": int(min(next_time, end_dt).timestamp() * 1000),
                }
                try:
                    response = requests.get("https://api.binance.com/api/v3/klines", params=params, timeout=30)
                    response.raise_for_status()
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
                df = df[["open_time", "open", "high", "low", "close", "volume"]]
                df = df.rename(columns={"open_time": "timestamp"})
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

                cache_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(cache_path, index=False)

            if df is None or df.empty:
                break

            all_frames.append(df)
            page += 1
            current = current + timedelta(minutes=limit * interval_minutes)

            if not self.config.paginate:
                break

            if self.config.sanity_mode:
                break

        if not all_frames:
            return None

        combined = pd.concat(all_frames, ignore_index=True)
        return combined

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _page_cache_path(self, symbol: str, page: int) -> Path:
        filename = f"{symbol}_{self.config.timeframe}_{self.config.start}_{self.config.end}_p{page:04d}.parquet"
        return self.cache_dir / filename

    def _full_symbol_path(self, symbol: str) -> Path:
        filename = f"{symbol}_{self.config.timeframe}_{self.config.start}_{self.config.end}_full.parquet"
        return self.cache_dir / filename

    def _persist_full_symbol(self, symbol: str, df: pd.DataFrame) -> None:
        path = self._full_symbol_path(symbol)
        df.to_parquet(path, index=False)

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        keep_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        df = df.copy()

        # Normalize column names
        rename_map = {col: col.lower() for col in df.columns}
        df = df.rename(columns=rename_map)

        if "t" in df.columns and "timestamp" not in df.columns:
            df["timestamp"] = pd.to_datetime(df["t"], unit="ms")
        if "time" in df.columns and "timestamp" not in df.columns:
            df["timestamp"] = pd.to_datetime(df["time"])

        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        df = df.sort_values("timestamp").drop_duplicates("timestamp")

        start_dt = self.config.resolved_datetime(self.config.start)
        end_dt = self.config.resolved_datetime(self.config.end)
        df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)]

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=keep_cols)
        df = df[keep_cols]

        if self.config.max_bars_per_symbol and len(df) > self.config.max_bars_per_symbol:
            df = df.tail(self.config.max_bars_per_symbol)

        if self.config.sanity_mode:
            df = df.tail(self.config.lookback_sanity)

        return df.reset_index(drop=True)

    def _build_manifest(self, market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        specs: List[WindowSpec] = []

        train_start, train_end = self.config.resolved_range(self.config.train_range)
        val_start, val_end = self.config.resolved_range(self.config.val_range)
        test_start, test_end = self.config.resolved_range(self.config.test_range)

        for symbol, df in market_data.items():
            if len(df) < self.config.min_bars_for_windowing:
                continue

            timestamps = df["timestamp"].tolist()
            total = len(df)
            data_path = str(self._full_symbol_path(symbol))

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

                specs.append(
                    WindowSpec(
                        symbol=symbol,
                        split=split,
                        start_idx=start_idx,
                        end_idx=end_idx,
                        start_ts=window_start,
                        end_ts=window_end,
                        data_path=data_path,
                    )
                )

        if not specs:
            raise RuntimeError("No window specs generated - check data availability")

        manifest = pd.DataFrame([spec.__dict__ for spec in specs])
        return manifest

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

    def _can_short_circuit(self) -> bool:
        return self.config.use_cache and self._sentinel.exists() and self._manifest_path.exists()

    def _load_cached_symbols(self, manifest: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        market_data: Dict[str, pd.DataFrame] = {}
        for symbol, group in manifest.groupby("symbol"):
            data_path = Path(group.iloc[0]["data_path"])
            if not data_path.is_absolute():
                data_path = (self.cache_dir / data_path).resolve()

            if data_path.exists():
                df = pd.read_parquet(data_path)
            else:
                df = self._rebuild_from_pages(symbol)
                if df is not None:
                    df.to_parquet(data_path, index=False)

            if df is None or df.empty:
                continue

            cleaned = self._clean_dataframe(df)
            if cleaned.empty:
                continue

            market_data[symbol] = cleaned

        if not market_data:
            raise RuntimeError("Cached manifest present but no symbol data found")

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


__all__ = ["FullHistoryDataManager", "WindowSpec"]
