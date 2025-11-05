"""Central configuration for full-history data preparation and training."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Tuple


def _resolve_today(value: str) -> str:
    """Resolve placeholders like "<idag>" or "<today>" to today's ISO date."""
    today = date.today().isoformat()
    placeholders = {"<idag>", "<today>", "<TODAY>", "<IDAG>"}
    if value in placeholders:
        return today
    return value


def _resolve_range(range_tuple: Tuple[str, str]) -> Tuple[str, str]:
    start, end = range_tuple
    return _resolve_today(start), _resolve_today(end)


@dataclass
class FullHistoryConfig:
    """Configuration switches controlling data prep and training windows."""

    full_history: bool = True
    start: str = "2019-01-01"
    end: str = field(default_factory=lambda: date.today().isoformat())

    paginate: bool = True
    chunk_size: int = 50_000
    max_bars_per_symbol: Optional[int] = None

    cache_dir: Path = field(default_factory=lambda: Path("./cache"))
    cache_version: str = "v1"
    use_cache: bool = True

    window_size: int = 4096
    window_stride: int = 256
    min_bars_for_windowing: int = 8192

    train_range: Tuple[str, str] = ("2019-01-01", "2023-12-31")
    val_range: Tuple[str, str] = ("2024-01-01", "2024-12-31")
    test_range: Tuple[str, str] = ("2025-01-01", "<idag>")

    sanity_mode: bool = False
    lookback_sanity: int = 1000

    timeframe: str = "1m"
    sources: Tuple[str, ...] = ("polygon", "binance", "yahoo")
    coverage_threshold: float = 0.95

    max_retries: int = 5
    initial_backoff: float = 1.0
    max_backoff: float = 30.0

    force_full_prep: bool = field(
        default_factory=lambda: os.getenv("FORCE_FULL_PREP", "").lower() in {"1", "true", "yes"}
    )

    def __post_init__(self) -> None:
        start_env = os.getenv("START_DATE") or os.getenv("FULL_HISTORY_START_DATE")
        end_env = os.getenv("END_DATE") or os.getenv("FULL_HISTORY_END_DATE")
        cache_dir_env = os.getenv("CACHE_DIR") or os.getenv("FULL_HISTORY_CACHE_DIR")
        cache_version_env = os.getenv("CACHE_VERSION") or os.getenv("FULL_HISTORY_CACHE_VERSION")
        if start_env:
            self.start = start_env
        if end_env:
            self.end = end_env
        if cache_dir_env:
            self.cache_dir = Path(cache_dir_env)
        if cache_version_env:
            self.cache_version = cache_version_env
        self.resolve_dates()

    def resolve_dates(self) -> None:
        """Resolve placeholders in ranges and end date."""
        self.start = _resolve_today(self.start)
        self.end = _resolve_today(self.end)
        self.train_range = _resolve_range(self.train_range)
        self.val_range = _resolve_range(self.val_range)
        self.test_range = _resolve_range(self.test_range)

    @property
    def cache_path(self) -> Path:
        base = Path(self.cache_dir)
        if self.cache_version:
            return base / self.cache_version
        return base

    def ensure_cache_dir(self) -> None:
        self.cache_path.mkdir(parents=True, exist_ok=True)

    def resolved_datetime(self, value: str) -> datetime:
        """Convert config date strings to ``datetime`` objects."""
        return datetime.fromisoformat(_resolve_today(value))

    def resolved_range(self, range_tuple: Tuple[str, str]) -> Tuple[datetime, datetime]:
        start, end = _resolve_range(range_tuple)
        return datetime.fromisoformat(start), datetime.fromisoformat(end)


__all__ = ["FullHistoryConfig"]
