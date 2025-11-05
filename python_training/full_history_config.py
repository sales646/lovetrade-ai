"""Central configuration for full-history data preparation and training."""
from __future__ import annotations

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

    def resolve_dates(self) -> None:
        """Resolve placeholders in ranges and end date."""
        self.end = _resolve_today(self.end)
        self.train_range = _resolve_range(self.train_range)
        self.val_range = _resolve_range(self.val_range)
        self.test_range = _resolve_range(self.test_range)

    @property
    def cache_path(self) -> Path:
        return Path(self.cache_dir)

    def ensure_cache_dir(self) -> None:
        self.cache_path.mkdir(parents=True, exist_ok=True)

    def resolved_datetime(self, value: str) -> datetime:
        """Convert config date strings to ``datetime`` objects."""
        return datetime.fromisoformat(_resolve_today(value))

    def resolved_range(self, range_tuple: Tuple[str, str]) -> Tuple[datetime, datetime]:
        start, end = _resolve_range(range_tuple)
        return datetime.fromisoformat(start), datetime.fromisoformat(end)


__all__ = ["FullHistoryConfig"]
