"""Utilities for managing sequential progress bars per training phase.

This module was reintroduced as part of the BC→PPO pipeline refresh so the
training CLI can be re-run without regressing into multi-bar spam. Keeping a
succinct explanation here helps future replays of the rollout understand why
we centralize phase progress management in a reusable helper.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock
from typing import Dict, Iterator, Optional

from tqdm import tqdm


@dataclass
class _ProgressState:
    bar: tqdm
    total: Optional[int]


class PhaseProgress:
    """Centralized manager that ensures only one bar per phase."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._bars: Dict[str, _ProgressState] = {}

    def start(self, phase: str, total: Optional[int] = None, unit: Optional[str] = None) -> tqdm:
        with self._lock:
            if phase in self._bars:
                bar = self._bars[phase].bar
                if total is not None and bar.total != total:
                    bar.total = total
                return bar
            bar = tqdm(total=total, desc=phase, unit=unit, leave=False)
            self._bars[phase] = _ProgressState(bar=bar, total=total)
            return bar

    def update(self, phase: str, n: int = 1) -> None:
        with self._lock:
            state = self._bars.get(phase)
            if state is None:
                return
            bar = state.bar
            if bar.total is not None:
                remaining = max(bar.total - bar.n, 0)
                increment = min(n, remaining)
            else:
                increment = n
            if increment > 0:
                bar.update(increment)

    def close(self, phase: str) -> None:
        with self._lock:
            state = self._bars.pop(phase, None)
            if state:
                state.bar.close()

    def close_all(self) -> None:
        with self._lock:
            phases = list(self._bars.keys())
        for phase in phases:
            self.close(phase)

    @contextmanager
    def track(
        self,
        phase: str,
        total: Optional[int] = None,
        unit: Optional[str] = None,
    ) -> Iterator[tqdm]:
        bar = self.start(phase, total=total, unit=unit)
        try:
            yield bar
        finally:
            self.close(phase)


__all__ = ["PhaseProgress"]
