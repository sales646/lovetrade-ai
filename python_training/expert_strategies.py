"""Rule-based expert strategies for generating behavioral cloning labels."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _rolling_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=1).mean()


@dataclass
class StrategyOutput:
    action: np.ndarray
    size_bucket: np.ndarray
    stop_bucket: np.ndarray
    take_bucket: np.ndarray


class VWAPReversionStrategy:
    def generate(self, df: pd.DataFrame) -> StrategyOutput:
        vwap = (df["close"] * df["volume"]).cumsum() / (df["volume"].cumsum() + 1e-9)
        deviation = (df["close"] - vwap) / (vwap + 1e-9)
        action = np.where(deviation < -0.002, 1, np.where(deviation > 0.002, 2, 0))

        atr = _atr(df)
        volatility = (atr / (df["close"] + 1e-9)).clip(0, 0.05)
        size_bucket = np.digitize(volatility, bins=[0.005, 0.01, 0.02, 0.035])
        stop_bucket = np.digitize(volatility * 10000, bins=[10, 20, 35, 50])
        take_bucket = np.digitize(volatility * 16000, bins=[20, 40, 60, 80])

        return StrategyOutput(action=action, size_bucket=size_bucket, stop_bucket=stop_bucket, take_bucket=take_bucket)


class RSIMeanReversionStrategy:
    def generate(self, df: pd.DataFrame) -> StrategyOutput:
        rsi = _rolling_rsi(df["close"]).fillna(50)
        ema_fast = _ema(df["close"], span=8)
        ema_slow = _ema(df["close"], span=21)
        trend = ema_fast - ema_slow

        long_signal = (rsi < 35) & (trend > 0)
        short_signal = (rsi > 65) & (trend < 0)
        action = np.where(long_signal, 1, np.where(short_signal, 2, 0))

        atr = _atr(df)
        normalized_rsi = (rsi - 50).abs() / 50
        size_bucket = np.digitize(normalized_rsi, bins=[0.1, 0.2, 0.35, 0.5])
        stop_bucket = np.digitize((atr / (df["close"] + 1e-9)) * 10000, bins=[8, 16, 24, 32])
        take_bucket = np.digitize((atr / (df["close"] + 1e-9)) * 14000, bins=[16, 32, 48, 64])

        return StrategyOutput(action=action, size_bucket=size_bucket, stop_bucket=stop_bucket, take_bucket=take_bucket)


class TrendPullbackStrategy:
    def generate(self, df: pd.DataFrame) -> StrategyOutput:
        ema50 = _ema(df["close"], span=50)
        ema200 = _ema(df["close"], span=200)
        ema20 = _ema(df["close"], span=20)
        trend_state = ema50 - ema200
        pullback = (df["close"] - ema20) / (ema20 + 1e-9)

        long_signal = (trend_state > 0) & (pullback < -0.01)
        short_signal = (trend_state < 0) & (pullback > 0.01)
        action = np.where(long_signal, 1, np.where(short_signal, 2, 0))

        trend_strength = trend_state.abs() / (df["close"] + 1e-9)
        size_bucket = np.digitize(trend_strength, bins=[0.002, 0.004, 0.006, 0.008])
        stop_bucket = np.digitize(trend_strength * 10000, bins=[8, 16, 32, 48])
        take_bucket = np.digitize(trend_strength * 14000, bins=[16, 32, 56, 80])

        return StrategyOutput(action=action, size_bucket=size_bucket, stop_bucket=stop_bucket, take_bucket=take_bucket)


class ExpertEnsemble:
    """Combine multiple strategies with tie-breaking rules."""

    def __init__(self) -> None:
        self.strategies = [
            VWAPReversionStrategy(),
            RSIMeanReversionStrategy(),
            TrendPullbackStrategy(),
        ]

    def generate(self, df: pd.DataFrame) -> StrategyOutput:
        actions: List[np.ndarray] = []
        sizes: List[np.ndarray] = []
        stops: List[np.ndarray] = []
        takes: List[np.ndarray] = []

        for strategy in self.strategies:
            output = strategy.generate(df)
            actions.append(output.action)
            sizes.append(output.size_bucket)
            stops.append(output.stop_bucket)
            takes.append(output.take_bucket)

        stacked_actions = np.stack(actions, axis=0)
        stacked_sizes = np.stack(sizes, axis=0)
        stacked_stops = np.stack(stops, axis=0)
        stacked_takes = np.stack(takes, axis=0)

        action = np.apply_along_axis(lambda x: np.bincount(x, minlength=3).argmax(), axis=0, arr=stacked_actions)
        size_bucket = np.median(stacked_sizes, axis=0).astype(int)
        stop_bucket = np.median(stacked_stops, axis=0).astype(int)
        take_bucket = np.median(stacked_takes, axis=0).astype(int)

        return StrategyOutput(
            action=action,
            size_bucket=size_bucket,
            stop_bucket=stop_bucket,
            take_bucket=take_bucket,
        )


def generate_expert_labels(df: pd.DataFrame) -> StrategyOutput:
    ensemble = ExpertEnsemble()
    return ensemble.generate(df)


__all__ = ["generate_expert_labels", "StrategyOutput"]
