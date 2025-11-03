#!/usr/bin/env python3
"""
Robustness Testing Suite for Trading Policies
Stress tests policies against various market scenarios
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Callable
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class StressScenario(Enum):
    """Different stress test scenarios"""
    MARKET_CRASH = "market_crash"  # -30% rapid decline
    FLASH_CRASH = "flash_crash"    # -10% in minutes, then recovery
    VOLATILITY_SPIKE = "volatility_spike"  # 3x normal volatility
    LOW_LIQUIDITY = "low_liquidity"  # Wide spreads, high slippage
    TRENDING_BULL = "trending_bull"  # Sustained uptrend
    TRENDING_BEAR = "trending_bear"  # Sustained downtrend
    SIDEWAYS_CHOPPY = "sideways_choppy"  # Range-bound, whipsaw
    GAP_UP = "gap_up"  # +5% overnight gap
    GAP_DOWN = "gap_down"  # -5% overnight gap


@dataclass
class StressTestResult:
    """Results from a stress test"""
    scenario: StressScenario
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    total_trades: int
    avg_trade_duration: float
    passed: bool
    failure_reason: str = None


class RobustnessTester:
    """
    Comprehensive robustness testing for trading policies
    """
    
    def __init__(
        self,
        min_sharpe: float = 0.5,
        max_drawdown: float = 0.15,
        min_win_rate: float = 0.45,
        min_trades: int = 10
    ):
        """
        Args:
            min_sharpe: Minimum acceptable Sharpe ratio
            max_drawdown: Maximum acceptable drawdown (as fraction)
            min_win_rate: Minimum acceptable win rate
            min_trades: Minimum number of trades to be valid
        """
        self.min_sharpe = min_sharpe
        self.max_drawdown = max_drawdown
        self.min_win_rate = min_win_rate
        self.min_trades = min_trades
    
    def generate_stress_data(
        self,
        scenario: StressScenario,
        base_data: np.ndarray,
        n_samples: int = 1000
    ) -> np.ndarray:
        """
        Generate synthetic price data for stress scenario
        
        Args:
            scenario: Type of stress test
            base_data: Original price data (OHLCV)
            n_samples: Number of samples to generate
        
        Returns:
            Modified OHLCV data representing stress scenario
        """
        if scenario == StressScenario.MARKET_CRASH:
            return self._simulate_market_crash(base_data, crash_pct=-0.30, duration=50)
        
        elif scenario == StressScenario.FLASH_CRASH:
            return self._simulate_flash_crash(base_data, crash_pct=-0.10, recovery_duration=20)
        
        elif scenario == StressScenario.VOLATILITY_SPIKE:
            return self._simulate_volatility_spike(base_data, vol_multiplier=3.0)
        
        elif scenario == StressScenario.LOW_LIQUIDITY:
            return self._simulate_low_liquidity(base_data, spread_multiplier=5.0)
        
        elif scenario == StressScenario.TRENDING_BULL:
            return self._simulate_trend(base_data, direction=1, strength=0.02)
        
        elif scenario == StressScenario.TRENDING_BEAR:
            return self._simulate_trend(base_data, direction=-1, strength=0.02)
        
        elif scenario == StressScenario.SIDEWAYS_CHOPPY:
            return self._simulate_choppy_range(base_data, range_pct=0.05)
        
        elif scenario == StressScenario.GAP_UP:
            return self._simulate_gap(base_data, gap_pct=0.05)
        
        elif scenario == StressScenario.GAP_DOWN:
            return self._simulate_gap(base_data, gap_pct=-0.05)
        
        else:
            raise ValueError(f"Unknown scenario: {scenario}")
    
    def _simulate_market_crash(
        self,
        data: np.ndarray,
        crash_pct: float = -0.30,
        duration: int = 50
    ) -> np.ndarray:
        """Simulate gradual market crash"""
        crash_data = data.copy()
        n_samples = len(crash_data)
        
        # Apply exponential decay to simulate accelerating decline
        crash_multipliers = np.exp(np.linspace(0, np.log(1 + crash_pct), duration))
        
        for i in range(min(duration, n_samples)):
            multiplier = crash_multipliers[i]
            crash_data[i, :4] *= multiplier  # Apply to OHLC
        
        # Continued depressed prices after crash
        if duration < n_samples:
            final_multiplier = 1 + crash_pct
            crash_data[duration:, :4] *= final_multiplier
        
        return crash_data
    
    def _simulate_flash_crash(
        self,
        data: np.ndarray,
        crash_pct: float = -0.10,
        recovery_duration: int = 20
    ) -> np.ndarray:
        """Simulate sudden crash with rapid recovery"""
        crash_data = data.copy()
        
        # Sudden drop
        crash_data[0, :4] *= (1 + crash_pct)
        
        # Gradual recovery
        recovery_multipliers = np.linspace(1 + crash_pct, 1.0, recovery_duration)
        for i in range(min(recovery_duration, len(crash_data))):
            crash_data[i, :4] *= recovery_multipliers[i]
        
        return crash_data
    
    def _simulate_volatility_spike(
        self,
        data: np.ndarray,
        vol_multiplier: float = 3.0
    ) -> np.ndarray:
        """Increase volatility while preserving trend"""
        vol_data = data.copy()
        
        # Calculate returns
        returns = np.diff(vol_data[:, 3]) / vol_data[:-1, 3]  # Close-to-close returns
        
        # Amplify returns
        amplified_returns = returns * vol_multiplier
        
        # Reconstruct prices
        vol_data[1:, 3] = vol_data[0, 3] * np.cumprod(1 + amplified_returns)
        
        # Adjust OHLV proportionally
        price_ratios = vol_data[1:, 3] / data[1:, 3]
        for i in range(3):  # O, H, L
            vol_data[1:, i] = data[1:, i] * price_ratios
        
        return vol_data
    
    def _simulate_low_liquidity(
        self,
        data: np.ndarray,
        spread_multiplier: float = 5.0
    ) -> np.ndarray:
        """Simulate low liquidity with wider spreads"""
        liq_data = data.copy()
        
        # Widen high-low range
        mid = (liq_data[:, 1] + liq_data[:, 2]) / 2  # Mid price
        range_size = (liq_data[:, 1] - liq_data[:, 2]) * spread_multiplier
        
        liq_data[:, 1] = mid + range_size / 2  # New high
        liq_data[:, 2] = mid - range_size / 2  # New low
        
        # Reduce volume
        liq_data[:, 4] /= spread_multiplier
        
        return liq_data
    
    def _simulate_trend(
        self,
        data: np.ndarray,
        direction: int = 1,
        strength: float = 0.02
    ) -> np.ndarray:
        """Simulate sustained trend (bull or bear)"""
        trend_data = data.copy()
        
        # Apply compound trend
        trend_multipliers = (1 + direction * strength) ** np.arange(len(trend_data))
        
        for i in range(4):  # OHLC
            trend_data[:, i] *= trend_multipliers[:, np.newaxis]
        
        return trend_data
    
    def _simulate_choppy_range(
        self,
        data: np.ndarray,
        range_pct: float = 0.05
    ) -> np.ndarray:
        """Simulate range-bound choppy market"""
        choppy_data = data.copy()
        
        # Oscillate around mean with random noise
        mean_price = np.mean(choppy_data[:, 3])
        oscillation = np.sin(np.linspace(0, 10 * np.pi, len(choppy_data))) * range_pct * mean_price
        noise = np.random.normal(0, 0.01 * mean_price, len(choppy_data))
        
        choppy_data[:, 3] = mean_price + oscillation + noise
        
        # Adjust OHLV
        for i in range(3):
            choppy_data[:, i] = choppy_data[:, 3] * (data[:, i] / data[:, 3])
        
        return choppy_data
    
    def _simulate_gap(
        self,
        data: np.ndarray,
        gap_pct: float = 0.05
    ) -> np.ndarray:
        """Simulate price gap at open"""
        gap_data = data.copy()
        
        # Apply gap to first bar
        gap_data[0, :4] *= (1 + gap_pct)
        
        return gap_data
    
    def run_stress_test(
        self,
        policy: Callable,
        scenario: StressScenario,
        base_data: np.ndarray,
        n_episodes: int = 5
    ) -> StressTestResult:
        """
        Run stress test on a policy
        
        Args:
            policy: Trading policy (function that takes state and returns action)
            scenario: Stress scenario to test
            base_data: Base OHLCV data
            n_episodes: Number of test episodes
        
        Returns:
            StressTestResult with performance metrics
        """
        logger.info(f"Running stress test: {scenario.value}")
        
        # Generate stress data
        stress_data = self.generate_stress_data(scenario, base_data)
        
        # Run multiple episodes and aggregate results
        episode_results = []
        
        for ep in range(n_episodes):
            result = self._run_single_episode(policy, stress_data)
            episode_results.append(result)
        
        # Aggregate metrics
        avg_return = np.mean([r['total_return'] for r in episode_results])
        max_dd = np.max([r['max_drawdown'] for r in episode_results])
        avg_sharpe = np.mean([r['sharpe_ratio'] for r in episode_results])
        avg_win_rate = np.mean([r['win_rate'] for r in episode_results])
        total_trades = sum([r['total_trades'] for r in episode_results])
        avg_duration = np.mean([r['avg_trade_duration'] for r in episode_results])
        
        # Check pass/fail criteria
        passed = True
        failure_reason = None
        
        if avg_sharpe < self.min_sharpe:
            passed = False
            failure_reason = f"Sharpe ratio {avg_sharpe:.2f} < {self.min_sharpe}"
        elif max_dd > self.max_drawdown:
            passed = False
            failure_reason = f"Drawdown {max_dd:.2%} > {self.max_drawdown:.2%}"
        elif avg_win_rate < self.min_win_rate:
            passed = False
            failure_reason = f"Win rate {avg_win_rate:.2%} < {self.min_win_rate:.2%}"
        elif total_trades < self.min_trades:
            passed = False
            failure_reason = f"Total trades {total_trades} < {self.min_trades}"
        
        return StressTestResult(
            scenario=scenario,
            total_return=avg_return,
            max_drawdown=max_dd,
            sharpe_ratio=avg_sharpe,
            win_rate=avg_win_rate,
            total_trades=total_trades,
            avg_trade_duration=avg_duration,
            passed=passed,
            failure_reason=failure_reason
        )
    
    def _run_single_episode(
        self,
        policy: Callable,
        data: np.ndarray
    ) -> Dict:
        """Run single trading episode and return metrics"""
        # Simplified trading simulation
        # In practice, this would use the full trading environment
        
        equity = 1.0
        peak_equity = 1.0
        max_drawdown = 0.0
        returns = []
        trades = []
        
        position = 0  # -1: short, 0: neutral, 1: long
        
        for i in range(len(data) - 1):
            # Get action from policy
            state = data[i]  # Simplified state
            action = policy(state)  # 0: sell, 1: hold, 2: buy
            
            # Execute trade
            if action == 2 and position <= 0:  # Buy
                position = 1
                trades.append({'entry': data[i, 3], 'type': 'long'})
            elif action == 0 and position >= 0:  # Sell
                position = -1
                trades.append({'entry': data[i, 3], 'type': 'short'})
            
            # Calculate return
            if position != 0:
                price_return = (data[i+1, 3] - data[i, 3]) / data[i, 3]
                trade_return = price_return * position
                returns.append(trade_return)
                
                equity *= (1 + trade_return)
                peak_equity = max(peak_equity, equity)
                drawdown = (peak_equity - equity) / peak_equity
                max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate metrics
        total_return = equity - 1.0
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252) if returns else 0
        
        winning_trades = sum(1 for r in returns if r > 0)
        win_rate = winning_trades / len(returns) if returns else 0
        
        return {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'win_rate': win_rate,
            'total_trades': len(trades),
            'avg_trade_duration': len(data) / max(len(trades), 1)
        }
    
    def run_full_suite(
        self,
        policy: Callable,
        base_data: np.ndarray
    ) -> Dict[StressScenario, StressTestResult]:
        """Run all stress tests and return comprehensive report"""
        results = {}
        
        for scenario in StressScenario:
            result = self.run_stress_test(policy, scenario, base_data)
            results[scenario] = result
            
            status = "✓ PASSED" if result.passed else "✗ FAILED"
            logger.info(
                f"{status} {scenario.value}: "
                f"Return={result.total_return:.2%}, DD={result.max_drawdown:.2%}, "
                f"Sharpe={result.sharpe_ratio:.2f}, WinRate={result.win_rate:.2%}"
            )
            
            if not result.passed:
                logger.warning(f"  Reason: {result.failure_reason}")
        
        # Summary
        passed_count = sum(1 for r in results.values() if r.passed)
        total_count = len(results)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"ROBUSTNESS TEST SUMMARY: {passed_count}/{total_count} tests passed")
        logger.info(f"{'='*60}")
        
        return results
