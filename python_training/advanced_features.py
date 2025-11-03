#!/usr/bin/env python3
"""
Advanced Feature Engineering for Trading
Order flow, bid-ask dynamics, options data, and more
"""

import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class AdvancedFeatureExtractor:
    """
    Extract advanced market microstructure and derivatives features
    """
    
    def __init__(self):
        self.feature_cache = {}
    
    def extract_order_flow_features(
        self,
        bars: List[Dict],
        window: int = 20
    ) -> Dict[str, float]:
        """
        Extract order flow imbalance features
        
        Order flow imbalance = (Buy Volume - Sell Volume) / Total Volume
        Inferred from price movement and volume
        """
        if len(bars) < 2:
            return {
                'order_flow_imbalance': 0.0,
                'cumulative_delta': 0.0,
                'delta_zscore': 0.0
            }
        
        # Infer buy/sell volume from price changes
        deltas = []
        for i in range(1, min(len(bars), window + 1)):
            curr = bars[-i]
            prev = bars[-(i+1)]
            
            # If close > open, assume more buying
            price_change = curr['close'] - prev['close']
            volume = curr['volume']
            
            # Delta = volume * sign(price_change)
            delta = volume * np.sign(price_change)
            deltas.append(delta)
        
        if not deltas:
            return {
                'order_flow_imbalance': 0.0,
                'cumulative_delta': 0.0,
                'delta_zscore': 0.0
            }
        
        cumulative_delta = sum(deltas)
        total_volume = sum(bars[-i]['volume'] for i in range(1, min(len(bars), window + 1)))
        
        order_flow_imbalance = cumulative_delta / (total_volume + 1e-8)
        
        # Z-score of delta
        delta_mean = np.mean(deltas)
        delta_std = np.std(deltas) + 1e-8
        delta_zscore = (deltas[0] - delta_mean) / delta_std
        
        return {
            'order_flow_imbalance': order_flow_imbalance,
            'cumulative_delta': cumulative_delta / 1e6,  # Normalize
            'delta_zscore': delta_zscore
        }
    
    def extract_bid_ask_features(
        self,
        bars: List[Dict],
        typical_spread_pct: float = 0.001  # 0.1% typical spread
    ) -> Dict[str, float]:
        """
        Estimate bid-ask spread dynamics
        
        Uses high-low range as proxy for spread
        """
        if not bars:
            return {
                'estimated_spread_pct': typical_spread_pct,
                'spread_volatility': 0.0,
                'spread_zscore': 0.0
            }
        
        # Use high-low range as spread proxy
        recent_bars = bars[-20:]
        spreads = []
        
        for bar in recent_bars:
            high = bar['high']
            low = bar['low']
            mid = (high + low) / 2
            
            if mid > 0:
                spread_pct = (high - low) / mid
                spreads.append(spread_pct)
        
        if not spreads:
            return {
                'estimated_spread_pct': typical_spread_pct,
                'spread_volatility': 0.0,
                'spread_zscore': 0.0
            }
        
        avg_spread = np.mean(spreads)
        spread_vol = np.std(spreads)
        
        current_spread = spreads[-1]
        spread_zscore = (current_spread - avg_spread) / (spread_vol + 1e-8)
        
        return {
            'estimated_spread_pct': avg_spread,
            'spread_volatility': spread_vol,
            'spread_zscore': spread_zscore
        }
    
    def extract_volume_profile_features(
        self,
        bars: List[Dict],
        n_bins: int = 10
    ) -> Dict[str, float]:
        """
        Intraday volume profile features
        
        Analyzes where volume is concentrated relative to price
        """
        if len(bars) < 10:
            return {
                'volume_at_price_ratio': 0.5,
                'poc_distance_pct': 0.0,  # Point of Control distance
                'value_area_position': 0.5
            }
        
        recent_bars = bars[-50:]
        
        # Get price range
        prices = [bar['close'] for bar in recent_bars]
        volumes = [bar['volume'] for bar in recent_bars]
        
        price_min = min(prices)
        price_max = max(prices)
        price_range = price_max - price_min
        
        if price_range == 0:
            return {
                'volume_at_price_ratio': 0.5,
                'poc_distance_pct': 0.0,
                'value_area_position': 0.5
            }
        
        # Create volume profile histogram
        bins = np.linspace(price_min, price_max, n_bins + 1)
        volume_profile = np.zeros(n_bins)
        
        for price, volume in zip(prices, volumes):
            bin_idx = min(int((price - price_min) / price_range * n_bins), n_bins - 1)
            volume_profile[bin_idx] += volume
        
        # Point of Control (POC) = price level with most volume
        poc_idx = np.argmax(volume_profile)
        poc_price = price_min + (poc_idx + 0.5) * (price_range / n_bins)
        
        current_price = prices[-1]
        poc_distance_pct = (current_price - poc_price) / current_price
        
        # Value area position (where is current price in volume distribution)
        cumulative_volume = np.cumsum(volume_profile)
        total_volume = cumulative_volume[-1]
        
        current_bin = min(int((current_price - price_min) / price_range * n_bins), n_bins - 1)
        value_area_position = cumulative_volume[current_bin] / (total_volume + 1e-8)
        
        # Volume at current price level ratio
        volume_at_price_ratio = volume_profile[current_bin] / (np.max(volume_profile) + 1e-8)
        
        return {
            'volume_at_price_ratio': volume_at_price_ratio,
            'poc_distance_pct': poc_distance_pct,
            'value_area_position': value_area_position
        }
    
    def extract_options_features(
        self,
        symbol: str,
        current_price: float,
        options_data: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Extract options market features (if available)
        
        In practice, would fetch from options API
        For now, returns defaults
        """
        # Placeholder - in production would call options data API
        if options_data is None:
            return {
                'implied_volatility_rank': 0.5,  # 0-1 scale
                'put_call_ratio': 1.0,            # Neutral
                'options_skew': 0.0,              # No skew
                'gamma_exposure': 0.0             # Net gamma
            }
        
        return {
            'implied_volatility_rank': options_data.get('iv_rank', 0.5),
            'put_call_ratio': options_data.get('pcr', 1.0),
            'options_skew': options_data.get('skew', 0.0),
            'gamma_exposure': options_data.get('gamma', 0.0)
        }
    
    def extract_sector_rotation_features(
        self,
        symbol: str,
        sector_performances: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Sector rotation and relative strength features
        """
        # Map symbols to sectors
        sector_map = {
            'AAPL': 'Tech', 'MSFT': 'Tech', 'GOOGL': 'Tech', 'NVDA': 'Tech',
            'JPM': 'Finance', 'BAC': 'Finance', 'GS': 'Finance',
            'XOM': 'Energy', 'CVX': 'Energy',
            # ... would have full mapping
        }
        
        symbol_sector = sector_map.get(symbol, 'Unknown')
        
        if sector_performances is None:
            return {
                'sector_relative_strength': 0.0,
                'sector_momentum': 0.0,
                'sector_rank': 0.5
            }
        
        sector_perf = sector_performances.get(symbol_sector, 0.0)
        all_perfs = list(sector_performances.values())
        
        # Relative strength vs market
        market_perf = np.mean(all_perfs)
        sector_relative_strength = sector_perf - market_perf
        
        # Sector rank (0 = worst, 1 = best)
        sorted_perfs = sorted(all_perfs)
        sector_rank = sorted_perfs.index(sector_perf) / len(sorted_perfs) if all_perfs else 0.5
        
        return {
            'sector_relative_strength': sector_relative_strength,
            'sector_momentum': sector_perf,
            'sector_rank': sector_rank
        }
    
    def extract_all_advanced_features(
        self,
        bars: List[Dict],
        symbol: str
    ) -> Dict[str, float]:
        """
        Extract all advanced features in one call
        """
        features = {}
        
        # Order flow
        features.update(self.extract_order_flow_features(bars))
        
        # Bid-ask
        features.update(self.extract_bid_ask_features(bars))
        
        # Volume profile
        features.update(self.extract_volume_profile_features(bars))
        
        # Options (placeholder)
        current_price = bars[-1]['close'] if bars else 100.0
        features.update(self.extract_options_features(symbol, current_price))
        
        # Sector rotation (placeholder)
        features.update(self.extract_sector_rotation_features(symbol))
        
        return features
