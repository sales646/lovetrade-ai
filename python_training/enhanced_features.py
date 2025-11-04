#!/usr/bin/env python3
"""Enhanced feature engineering with macro and sentiment data"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime

class EnhancedFeatureEngine:
    """Creates comprehensive feature set combining price, macro, and sentiment"""
    
    def __init__(self):
        self.macro_features = []
        self.sentiment_features = []
        
    def add_technical_features(self, ohlcv: np.ndarray, window: int = 20) -> Dict[str, np.ndarray]:
        """
        Add technical indicators
        Input: (T, 6) array [open, high, low, close, volume, transactions]
        """
        close = ohlcv[:, 3]
        high = ohlcv[:, 1]
        low = ohlcv[:, 2]
        volume = ohlcv[:, 4]
        
        features = {}
        
        # Price-based
        features['returns'] = np.diff(close, prepend=close[0]) / (close + 1e-8)
        features['log_returns'] = np.log(close / (np.roll(close, 1) + 1e-8))
        features['log_returns'][0] = 0
        
        # Volatility
        features['volatility'] = pd.Series(features['returns']).rolling(window).std().values
        features['volatility'] = np.nan_to_num(features['volatility'])
        
        # Momentum
        features['rsi'] = self._calculate_rsi(close, window)
        features['macd'] = self._calculate_macd(close)
        
        # Volume
        features['volume_ma'] = pd.Series(volume).rolling(window).mean().values
        features['volume_ratio'] = volume / (features['volume_ma'] + 1e-8)
        features['volume_ratio'] = np.nan_to_num(features['volume_ratio'])
        
        # Trend
        features['sma_20'] = pd.Series(close).rolling(20).mean().values
        features['sma_50'] = pd.Series(close).rolling(50).mean().values
        features['trend'] = (features['sma_20'] / (features['sma_50'] + 1e-8)) - 1
        features['trend'] = np.nan_to_num(features['trend'])
        
        # ATR (Average True Range)
        tr = np.maximum(high - low, 
                       np.maximum(np.abs(high - np.roll(close, 1)),
                                 np.abs(low - np.roll(close, 1))))
        features['atr'] = pd.Series(tr).rolling(window).mean().values
        features['atr'] = np.nan_to_num(features['atr'])
        
        return features
    
    def _calculate_rsi(self, prices: np.ndarray, window: int = 14) -> np.ndarray:
        """Calculate RSI indicator"""
        deltas = np.diff(prices, prepend=prices[0])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = pd.Series(gains).rolling(window).mean().values
        avg_losses = pd.Series(losses).rolling(window).mean().values
        
        rs = avg_gains / (avg_losses + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        return np.nan_to_num(rsi)
    
    def _calculate_macd(self, prices: np.ndarray, 
                       fast: int = 12, slow: int = 26, signal: int = 9) -> np.ndarray:
        """Calculate MACD indicator"""
        ema_fast = pd.Series(prices).ewm(span=fast).mean().values
        ema_slow = pd.Series(prices).ewm(span=slow).mean().values
        macd = ema_fast - ema_slow
        signal_line = pd.Series(macd).ewm(span=signal).mean().values
        
        return macd - signal_line
    
    def add_macro_features(self, macro_state: Dict[str, float]) -> np.ndarray:
        """Convert macro state to feature vector"""
        features = [
            macro_state.get('fed_rate', 0.0) / 10.0,  # Normalize
            macro_state.get('treasury_10y', 0.0) / 10.0,
            macro_state.get('yield_curve', 0.0) / 5.0,
            macro_state.get('unemployment', 0.0) / 20.0,
            macro_state.get('inflation', 0.0) / 10.0,
            macro_state.get('vix', 20.0) / 100.0,
            macro_state.get('recession_signal', 0.0),
        ]
        return np.array(features, dtype=np.float32)
    
    def add_regime_features(self, returns: np.ndarray, window: int = 50) -> Dict[str, np.ndarray]:
        """Detect market regime (bull/bear, high/low volatility)"""
        vol = pd.Series(returns).rolling(window).std().values
        trend = pd.Series(returns).rolling(window).mean().values
        
        # Regime classification
        high_vol = vol > np.nanpercentile(vol, 75)
        bull_market = trend > 0
        
        return {
            'volatility_regime': high_vol.astype(float),
            'trend_regime': bull_market.astype(float),
            'vol_percentile': pd.Series(vol).rank(pct=True).values,
        }
    
    def build_state_vector(self,
                          ohlcv: np.ndarray,
                          position: int,
                          macro_state: Dict[str, float],
                          lookback: int = 20) -> np.ndarray:
        """
        Build comprehensive state vector for RL
        
        Returns: Extended state vector with ~70 features (was 52)
        """
        # Technical features
        tech_features = self.add_technical_features(ohlcv, window=lookback)
        
        # Get most recent values
        current_idx = len(ohlcv) - 1
        
        # Price features (normalized)
        close_prices = ohlcv[:, 3]
        current_price = close_prices[current_idx]
        price_norm = close_prices[-lookback:] / (current_price + 1e-8)
        
        # Technical indicators (most recent)
        technical = [
            tech_features['returns'][current_idx],
            tech_features['volatility'][current_idx],
            tech_features['rsi'][current_idx] / 100.0,
            tech_features['macd'][current_idx],
            tech_features['volume_ratio'][current_idx],
            tech_features['trend'][current_idx],
            tech_features['atr'][current_idx] / (current_price + 1e-8),
        ]
        
        # Regime features
        regime = self.add_regime_features(tech_features['returns'])
        regime_features = [
            regime['volatility_regime'][current_idx],
            regime['trend_regime'][current_idx],
            regime['vol_percentile'][current_idx],
        ]
        
        # Macro features
        macro_features = self.add_macro_features(macro_state)
        
        # Position features
        position_features = [
            float(position),  # Current position
            float(position > 0),  # Is long
            float(position < 0),  # Is short
        ]
        
        # Combine all features
        state = np.concatenate([
            price_norm,  # 20 features
            technical,   # 7 features
            regime_features,  # 3 features
            macro_features,   # 7 features
            position_features # 3 features
        ])
        
        return state.astype(np.float32)


if __name__ == "__main__":
    print("🧪 Testing Enhanced Feature Engineering\n")
    
    # Create sample OHLCV data
    T = 1000
    np.random.seed(42)
    
    prices = 100 + np.cumsum(np.random.randn(T) * 0.02)
    volume = np.random.randint(1000, 10000, T)
    
    ohlcv = np.column_stack([
        prices,  # open
        prices * 1.01,  # high
        prices * 0.99,  # low
        prices,  # close
        volume,  # volume
        np.random.randint(10, 100, T)  # transactions
    ])
    
    # Test feature engine
    engine = EnhancedFeatureEngine()
    
    # Sample macro state
    macro_state = {
        'fed_rate': 5.25,
        'treasury_10y': 4.5,
        'yield_curve': 0.5,
        'unemployment': 3.8,
        'inflation': 3.2,
        'vix': 18.5,
        'recession_signal': 0.0,
    }
    
    # Build state vector
    state = engine.build_state_vector(ohlcv, position=0, macro_state=macro_state)
    
    print(f"📊 State vector shape: {state.shape}")
    print(f"📊 State vector range: [{state.min():.3f}, {state.max():.3f}]")
    print(f"📊 Sample state (first 10 features): {state[:10]}")
