#!/usr/bin/env python3
"""
Ensemble of Specialized Trading Policies
Multiple expert agents for different market regimes and trading styles
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class TradingStyle(Enum):
    """Different trading strategy styles"""
    TREND_FOLLOWER = "trend_follower"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"


class MarketRegime(Enum):
    """Different market conditions"""
    BULL_TRENDING = "bull_trending"
    BEAR_TRENDING = "bear_trending"
    CHOPPY_RANGE = "choppy_range"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


class SpecializedPolicy(nn.Module):
    """
    A specialized trading policy for a specific style or regime
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int = 3,  # SELL, HOLD, BUY
        style: TradingStyle = TradingStyle.TREND_FOLLOWER,
        dropout: float = 0.1
    ):
        super().__init__()
        self.style = style
        
        # Build network
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class RegimeDetector(nn.Module):
    """
    Detects current market regime from features
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        n_regimes: int = 5
    ):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_regimes),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns probability distribution over regimes"""
        return self.network(x)
    
    def detect_regime(self, x: torch.Tensor) -> int:
        """Returns most likely regime index"""
        probs = self.forward(x)
        return torch.argmax(probs, dim=-1).item()


class PolicyEnsemble:
    """
    Ensemble of specialized policies with intelligent routing
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 256],
        device: str = 'cuda'
    ):
        self.device = device
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # Create specialized policies
        self.policies = {
            TradingStyle.TREND_FOLLOWER: SpecializedPolicy(
                input_dim, hidden_dims, style=TradingStyle.TREND_FOLLOWER
            ).to(device),
            
            TradingStyle.MEAN_REVERSION: SpecializedPolicy(
                input_dim, hidden_dims, style=TradingStyle.MEAN_REVERSION
            ).to(device),
            
            TradingStyle.BREAKOUT: SpecializedPolicy(
                input_dim, hidden_dims, style=TradingStyle.BREAKOUT
            ).to(device),
            
            TradingStyle.MOMENTUM: SpecializedPolicy(
                input_dim, hidden_dims, style=TradingStyle.MOMENTUM
            ).to(device),
            
            TradingStyle.VOLATILITY: SpecializedPolicy(
                input_dim, hidden_dims, style=TradingStyle.VOLATILITY
            ).to(device),
        }
        
        # Regime detector
        self.regime_detector = RegimeDetector(
            input_dim,
            hidden_dim=256,
            n_regimes=len(MarketRegime)
        ).to(device)
        
        # Policy weights (learned through training)
        self.policy_weights = {style: 1.0 for style in TradingStyle}
        
        # Performance tracking
        self.policy_performance = {
            style: {'wins': 0, 'total': 0, 'cumulative_return': 0.0}
            for style in TradingStyle
        }
    
    def predict(
        self,
        state: torch.Tensor,
        regime: MarketRegime = None,
        mode: str = 'weighted_vote'
    ) -> Tuple[int, float, Dict]:
        """
        Predict action using ensemble
        
        Args:
            state: Current state features
            regime: Optional market regime (auto-detected if None)
            mode: 'weighted_vote', 'best_only', or 'regime_based'
        
        Returns:
            action, confidence, policy_contributions
        """
        # Detect regime if not provided
        if regime is None:
            regime_idx = self.regime_detector.detect_regime(state)
            regime = list(MarketRegime)[regime_idx]
        
        # Get predictions from all policies
        predictions = {}
        confidences = {}
        
        with torch.no_grad():
            for style, policy in self.policies.items():
                logits = policy(state)
                probs = torch.softmax(logits, dim=-1)
                action = torch.argmax(probs).item()
                confidence = probs.max().item()
                
                predictions[style] = action
                confidences[style] = confidence
        
        # Ensemble decision based on mode
        if mode == 'weighted_vote':
            action, confidence = self._weighted_vote(predictions, confidences)
        elif mode == 'best_only':
            action, confidence = self._best_policy_only(predictions, confidences)
        elif mode == 'regime_based':
            action, confidence = self._regime_based_selection(
                predictions, confidences, regime
            )
        else:
            raise ValueError(f"Unknown ensemble mode: {mode}")
        
        return action, confidence, {
            'regime': regime.value,
            'predictions': predictions,
            'confidences': confidences
        }
    
    def _weighted_vote(
        self,
        predictions: Dict[TradingStyle, int],
        confidences: Dict[TradingStyle, float]
    ) -> Tuple[int, float]:
        """Weighted voting based on confidence and historical performance"""
        vote_scores = {0: 0.0, 1: 0.0, 2: 0.0}  # SELL, HOLD, BUY
        
        for style, action in predictions.items():
            weight = confidences[style] * self.policy_weights[style]
            vote_scores[action] += weight
        
        action = max(vote_scores, key=vote_scores.get)
        confidence = vote_scores[action] / sum(vote_scores.values())
        
        return action, confidence
    
    def _best_policy_only(
        self,
        predictions: Dict[TradingStyle, int],
        confidences: Dict[TradingStyle, float]
    ) -> Tuple[int, float]:
        """Use only the most confident policy"""
        best_style = max(confidences, key=confidences.get)
        return predictions[best_style], confidences[best_style]
    
    def _regime_based_selection(
        self,
        predictions: Dict[TradingStyle, int],
        confidences: Dict[TradingStyle, float],
        regime: MarketRegime
    ) -> Tuple[int, float]:
        """Select policy based on market regime"""
        # Map regimes to preferred trading styles
        regime_style_map = {
            MarketRegime.BULL_TRENDING: TradingStyle.TREND_FOLLOWER,
            MarketRegime.BEAR_TRENDING: TradingStyle.TREND_FOLLOWER,
            MarketRegime.CHOPPY_RANGE: TradingStyle.MEAN_REVERSION,
            MarketRegime.HIGH_VOLATILITY: TradingStyle.VOLATILITY,
            MarketRegime.LOW_VOLATILITY: TradingStyle.MOMENTUM,
        }
        
        preferred_style = regime_style_map.get(regime, TradingStyle.TREND_FOLLOWER)
        return predictions[preferred_style], confidences[preferred_style]
    
    def update_performance(
        self,
        style: TradingStyle,
        reward: float,
        action_was_correct: bool
    ):
        """Update performance tracking for adaptive weighting"""
        perf = self.policy_performance[style]
        perf['total'] += 1
        perf['cumulative_return'] += reward
        
        if action_was_correct:
            perf['wins'] += 1
        
        # Update weights based on recent performance
        if perf['total'] >= 10:  # Wait for minimum samples
            win_rate = perf['wins'] / perf['total']
            avg_return = perf['cumulative_return'] / perf['total']
            
            # Combine win rate and returns for weight
            self.policy_weights[style] = win_rate * (1 + avg_return)
    
    def get_policy_stats(self) -> Dict:
        """Get performance statistics for all policies"""
        stats = {}
        for style, perf in self.policy_performance.items():
            if perf['total'] > 0:
                stats[style.value] = {
                    'win_rate': perf['wins'] / perf['total'],
                    'avg_return': perf['cumulative_return'] / perf['total'],
                    'weight': self.policy_weights[style],
                    'trades': perf['total']
                }
        return stats
    
    def save(self, path: str):
        """Save all policies and regime detector"""
        checkpoint = {
            'policies': {
                style.value: policy.state_dict()
                for style, policy in self.policies.items()
            },
            'regime_detector': self.regime_detector.state_dict(),
            'policy_weights': {style.value: w for style, w in self.policy_weights.items()},
            'policy_performance': {
                style.value: perf 
                for style, perf in self.policy_performance.items()
            }
        }
        torch.save(checkpoint, path)
        logger.info(f"Ensemble saved to {path}")
    
    def load(self, path: str):
        """Load all policies and regime detector"""
        checkpoint = torch.load(path, map_location=self.device)
        
        for style_str, state_dict in checkpoint['policies'].items():
            style = TradingStyle(style_str)
            self.policies[style].load_state_dict(state_dict)
        
        self.regime_detector.load_state_dict(checkpoint['regime_detector'])
        
        self.policy_weights = {
            TradingStyle(k): v 
            for k, v in checkpoint['policy_weights'].items()
        }
        
        self.policy_performance = {
            TradingStyle(k): v 
            for k, v in checkpoint['policy_performance'].items()
        }
        
        logger.info(f"Ensemble loaded from {path}")
