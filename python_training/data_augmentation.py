#!/usr/bin/env python3
"""
Advanced Data Augmentation for Financial Time Series
Provides robust augmentation techniques to increase training data diversity
"""

import numpy as np
import torch
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class TimeSeriesAugmenter:
    """
    Advanced augmentation for financial time series while preserving 
    statistical properties and market dynamics
    """
    
    def __init__(
        self,
        temporal_jitter_prob: float = 0.3,
        price_scaling_prob: float = 0.3,
        noise_injection_prob: float = 0.2,
        regime_sampling_prob: float = 0.4,
        seed: int = 42
    ):
        self.temporal_jitter_prob = temporal_jitter_prob
        self.price_scaling_prob = price_scaling_prob
        self.noise_injection_prob = noise_injection_prob
        self.regime_sampling_prob = regime_sampling_prob
        self.rng = np.random.RandomState(seed)
    
    def temporal_jitter(
        self, 
        obs_features: np.ndarray,
        max_shift: int = 3
    ) -> np.ndarray:
        """
        Shift time series by small random amount (±max_shift steps)
        Preserves patterns while adding temporal diversity
        """
        if self.rng.rand() > self.temporal_jitter_prob:
            return obs_features
        
        shift = self.rng.randint(-max_shift, max_shift + 1)
        if shift > 0:
            # Shift forward: pad with first value
            return np.concatenate([
                np.repeat(obs_features[:1], shift, axis=0),
                obs_features[:-shift]
            ])
        elif shift < 0:
            # Shift backward: pad with last value
            return np.concatenate([
                obs_features[-shift:],
                np.repeat(obs_features[-1:], -shift, axis=0)
            ])
        return obs_features
    
    def price_scaling(
        self,
        obs_features: np.ndarray,
        feature_indices: Dict[str, List[int]],
        scale_range: Tuple[float, float] = (0.95, 1.05)
    ) -> np.ndarray:
        """
        Apply consistent scaling to price-based features
        Simulates different price levels while preserving relative movements
        """
        if self.rng.rand() > self.price_scaling_prob:
            return obs_features
        
        scale = self.rng.uniform(*scale_range)
        
        # Scale price features: open, high, low, close, vwap, emas
        price_cols = (
            feature_indices.get('price', []) + 
            feature_indices.get('vwap', []) + 
            feature_indices.get('ema', [])
        )
        
        augmented = obs_features.copy()
        for col in price_cols:
            if col < augmented.shape[1]:
                augmented[:, col] *= scale
        
        return augmented
    
    def noise_injection(
        self,
        obs_features: np.ndarray,
        noise_std: float = 0.01
    ) -> np.ndarray:
        """
        Add small Gaussian noise to simulate market microstructure noise
        """
        if self.rng.rand() > self.noise_injection_prob:
            return obs_features
        
        noise = self.rng.normal(0, noise_std, obs_features.shape)
        return obs_features + noise
    
    def regime_based_sampling(
        self,
        trajectories: List[Dict],
        target_samples: int,
        regime_key: str = 'regime_tag'
    ) -> List[Dict]:
        """
        Oversample from underrepresented market regimes
        Ensures balanced training across different market conditions
        """
        if self.rng.rand() > self.regime_sampling_prob:
            return trajectories
        
        # Group by regime
        regime_groups: Dict[str, List[Dict]] = {}
        for traj in trajectories:
            regime = traj.get(regime_key, 'unknown')
            if regime not in regime_groups:
                regime_groups[regime] = []
            regime_groups[regime].append(traj)
        
        # Calculate sampling weights (inverse frequency)
        regime_counts = {k: len(v) for k, v in regime_groups.items()}
        total = sum(regime_counts.values())
        regime_weights = {
            k: total / (len(regime_groups) * v) 
            for k, v in regime_counts.items()
        }
        
        # Sample with replacement based on weights
        sampled = []
        for _ in range(target_samples):
            # Pick regime
            regimes = list(regime_groups.keys())
            weights = [regime_weights[r] for r in regimes]
            weights = np.array(weights) / sum(weights)
            
            regime = self.rng.choice(regimes, p=weights)
            # Pick random sample from that regime
            traj = self.rng.choice(regime_groups[regime])
            sampled.append(traj)
        
        logger.info(f"Regime sampling: {regime_counts} → {target_samples} balanced samples")
        return sampled
    
    def augment_trajectory(
        self,
        trajectory: Dict,
        feature_indices: Dict[str, List[int]]
    ) -> Dict:
        """
        Apply full augmentation pipeline to a single trajectory
        """
        obs_features = np.array(trajectory['obs_features'])
        
        # Apply augmentations
        obs_features = self.temporal_jitter(obs_features)
        obs_features = self.price_scaling(obs_features, feature_indices)
        obs_features = self.noise_injection(obs_features)
        
        # Return augmented trajectory
        augmented = trajectory.copy()
        augmented['obs_features'] = obs_features.tolist()
        return augmented
    
    def augment_batch(
        self,
        trajectories: List[Dict],
        feature_indices: Dict[str, List[int]],
        augmentation_factor: int = 3
    ) -> List[Dict]:
        """
        Augment entire batch with multiple variants per trajectory
        
        Args:
            trajectories: List of original trajectories
            feature_indices: Mapping of feature names to column indices
            augmentation_factor: How many augmented copies to create per trajectory
        
        Returns:
            Augmented trajectories (original + augmented variants)
        """
        augmented_trajectories = list(trajectories)  # Keep originals
        
        for traj in trajectories:
            for _ in range(augmentation_factor - 1):  # -1 because we kept original
                aug_traj = self.augment_trajectory(traj, feature_indices)
                augmented_trajectories.append(aug_traj)
        
        logger.info(
            f"Augmented {len(trajectories)} → {len(augmented_trajectories)} trajectories "
            f"({augmentation_factor}x factor)"
        )
        
        return augmented_trajectories


def get_feature_indices() -> Dict[str, List[int]]:
    """
    Define which columns correspond to which feature types
    Used for targeted augmentation
    """
    return {
        'price': [0, 1, 2, 3],  # open, high, low, close
        'vwap': [9],
        'ema': [5, 6],
        'volume': [4],
        'indicators': [7, 8, 10, 11, 12, 13],  # rsi, atr, etc.
        'news': [15, 16, 17],
        'macro': [18, 19, 20, 21],
        'time': [22, 23, 24]
    }
