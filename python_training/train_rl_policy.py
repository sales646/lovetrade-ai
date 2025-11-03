#!/usr/bin/env python3
"""
RL Trading Policy Training Pipeline
Behavior Cloning + PPO Finetuning with Walk-Forward Validation
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from supabase import create_client, Client
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
from gymnasium import spaces
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Detect device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# ============================================================================
# Early Stopping
# ============================================================================

class EarlyStopping:
    """
    Advanced Early Stopping with:
    - Patience-based stopping on validation metric
    - Divergence detection (train vs val)
    - Checkpoint saving every N epochs
    - Top-K checkpoint ensemble tracking
    """
    
    def __init__(
        self,
        patience: int = 1000,
        min_delta: float = 0.001,
        checkpoint_every: int = 1000,
        top_k: int = 3,
        metric_name: str = "val_loss",
        mode: str = "min"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_every = checkpoint_every
        self.top_k = top_k
        self.metric_name = metric_name
        self.mode = mode
        
        self.counter = 0
        self.best_metric = float('inf') if mode == "min" else float('-inf')
        self.best_epoch = 0
        self.best_checkpoint = None
        self.top_checkpoints = []  # List of (metric, epoch, path)
        
        self.train_metrics = []
        self.val_metrics = []
    
    def __call__(
        self,
        epoch: int,
        val_metric: float,
        train_metric: Optional[float] = None,
        checkpoint_path: Optional[str] = None
    ) -> bool:
        """
        Returns True if training should stop
        """
        
        # Track metrics for divergence detection
        self.val_metrics.append(val_metric)
        if train_metric is not None:
            self.train_metrics.append(train_metric)
        
        # Check if improved
        is_better = (
            (self.mode == "min" and val_metric < self.best_metric - self.min_delta) or
            (self.mode == "max" and val_metric > self.best_metric + self.min_delta)
        )
        
        if is_better:
            self.best_metric = val_metric
            self.best_epoch = epoch
            self.counter = 0
            
            if checkpoint_path:
                self.best_checkpoint = checkpoint_path
                # Add to top-K
                self.top_checkpoints.append((val_metric, epoch, checkpoint_path))
                # Sort and keep top K
                if self.mode == "min":
                    self.top_checkpoints.sort(key=lambda x: x[0])
                else:
                    self.top_checkpoints.sort(key=lambda x: x[0], reverse=True)
                self.top_checkpoints = self.top_checkpoints[:self.top_k]
                
            logger.info(f"✓ New best {self.metric_name}: {val_metric:.6f} at epoch {epoch}")
        else:
            self.counter += 1
        
        # Checkpoint saving every N epochs
        if epoch > 0 and epoch % self.checkpoint_every == 0 and checkpoint_path:
            logger.info(f"💾 Checkpoint saved at epoch {epoch}")
        
        # Divergence detection: if train-val gap is widening rapidly
        if len(self.train_metrics) >= 5 and len(self.val_metrics) >= 5:
            recent_train = np.mean(self.train_metrics[-5:])
            recent_val = np.mean(self.val_metrics[-5:])
            gap = abs(recent_val - recent_train)
            
            if self.mode == "min" and gap > 0.3:  # Val loss >> Train loss
                logger.warning(f"⚠️ Divergence detected! Train-Val gap: {gap:.4f}")
                return True
        
        # Patience exceeded
        if self.counter >= self.patience:
            logger.info(
                f"🛑 Early stopping triggered after {epoch} epochs. "
                f"Best {self.metric_name}: {self.best_metric:.6f} at epoch {self.best_epoch}"
            )
            return True
        
        return False
    
    def get_best_checkpoint(self) -> Optional[str]:
        return self.best_checkpoint
    
    def get_top_checkpoints(self) -> List[Tuple[float, int, str]]:
        return self.top_checkpoints

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TrainingConfig:
    # Supabase
    supabase_url: str = os.getenv("SUPABASE_URL", "")
    supabase_key: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    
    # Data - OPTIMIZED FOR MAXIMUM MARKET PREDICTION
    symbols: List[str] = None
    timeframe: str = "5m"
    train_days: int = 30
    val_days: int = 10
    test_days: int = 5
    embargo_days: int = 1
    
    # Features - INCREASED FOR BETTER CONTEXT
    frame_stack_size: int = 64  # Doubled from 32 for more historical context
    feature_dim: int = 25  # Expanded: 15 technical + 5 news/macro + 5 time features
    
    # BC Training - OPTIMIZED PER PLAN
    bc_epochs: int = 5000  # From plan: 5000 epochs with checkpoint every 1000
    bc_batch_size: int = 512  # Increased for faster GPU training
    bc_lr: float = 2e-4  # Stable learning rate
    bc_weight_decay: float = 1e-5
    bc_early_stop_patience: int = 1000  # From plan: patience 800-1000
    
    # PPO Training - MAXIMIZED PER PLAN (10-20M timesteps)
    ppo_total_timesteps: int = 10_000_000  # From plan: 10-20M timesteps
    ppo_n_steps: int = 4096  # Increased from 2048 for better rollouts
    ppo_batch_size: int = 4096  # Increased from 2048 for GPU efficiency
    ppo_learning_rate: float = 2e-4  # Slightly lower for stability
    ppo_gamma: float = 0.995  # Higher discount for long-term planning
    ppo_gae_lambda: float = 0.98  # Higher for better advantage estimation
    ppo_clip_range: float = 0.15  # Tighter for more conservative updates
    ppo_vf_coef: float = 0.5
    ppo_ent_coef: float = 0.005  # Lower for more exploitation
    ppo_max_grad_norm: float = 0.5
    ppo_early_stop_patience: int = 10  # Stop if no improvement after 10 rollouts
    
    # Reward shaping - OPTIMIZED FOR SHARPE RATIO
    lambda_risk: float = 0.3  # Higher risk penalty for better risk-adjusted returns
    reward_clip: Tuple[float, float] = (-3.0, 3.0)  # Tighter clipping
    
    # Walk-forward
    n_seeds: int = 5  # More seeds for robustness
    
    def __post_init__(self):
        if self.symbols is None:
            # EXPANDED SYMBOL LIST FOR DIVERSIFICATION
            self.symbols = [
                "SPY", "QQQ", "IWM", "DIA",  # Indices
                "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META",  # Large caps
                "AMD", "NFLX", "INTC", "CSCO",  # Tech
                "JPM", "BAC", "GS",  # Finance
                "XOM", "CVX",  # Energy
            ]

# ============================================================================
# Data Loading
# ============================================================================

class TrajectoryDataset(Dataset):
    """Dataset for expert trajectories"""
    
    def __init__(self, trajectories: List[Dict], config: TrainingConfig):
        self.config = config
        self.trajectories = trajectories
        
        # Parse features and actions
        self.obs = []
        self.actions = []
        self.rewards = []
        self.weights = []
        
        for traj in trajectories:
            obs_features = traj['obs_features']
            action = traj['action'] + 1  # Convert {-1,0,1} to {0,1,2}
            reward = traj['reward']
            entry_quality = traj['entry_quality']
            rr_ratio = traj['rr_ratio']
            
            # Extract frame stack
            frame_stack = self._parse_frame_stack(obs_features)
            
            self.obs.append(frame_stack)
            self.actions.append(action)
            self.rewards.append(reward)
            
            # Weight calculation: upweight HOLD and high-quality trades
            weight = 1.0
            if action == 1:  # HOLD
                weight = 2.0
            elif rr_ratio > 1.5 and entry_quality > 0.7:
                weight = 2.5
            
            self.weights.append(weight)
        
        self.obs = np.array(self.obs, dtype=np.float32)
        self.actions = np.array(self.actions, dtype=np.int64)
        self.rewards = np.array(self.rewards, dtype=np.float32)
        self.weights = np.array(self.weights, dtype=np.float32)
        
        logger.info(f"Loaded {len(self)} trajectories")
        logger.info(f"Action distribution: {np.bincount(self.actions)}")
    
    def _parse_frame_stack(self, obs_features: Dict) -> np.ndarray:
        """Extract frame stack features with normalization + news/macro/time context"""
        frame_stack = obs_features.get('frame_stack', [])
        
        # Get latest news sentiment for this symbol (cache or default)
        news_sentiment = obs_features.get('news_sentiment', 0)
        news_confidence = obs_features.get('news_confidence', 0)
        
        # Get macro data (cache or default)
        vix = obs_features.get('vix', 15) / 50  # Normalize VIX
        spy_change = obs_features.get('spy_change_pct', 0) / 10  # Normalize SPY change
        market_risk = 1.0 if obs_features.get('risk_off', False) else 0.0
        
        # Time features
        timestamp = obs_features.get('timestamp', '')
        hour = int(timestamp[11:13]) if len(timestamp) > 13 else 12
        day_of_week = obs_features.get('day_of_week', 2)  # 0=Monday, 4=Friday
        
        is_market_open = 1.0 if 9.5 <= hour <= 16 else 0.0  # Market hours 9:30-16:00
        is_premarket = 1.0 if 4 <= hour < 9.5 else 0.0
        is_afterhours = 1.0 if 16 < hour <= 20 else 0.0
        end_of_week = 1.0 if day_of_week >= 3 else 0.0  # Thursday-Friday
        
        features = []
        for frame in frame_stack[-self.config.frame_stack_size:]:
            # Normalize technical features
            close = frame.get('close', 0)
            features.append([
                # Technical (6 features)
                close / 100.0,  # Normalize price
                frame.get('volume', 0) / 1e6,  # Normalize volume to millions
                (frame.get('rsi_14', 50) - 50) / 50,  # Center RSI around 0
                frame.get('atr_14', 0) / close if close > 0 else 0,  # ATR as % of price
                frame.get('vwap_distance_pct', 0) / 10,  # Scale VWAP distance
                frame.get('volume_zscore', 0) / 3,  # Normalize z-score
                
                # News/Macro context (5 features) - same for all frames in stack
                news_sentiment,
                news_confidence,
                vix,
                spy_change,
                market_risk,
                
                # Time features (5 features) - same for all frames
                is_market_open,
                is_premarket,
                is_afterhours,
                end_of_week,
                hour / 24.0,  # Normalized hour
            ])
        
        # Pad if necessary
        while len(features) < self.config.frame_stack_size:
            features.insert(0, [0] * 16)  # 16 features per frame
        
        return np.array(features, dtype=np.float32).flatten()
    
    def __len__(self) -> int:
        return len(self.obs)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int, float, float]:
        return self.obs[idx], self.actions[idx], self.rewards[idx], self.weights[idx]


def load_trajectories_from_supabase(
    client: Client,
    symbols: List[str],
    timeframe: str,
    start_date: datetime,
    end_date: datetime
) -> List[Dict]:
    """Load trajectories from Supabase"""
    
    logger.info(f"Loading trajectories for {symbols} from {start_date} to {end_date}")
    
    response = client.table("expert_trajectories").select("*").in_("symbol", symbols).eq("timeframe", timeframe).gte("timestamp", start_date.isoformat()).lte("timestamp", end_date.isoformat()).execute()
    
    trajectories = response.data
    logger.info(f"Loaded {len(trajectories)} trajectories")
    
    return trajectories

# ============================================================================
# Behavior Cloning
# ============================================================================

class BCPolicy(nn.Module):
    """Policy network for behavior cloning"""
    
    def __init__(self, obs_dim: int, action_dim: int = 3):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)


def train_bc(
    train_dataset: TrajectoryDataset,
    val_dataset: TrajectoryDataset,
    config: TrainingConfig,
    run_id: str,
    device: torch.device = torch.device("cpu")
) -> BCPolicy:
    """Train behavior cloning policy with advanced early stopping"""
    
    logger.info("Starting Behavior Cloning training")
    
    obs_dim = train_dataset.obs.shape[1]
    policy = BCPolicy(obs_dim).to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=config.bc_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.bc_batch_size)
    
    optimizer = optim.Adam(policy.parameters(), lr=config.bc_lr, weight_decay=config.bc_weight_decay)
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    # Initialize advanced early stopping
    early_stopping = EarlyStopping(
        patience=config.bc_early_stop_patience,
        min_delta=0.001,
        checkpoint_every=1000,
        top_k=3,
        metric_name="val_loss",
        mode="min"
    )
    
    for epoch in range(config.bc_epochs):
        # Training
        policy.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for obs, actions, rewards, weights in train_loader:
            obs, actions, weights = obs.to(device), actions.to(device), weights.to(device)
            
            optimizer.zero_grad()
            
            logits = policy(obs)
            loss = criterion(logits, actions)
            weighted_loss = (loss * weights).mean()
            
            weighted_loss.backward()
            optimizer.step()
            
            train_loss += weighted_loss.item()
            train_correct += (logits.argmax(dim=1) == actions).sum().item()
            train_total += len(actions)
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation
        policy.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for obs, actions, rewards, weights in val_loader:
                obs, actions, weights = obs.to(device), actions.to(device), weights.to(device)
                
                logits = policy(obs)
                loss = criterion(logits, actions)
                weighted_loss = (loss * weights).mean()
                
                val_loss += weighted_loss.item()
                val_correct += (logits.argmax(dim=1) == actions).sum().item()
                val_total += len(actions)
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        logger.info(
            f"BC Epoch {epoch+1}/{config.bc_epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )
        
        # Save checkpoint
        checkpoint_path = f"checkpoints/policy_bc_{run_id}_epoch{epoch+1}.pt"
        if (epoch + 1) % 1000 == 0 or epoch == 0:  # Save every 1000 epochs
            torch.save(policy.state_dict(), checkpoint_path)
        
        # Early stopping check
        should_stop = early_stopping(
            epoch=epoch+1,
            val_metric=val_loss,
            train_metric=train_loss,
            checkpoint_path=checkpoint_path if val_loss == early_stopping.best_metric else None
        )
        
        # Additional stopping criteria from plan
        if val_acc > 0.85:
            logger.info(f"🎯 Stopping: Validation accuracy exceeded 85% ({val_acc:.2%})")
            break
        
        if train_loss < 0.01:
            logger.info(f"🎯 Stopping: Training loss below threshold ({train_loss:.6f})")
            break
        
        if should_stop:
            break
    
    # Load best checkpoint from top-K ensemble
    best_checkpoint = early_stopping.get_best_checkpoint()
    if best_checkpoint and os.path.exists(best_checkpoint):
        policy.load_state_dict(torch.load(best_checkpoint, map_location=device))
        logger.info(f"✅ Loaded best checkpoint: {best_checkpoint}")
    
    # Log top checkpoints for ensemble
    top_checkpoints = early_stopping.get_top_checkpoints()
    logger.info(f"📊 Top-3 checkpoints for ensemble:")
    for metric, ep, path in top_checkpoints:
        logger.info(f"  - Epoch {ep}: val_loss={metric:.6f} ({path})")
    
    return policy

# ============================================================================
# PPO Environment
# ============================================================================

class TradingEnv(gym.Env):
    """Gym environment for trading with proper metrics tracking"""
    
    def __init__(self, trajectories: List[Dict], config: TrainingConfig):
        super().__init__()
        
        self.config = config
        self.trajectories = trajectories
        self.current_idx = 0
        
        # Episode tracking
        self.episode_trades = []
        self.episode_actions = []
        self.episode_returns = []
        self.equity_curve = []
        self.initial_equity = 100000.0
        self.current_equity = self.initial_equity
        
        # Action space: {0: SELL, 1: HOLD, 2: BUY}
        self.action_space = spaces.Discrete(3)
        
        # Observation space: flattened frame stack (64 bars * 16 features = 1024)
        obs_dim = config.frame_stack_size * 16  # 16 features per bar (tech+news+macro+time)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        self.steps_in_episode = 0
        self.max_episode_steps = 100
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        
        # Reset episode tracking
        episode_info = self._get_episode_info()
        
        self.episode_trades = []
        self.episode_actions = []
        self.episode_returns = []
        self.equity_curve = [self.initial_equity]
        self.current_equity = self.initial_equity
        self.steps_in_episode = 0
        
        self.current_idx = np.random.randint(0, len(self.trajectories))
        return self._get_obs(), episode_info
    
    def step(self, action: int):
        # Track action
        self.episode_actions.append(action)
        
        traj = self.trajectories[self.current_idx]
        expert_action = traj['action']  # {-1: sell, 0: hold, 1: buy}
        base_reward = traj['reward']
        
        # Enhanced reward shaping for better market prediction
        obs_features = traj['obs_features']
        atr = obs_features['current'].get('atr_14', 1.0)
        
        # Risk-adjusted reward with Sharpe-like scaling
        pnl = base_reward * 0.01 * self.current_equity
        self.current_equity += pnl
        
        # Calculate rolling Sharpe-like metric for reward bonus
        self.episode_returns.append(base_reward)
        if len(self.episode_returns) > 10:
            recent_returns = self.episode_returns[-10:]
            sharpe_bonus = (np.mean(recent_returns) / (np.std(recent_returns) + 1e-6)) * 0.2
        else:
            sharpe_bonus = 0
        
        # Final reward with Sharpe bonus and risk penalty
        normalized_reward = base_reward - (self.config.lambda_risk * abs(base_reward))
        normalized_reward += sharpe_bonus
        normalized_reward = np.clip(normalized_reward, *self.config.reward_clip)
        
        self.equity_curve.append(self.current_equity)
        
        # Track trade if action is BUY or SELL
        if action != 1:  # Not HOLD
            trade = {
                'action': action,
                'reward': base_reward,
                'pnl': pnl,
                'correct': (action - 1) == expert_action  # Convert action space
            }
            self.episode_trades.append(trade)
        
        # Move to next trajectory
        self.current_idx = (self.current_idx + 1) % len(self.trajectories)
        self.steps_in_episode += 1
        
        # Episode ends after max_episode_steps
        terminated = self.steps_in_episode >= self.max_episode_steps
        truncated = False
        
        info = {}
        if terminated:
            info['episode'] = self._get_episode_info()
        
        return self._get_obs(), normalized_reward, terminated, truncated, info
    
    def _get_episode_info(self) -> Dict:
        """Calculate episode statistics"""
        if len(self.episode_returns) == 0:
            return {
                'r': 0.0,
                'l': self.steps_in_episode,
                'trades': 0,
                'win_rate': 0.0,
                'sharpe': 0.0,
                'profit_factor': 0.0,
                'max_dd': 0.0,
                'action_dist': [0, 0, 0]
            }
        
        # Calculate metrics
        total_return = sum(self.episode_returns)
        
        # Trading performance
        winning_trades = sum(1 for t in self.episode_trades if t['pnl'] > 0)
        total_trades = len(self.episode_trades)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Profit factor
        gross_profit = sum(t['pnl'] for t in self.episode_trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in self.episode_trades if t['pnl'] < 0))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0
        
        # Sharpe ratio
        returns = np.array(self.episode_returns)
        sharpe = (np.mean(returns) / (np.std(returns) + 1e-6)) * np.sqrt(252) if len(returns) > 1 else 0
        
        # Max drawdown
        equity = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity)
        drawdown = (running_max - equity) / running_max * 100
        max_dd = np.max(drawdown) if len(drawdown) > 0 else 0
        
        # Action distribution
        action_counts = [
            sum(1 for a in self.episode_actions if a == 0),  # SELL
            sum(1 for a in self.episode_actions if a == 1),  # HOLD
            sum(1 for a in self.episode_actions if a == 2),  # BUY
        ]
        
        return {
            'r': float(total_return),
            'l': self.steps_in_episode,
            'trades': total_trades,
            'win_rate': float(win_rate),
            'sharpe': float(sharpe),
            'profit_factor': float(profit_factor),
            'max_dd': float(max_dd),
            'action_dist': action_counts
        }
    
    def _get_obs(self) -> np.ndarray:
        traj = self.trajectories[self.current_idx]
        obs_features = traj['obs_features']
        
        # Extract context data
        news_sentiment = obs_features.get('news_sentiment', 0)
        news_confidence = obs_features.get('news_confidence', 0)
        vix = obs_features.get('vix', 15) / 50
        spy_change = obs_features.get('spy_change_pct', 0) / 10
        market_risk = 1.0 if obs_features.get('risk_off', False) else 0.0
        
        # Time features
        timestamp = obs_features.get('timestamp', '')
        hour = int(timestamp[11:13]) if len(timestamp) > 13 else 12
        day_of_week = obs_features.get('day_of_week', 2)
        is_market_open = 1.0 if 9.5 <= hour <= 16 else 0.0
        is_premarket = 1.0 if 4 <= hour < 9.5 else 0.0
        is_afterhours = 1.0 if 16 < hour <= 20 else 0.0
        end_of_week = 1.0 if day_of_week >= 3 else 0.0
        
        frame_stack = obs_features.get('frame_stack', [])
        features = []
        
        for frame in frame_stack[-self.config.frame_stack_size:]:
            # Normalize features same as in dataset
            close = frame.get('close', 0)
            features.append([
                close / 100.0,
                frame.get('volume', 0) / 1e6,
                (frame.get('rsi_14', 50) - 50) / 50,
                frame.get('atr_14', 0) / close if close > 0 else 0,
                frame.get('vwap_distance_pct', 0) / 10,
                frame.get('volume_zscore', 0) / 3,
                news_sentiment,
                news_confidence,
                vix,
                spy_change,
                market_risk,
                is_market_open,
                is_premarket,
                is_afterhours,
                end_of_week,
                hour / 24.0,
            ])
        
        while len(features) < self.config.frame_stack_size:
            features.insert(0, [0] * 16)
        
        return np.array(features, dtype=np.float32).flatten()

# ============================================================================
# PPO Training
# ============================================================================

class MetricsCallback(BaseCallback):
    """Callback to log comprehensive metrics during PPO training with early stopping"""
    
    def __init__(
        self,
        client: Client,
        run_id: str,
        val_env: Optional[gym.Env] = None,
        early_stopping_patience: int = 10,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.client = client
        self.run_id = run_id
        self.val_env = val_env
        
        # Collect episode info from completed episodes
        self.episode_infos = []
        self.step_count = 0
        self.last_log_step = 0
        self.log_interval = 2048  # Log every rollout (PPO default)
        
        # Early stopping for PPO
        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=0.01,
            checkpoint_every=5,  # Every 5 rollouts
            top_k=3,
            metric_name="val_sharpe",
            mode="max"
        )
        
        self.rollout_count = 0
        self.best_sharpe = float('-inf')
    
    def _on_step(self) -> bool:
        self.step_count += 1
        
        # Collect episode completion info
        if "infos" in self.locals:
            for info in self.locals["infos"]:
                if isinstance(info, dict) and "episode" in info:
                    self.episode_infos.append(info["episode"])
        
        return True
    
    def _on_rollout_end(self) -> None:
        """Log comprehensive metrics and check early stopping after each rollout"""
        
        if len(self.episode_infos) == 0:
            return
        
        self.rollout_count += 1
        
        # Aggregate episode statistics
        mean_reward = np.mean([ep['r'] for ep in self.episode_infos])
        mean_length = np.mean([ep['l'] for ep in self.episode_infos])
        
        # Trading metrics
        total_trades = sum(ep.get('trades', 0) for ep in self.episode_infos)
        win_rates = [ep.get('win_rate', 0) for ep in self.episode_infos]
        mean_win_rate = np.mean(win_rates) if win_rates else 0
        
        sharpe_ratios = [ep.get('sharpe', 0) for ep in self.episode_infos if not np.isnan(ep.get('sharpe', 0))]
        mean_sharpe = np.mean(sharpe_ratios) if sharpe_ratios else 0
        
        profit_factors = [ep.get('profit_factor', 0) for ep in self.episode_infos if not np.isnan(ep.get('profit_factor', 0))]
        mean_profit_factor = np.mean(profit_factors) if profit_factors else 0
        
        max_drawdowns = [ep.get('max_dd', 0) for ep in self.episode_infos]
        mean_max_dd = np.mean(max_drawdowns) if max_drawdowns else 0
        
        # Action distribution (aggregate across all episodes)
        total_actions = [0, 0, 0]
        for ep in self.episode_infos:
            action_dist = ep.get('action_dist', [0, 0, 0])
            for i in range(3):
                total_actions[i] += action_dist[i]
        
        total_action_count = sum(total_actions)
        action_pcts = [a / total_action_count * 100 if total_action_count > 0 else 0 for a in total_actions]
        
        # Get PPO training metrics from logger
        policy_loss = self.model.logger.name_to_value.get("train/policy_loss", 0)
        value_loss = self.model.logger.name_to_value.get("train/value_loss", 0)
        entropy = self.model.logger.name_to_value.get("train/entropy", 0)
        clip_fraction = self.model.logger.name_to_value.get("train/clip_fraction", 0)
        approx_kl = self.model.logger.name_to_value.get("train/approx_kl", 0)
        
        # Calculate epoch number
        epoch = self.step_count // self.log_interval
        
        metrics = {
            "run_id": self.run_id,
            "epoch": int(epoch),
            "split": "train",
            "mean_reward": float(mean_reward),
            "win_rate": float(mean_win_rate),
            "sharpe_ratio": float(mean_sharpe),
            "profit_factor": float(mean_profit_factor),
            "max_drawdown": float(mean_max_dd),
            "policy_loss": float(policy_loss),
            "value_loss": float(value_loss),
            "entropy": float(entropy),
            "action_sell_pct": float(action_pcts[0]),
            "action_hold_pct": float(action_pcts[1]),
            "action_buy_pct": float(action_pcts[2]),
        }
        
        try:
            self.client.table("training_metrics").insert(metrics).execute()
            logger.info(
                f"PPO Rollout {self.rollout_count}: reward={mean_reward:.3f}, win_rate={mean_win_rate:.1f}%, "
                f"sharpe={mean_sharpe:.2f}, pf={mean_profit_factor:.2f}, "
                f"entropy={entropy:.4f}, approx_kl={approx_kl:.6f}"
            )
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
        
        # Early stopping checks based on plan
        
        # 1. Entropy collapse detection (policy collapsed to deterministic)
        if entropy < 0.1:
            logger.warning(f"⚠️ Policy collapsed! Entropy={entropy:.4f} < 0.1")
            return False  # Stop training
        
        # 2. KL divergence too high (instability)
        if approx_kl > 0.5:
            logger.warning(f"⚠️ Training unstable! KL divergence={approx_kl:.4f} > 0.5")
            return False
        
        # 3. Validation-based early stopping on Sharpe ratio
        checkpoint_path = f"checkpoints/policy_ppo_{self.run_id}_rollout{self.rollout_count}.zip"
        if self.rollout_count % 5 == 0:  # Save every 5 rollouts
            self.model.save(checkpoint_path)
        
        should_stop = self.early_stopping(
            epoch=self.rollout_count,
            val_metric=mean_sharpe,
            checkpoint_path=checkpoint_path if mean_sharpe > self.best_sharpe else None
        )
        
        if mean_sharpe > self.best_sharpe:
            self.best_sharpe = mean_sharpe
        
        if should_stop:
            logger.info("🛑 PPO early stopping triggered")
            return False  # Stop training
        
        # Clear collected episodes
        self.episode_infos = []
        self.last_log_step = self.step_count
        
        return True


def train_ppo(
    train_trajectories: List[Dict],
    val_trajectories: List[Dict],
    config: TrainingConfig,
    bc_policy: BCPolicy,
    run_id: str,
    client: Client
) -> PPO:
    """Train PPO policy starting from BC weights with early stopping"""
    
    logger.info("Starting PPO training")
    
    # Create environment
    env = DummyVecEnv([lambda: TradingEnv(train_trajectories, config)])
    
    # Initialize PPO with GPU support
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config.ppo_learning_rate,
        n_steps=config.ppo_n_steps,
        batch_size=config.ppo_batch_size,
        gamma=config.ppo_gamma,
        gae_lambda=config.ppo_gae_lambda,
        clip_range=config.ppo_clip_range,
        vf_coef=config.ppo_vf_coef,
        ent_coef=config.ppo_ent_coef,
        max_grad_norm=config.ppo_max_grad_norm,
        verbose=1,
        device="auto",  # Auto-detect GPU/CPU
    )
    
    # TODO: Load BC weights into PPO policy (requires model surgery)
    
    # Train with callback and early stopping
    callback = MetricsCallback(
        client,
        run_id,
        early_stopping_patience=config.ppo_early_stop_patience
    )
    
    try:
        model.learn(total_timesteps=config.ppo_total_timesteps, callback=callback)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    
    # Load best checkpoint from early stopping
    best_checkpoint = callback.early_stopping.get_best_checkpoint()
    if best_checkpoint and os.path.exists(best_checkpoint):
        model = PPO.load(best_checkpoint, env=env)
        logger.info(f"✅ Loaded best PPO checkpoint: {best_checkpoint}")
    
    # Log top checkpoints
    top_checkpoints = callback.early_stopping.get_top_checkpoints()
    logger.info(f"📊 Top-3 PPO checkpoints for ensemble:")
    for metric, rollout, path in top_checkpoints:
        logger.info(f"  - Rollout {rollout}: sharpe={metric:.2f} ({path})")
    
    # Save final model
    final_path = f"checkpoints/policy_ppo_{run_id}_final.zip"
    model.save(final_path)
    logger.info(f"💾 Final model saved: {final_path}")
    
    return model

# ============================================================================
# Main Training Pipeline
# ============================================================================

def main():
    config = TrainingConfig()
    
    # Initialize Supabase
    client = create_client(config.supabase_url, config.supabase_key)
    
    # Create run
    run_name = f"rl_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_response = client.table("training_runs").insert({
        "run_name": run_name,
        "status": "running",
        "phase": "data_loading",
        "hyperparams": {
            "bc_lr": config.bc_lr,
            "ppo_lr": config.ppo_learning_rate,
            "lambda_risk": config.lambda_risk,
        },
    }).execute()
    
    run_id = run_response.data[0]['id']
    logger.info(f"Created training run: {run_id}")
    
    # Define time windows based on available data
    # We have data from 2025-10-02 to 2025-10-03
    train_start = datetime(2025, 10, 2, 13, 0, 0)
    train_end = datetime(2025, 10, 3, 0, 0, 0)
    val_start = datetime(2025, 10, 3, 0, 0, 0)
    val_end = datetime(2025, 10, 3, 17, 0, 0)
    
    # Load data
    train_trajectories = load_trajectories_from_supabase(
        client, config.symbols, config.timeframe, train_start, train_end
    )
    val_trajectories = load_trajectories_from_supabase(
        client, config.symbols, config.timeframe, val_start, val_end
    )
    
    if not train_trajectories or not val_trajectories:
        logger.error("No trajectories found. Please run generate-trajectories first.")
        return
    
    # Create datasets
    train_dataset = TrajectoryDataset(train_trajectories, config)
    val_dataset = TrajectoryDataset(val_trajectories, config)
    
    # Update phase
    client.table("training_runs").update({"phase": "behavior_cloning"}).eq("id", run_id).execute()
    
    # Train BC
    os.makedirs("checkpoints", exist_ok=True)
    bc_policy = train_bc(train_dataset, val_dataset, config, run_id, device)
    
    # Update phase
    client.table("training_runs").update({"phase": "ppo_finetuning"}).eq("id", run_id).execute()
    
    # Train PPO
    ppo_model = train_ppo(train_trajectories, val_trajectories, config, bc_policy, run_id, client)
    
    # Complete run
    client.table("training_runs").update({
        "status": "completed",
        "phase": "completed",
        "completed_at": datetime.now().isoformat(),
    }).eq("id", run_id).execute()
    
    logger.info(f"Training completed: {run_id}")


if __name__ == "__main__":
    main()
