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
# Configuration
# ============================================================================

@dataclass
class TrainingConfig:
    # Supabase
    supabase_url: str = os.getenv("SUPABASE_URL", "")
    supabase_key: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    
    # Data
    symbols: List[str] = None
    timeframe: str = "5m"
    train_days: int = 20
    val_days: int = 7
    test_days: int = 3
    embargo_days: int = 1
    
    # Features
    frame_stack_size: int = 32
    feature_dim: int = 15  # Technical indicators per bar
    
    # BC Training
    bc_epochs: int = 50
    bc_batch_size: int = 256
    bc_lr: float = 3e-4
    bc_weight_decay: float = 1e-5
    bc_early_stop_patience: int = 5
    
    # PPO Training
    ppo_total_timesteps: int = 100000
    ppo_n_steps: int = 2048
    ppo_batch_size: int = 2048
    ppo_learning_rate: float = 3e-4
    ppo_gamma: float = 0.99
    ppo_gae_lambda: float = 0.95
    ppo_clip_range: float = 0.2
    ppo_vf_coef: float = 0.5
    ppo_ent_coef: float = 0.01
    ppo_max_grad_norm: float = 0.5
    
    # Reward shaping
    lambda_risk: float = 0.2
    reward_clip: Tuple[float, float] = (-5.0, 5.0)
    
    # Walk-forward
    n_seeds: int = 3
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["SPY", "QQQ", "AAPL", "TSLA"]

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
        """Extract frame stack features"""
        frame_stack = obs_features.get('frame_stack', [])
        
        features = []
        for frame in frame_stack[-self.config.frame_stack_size:]:
            features.append([
                frame.get('close', 0),
                frame.get('volume', 0),
                frame.get('rsi_14', 50),
                frame.get('atr_14', 0),
                frame.get('vwap_distance_pct', 0),
                frame.get('volume_zscore', 0),
            ])
        
        # Pad if necessary
        while len(features) < self.config.frame_stack_size:
            features.insert(0, [0] * 6)
        
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
    """Train behavior cloning policy"""
    
    logger.info("Starting Behavior Cloning training")
    
    obs_dim = train_dataset.obs.shape[1]
    policy = BCPolicy(obs_dim).to(device)  # Move to GPU
    
    train_loader = DataLoader(train_dataset, batch_size=config.bc_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.bc_batch_size)
    
    optimizer = optim.Adam(policy.parameters(), lr=config.bc_lr, weight_decay=config.bc_weight_decay)
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.bc_epochs):
        # Training
        policy.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for obs, actions, rewards, weights in train_loader:
            # Move to device
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
                # Move to device
                obs, actions, weights = obs.to(device), actions.to(device), weights.to(device)
                
                logits = policy(obs)
                loss = criterion(logits, actions)
                weighted_loss = (loss * weights).mean()
                
                val_loss += weighted_loss.item()
                val_correct += (logits.argmax(dim=1) == actions).sum().item()
                val_total += len(actions)
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        logger.info(f"Epoch {epoch+1}/{config.bc_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(policy.state_dict(), f"checkpoints/policy_bc_{run_id}.pt")
        else:
            patience_counter += 1
            if patience_counter >= config.bc_early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    policy.load_state_dict(torch.load(f"checkpoints/policy_bc_{run_id}.pt", map_location=device))
    
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
        
        # Observation space: flattened frame stack
        obs_dim = config.frame_stack_size * 6  # 6 features per bar
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
        reward = traj['reward']
        
        # Normalize reward by ATR
        obs_features = traj['obs_features']
        atr = obs_features['current'].get('atr_14', 1.0)
        normalized_reward = np.clip(reward / max(atr, 0.01), *self.config.reward_clip)
        
        # Track returns and equity
        self.episode_returns.append(normalized_reward)
        pnl = reward * 0.01 * self.current_equity  # Convert to dollar P&L
        self.current_equity += pnl
        self.equity_curve.append(self.current_equity)
        
        # Track trade if action is BUY or SELL
        if action != 1:  # Not HOLD
            trade = {
                'action': action,
                'reward': reward,
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
        
        frame_stack = obs_features.get('frame_stack', [])
        features = []
        
        for frame in frame_stack[-self.config.frame_stack_size:]:
            features.append([
                frame.get('close', 0),
                frame.get('volume', 0),
                frame.get('rsi_14', 50),
                frame.get('atr_14', 0),
                frame.get('vwap_distance_pct', 0),
                frame.get('volume_zscore', 0),
            ])
        
        while len(features) < self.config.frame_stack_size:
            features.insert(0, [0] * 6)
        
        return np.array(features, dtype=np.float32).flatten()

# ============================================================================
# PPO Training
# ============================================================================

class MetricsCallback(BaseCallback):
    """Callback to log comprehensive metrics during PPO training"""
    
    def __init__(self, client: Client, run_id: str, verbose: int = 0):
        super().__init__(verbose)
        self.client = client
        self.run_id = run_id
        
        # Collect episode info from completed episodes
        self.episode_infos = []
        self.step_count = 0
        self.last_log_step = 0
        self.log_interval = 2048  # Log every rollout (PPO default)
    
    def _on_step(self) -> bool:
        self.step_count += 1
        
        # Collect episode completion info
        if "infos" in self.locals:
            for info in self.locals["infos"]:
                if isinstance(info, dict) and "episode" in info:
                    self.episode_infos.append(info["episode"])
        
        return True
    
    def _on_rollout_end(self) -> None:
        """Log comprehensive metrics after each rollout"""
        
        if len(self.episode_infos) == 0:
            return
        
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
                f"Epoch {epoch}: reward={mean_reward:.3f}, win_rate={mean_win_rate:.1f}%, "
                f"sharpe={mean_sharpe:.2f}, pf={mean_profit_factor:.2f}, "
                f"trades={total_trades}"
            )
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
        
        # Clear collected episodes
        self.episode_infos = []
        self.last_log_step = self.step_count


def train_ppo(
    train_trajectories: List[Dict],
    val_trajectories: List[Dict],
    config: TrainingConfig,
    bc_policy: BCPolicy,
    run_id: str,
    client: Client
) -> PPO:
    """Train PPO policy starting from BC weights"""
    
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
    
    # Train with callback
    callback = MetricsCallback(client, run_id)
    model.learn(total_timesteps=config.ppo_total_timesteps, callback=callback)
    
    # Save model
    model.save(f"checkpoints/policy_ppo_{run_id}.zip")
    
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
