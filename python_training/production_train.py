#!/usr/bin/env python3
"""Production training script using full-history cached data and stable RL."""

from __future__ import annotations

import math
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client
from trading_environment import create_trading_env
from transformer_policy import TransformerPolicy

from full_history_config import FullHistoryConfig
from full_history_data import FullHistoryDataManager

load_dotenv()


@dataclass
class TrainingMetrics:
    seed: int
    best_epoch: int
    best_sortino: float
    checkpoint: Path
    train: Dict[str, float]
    val: Dict[str, float]
    test: Dict[str, float]


class ProductionTrainer:
    """Train RL agent on real market data with actual outcomes."""

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Full history settings
        self.settings = FullHistoryConfig()
        self.stock_tickers = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD",
            "SPY", "QQQ", "IWM", "DIA",
            "JPM", "BAC", "GS", "WFC",
            "XOM", "CVX", "COP",
            "UNH", "JNJ", "PFE",
        ]
        self.crypto_tickers = [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
            "ADAUSDT", "DOGEUSDT", "MATICUSDT", "DOTUSDT", "AVAXUSDT",
        ]
        self.data_manager = FullHistoryDataManager(
            self.settings,
            stocks=self.stock_tickers,
            crypto=self.crypto_tickers,
        )
        self.env_manifest = pd.DataFrame()
        self.env_specs = {}

        # Supabase for logging
        self.supabase = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        )

        self.rank, self.world_size = self._setup_distributed()
        self.device = self._resolve_device()
        self.training_dir = self._resolve_training_dir()
        self.progress = PhaseProgress()
        self.supabase = SupabaseLogger()

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.training_seeds: Sequence[int] = self.config.get("seeds", [0, 1, 2])
        self.epochs: int = self.config.get("epochs", 10)
        self.train_steps_per_epoch: int = self.config.get("steps_per_epoch", 64)
        self.grad_accum_steps: int = self.config.get("grad_accum_steps", 1)
        self.grad_clip: float = self.config.get("grad_clip", 1.0)
        self.entropy_coef: float = self.config.get("entropy_coef", 0.01)
        self.value_coef: float = self.config.get("value_coef", 0.5)
        self.discount_gamma: float = self.config.get("gamma", 0.99)
        self.eval_windows: int = self.config.get("eval_windows", 64)
        self.patience: int = self.config.get("patience", 12)
        ppo_cfg = self.config.get("ppo", {})
        self.clip_range: float = ppo_cfg.get("clip_range", 0.2)
        self.kl_coef: float = ppo_cfg.get("kl_coef", 1.0)
        self.kl_target: Optional[float] = ppo_cfg.get("kl_target", 0.01)
        self.kl_adapt_rate: float = ppo_cfg.get("kl_adapt_rate", 1.5)

        self.market_data: Dict[str, pd.DataFrame] = {}
        self.env_manifest = pd.DataFrame()
        self.env_specs: Dict[str, List[Dict]] = {}
        self.policy: Optional[TransformerPolicy] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[LambdaLR] = None
        self.bc_metrics: Dict[str, float] = {}
        self.bc_checkpoint: Optional[Path] = None
        self.bc_dataset = None
        self.bc_reference: Optional[TransformerPolicy] = None
        self.bc_state_dict: Optional[Dict[str, torch.Tensor]] = None

        if self.rank == 0:
            print(f"🖥️  Device: {self.device}")
            if torch.cuda.is_available():
                print(f"   GPU: {torch.cuda.get_device_name(0)}")
                print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _build_history_config(self, overrides: Dict) -> FullHistoryConfig:
        cfg = FullHistoryConfig()
        for key, value in overrides.items():
            setattr(cfg, key, value)
        return cfg

    def _default_stock_tickers(self) -> List[str]:
        return [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD",
            "SPY", "QQQ", "IWM", "DIA",
            "JPM", "BAC", "GS", "WFC",
            "XOM", "CVX", "COP",
            "UNH", "JNJ", "PFE",
        ]

    def _default_crypto_tickers(self) -> List[str]:
        return [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
            "ADAUSDT", "DOGEUSDT", "MATICUSDT", "DOTUSDT", "AVAXUSDT",
        ]

    def _setup_distributed(self) -> Sequence[int]:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        if world_size > 1 and not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
        return rank, world_size

    def _resolve_device(self) -> torch.device:
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            torch.cuda.set_device(local_rank)
            return torch.device(f"cuda:{local_rank}")
        return torch.device("cpu")

    def _resolve_training_dir(self) -> Path:
        directory = Path(self.config.get("training_dir", self.data_manager.cache_dir / "training"))
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def prepare_data(self):
        """Fetch and prepare real market data"""
        print("\n" + "="*70)
        print("📊 PREPARING REAL MARKET DATA")
        print("="*70)

        try:
            market_data, manifest = self.data_manager.prepare()
        except Exception as exc:
            print(f"❌ Data preparation failed: {exc}")
            raise

        self.market_data = market_data
        self.env_manifest = manifest
        self.news_data = pd.DataFrame()
        self.symbols = sorted(self.market_data.keys())

        split_counts = manifest['split'].value_counts().to_dict()
        print(f"\n✅ Prepared {len(self.symbols)} symbols")
        print(f"   Windows: {len(manifest)} (train={split_counts.get('TRAIN', 0)}, "
              f"val={split_counts.get('VAL', 0)}, test={split_counts.get('TEST', 0)})")

        print("✅ PREP DONE — starting training now")

        return self.market_data, self.news_data
    
    def create_environments(self):
        """Create trading environments for each symbol"""
        print("\n📦 Creating trading environments...")

        if self.env_manifest.empty:
            raise RuntimeError("Environment manifest is empty - run prepare_data first")

        self.env_specs = {
            split: df.reset_index(drop=True).to_dict("records")
            for split, df in self.env_manifest.groupby("split")
        }

        total_specs = sum(len(specs) for specs in self.env_specs.values())
        print(f"✅ Manifest ready with {total_specs} windows")

        self.train_specs = self.env_specs.get("TRAIN", [])
        self.val_specs = self.env_specs.get("VAL", [])
        self.test_specs = self.env_specs.get("TEST", [])

    def _create_env_from_spec(self, spec: dict):
        symbol = spec['symbol']
        window_df = self.market_data[symbol].iloc[spec['start_idx']:spec['end_idx']]
        env_kwargs = self.config.get('env_config', {}).copy()

        return create_trading_env(
            symbols=[symbol],
            phase=spec['split'].lower(),
            external_data={symbol: window_df.to_dict('records')},
            **env_kwargs
        )

    @staticmethod
    def _reset_env(env, phase: str):
        result = env.reset(phase=phase.lower())
        if isinstance(result, tuple) and len(result) == 2:
            return result
        return result, {}

    @staticmethod
    def _step_env(env, action: int):
        result = env.step(action)
        if isinstance(result, tuple) and len(result) == 5:
            return result
        obs, reward, done, info = result
        return obs, reward, done, False, info
    
    def initialize_policy(self):
        """Initialize transformer policy"""
        print("\n🧠 Initializing Transformer Policy...")

        if not self.train_specs:
            raise RuntimeError("No training window specs available for policy initialization")

        sample_env = self._create_env_from_spec(self.train_specs[0])
        sample_obs, _ = self._reset_env(sample_env, phase="train")
        obs_dim = sample_obs.shape[0]
        act_dim = 3  # Environment uses three discrete actions: sell, hold, buy

        d_model = self.config.get('d_model', self.config.get('hidden_dim', 256))
        nhead = self.config.get('nhead', self.config.get('num_heads', 8))
        num_layers = self.config.get('num_layers', 4)

        self.policy = TransformerPolicy(
            state_dim=obs_dim,
            action_dim=act_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.config.get('learning_rate', 3e-4)
        )
        
        print(f"✅ Policy initialized")
        print(f"   Obs dim: {obs_dim}")
        print(f"   Action dim: {act_dim}")
        print(f"   Model dim: {d_model}")
        print(f"   Parameters: {sum(p.numel() for p in self.policy.parameters()):,}")
    
    def collect_trajectories(self, num_episodes: int):
        """Collect trajectories from real market data"""
        print(f"\n🎬 Collecting {num_episodes} episodes from real market data...")

        trajectories = []

        with tqdm(total=num_episodes, desc="Episodes") as pbar:
            for episode in range(num_episodes):
                if not self.train_specs:
                    raise RuntimeError("No training windows available for trajectory collection")

                spec = self.train_specs[np.random.randint(0, len(self.train_specs))]
                env = self._create_env_from_spec(spec)

                obs, info = self._reset_env(env, phase=spec['split'])
                done = False
                episode_data = {
                    'symbol': spec['symbol'],
                    'observations': [],
                    'actions': [],
                    'log_probs': [],
                    'entropies': [],
                    'values': [],
                    'rewards': [],
                    'dones': [],
                    'infos': []
                }
                
                while not done:
                    with torch.no_grad():
                        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                        action_logits = self.policy.get_action_logits(obs_tensor)
                        dist = torch.distributions.Categorical(logits=action_logits)
                        action_tensor = dist.sample()
                        action = int(action_tensor.item())
                        log_prob = dist.log_prob(action_tensor).item()
                        entropy = dist.entropy().item()
                        value = self.policy.get_value(obs_tensor).item()

                    # Environment step
                    next_obs, reward, terminated, truncated, info = self._step_env(env, action)
                    done = terminated or truncated

                    # Store real outcome
                    episode_data['observations'].append(obs)
                    episode_data['actions'].append(action)
                    episode_data['log_probs'].append(log_prob)
                    episode_data['entropies'].append(entropy)
                    episode_data['values'].append(value)
                    episode_data['rewards'].append(reward)
                    episode_data['dones'].append(done)
                    episode_data['infos'].append(info)
                    
                    obs = next_obs
                
                trajectories.append(episode_data)
                pbar.update(1)
                pbar.set_postfix({
                    'symbol': spec['symbol'],
                    'steps': len(episode_data['rewards']),
                    'return': sum(episode_data['rewards'])
                })
        
        return trajectories
    
    def compute_advantages(self, trajectories):
        """Compute advantages using GAE"""
        gamma = self.config.get('gamma', 0.99)
        gae_lambda = self.config.get('gae_lambda', 0.95)
        
        for traj in trajectories:
            rewards = np.array(traj['rewards'], dtype=np.float32)
            dones = np.array(traj['dones'], dtype=np.float32)
            values = np.array(traj.get('values', [0.0] * len(rewards)), dtype=np.float32)

            values = np.append(values, 0.0)
            advantages = np.zeros_like(rewards, dtype=np.float32)
            returns = np.zeros_like(rewards, dtype=np.float32)
            gae = 0.0

            for step in reversed(range(len(rewards))):
                mask = 1.0 - dones[step]
                delta = rewards[step] + gamma * values[step + 1] * mask - values[step]
                gae = delta + gamma * gae_lambda * mask * gae
                advantages[step] = gae
                returns[step] = advantages[step] + values[step]

            traj['advantages'] = advantages
            traj['returns'] = returns
        
        return trajectories
    
    def update_policy(self, trajectories):
        """Update policy using PPO"""
        print("\n📈 Updating policy...")
        
        # Prepare batch data
        all_obs = []
        all_actions = []
        all_advantages = []
        all_returns = []
        all_log_probs = []

        for traj in trajectories:
            all_obs.extend(traj['observations'])
            all_actions.extend(traj['actions'])
            all_advantages.extend(traj['advantages'])
            all_returns.extend(traj['returns'])
            all_log_probs.extend(traj.get('log_probs', [0.0] * len(traj['actions'])))

        # Convert to tensors
        obs_tensor = torch.FloatTensor(np.array(all_obs)).to(self.device)
        actions_tensor = torch.LongTensor(np.array(all_actions)).to(self.device)
        advantages_tensor = torch.FloatTensor(np.array(all_advantages)).to(self.device)
        returns_tensor = torch.FloatTensor(np.array(all_returns)).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(np.array(all_log_probs)).to(self.device)

        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        # PPO update
        num_updates = self.config.get('num_policy_updates', 10)
        batch_size = self.config.get('batch_size', 256)
        num_samples = len(all_obs)

        policy_losses = []
        value_losses = []
        entropies = []
        total_losses = []
        approx_kls = []
        clip_fracs = []

        for update in range(num_updates):
            if num_samples == 0:
                break
            # Mini-batch sampling
            indices = np.random.permutation(num_samples)[: min(batch_size, num_samples)]

            obs_batch = obs_tensor[indices]
            actions_batch = actions_tensor[indices]
            advantages_batch = advantages_tensor[indices]
            returns_batch = returns_tensor[indices]
            old_log_probs_batch = old_log_probs_tensor[indices]

            # Forward pass
            action_logits = self.policy.get_action_logits(obs_batch)
            dist = torch.distributions.Categorical(logits=action_logits)
            log_probs = dist.log_prob(actions_batch)
            entropy = dist.entropy().mean()
            values = self.policy.get_value(obs_batch)

            ratios = torch.exp(log_probs - old_log_probs_batch)
            clipped_ratios = torch.clamp(ratios, 1 - self.clip_range, 1 + self.clip_range)
            surrogate = torch.min(ratios * advantages_batch, clipped_ratios * advantages_batch)
            policy_loss = -surrogate.mean()
            value_loss = F.mse_loss(values, returns_batch)
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            approx_kl = (old_log_probs_batch - log_probs).mean()
            clip_frac = (torch.abs(ratios - 1.0) > self.clip_range).float().mean()

            # Update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(entropy.item())
            total_losses.append(loss.item())
            approx_kls.append(approx_kl.item())
            clip_fracs.append(clip_frac.item())

        if policy_losses:
            avg_policy_loss = float(np.mean(policy_losses))
            avg_value_loss = float(np.mean(value_losses))
            avg_entropy = float(np.mean(entropies))
            avg_loss = float(np.mean(total_losses))
            avg_kl = float(np.mean(approx_kls))
            avg_clip = float(np.mean(clip_fracs))
        else:
            avg_policy_loss = avg_value_loss = avg_entropy = avg_loss = avg_kl = avg_clip = 0.0

        print(f"✅ Policy updated ({min(num_updates, len(policy_losses))} iterations)")
        print(f"   Final loss: {avg_loss:.4f}")
        print(f"   Policy loss: {avg_policy_loss:.4f}")
        print(f"   Value loss: {avg_value_loss:.4f}")
        print(f"   Entropy: {avg_entropy:.4f}")
        print(f"   Approx KL: {avg_kl:.4f} | Clip frac: {avg_clip:.3f}")

        return {
            "loss": avg_loss,
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "entropy": avg_entropy,
            "approx_kl": avg_kl,
            "clip_fraction": avg_clip,
        }
    
    def evaluate(self, num_episodes: int = 10):
        """Evaluate policy on real market data"""
        print(f"\n📊 Evaluating on {num_episodes} episodes...")

        eval_returns = []
        eval_lengths = []

        eval_pool = self.val_specs or self.test_specs or self.train_specs
        if not eval_pool:
            return {'mean_return': 0.0, 'std_return': 0.0, 'mean_length': 0.0}

        for _ in range(num_episodes):
            spec = eval_pool[np.random.randint(0, len(eval_pool))]
            env = self._create_env_from_spec(spec)

            obs, _ = self._reset_env(env, phase=spec['split'])
            done = False
            episode_return = 0
            episode_length = 0

            while not done:
                with torch.no_grad():
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                    action_logits = self.policy.get_action_logits(obs_tensor)
                    action = int(torch.argmax(action_logits, dim=-1).item())  # Greedy for eval

                obs, reward, terminated, truncated, _ = self._step_env(env, action)
                done = terminated or truncated
                episode_return += reward
                episode_length += 1

            eval_returns.append(episode_return)
            eval_lengths.append(episode_length)
        
        metrics = {
            'mean_return': np.mean(eval_returns),
            'std_return': np.std(eval_returns),
            'mean_length': np.mean(eval_lengths)
        }
        
        print(f"✅ Evaluation complete")
        print(f"   Mean return: {metrics['mean_return']:.2f} ± {metrics['std_return']:.2f}")
        print(f"   Mean length: {metrics['mean_length']:.1f} steps")
        
        return metrics
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*70)
        print("🚀 STARTING PRODUCTION TRAINING")
        print("="*70)
        
        # 1. Prepare data
        self.prepare_data()
        
        # 2. Create environments
        self.create_environments()

        # 3. Initialize policy
        self.initialize_policy()

        if not self.train_specs:
            raise RuntimeError("No training window specifications available")

        print("🚀 TRAINING START")

        # 4. Training loop
        num_iterations = self.config.get('num_iterations', 100)
        episodes_per_iter = self.config.get('episodes_per_iteration', 50)
        
        for iteration in range(num_iterations):
            print(f"\n{'='*70}")
            print(f"ITERATION {iteration + 1}/{num_iterations}")
            print(f"{'='*70}")
            
            # Collect trajectories
            trajectories = self.collect_trajectories(episodes_per_iter)
            
            # Compute advantages
            trajectories = self.compute_advantages(trajectories)
            
            # Update policy
            train_metrics = self.update_policy(trajectories)
            
            # Evaluate
            if (iteration + 1) % self.config.get('eval_frequency', 10) == 0:
                eval_metrics = self.evaluate(num_episodes=20)
                
                # Log to Supabase
                self.log_metrics(iteration + 1, train_metrics, eval_metrics)
            
            # Save checkpoint
            if (iteration + 1) % self.config.get('save_frequency', 20) == 0:
                self.save_checkpoint(iteration + 1)
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE")
        print("="*70)
    
    def log_metrics(self, iteration: int, train_metrics: dict, eval_metrics: dict):
        """Log metrics to Supabase"""
        try:
            self.supabase.table('training_metrics').insert({
                'run_id': self.config.get('run_id', 'production_run'),
                'epoch': iteration,
                'split': 'train',
                'policy_loss': train_metrics['policy_loss'],
                'entropy': train_metrics['entropy'],
                'mean_reward': eval_metrics['mean_return']
            }).execute()
        except Exception as e:
            print(f"⚠️  Failed to log metrics: {e}")
    
    def save_checkpoint(self, iteration: int):
        """Save model checkpoint"""
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        path = f"{checkpoint_dir}/policy_iter_{iteration}.pt"
        torch.save({
            'iteration': iteration,
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)
        
        print(f"💾 Checkpoint saved: {path}")


def main():
    """Main entry point"""
    
    config = {
        # Data config
        'data_days': 60,  # 2 months of recent data
        'min_bars_per_symbol': 1000,
        
        # Environment config
        'env_config': {
            'initial_balance': 100000,
            'commission': 0.001,
        },
        
        # Policy config
        'hidden_dim': 256,
        'num_layers': 4,
        'num_heads': 8,
        'learning_rate': 3e-4,
        
        # Training config
        'num_iterations': 100,
        'episodes_per_iteration': 50,
        'batch_size': 256,
        'num_policy_updates': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'entropy_coef': 0.01,
        
        # Logging
        'eval_frequency': 10,
        'save_frequency': 20,
        'run_id': f'production_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    }
    
    trainer = ProductionTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
