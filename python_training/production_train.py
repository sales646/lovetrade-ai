#!/usr/bin/env python3
"""
Production Training Script - Uses ONLY real market data
No simulations, real outcomes only
"""

import os
import sys
import torch
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from supabase import create_client
from production_data_fetcher import ProductionDataFetcher
from trading_environment import create_trading_env
from transformer_policy import TransformerPolicy
from tqdm import tqdm

load_dotenv()


class ProductionTrainer:
    """Train RL agent on real market data with actual outcomes"""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Supabase for logging
        self.supabase = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        )
        
        print(f"🖥️  Device: {self.device}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def prepare_data(self):
        """Fetch and prepare real market data"""
        print("\n" + "="*70)
        print("📊 PREPARING REAL MARKET DATA")
        print("="*70)
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.config['data_days'])
        
        # Fetch data
        fetcher = ProductionDataFetcher(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )
        
        market_data, news_data = fetcher.fetch_all()
        
        # Filter symbols with enough data
        min_bars = self.config.get('min_bars_per_symbol', 1000)
        valid_symbols = [s for s, df in market_data.items() if len(df) >= min_bars]
        
        print(f"\n✅ Valid symbols for training: {len(valid_symbols)}/{len(market_data)}")
        print(f"   Min bars required: {min_bars:,}")
        
        # Store for later use
        self.market_data = {s: market_data[s] for s in valid_symbols}
        self.news_data = news_data
        self.symbols = valid_symbols
        
        return self.market_data, self.news_data
    
    def create_environments(self):
        """Create trading environments for each symbol"""
        print("\n📦 Creating trading environments...")
        
        self.envs = {}
        for symbol in tqdm(self.symbols, desc="Environments"):
            env = create_trading_env(
                symbol=symbol,
                data_df=self.market_data[symbol],
                news_df=self.news_data[self.news_data['symbol'] == symbol] if not self.news_data.empty else None,
                config=self.config.get('env_config', {})
            )
            self.envs[symbol] = env
        
        print(f"✅ Created {len(self.envs)} environments")
    
    def initialize_policy(self):
        """Initialize transformer policy"""
        print("\n🧠 Initializing Transformer Policy...")
        
        # Get observation space from first environment
        sample_env = self.envs[self.symbols[0]]
        obs_dim = sample_env.observation_space.shape[0]
        act_dim = sample_env.action_space.n
        
        self.policy = TransformerPolicy(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dim=self.config.get('hidden_dim', 256),
            num_layers=self.config.get('num_layers', 4),
            num_heads=self.config.get('num_heads', 8)
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.config.get('learning_rate', 3e-4)
        )
        
        print(f"✅ Policy initialized")
        print(f"   Obs dim: {obs_dim}")
        print(f"   Action dim: {act_dim}")
        print(f"   Hidden dim: {self.config.get('hidden_dim', 256)}")
        print(f"   Parameters: {sum(p.numel() for p in self.policy.parameters()):,}")
    
    def collect_trajectories(self, num_episodes: int):
        """Collect trajectories from real market data"""
        print(f"\n🎬 Collecting {num_episodes} episodes from real market data...")
        
        trajectories = []
        
        with tqdm(total=num_episodes, desc="Episodes") as pbar:
            for episode in range(num_episodes):
                # Random symbol each episode for diversity
                symbol = np.random.choice(self.symbols)
                env = self.envs[symbol]
                
                obs, info = env.reset()
                done = False
                episode_data = {
                    'symbol': symbol,
                    'observations': [],
                    'actions': [],
                    'rewards': [],
                    'dones': [],
                    'infos': []
                }
                
                while not done:
                    # Policy forward pass
                    with torch.no_grad():
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                        action_probs = self.policy(obs_tensor).cpu().numpy()[0]
                        action = np.random.choice(len(action_probs), p=action_probs)
                    
                    # Environment step
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
                    # Store real outcome
                    episode_data['observations'].append(obs)
                    episode_data['actions'].append(action)
                    episode_data['rewards'].append(reward)
                    episode_data['dones'].append(done)
                    episode_data['infos'].append(info)
                    
                    obs = next_obs
                
                trajectories.append(episode_data)
                pbar.update(1)
                pbar.set_postfix({
                    'symbol': symbol,
                    'steps': len(episode_data['rewards']),
                    'return': sum(episode_data['rewards'])
                })
        
        return trajectories
    
    def compute_advantages(self, trajectories):
        """Compute advantages using GAE"""
        gamma = self.config.get('gamma', 0.99)
        gae_lambda = self.config.get('gae_lambda', 0.95)
        
        for traj in trajectories:
            rewards = np.array(traj['rewards'])
            dones = np.array(traj['dones'])
            
            # Simple advantage = discounted returns
            advantages = []
            returns = []
            running_return = 0
            
            for r, done in zip(reversed(rewards), reversed(dones)):
                running_return = r + gamma * running_return * (1 - done)
                returns.insert(0, running_return)
            
            returns = np.array(returns)
            advantages = returns - returns.mean()
            
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
        
        for traj in trajectories:
            all_obs.extend(traj['observations'])
            all_actions.extend(traj['actions'])
            all_advantages.extend(traj['advantages'])
            all_returns.extend(traj['returns'])
        
        # Convert to tensors
        obs_tensor = torch.FloatTensor(np.array(all_obs)).to(self.device)
        actions_tensor = torch.LongTensor(np.array(all_actions)).to(self.device)
        advantages_tensor = torch.FloatTensor(np.array(all_advantages)).to(self.device)
        
        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # PPO update
        num_updates = self.config.get('num_policy_updates', 10)
        batch_size = self.config.get('batch_size', 256)
        
        for update in range(num_updates):
            # Mini-batch sampling
            indices = np.random.permutation(len(all_obs))[:batch_size]
            
            obs_batch = obs_tensor[indices]
            actions_batch = actions_tensor[indices]
            advantages_batch = advantages_tensor[indices]
            
            # Forward pass
            action_probs = self.policy(obs_batch)
            log_probs = torch.log(action_probs.gather(1, actions_batch.unsqueeze(1)) + 1e-8)
            
            # Policy loss
            policy_loss = -(log_probs.squeeze() * advantages_batch).mean()
            
            # Entropy bonus for exploration
            entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=1).mean()
            entropy_coef = self.config.get('entropy_coef', 0.01)
            
            loss = policy_loss - entropy_coef * entropy
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
        
        print(f"✅ Policy updated ({num_updates} iterations)")
        print(f"   Final loss: {loss.item():.4f}")
        print(f"   Policy loss: {policy_loss.item():.4f}")
        print(f"   Entropy: {entropy.item():.4f}")
        
        return {
            'policy_loss': policy_loss.item(),
            'entropy': entropy.item(),
            'total_loss': loss.item()
        }
    
    def evaluate(self, num_episodes: int = 10):
        """Evaluate policy on real market data"""
        print(f"\n📊 Evaluating on {num_episodes} episodes...")
        
        eval_returns = []
        eval_lengths = []
        
        for _ in range(num_episodes):
            symbol = np.random.choice(self.symbols)
            env = self.envs[symbol]
            
            obs, _ = env.reset()
            done = False
            episode_return = 0
            episode_length = 0
            
            while not done:
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    action_probs = self.policy(obs_tensor).cpu().numpy()[0]
                    action = np.argmax(action_probs)  # Greedy for eval
                
                obs, reward, terminated, truncated, _ = env.step(action)
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
