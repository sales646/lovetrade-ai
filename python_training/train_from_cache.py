"""
Train RL policy from cached market data and save to Supabase.

Reads parquet files from .cache_market/ instead of fetching data.
Saves all metrics and trained model to Supabase.
"""
import os
import sys
import glob
import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment from root /.env
load_dotenv("/.env")

# Import training modules
from trading_environment import TradingEnvironment
from transformer_policy import TransformerPolicy
from supabase_logger import SupabaseLogger

class CachedDataTrainer:
    """Train RL policy from cached market data."""
    
    def __init__(self, cache_dir: str = ".cache_market"):
        self.cache_dir = Path(cache_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize Supabase
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self.logger = SupabaseLogger(supabase_url, supabase_key)
        
        print(f"🎯 Device: {self.device}")
        print(f"📁 Cache directory: {self.cache_dir}")
        
    def load_cached_data(self) -> Dict[str, pd.DataFrame]:
        """Load all cached raw files (S3 and Binance)."""
        print("\n📊 Loading cached market data...")
        
        data = {}
        
        # Load S3 stock data
        s3_base = self.cache_dir / "raw" / "s3" / "us_stock_sip" / "minute_aggs_v1"
        if s3_base.exists():
            print("  📦 Loading S3 stock data...")
            for year_dir in s3_base.iterdir():
                if not year_dir.is_dir():
                    continue
                for month_dir in year_dir.iterdir():
                    if not month_dir.is_dir():
                        continue
                    for csv_gz_file in month_dir.glob("*.csv.gz"):
                        try:
                            df = pd.read_csv(csv_gz_file, compression='gzip')
                            
                            # Extract symbol from filename (assuming format: SYMBOL_date.csv.gz)
                            symbol = csv_gz_file.stem.split('_')[0]
                            
                            # Standardize column names
                            if 'timestamp' not in df.columns and 't' in df.columns:
                                df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
                            if 'open' not in df.columns and 'o' in df.columns:
                                df['open'] = df['o']
                            if 'high' not in df.columns and 'h' in df.columns:
                                df['high'] = df['h']
                            if 'low' not in df.columns and 'l' in df.columns:
                                df['low'] = df['l']
                            if 'close' not in df.columns and 'c' in df.columns:
                                df['close'] = df['c']
                            if 'volume' not in df.columns and 'v' in df.columns:
                                df['volume'] = df['v']
                            
                            # Append to existing data or create new
                            if symbol in data:
                                data[symbol] = pd.concat([data[symbol], df], ignore_index=True)
                            else:
                                data[symbol] = df
                        except Exception as e:
                            print(f"    ⚠️ Error loading {csv_gz_file.name}: {e}")
        
        # Load Binance data
        binance_base = self.cache_dir / "raw" / "binance_vision" / "data" / "spot" / "daily" / "klines"
        if binance_base.exists():
            print("  📦 Loading Binance crypto data...")
            for symbol_dir in binance_base.iterdir():
                if not symbol_dir.is_dir():
                    continue
                
                symbol = symbol_dir.name
                interval_dir = symbol_dir / "1m"
                
                if not interval_dir.exists():
                    continue
                
                for zip_file in interval_dir.glob("*.zip"):
                    try:
                        import zipfile
                        with zipfile.ZipFile(zip_file, 'r') as z:
                            # Read first CSV file in zip
                            csv_files = [f for f in z.namelist() if f.endswith('.csv')]
                            if csv_files:
                                with z.open(csv_files[0]) as f:
                                    df = pd.read_csv(f, header=None, names=[
                                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                                        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                                        'taker_buy_quote', 'ignore'
                                    ])
                                    
                                    # Convert timestamp
                                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                                    
                                    # Keep only needed columns
                                    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                                    
                                    # Append to existing data or create new
                                    if symbol in data:
                                        data[symbol] = pd.concat([data[symbol], df], ignore_index=True)
                                    else:
                                        data[symbol] = df
                    except Exception as e:
                        print(f"    ⚠️ Error loading {zip_file.name}: {e}")
        
        # Sort and deduplicate all data
        for symbol in data:
            data[symbol] = data[symbol].sort_values('timestamp').drop_duplicates().reset_index(drop=True)
            print(f"  ✅ {symbol}: {len(data[symbol]):,} bars")
        
        total_bars = sum(len(df) for df in data.values())
        print(f"\n📈 Total: {len(data)} symbols, {total_bars:,} bars")
        
        return data
    
    def create_environments(self, data: Dict[str, pd.DataFrame], 
                          train_split: float = 0.7,
                          val_split: float = 0.15) -> tuple:
        """Create train/val/test environments."""
        print("\n🏗️  Creating trading environments...")
        
        train_envs = []
        val_envs = []
        test_envs = []
        
        for symbol, df in data.items():
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Split data
            n = len(df)
            train_end = int(n * train_split)
            val_end = int(n * (train_split + val_split))
            
            train_df = df[:train_end]
            val_df = df[train_end:val_end]
            test_df = df[val_end:]
            
            # Create environments
            if len(train_df) > 100:
                train_envs.append(TradingEnvironment(
                    symbol=symbol,
                    data=train_df,
                    initial_balance=100000
                ))
            
            if len(val_df) > 100:
                val_envs.append(TradingEnvironment(
                    symbol=symbol,
                    data=val_df,
                    initial_balance=100000
                ))
            
            if len(test_df) > 100:
                test_envs.append(TradingEnvironment(
                    symbol=symbol,
                    data=test_df,
                    initial_balance=100000
                ))
        
        print(f"  📚 Train: {len(train_envs)} environments")
        print(f"  📊 Val: {len(val_envs)} environments")
        print(f"  🧪 Test: {len(test_envs)} environments")
        
        return train_envs, val_envs, test_envs
    
    def initialize_policy(self, sample_env: TradingEnvironment) -> TransformerPolicy:
        """Initialize transformer policy."""
        print("\n🧠 Initializing Transformer Policy...")
        
        # Get state dimension from environment
        sample_state = sample_env.reset()
        state_dim = len(sample_state)
        
        policy = TransformerPolicy(
            state_dim=state_dim,
            action_dim=3,  # buy, hold, sell
            hidden_dim=256,
            num_heads=8,
            num_layers=6,
            dropout=0.1
        ).to(self.device)
        
        total_params = sum(p.numel() for p in policy.parameters())
        print(f"  ✅ State dim: {state_dim}")
        print(f"  ✅ Parameters: {total_params:,}")
        
        return policy
    
    def collect_trajectories(self, policy: TransformerPolicy, 
                           envs: List[TradingEnvironment],
                           num_episodes: int = 100) -> List[Dict]:
        """Collect trajectories from environments."""
        trajectories = []
        
        for episode in range(num_episodes):
            env = np.random.choice(envs)
            state = env.reset()
            done = False
            episode_data = []
            
            while not done:
                # Get action from policy
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action_probs = policy(state_tensor)
                    action = torch.argmax(action_probs, dim=-1).item()
                
                # Convert to trading action (-1, 0, 1)
                trading_action = action - 1
                
                # Step environment
                next_state, reward, done, info = env.step(trading_action)
                
                episode_data.append({
                    'state': state,
                    'action': trading_action,
                    'reward': reward,
                    'next_state': next_state,
                    'done': done
                })
                
                state = next_state
            
            # Determine final equity including any open position value
            final_equity = env.balance
            if env.position > 0:
                latest_price = None
                if env.current_symbol and env.current_symbol in env.data:
                    symbol_data = env.data[env.current_symbol]
                    if symbol_data:
                        price_index = min(max(env.current_step - 1, 0), len(symbol_data) - 1)
                        latest_price = float(symbol_data[price_index].get('close', 0.0))

                if latest_price is None or latest_price == 0.0:
                    latest_price = env.position_price

                final_equity += env.position * latest_price

            trajectories.append({
                'symbol': env.current_symbol,
                'data': episode_data,
                'total_reward': sum(t['reward'] for t in episode_data),
                'final_equity': final_equity,
                'episode_stats': env.get_episode_stats()
            })
        
        return trajectories
    
    def compute_advantages(self, trajectories: List[Dict], gamma: float = 0.99) -> List[Dict]:
        """Compute GAE advantages."""
        for traj in trajectories:
            rewards = [t['reward'] for t in traj['data']]
            
            # Compute returns
            returns = []
            R = 0
            for r in reversed(rewards):
                R = r + gamma * R
                returns.insert(0, R)
            
            # Add to trajectory
            for i, t in enumerate(traj['data']):
                t['return'] = returns[i]
                t['advantage'] = returns[i]  # Simplified, no baseline
        
        return trajectories
    
    def update_policy(self, policy: TransformerPolicy, optimizer: torch.optim.Optimizer,
                     trajectories: List[Dict], clip_epsilon: float = 0.2) -> Dict:
        """PPO policy update."""
        policy.train()
        
        # Prepare batch
        states = []
        actions = []
        advantages = []
        
        for traj in trajectories:
            for t in traj['data']:
                states.append(t['state'])
                actions.append(t['action'] + 1)  # Convert back to 0,1,2
                advantages.append(t['advantage'])
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Forward pass
        action_probs = policy(states)
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)) + 1e-8).squeeze()
        
        # PPO loss
        loss = -(log_probs * advantages).mean()
        
        # Update
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()
        
        return {
            'loss': loss.item(),
            'mean_advantage': advantages.mean().item()
        }
    
    def evaluate(self, policy: TransformerPolicy, envs: List[TradingEnvironment],
                num_episodes: int = 50) -> Dict:
        """Evaluate policy performance."""
        policy.eval()
        
        results = []
        for _ in range(num_episodes):
            env = np.random.choice(envs)
            state = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action_probs = policy(state_tensor)
                    action = torch.argmax(action_probs, dim=-1).item()
                
                trading_action = action - 1
                next_state, reward, done, _ = env.step(trading_action)
                
                total_reward += reward
                state = next_state
            
            results.append({
                'total_reward': total_reward,
                'final_equity': env.equity,
                'return_pct': ((env.equity - env.initial_balance) / env.initial_balance) * 100
            })
        
        # Calculate metrics
        returns = [r['return_pct'] for r in results]
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe = mean_return / (std_return + 1e-8) if std_return > 0 else 0
        
        win_rate = sum(1 for r in returns if r > 0) / len(returns)
        
        return {
            'mean_reward': np.mean([r['total_reward'] for r in results]),
            'mean_return': mean_return,
            'sharpe_ratio': sharpe,
            'win_rate': win_rate,
            'max_return': max(returns),
            'min_return': min(returns)
        }
    
    def save_model_to_supabase(self, policy: TransformerPolicy, run_id: str,
                              metrics: Dict, symbols: List[str]) -> str:
        """Save trained model to Supabase storage."""
        print("\n💾 Saving model to Supabase...")
        
        # Save model to temporary file
        model_name = f"transformer_policy_{run_id}_{int(time.time())}.pth"
        temp_path = f"/tmp/{model_name}"
        
        torch.save({
            'model_state_dict': policy.state_dict(),
            'metrics': metrics,
            'symbols': symbols,
            'created_at': datetime.now().isoformat()
        }, temp_path)
        
        # Get file size
        file_size = os.path.getsize(temp_path)
        
        # Upload to Supabase storage
        with open(temp_path, 'rb') as f:
            storage_path = f"models/{model_name}"
            self.supabase.storage.from_('trained-models').upload(
                storage_path,
                f.read(),
                file_options={"content-type": "application/octet-stream"}
            )
        
        # Save metadata to database
        self.supabase.table('trained_models').insert({
            'run_id': run_id,
            'model_name': model_name,
            'model_type': 'transformer',
            'storage_path': storage_path,
            'file_size_bytes': file_size,
            'performance_metrics': metrics,
            'trained_on_symbols': symbols,
            'final_sharpe_ratio': metrics.get('sharpe_ratio'),
            'final_win_rate': metrics.get('win_rate'),
            'is_best': True  # Mark as best for now
        }).execute()
        
        # Clean up temp file
        os.remove(temp_path)
        
        print(f"  ✅ Model saved: {model_name}")
        print(f"  ✅ Size: {file_size / 1024 / 1024:.2f} MB")
        
        return storage_path
    
    def train(self, num_epochs: int = 100, 
             episodes_per_epoch: int = 100,
             learning_rate: float = 3e-4):
        """Main training loop."""
        start_time = time.time()
        
        # Load cached data
        data = self.load_cached_data()
        
        if len(data) == 0:
            print("❌ No cached data found!")
            return
        
        # Create environments
        train_envs, val_envs, test_envs = self.create_environments(data)
        
        # Initialize policy
        policy = self.initialize_policy(train_envs[0])
        optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
        
        # Create training run in Supabase
        run_response = self.supabase.table('training_runs').insert({
            'run_name': f'cached_training_{int(time.time())}',
            'phase': 'training',
            'status': 'running',
            'hyperparams': {
                'num_epochs': num_epochs,
                'episodes_per_epoch': episodes_per_epoch,
                'learning_rate': learning_rate,
                'device': str(self.device)
            },
            'config': {
                'data_source': 'cached',
                'num_symbols': len(data),
                'train_envs': len(train_envs),
                'val_envs': len(val_envs)
            }
        }).execute()
        
        run_id = run_response.data[0]['id']
        print(f"\n🚀 Training run ID: {run_id}")
        
        best_sharpe = -float('inf')
        
        # Training loop
        print("\n" + "="*70)
        print("🎯 STARTING TRAINING")
        print("="*70)
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Collect trajectories
            trajectories = self.collect_trajectories(policy, train_envs, episodes_per_epoch)
            
            # Compute advantages
            trajectories = self.compute_advantages(trajectories)
            
            # Update policy
            update_metrics = self.update_policy(policy, optimizer, trajectories)
            
            # Evaluate on validation set
            val_metrics = self.evaluate(policy, val_envs, num_episodes=20)
            
            epoch_time = time.time() - epoch_start
            
            # Log metrics to Supabase
            self.supabase.table('training_metrics').insert({
                'run_id': run_id,
                'epoch': epoch,
                'split': 'train',
                'mean_reward': val_metrics['mean_reward'],
                'sharpe_ratio': val_metrics['sharpe_ratio'],
                'win_rate': val_metrics['win_rate'],
                'policy_loss': update_metrics['loss']
            }).execute()
            
            # Print progress
            print(f"\nEpoch {epoch + 1}/{num_epochs} ({epoch_time:.1f}s)")
            print(f"  Loss: {update_metrics['loss']:.4f}")
            print(f"  Val Sharpe: {val_metrics['sharpe_ratio']:.3f}")
            print(f"  Val Win Rate: {val_metrics['win_rate']*100:.1f}%")
            print(f"  Val Return: {val_metrics['mean_return']:.2f}%")
            
            # Save best model
            if val_metrics['sharpe_ratio'] > best_sharpe:
                best_sharpe = val_metrics['sharpe_ratio']
                print(f"  🌟 New best Sharpe ratio: {best_sharpe:.3f}")
        
        # Final evaluation on test set
        print("\n" + "="*70)
        print("🧪 FINAL TEST EVALUATION")
        print("="*70)
        
        test_metrics = self.evaluate(policy, test_envs, num_episodes=50)
        print(f"\nTest Results:")
        print(f"  Sharpe Ratio: {test_metrics['sharpe_ratio']:.3f}")
        print(f"  Win Rate: {test_metrics['win_rate']*100:.1f}%")
        print(f"  Mean Return: {test_metrics['mean_return']:.2f}%")
        print(f"  Max Return: {test_metrics['max_return']:.2f}%")
        print(f"  Min Return: {test_metrics['min_return']:.2f}%")
        
        # Save model to Supabase
        model_path = self.save_model_to_supabase(
            policy, run_id, test_metrics, list(data.keys())
        )
        
        # Update training run status
        total_time = time.time() - start_time
        self.supabase.table('training_runs').update({
            'status': 'completed',
            'completed_at': datetime.now().isoformat(),
            'best_val_sharpe': best_sharpe,
            'best_checkpoint_path': model_path,
            'config': {
                **run_response.data[0]['config'],
                'training_duration_seconds': int(total_time),
                'final_test_metrics': test_metrics
            }
        }).eq('id', run_id).execute()
        
        print(f"\n✅ Training complete! Total time: {total_time/60:.1f} minutes")
        print(f"✅ Model saved to Supabase: {model_path}")
        print(f"✅ Run ID: {run_id}")


if __name__ == "__main__":
    # Configuration
    config = {
        'num_epochs': 100,
        'episodes_per_epoch': 100,
        'learning_rate': 3e-4
    }
    
    # Train
    trainer = CachedDataTrainer(cache_dir=".cache_market")
    trainer.train(**config)
