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
import torch.nn.functional as F
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
from bc_pretrain import pretrain_bc

class CachedDataTrainer:
    """Train RL policy from cached market data."""
    
    def __init__(self, cache_dir: str = ".cache_market"):
        self.cache_dir = Path(cache_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize Supabase
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self.logger = SupabaseLogger()
        
        print(f"🎯 Device: {self.device}")
        print(f"📁 Cache directory: {self.cache_dir}")
        
    def _normalize_timestamp_column(self, df: pd.DataFrame, column: str) -> pd.Series:
        """Return a timezone-naive datetime series from the provided column."""
        series = df[column]

        if pd.api.types.is_datetime64_any_dtype(series):
            return series

        numeric_series = pd.to_numeric(series, errors='coerce')
        if numeric_series.notna().any():
            magnitude = numeric_series.abs().max()

            if pd.isna(magnitude):
                return pd.to_datetime(series, errors='coerce')

            if magnitude >= 10 ** 18:
                unit = 'ns'
            elif magnitude >= 10 ** 15:
                unit = 'us'
            else:
                unit = 'ms'

            return pd.to_datetime(numeric_series, unit=unit, errors='coerce')

        return pd.to_datetime(series, errors='coerce')

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
                            
                            # Normalize timestamp column before further processing
                            if 'timestamp' in df.columns:
                                df['timestamp'] = self._normalize_timestamp_column(df, 'timestamp')
                            elif 't' in df.columns:
                                df['timestamp'] = self._normalize_timestamp_column(df, 't')

                            # Extract symbol from filename (assuming format: SYMBOL_date.csv.gz)
                            symbol = csv_gz_file.stem.split('_')[0]

                            # Standardize column names
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
                                    
                                    # Normalize timestamp to handle mixed resolutions
                                    df['timestamp'] = self._normalize_timestamp_column(df, 'timestamp')
                                    
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

            def _prepare_records(split_df: pd.DataFrame):
                if len(split_df) <= 100:
                    return None
                return split_df.to_dict('records')

            train_records = _prepare_records(train_df)
            if train_records is not None:
                train_envs.append(
                    TradingEnvironment(
                        symbols=[symbol],
                        external_data={symbol: train_records},
                        walk_forward=False,
                        enable_multi_market=False,
                        initial_balance=100000
                    )
                )

            val_records = _prepare_records(val_df)
            if val_records is not None:
                val_envs.append(
                    TradingEnvironment(
                        symbols=[symbol],
                        external_data={symbol: val_records},
                        walk_forward=False,
                        enable_multi_market=False,
                        initial_balance=100000
                    )
                )

            test_records = _prepare_records(test_df)
            if test_records is not None:
                test_envs.append(
                    TradingEnvironment(
                        symbols=[symbol],
                        external_data={symbol: test_records},
                        walk_forward=False,
                        enable_multi_market=False,
                        initial_balance=100000
                    )
                )
        
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

        if len(envs) == 0:
            return trajectories

        policy.eval()

        for _ in range(num_episodes):
            env = np.random.choice(envs)
            state = env.reset()
            done = False
            episode_data = []

            while not done:
                state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    action, size, value, log_prob, entropy, _ = policy(state_tensor)
                    action = int(action.item())
                    size = int(size.item())
                    value = float(value.item())
                    log_prob = float(log_prob.item())

                next_state, reward, done, info = env.step(action)

                if done:
                    next_value = 0.0
                else:
                    next_state_tensor = torch.as_tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
                    with torch.no_grad():
                        next_value = float(policy.get_value(next_state_tensor).item())

                episode_data.append({
                    'state': state,
                    'action': action,
                    'size': size,
                    'log_prob': log_prob,
                    'value': value,
                    'reward': reward,
                    'done': done,
                    'next_value': next_value
                })

                state = next_state

            final_symbol = env.current_symbol
            final_price = 0.0
            if final_symbol in env.data and env.current_step > 0:
                last_index = min(env.current_step - 1, len(env.data[final_symbol]) - 1)
                final_price = float(env.data[final_symbol][last_index]['close'])
            final_equity = env.balance + env.position * final_price

            trajectories.append({
                'symbol': final_symbol,
                'data': episode_data,
                'total_reward': sum(t['reward'] for t in episode_data),
                'final_equity': final_equity
            })

        return trajectories

    @staticmethod
    def _compute_env_equity(env: TradingEnvironment) -> float:
        """Estimate current equity from balance and open position."""
        balance = getattr(env, 'balance', 0.0)
        position = getattr(env, 'position', 0.0)
        symbol = getattr(env, 'current_symbol', None)
        data = getattr(env, 'data', {})

        if symbol and symbol in data and len(data[symbol]) > 0:
            current_step = max(0, getattr(env, 'current_step', 0) - 1)
            current_step = min(current_step, len(data[symbol]) - 1)
            last_price = float(data[symbol][current_step].get('close', 0.0))
        else:
            last_price = 0.0

        return float(balance) + float(position) * last_price
    
    def compute_advantages(self, trajectories: List[Dict], gamma: float = 0.99,
                           lam: float = 0.95) -> List[Dict]:
        """Compute GAE advantages using policy value estimates."""
        for traj in trajectories:
            advantage = 0.0
            returns = 0.0

            for step in reversed(traj['data']):
                td_error = step['reward'] + gamma * step['next_value'] - step['value']
                advantage = td_error + gamma * lam * advantage
                returns = step['value'] + advantage
                step['advantage'] = advantage
                step['return'] = returns

        return trajectories

    def update_policy(self, policy: TransformerPolicy, optimizer: torch.optim.Optimizer,
                     trajectories: List[Dict], clip_epsilon: float = 0.2,
                     value_coef: float = 0.5, entropy_coef: float = 0.01) -> Dict:
        """PPO policy update using stored trajectories."""
        policy.train()

        states = []
        actions = []
        sizes = []
        old_log_probs = []
        returns = []
        advantages = []

        for traj in trajectories:
            for step in traj['data']:
                states.append(step['state'])
                actions.append(step['action'])
                sizes.append(step['size'])
                old_log_probs.append(step['log_prob'])
                returns.append(step['return'])
                advantages.append(step['advantage'])

        if len(states) == 0:
            return {
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy': 0.0,
                'approx_kl': 0.0
            }

        states = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        sizes = torch.as_tensor(sizes, dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        log_probs, entropy, values, _ = policy.evaluate_actions(states, actions, sizes)

        ratios = torch.exp(log_probs - old_log_probs)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = F.mse_loss(values, returns)
        entropy_loss = -entropy.mean()

        loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()

        with torch.no_grad():
            approx_kl = 0.5 * torch.mean((old_log_probs - log_probs) ** 2)

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.mean().item(),
            'approx_kl': approx_kl.item()
        }
    
    def evaluate(self, policy: TransformerPolicy, envs: List[TradingEnvironment],
                num_episodes: int = 50) -> Dict:
        """Evaluate policy performance."""
        policy.eval()

        results = []
        if len(envs) == 0:
            return {
                'mean_reward': 0.0,
                'mean_return': 0.0,
                'sharpe_ratio': 0.0,
                'win_rate': 0.0,
                'max_return': 0.0,
                'min_return': 0.0
            }
        for _ in range(num_episodes):
            env = np.random.choice(envs)
            state = env.reset()
            done = False
            total_reward = 0

            while not done:
                state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    action = int(policy.get_action(state_tensor, deterministic=True).item())

                next_state, reward, done, _ = env.step(action)

                total_reward += reward
                state = next_state

            final_symbol = env.current_symbol
            final_price = 0.0
            if final_symbol in env.data and env.current_step > 0:
                last_index = min(env.current_step - 1, len(env.data[final_symbol]) - 1)
                final_price = float(env.data[final_symbol][last_index]['close'])
            final_equity = env.balance + env.position * final_price

            results.append({
                'total_reward': total_reward,
                'final_equity': final_equity,
                'return_pct': ((final_equity - env.initial_balance) / env.initial_balance) * 100
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

    def maybe_run_bc_pretraining(self, policy: TransformerPolicy,
                                 data: Dict[str, pd.DataFrame],
                                 run_id: Optional[str] = None) -> Optional[Dict]:
        """Run BC pretraining if checkpoint missing and load weights."""
        checkpoint_path = Path("checkpoints/bc/bc_best.pt")
        symbol_records = {
            symbol: df.sort_values('timestamp').to_dict('records')
            for symbol, df in data.items() if len(df) > 0
        }

        if len(symbol_records) == 0:
            print("⚠️  No data available for BC pretraining.")
            return None

        if checkpoint_path.exists():
            print("📦 Loading existing BC checkpoint...")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            policy.load_state_dict(checkpoint['model_state_dict'])
            policy.to(self.device)
            policy.eval()
            metrics = {
                'final_loss': checkpoint.get('loss'),
                'final_accuracy': checkpoint.get('accuracy'),
                'epochs_trained': checkpoint.get('epoch', 0) + 1,
            }
        else:
            bc_env = TradingEnvironment(
                symbols=list(symbol_records.keys()),
                external_data=symbol_records,
                walk_forward=False,
                enable_multi_market=True,
                initial_balance=100000
            )
            metrics = pretrain_bc(
                policy,
                symbols=list(symbol_records.keys()),
                env=bc_env,
                device=str(self.device)
            )
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                policy.load_state_dict(checkpoint['model_state_dict'])
            policy.to(self.device)
            policy.eval()

        if run_id is not None and metrics is not None:
            self.supabase.table('training_metrics').insert({
                'run_id': run_id,
                'epoch': 0,
                'split': 'bc',
                'mean_reward': None,
                'sharpe_ratio': None,
                'win_rate': metrics.get('final_accuracy'),
                'policy_loss': metrics.get('final_loss')
            }).execute()

        return metrics
    
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
        if len(train_envs) == 0:
            print("❌ No training environments could be created.")
            return

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

        bc_metrics = self.maybe_run_bc_pretraining(policy, data, run_id)
        if bc_metrics:
            print("\n🎓 BC Pretraining Summary:")
            for key, value in bc_metrics.items():
                print(f"  {key}: {value}")

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
                'policy_loss': update_metrics['policy_loss'],
                'value_loss': update_metrics['value_loss'],
                'entropy': update_metrics['entropy'],
                'approx_kl': update_metrics['approx_kl']
            }).execute()

            # Print progress
            print(f"\nEpoch {epoch + 1}/{num_epochs} ({epoch_time:.1f}s)")
            print(f"  Policy Loss: {update_metrics['policy_loss']:.4f}")
            print(f"  Value Loss: {update_metrics['value_loss']:.4f}")
            print(f"  Entropy: {update_metrics['entropy']:.4f}")
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
