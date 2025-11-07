"""Distributed Orchestrator with BC+PPO Pipeline and Auto-Discovery"""
import os
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from dotenv import load_dotenv

load_dotenv()

from distributed_training import DistributedTrainer, check_gpu_availability
from pbt_scheduler import AdaptivePBTScheduler
from transformer_policy import TransformerPolicy, LightweightTransformerPolicy
from gpu_monitor import GPUMonitor, LoadBalancer
from trading_environment import create_trading_env
from s3_data_loader import S3MarketDataLoader


# Module-level function for multiprocessing pickle compatibility
def _create_env_for_training(symbols, enable_multi_market=True, phase="train", external_data=None):
    """Environment factory for distributed training"""
    return create_trading_env(
        symbols=symbols,
        enable_multi_market=enable_multi_market,
        phase=phase,
        external_data=external_data
    )


class DistributedRLOrchestrator:
    """Master orchestrator with BC→PPO pipeline"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.use_cache = self.config.get('use_cache', Path(".cache_market").exists())
        self.config['use_cache'] = self.use_cache
        self.distributed_trainer = None
        self.pbt_scheduler = None
        self.gpu_monitor = None
        self.load_balancer = None
        self.run_id = f"pnu_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.checkpoint_dir = Path(f"checkpoints/{self.run_id}")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.external_data = None
    
    def _default_config(self) -> Dict:
        """A100-optimized configuration"""
        return {
            'world_size': 2, 'envs_per_gpu': 256, 'use_bf16': True, 'use_tf32': True,
            'pbt_enabled': True, 'population_size': 2, 'exploit_interval': 5,
            'model_type': 'transformer', 'state_dim': 52, 'action_dim': 3,
            'd_model': 1024, 'nhead': 16, 'num_layers': 8, 'dim_feedforward': 4096, 'dropout': 0.1,
            'total_timesteps': 50_000_000, 'epochs': 100, 'steps_per_rollout': 8192, 'batch_size': 32768,
            'learning_rate': 3e-4, 'gamma': 0.99, 'gae_lambda': 0.95, 'clip_param': 0.2, 'ppo_epochs': 4,
            'auto_discover_symbols': True, 'symbols': [], 'timeframe': '1Min',
            'augment_data': True, 'enable_multi_market': True, 'crypto_stock_ratio': 0.7,
            'bc_pretrain': True, 'bc_epochs': 5000, 'confidence_threshold': 0.6,
            'log_interval': 10, 'save_interval': 50
        }
    
    def setup(self):
        """Initialize all components"""
        print("\n🚀 PNU Training System Setup")
        print("="*70)
        
        if self.use_cache:
            print("📦 Loading market data from local cache...")
            from train_from_cache import CachedDataTrainer

            cache_dir = Path(self.config.get('cache_dir', ".cache_market"))
            cached_data = CachedDataTrainer(cache_dir).load_cached_data() or {}
            self.external_data = cached_data
            self.config['symbols'] = list(cached_data.keys())
            print(f"✅ Loaded {len(self.config['symbols'])} symbols from cache")
        else:
            # Use ProductionDataFetcher for real market data
            print("📥 Fetching REAL market data (Polygon S3 + Binance + Yahoo)...")
            from production_data_fetcher import ProductionDataFetcher
            from datetime import datetime, timedelta

            # Yahoo Finance 1-minute data is limited to 7 days
            # For longer ranges, Polygon S3 and Binance provide historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)  # Last 7 days for Yahoo 1-min data

            fetcher = ProductionDataFetcher(
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d")
            )

            market_data, news_data = fetcher.fetch_all()

            # Filter symbols with enough bars (min 1000)
            min_bars = 1000
            valid_symbols = [s for s, df in market_data.items() if len(df) >= min_bars]

            # Store data in Supabase for training
            print(f"💾 Storing {len(valid_symbols)} symbols to Supabase...")
            self._store_data_to_supabase(market_data, news_data, valid_symbols)

            self.config['symbols'] = valid_symbols
            print(f"✅ Ready with {len(valid_symbols)} symbols")
            print(f"   Data range: {start_date.date()} to {end_date.date()}")
        
        # GPU setup
        gpu_info = check_gpu_availability()
        if gpu_info['available']:
            self.config['world_size'] = min(self.config['world_size'], gpu_info['count'])
            print(f"✅ Using {self.config['world_size']} GPUs")
        else:
            print("⚠️ No GPUs detected. Switching to CPU training mode.")
            self.config['world_size'] = 1
            self.config['use_bf16'] = False

        # Initialize components
        self.gpu_monitor = GPUMonitor(refresh_interval=2.0)
        self.gpu_monitor.start()
        self.load_balancer = LoadBalancer(self.gpu_monitor)
        
        self.distributed_trainer = DistributedTrainer(
            world_size=self.config['world_size'],
            use_bf16=self.config['use_bf16'],
            envs_per_gpu=self.config['envs_per_gpu']
        )
        
        if self.config['pbt_enabled']:
            self.pbt_scheduler = AdaptivePBTScheduler(
                population_size=self.config['population_size'],
                exploit_interval=self.config['exploit_interval']
            )
            base_hyperparams = {
                'learning_rate': self.config['learning_rate'],
                'gamma': self.config['gamma'],
                'clip_param': self.config['clip_param'],
            }
            self.pbt_scheduler.initialize_population(base_hyperparams)
        
        print("✅ Setup complete\n")
    
    def _store_data_to_supabase(self, market_data, news_data, valid_symbols):
        """Store fetched data to Supabase historical_bars"""
        from supabase import create_client
        supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_ROLE_KEY"))
        
        for symbol in valid_symbols:
            df = market_data[symbol]
            
            # Convert to historical_bars format
            bars = []
            for _, row in df.iterrows():
                bars.append({
                    'symbol': symbol,
                    'timestamp': row['timestamp'].isoformat(),
                    'timeframe': '1Min',
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': int(row['volume'])
                })
            
            # Batch insert (1000 at a time) - skip duplicates
            batch_size = 1000
            for i in range(0, len(bars), batch_size):
                batch = bars[i:i+batch_size]
                try:
                    # Use upsert with ignore_duplicates to skip existing records
                    supabase.table('historical_bars').upsert(
                        batch, 
                        on_conflict='symbol,timeframe,timestamp',
                        ignore_duplicates=True
                    ).execute()
                except Exception as e:
                    # Silently skip duplicate errors, log others
                    if '23505' not in str(e):  # Not a duplicate key error
                        print(f"  ⚠️ Error storing {symbol} batch {i}: {e}")
        
        print("✅ Data stored to Supabase")
    
    def train(self):
        """BC→PPO training pipeline"""
        print("🎯 Starting BC→PPO Training Pipeline\n")
        
        # Phase 1: BC Pretraining
        if self.config.get('bc_pretrain', True):
            print("📚 PHASE 1: BC Pretraining")
            from bc_pretrain import pretrain_bc
            env = create_trading_env(
                symbols=self.config['symbols'],
                enable_multi_market=True,
                augment=True,
                external_data=self.external_data
            )
            
            from transformer_policy import TransformerPolicy
            policy = TransformerPolicy(
                state_dim=self.config['state_dim'], action_dim=self.config['action_dim'],
                d_model=self.config['d_model'], nhead=self.config['nhead'],
                num_layers=self.config['num_layers'], dim_feedforward=self.config['dim_feedforward']
            )
            
            bc_results = pretrain_bc(policy=policy, symbols=self.config['symbols'], env=env,
                                    epochs=self.config.get('bc_epochs', 5000))
            print(f"✅ BC Complete: {bc_results['final_accuracy']:.2%}\n")
        
        # Phase 2: PPO Training
        print("🎮 PHASE 2: PPO Training")
        
        # Store symbols for env creation
        symbols = self.config['symbols']
        
        try:
            for epoch in range(self.config['epochs']):
                print(f"\n📅 Epoch {epoch+1}/{self.config['epochs']}")
                
                # Use module-level function for pickling
                from functools import partial
                make_env = partial(
                    _create_env_for_training,
                    symbols=symbols,
                    external_data=self.external_data
                )
                
                metrics = self.distributed_trainer.launch(
                    config=self.config, model_class=TransformerPolicy, env_fn=make_env
                )
                
                # Log metrics to Supabase
                self._log_metrics_to_supabase(epoch + 1, metrics)
                
                if (epoch + 1) % 10 == 0:
                    print(f"💾 Saving checkpoint...")
                    torch.save({'epoch': epoch, 'config': self.config}, 
                              self.checkpoint_dir / f"epoch_{epoch+1}.pt")
                
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted")
        finally:
            self.cleanup()
    
    def _log_metrics_to_supabase(self, epoch: int, metrics: dict):
        """Log training metrics to Supabase"""
        from supabase import create_client
        supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_ROLE_KEY"))
        
        try:
            supabase.table('training_metrics').insert({
                'run_id': self.run_id,
                'epoch': epoch,
                'split': 'train',
                'policy_loss': metrics.get('policy_loss', 0),
                'value_loss': metrics.get('value_loss', 0),
                'entropy': metrics.get('entropy', 0),
                'mean_reward': metrics.get('mean_reward', 0),
                'metadata': {
                    'world_size': self.config['world_size'],
                    'envs_per_gpu': self.config['envs_per_gpu'],
                    'total_envs': self.config['world_size'] * self.config['envs_per_gpu']
                }
            }).execute()
        except Exception as e:
            print(f"⚠️ Failed to log to Supabase: {e}")
    
    def cleanup(self):
        """Cleanup"""
        if self.gpu_monitor:
            self.gpu_monitor.stop()


def main():
    orchestrator = DistributedRLOrchestrator()
    orchestrator.setup()
    orchestrator.train()


if __name__ == "__main__":
    main()
