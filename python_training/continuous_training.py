#!/usr/bin/env python3
"""
Continuous Training Loop - Optimized for 4x A100 GPUs
Repeats training automatically with data from all sources
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from dotenv import load_dotenv
import time

load_dotenv()

from unified_data_fetcher import UnifiedDataFetcher
from distributed_training import DistributedTrainer, check_gpu_availability
from transformer_policy import TransformerPolicy
from trading_environment import create_trading_env
from functools import partial


def _create_env_for_training(symbols, enable_multi_market=True, phase="train"):
    """Environment factory for distributed training"""
    return create_trading_env(
        symbols=symbols,
        enable_multi_market=enable_multi_market,
        phase=phase
    )


class ContinuousTrainer:
    """Continuous training loop optimized for A100 GPUs"""
    
    def __init__(self):
        self.config = self._a100_optimized_config()
        self.run_count = 0
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def _a100_optimized_config(self) -> Dict:
        """Maximum A100 GPU utilization"""
        return {
            # GPU Configuration - MAX POWER
            'world_size': 4,                    # 4x A100
            'envs_per_gpu': 512,               # 512 parallel envs per GPU = 2048 total
            'use_bf16': True,                  # BF16 for A100
            'use_tf32': True,                  # TF32 for matrix ops
            'gradient_accumulation_steps': 4,   # Effective batch size multiplier
            
            # Model - Large Transformer
            'state_dim': 52,
            'action_dim': 3,
            'd_model': 2048,                   # Large model
            'nhead': 32,                       # Many attention heads
            'num_layers': 12,                  # Deep network
            'dim_feedforward': 8192,           # Wide FFN
            'dropout': 0.1,
            
            # Training - Aggressive
            'total_timesteps': 100_000_000,    # 100M timesteps per run
            'epochs': 200,                     # 200 epochs per run
            'steps_per_rollout': 16384,        # Large rollouts
            'batch_size': 65536,               # Massive batches (64K)
            'learning_rate': 1e-4,             # Lower LR for stability
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_param': 0.2,
            'ppo_epochs': 8,                   # More PPO updates
            
            # Data
            'min_bars': 2000,                  # Minimum data quality
            'timeframe': '1Min',
            'enable_multi_market': True,
            
            # Logging
            'log_interval': 5,
            'save_interval': 25
        }
    
    def load_data(self):
        """Load data from all sources"""
        print("\n" + "="*70)
        print(f"🔄 RUN #{self.run_count + 1} - Loading fresh data from all sources")
        print("="*70)
        
        fetcher = UnifiedDataFetcher()
        
        # Load 2 years of data
        symbol_data = fetcher.fetch_all(
            start_date="2022-01-01",
            end_date="2024-01-31"
        )
        
        # Get symbols with sufficient data
        valid_symbols = fetcher.get_symbols_with_data(
            symbol_data, 
            min_bars=self.config['min_bars']
        )
        
        if len(valid_symbols) < 50:
            print(f"⚠️ Only {len(valid_symbols)} symbols available, need more data!")
            return []
        
        # Use all valid symbols
        self.config['symbols'] = valid_symbols
        
        print(f"\n✅ Loaded {len(valid_symbols)} symbols for training")
        print(f"   Total bars: {sum(len(symbol_data[s]) for s in valid_symbols):,}")
        
        return valid_symbols
    
    def train_one_run(self):
        """Execute one complete training run"""
        self.run_count += 1
        run_start = time.time()
        
        print("\n" + "="*70)
        print(f"🚀 STARTING RUN #{self.run_count}")
        print("="*70)
        print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load data
        symbols = self.load_data()
        if not symbols:
            print("❌ No data available, skipping run")
            return False
        
        # GPU info
        gpu_info = check_gpu_availability()
        if not gpu_info['available']:
            print("❌ No GPUs available!")
            return False
        
        print(f"\n💪 GPU Configuration:")
        print(f"   GPUs: {gpu_info['count']}")
        print(f"   Envs per GPU: {self.config['envs_per_gpu']}")
        print(f"   Total parallel envs: {gpu_info['count'] * self.config['envs_per_gpu']}")
        print(f"   Batch size: {self.config['batch_size']:,}")
        print(f"   Model size: {self.config['d_model']} dim, {self.config['num_layers']} layers")
        
        # Initialize trainer
        trainer = DistributedTrainer(
            world_size=self.config['world_size'],
            use_bf16=self.config['use_bf16'],
            envs_per_gpu=self.config['envs_per_gpu']
        )
        
        # Create environment factory
        make_env = partial(_create_env_for_training, symbols=symbols)
        
        # Run training
        print(f"\n🎯 Training for {self.config['epochs']} epochs...")
        
        try:
            for epoch in range(self.config['epochs']):
                print(f"\n📅 Epoch {epoch+1}/{self.config['epochs']}")
                
                metrics = trainer.launch(
                    config=self.config,
                    model_class=TransformerPolicy,
                    env_fn=make_env
                )
                
                # Save checkpoint
                if (epoch + 1) % self.config['save_interval'] == 0:
                    checkpoint_path = self.checkpoint_dir / f"run{self.run_count}_epoch{epoch+1}.pt"
                    print(f"💾 Saving checkpoint: {checkpoint_path}")
                    torch.save({
                        'run': self.run_count,
                        'epoch': epoch,
                        'config': self.config,
                        'symbols': symbols[:100]  # Save first 100 symbols for reference
                    }, checkpoint_path)
        
        except KeyboardInterrupt:
            print("\n⚠️ Training interrupted by user")
            return False
        
        except Exception as e:
            print(f"\n❌ Error during training: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Run complete
        run_time = time.time() - run_start
        print("\n" + "="*70)
        print(f"✅ RUN #{self.run_count} COMPLETE")
        print("="*70)
        print(f"⏰ Duration: {run_time/3600:.2f} hours")
        print(f"🎯 Epochs: {self.config['epochs']}")
        print(f"💾 Checkpoints saved: {self.config['epochs'] // self.config['save_interval']}")
        
        return True
    
    def run_continuous(self):
        """Run training continuously"""
        print("\n" + "="*80)
        print("🔄 CONTINUOUS TRAINING MODE - Will repeat until stopped")
        print("="*80)
        print("Press Ctrl+C to stop")
        print("="*80)
        
        try:
            while True:
                success = self.train_one_run()
                
                if not success:
                    print("\n⚠️ Run failed, waiting 60 seconds before retry...")
                    time.sleep(60)
                    continue
                
                print(f"\n✅ Completed {self.run_count} runs")
                print("🔄 Starting next run in 10 seconds...")
                time.sleep(10)
        
        except KeyboardInterrupt:
            print("\n\n" + "="*80)
            print("🛑 CONTINUOUS TRAINING STOPPED")
            print("="*80)
            print(f"✅ Total runs completed: {self.run_count}")
            print(f"💾 Checkpoints saved in: {self.checkpoint_dir}")


def main():
    # Set CUDA memory allocation strategy for better memory usage
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    trainer = ContinuousTrainer()
    trainer.run_continuous()


if __name__ == "__main__":
    main()
