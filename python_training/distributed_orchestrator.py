"""
Distributed Training Orchestrator
Integrates: Distributed Training, PBT, Transformer Policies, Advanced Rewards
"""
import os
import sys
import json
import torch
import numpy as np
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path

from distributed_training import DistributedTrainer, check_gpu_availability
from pbt_scheduler import AdaptivePBTScheduler
from transformer_policy import TransformerPolicy, LightweightTransformerPolicy
from advanced_rewards import ProfitOptimizedRewardShaper
from gpu_monitor import GPUMonitor, LoadBalancer, print_gpu_summary
from trading_environment import TradingEnvironment, create_trading_env


class DistributedRLOrchestrator:
    """
    Master orchestrator for distributed RL training
    
    Coordinates:
    - 8 GPU workers running PPO
    - Population-based training (PBT) for hyperparameter search
    - Transformer policy networks
    - Advanced profit-optimized reward shaping
    - GPU monitoring and load balancing
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        
        # Components
        self.distributed_trainer = None
        self.pbt_scheduler = None
        self.gpu_monitor = None
        self.load_balancer = None
        
        # State
        self.run_id = f"dist_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.checkpoint_dir = Path(f"checkpoints/{self.run_id}")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def _default_config(self) -> Dict:
        """Default configuration for distributed training"""
        return {
            # Distributed settings
            'world_size': 8,  # 8 GPUs
            'envs_per_gpu': 1000,  # 8000 total environments
            'use_bf16': True,
            
            # PBT settings
            'population_size': 8,  # One per GPU
            'exploit_interval': 5,
            'pbt_enabled': True,
            
            # Model settings
            'model_type': 'transformer',  # 'transformer', 'lightweight', 'mlp'
            'state_dim': 50,
            'action_dim': 3,  # position_size, stop_loss, take_profit
            'd_model': 256,
            'nhead': 8,
            'num_layers': 4,
            
            # Training settings
            'total_timesteps': 50_000_000,  # 50M timesteps
            'epochs': 100,
            'steps_per_rollout': 512,
            'batch_size': 256,
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_param': 0.2,
            'ppo_epochs': 4,
            
            # Reward shaping
            'reward_shaper': 'profit_optimized',
            'w_profit': 1.0,
            'w_sharpe': 2.0,
            'w_leverage': 0.5,
            'w_drawdown': -3.0,
            
            # Monitoring
            'log_interval': 10,
            'save_interval': 50,
            'eval_interval': 25
        }
    
    def setup(self):
        """Initialize all components"""
        print("🚀 Setting up Distributed RL Training System")
        print("="*70)
        
        # 1. Check GPU availability
        gpu_info = check_gpu_availability()
        print(f"\n📊 GPU Info:")
        print(f"  Available: {gpu_info['available']}")
        print(f"  Count: {gpu_info['count']}")
        
        if not gpu_info['available'] or gpu_info['count'] < self.config['world_size']:
            print(f"\n⚠️  Warning: Requested {self.config['world_size']} GPUs but only "
                  f"{gpu_info['count']} available")
            print("   Adjusting world_size to match available GPUs...")
            self.config['world_size'] = max(1, gpu_info['count'])
        
        # 2. Initialize GPU monitor
        self.gpu_monitor = GPUMonitor(refresh_interval=2.0)
        self.gpu_monitor.start()
        self.load_balancer = LoadBalancer(self.gpu_monitor)
        
        print_gpu_summary()
        
        # 3. Initialize distributed trainer
        self.distributed_trainer = DistributedTrainer(
            world_size=self.config['world_size'],
            use_bf16=self.config['use_bf16'],
            envs_per_gpu=self.config['envs_per_gpu']
        )
        
        print(f"\n🎯 Distributed Trainer:")
        print(f"  World size: {self.config['world_size']} GPUs")
        print(f"  Envs per GPU: {self.config['envs_per_gpu']}")
        print(f"  Total envs: {self.distributed_trainer.total_envs}")
        print(f"  BF16 precision: {self.config['use_bf16']}")
        
        # 4. Initialize PBT scheduler
        if self.config['pbt_enabled']:
            self.pbt_scheduler = AdaptivePBTScheduler(
                population_size=self.config['population_size'],
                exploit_interval=self.config['exploit_interval']
            )
            
            # Initialize population with base hyperparams
            base_hyperparams = {
                'learning_rate': self.config['learning_rate'],
                'gamma': self.config['gamma'],
                'gae_lambda': self.config['gae_lambda'],
                'clip_param': self.config['clip_param'],
                'batch_size': self.config['batch_size']
            }
            self.pbt_scheduler.initialize_population(base_hyperparams)
            
            print(f"\n🧬 PBT Scheduler:")
            print(f"  Population size: {self.config['population_size']}")
            print(f"  Exploit interval: {self.config['exploit_interval']}")
        
        # 5. Save config
        self._save_config()
        
        print("\n✅ Setup complete!")
        print("="*70 + "\n")
    
    def train(self):
        """Run full distributed training with PBT"""
        print(f"\n🎬 Starting Distributed Training Run: {self.run_id}")
        print("="*70 + "\n")
        
        try:
            for epoch in range(self.config['epochs']):
                print(f"\n📅 Epoch {epoch + 1}/{self.config['epochs']}")
                print("-"*70)
                
                # Check thermal throttling
                thermal_status = self.load_balancer.check_thermal_throttling()
                if thermal_status['is_throttling']:
                    print("🔥 Warning: Thermal throttling detected!")
                    for gpu in thermal_status['gpus']:
                        print(f"   GPU {gpu['gpu_id']}: {gpu['temperature']:.1f}°C")
                
                # Get current hyperparams from PBT (if enabled)
                if self.pbt_scheduler:
                    population = self.pbt_scheduler.population
                    hyperparams_per_worker = {p.id: p.hyperparams for p in population}
                else:
                    hyperparams_per_worker = {0: self.config}
                
                # Create model class based on config
                model_class = self._get_model_class()
                
                # Create environment factory
                env_fn = self._create_env_factory()
                
                # Launch distributed training for this epoch
                print(f"🚀 Launching {self.config['world_size']} workers...")
                epoch_config = self.config.copy()
                epoch_config['epochs'] = 1  # One epoch per iteration
                
                metrics = self.distributed_trainer.launch(
                    config=epoch_config,
                    model_class=model_class,
                    env_fn=env_fn
                )
                
                # Process metrics
                self._process_metrics(epoch, metrics)
                
                # PBT step (exploit & explore)
                if self.pbt_scheduler and epoch % self.pbt_scheduler.exploit_interval == 0:
                    print("\n🧬 PBT: Exploit & Explore phase")
                    
                    # Collect performances
                    performances = {
                        m['rank']: m['metrics']['mean_reward']
                        for m in metrics if 'metrics' in m
                    }
                    
                    self.pbt_scheduler.step(performances)
                    
                    best = self.pbt_scheduler.get_best()
                    print(f"   Best performer: Population {best.id} "
                          f"(reward={best.performance:.2f})")
                    print(f"   LR={best.hyperparams['learning_rate']:.6f}, "
                          f"Gamma={best.hyperparams['gamma']:.4f}")
                
                # Checkpointing
                if (epoch + 1) % self.config['save_interval'] == 0:
                    self._save_checkpoint(epoch)
                
                # GPU stats
                if (epoch + 1) % self.config['log_interval'] == 0:
                    self._log_gpu_stats()
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Training interrupted by user")
            self._save_checkpoint(epoch, prefix="interrupted")
        
        except Exception as e:
            print(f"\n\n❌ Training error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.cleanup()
    
    def _get_model_class(self):
        """Get model class based on config"""
        model_type = self.config.get('model_type', 'transformer')
        
        if model_type == 'transformer':
            return TransformerPolicy
        elif model_type == 'lightweight':
            return LightweightTransformerPolicy
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _create_env_factory(self):
        """Create environment factory function"""
        # Return the real trading environment factory (picklable)
        return create_trading_env
    
    def _process_metrics(self, epoch: int, metrics: list):
        """Process and log metrics"""
        if not metrics:
            return
        
        # Aggregate metrics across workers
        total_loss = sum(m['metrics']['loss'] for m in metrics if 'metrics' in m)
        total_reward = sum(m['metrics']['mean_reward'] for m in metrics if 'metrics' in m)
        n_workers = len([m for m in metrics if 'metrics' in m])
        
        if n_workers > 0:
            avg_loss = total_loss / n_workers
            avg_reward = total_reward / n_workers
            
            print(f"\n📊 Epoch {epoch} Results:")
            print(f"   Avg Loss: {avg_loss:.4f}")
            print(f"   Avg Reward: {avg_reward:.2f}")
            
            # Save metrics
            self._save_metrics(epoch, {
                'avg_loss': avg_loss,
                'avg_reward': avg_reward,
                'n_workers': n_workers
            })
    
    def _save_metrics(self, epoch: int, metrics: Dict):
        """Save metrics to file"""
        metrics_file = self.checkpoint_dir / "metrics.jsonl"
        with open(metrics_file, 'a') as f:
            f.write(json.dumps({
                'epoch': epoch,
                'timestamp': datetime.now().isoformat(),
                **metrics
            }) + '\n')
    
    def _save_checkpoint(self, epoch: int, prefix: str = ""):
        """Save training checkpoint"""
        checkpoint_name = f"{prefix}checkpoint_epoch_{epoch}.pt" if prefix else f"checkpoint_epoch_{epoch}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        checkpoint = {
            'epoch': epoch,
            'run_id': self.run_id,
            'config': self.config,
            'pbt_state': self.pbt_scheduler.__dict__ if self.pbt_scheduler else None
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Saved checkpoint: {checkpoint_path}")
    
    def _save_config(self):
        """Save configuration"""
        config_file = self.checkpoint_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def _log_gpu_stats(self):
        """Log GPU statistics"""
        summary = self.gpu_monitor.get_summary()
        if summary.get('available'):
            print(f"\n🎮 GPU Stats:")
            print(f"   Avg Utilization: {summary['avg_utilization']:.1f}%")
            print(f"   Avg Memory: {summary['avg_memory_used']:.1f} GB")
            print(f"   Avg Temp: {summary['avg_temperature']:.1f}°C")
    
    def cleanup(self):
        """Cleanup resources"""
        print("\n🧹 Cleaning up...")
        if self.gpu_monitor:
            self.gpu_monitor.stop()
        print("✅ Cleanup complete")


def main():
    """Main entry point"""
    # Use None to get full default config
    orchestrator = DistributedRLOrchestrator(config=None)
    orchestrator.setup()
    orchestrator.train()


if __name__ == "__main__":
    main()
