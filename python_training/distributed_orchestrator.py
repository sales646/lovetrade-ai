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


class DistributedRLOrchestrator:
    """Master orchestrator with BC→PPO pipeline"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.distributed_trainer = None
        self.pbt_scheduler = None
        self.gpu_monitor = None
        self.load_balancer = None
        self.run_id = f"pnu_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.checkpoint_dir = Path(f"checkpoints/{self.run_id}")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
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
        
        # Auto-discover symbols
        if self.config.get('auto_discover_symbols', False):
            from data_discovery import load_discovered_symbols
            symbols_data = load_discovered_symbols()
            crypto_ratio = self.config.get('crypto_stock_ratio', 0.7)
            n_crypto = int(len(symbols_data['crypto']) * crypto_ratio)
            n_stocks = len(symbols_data['stocks']) - n_crypto
            self.config['symbols'] = symbols_data['crypto'][:n_crypto] + symbols_data['stocks'][:n_stocks]
            print(f"✅ Discovered {len(self.config['symbols'])} symbols")
        
        # GPU setup
        gpu_info = check_gpu_availability()
        if gpu_info['available']:
            self.config['world_size'] = min(self.config['world_size'], gpu_info['count'])
            print(f"✅ Using {self.config['world_size']} GPUs")
        
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
    
    def train(self):
        """BC→PPO training pipeline"""
        print("🎯 Starting BC→PPO Training Pipeline\n")
        
        # Phase 1: BC Pretraining
        if self.config.get('bc_pretrain', True):
            print("📚 PHASE 1: BC Pretraining")
            from bc_pretrain import pretrain_bc
            env = create_trading_env(symbols=self.config['symbols'], enable_multi_market=True, augment=True)
            
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
        try:
            for epoch in range(self.config['epochs']):
                print(f"\n📅 Epoch {epoch+1}/{self.config['epochs']}")
                
                env_fn = lambda: create_trading_env(symbols=self.config['symbols'], 
                                                   enable_multi_market=True, phase="train")
                
                metrics = self.distributed_trainer.launch(
                    config=self.config, model_class=TransformerPolicy, env_fn=env_fn
                )
                
                if (epoch + 1) % 10 == 0:
                    print(f"💾 Saving checkpoint...")
                    torch.save({'epoch': epoch, 'config': self.config}, 
                              self.checkpoint_dir / f"epoch_{epoch+1}.pt")
                
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted")
        finally:
            self.cleanup()
    
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
