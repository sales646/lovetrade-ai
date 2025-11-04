#!/usr/bin/env python3
"""PNU Quick Training Launcher - BC→PPO with all available data"""

from distributed_orchestrator import DistributedRLOrchestrator

config = {
    'world_size': 4,  # Use all 4 A100 GPUs
    'envs_per_gpu': 2048,  # Massive parallelization for A100s - 8192 total environments
    'use_bf16': True,
    'use_tf32': True,
    'pbt_enabled': True,
    'population_size': 2,
    'exploit_interval': 5,
    'model_type': 'transformer',
    'state_dim': 52,
    'action_dim': 3,
    'd_model': 1024,
    'nhead': 16,
    'num_layers': 8,
    'dim_feedforward': 4096,
    'dropout': 0.1,
    'total_timesteps': 50_000_000,
    'epochs': 100,
    'steps_per_rollout': 8192,
    'batch_size': 65536,  # Increased for larger env count
    'learning_rate': 3e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_param': 0.2,
    'ppo_epochs': 4,
    
    # PNU settings
    'auto_discover_symbols': True,
    'enable_multi_market': True,
    'crypto_stock_ratio': 0.7,
    'bc_pretrain': True,
    'bc_epochs': 5000,
    'confidence_threshold': 0.6,
    'augment_data': True,
}

if __name__ == '__main__':
    print("🧠 PNU Training System - MAXIMUM SCALE")
    print(f"   BC Pretraining → PPO Reinforcement Learning")
    print(f"   Auto-discovering all symbols...")
    print(f"   🚀 A100 BEAST MODE: {config['envs_per_gpu']} envs/GPU")
    print(f"   📊 Total environments: {config['world_size'] * config['envs_per_gpu']}")
    print(f"   Crypto:Stock ratio: 70:30")
    
    orchestrator = DistributedRLOrchestrator(config=config)
    orchestrator.setup()
    orchestrator.train()
