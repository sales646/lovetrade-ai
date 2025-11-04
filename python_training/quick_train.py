#!/usr/bin/env python3
"""PNU Quick Training Launcher - BC→PPO with all available data"""

from distributed_orchestrator import DistributedRLOrchestrator

config = {
    'world_size': 4,  # Use all 4 A100 GPUs
    'envs_per_gpu': 4096,  # EXTREME parallelization - 16,384 total environments
    'use_bf16': True,
    'use_tf32': True,
    'pbt_enabled': True,
    'population_size': 2,
    'exploit_interval': 5,
    'model_type': 'transformer',
    'state_dim': 52,
    'action_dim': 3,
    'd_model': 2048,  # MASSIVE model - 2x larger
    'nhead': 32,  # 2x more attention heads
    'num_layers': 16,  # 2x deeper network
    'dim_feedforward': 8192,  # 2x larger feedforward
    'dropout': 0.1,
    'total_timesteps': 50_000_000,
    'epochs': 100,
    'steps_per_rollout': 32768,  # 4x larger rollouts
    'batch_size': 262144,  # 4x larger batches - MASSIVE
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
    print("🚀🚀🚀 PNU EXTREME SCALE TRAINING 🚀🚀🚀")
    print(f"   BC Pretraining → PPO Reinforcement Learning")
    print(f"   Auto-discovering all symbols...")
    print(f"   💪 INSANE MODE: {config['envs_per_gpu']} envs/GPU = {config['world_size'] * config['envs_per_gpu']:,} TOTAL")
    print(f"   🧠 MASSIVE MODEL: {config['d_model']}-dim, {config['num_layers']} layers")
    print(f"   📦 HUGE BATCHES: {config['batch_size']:,} samples")
    print(f"   ⚡ This WILL saturate your A100s!")
    print(f"   Crypto:Stock ratio: 70:30")
    print("\n" + "="*60)
    
    print("\n[STEP 1/3] Creating orchestrator...")
    orchestrator = DistributedRLOrchestrator(config=config)
    
    print("[STEP 2/3] Running setup (GPU monitor, load balancer, trainer)...")
    orchestrator.setup()
    print("✅ Setup complete!\n")
    
    print("[STEP 3/3] Starting training...")
    orchestrator.train()
