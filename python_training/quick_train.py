#!/usr/bin/env python3
"""Quick training launcher with multi-market support (stocks + crypto)"""

from distributed_orchestrator import DistributedRLOrchestrator

# Multi-market config: 70% crypto, 30% stock for optimal learning
config = {
    'world_size': 2,
    'envs_per_gpu': 16,  # FORCE 16 envs per GPU
    'use_bf16': True,
    'pbt_enabled': True,
    'population_size': 2,
    'exploit_interval': 5,
    'model_type': 'transformer',
    'state_dim': 52,  # Updated for multi-market features
    'action_dim': 3,
    'd_model': 1024,
    'nhead': 16,
    'num_layers': 8,
    'dim_feedforward': 4096,
    'dropout': 0.1,
    'total_timesteps': 50_000_000,
    'epochs': 100,
    'steps_per_rollout': 8192,
    'batch_size': 32768,
    'learning_rate': 3e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_param': 0.2,
    'ppo_epochs': 4,
    
    # Multi-market settings
    'enable_multi_market': True,
    'crypto_stock_ratio': 0.7,  # 70% crypto, 30% stock
}

print("🌍 Starting MULTI-MARKET training with FORCED config:")
print(f"   State dim: {config['state_dim']} (includes market_type + normalized_volatility)")
print(f"   Envs per GPU: {config['envs_per_gpu']}")
print(f"   Total envs: {config['world_size'] * config['envs_per_gpu']}")
print(f"   Multi-market: {config['enable_multi_market']}")
print(f"   Crypto:Stock ratio: {config['crypto_stock_ratio']*100:.0f}:{(1-config['crypto_stock_ratio'])*100:.0f}")

orchestrator = DistributedRLOrchestrator(config=config)
orchestrator.setup()
orchestrator.train()
