#!/usr/bin/env python3
"""Quick training launcher with correct GPU settings"""

from distributed_orchestrator import DistributedRLOrchestrator

# Force correct config
config = {
    'world_size': 2,
    'envs_per_gpu': 16,  # FORCE 16 envs per GPU
    'use_bf16': True,
    'pbt_enabled': True,
    'population_size': 2,
    'exploit_interval': 5,
    'model_type': 'transformer',
    'state_dim': 50,
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
}

print("🚀 Starting training with FORCED config:")
print(f"   Envs per GPU: {config['envs_per_gpu']}")
print(f"   Total envs: {config['world_size'] * config['envs_per_gpu']}")

orchestrator = DistributedRLOrchestrator(config=config)
orchestrator.setup()
orchestrator.train()
