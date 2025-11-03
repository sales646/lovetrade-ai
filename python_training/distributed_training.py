"""
Distributed RL Training System - 8 GPUs with DDP
Supports: 8000 parallel environments, BF16 precision, Transformer policies
"""
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import json

class DistributedTrainer:
    """Manages distributed training across multiple GPUs"""
    
    def __init__(
        self,
        world_size: int = 8,
        backend: str = "nccl",
        use_bf16: bool = True,
        envs_per_gpu: int = 1000
    ):
        self.world_size = world_size
        self.backend = backend
        self.use_bf16 = use_bf16
        self.envs_per_gpu = envs_per_gpu
        self.total_envs = world_size * envs_per_gpu
        
    def setup(self, rank: int, world_size: int):
        """Initialize distributed process group"""
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(self.backend, rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        
    def cleanup(self):
        """Clean up distributed process group"""
        dist.destroy_process_group()
        
    def train_worker(
        self,
        rank: int,
        world_size: int,
        config: Dict,
        model_class,
        env_fn,
        data_queue=None
    ):
        """Training worker for each GPU"""
        print(f"🚀 Starting worker on GPU {rank}/{world_size}")
        self.setup(rank, world_size)
        
        # Create model on this GPU
        try:
            model = model_class(config).to(rank)
        except TypeError:
            # Fallback if model_class doesn't accept config
            model = model_class(
                state_dim=config.get('state_dim', 50),
                action_dim=config.get('action_dim', 3)
            ).to(rank)
        
        if self.use_bf16:
            model = model.to(torch.bfloat16)
        
        # Wrap with DDP
        ddp_model = DDP(model, device_ids=[rank])
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            ddp_model.parameters(),
            lr=config.get("learning_rate", 3e-4),
            weight_decay=config.get("weight_decay", 1e-5)
        )
        
        # Training loop for this worker
        for epoch in range(config.get("epochs", 10)):
            # Generate rollouts on this GPU
            rollouts = self._collect_rollouts(
                rank, 
                ddp_model, 
                env_fn, 
                self.envs_per_gpu,
                config
            )
            
            # Train on rollouts
            metrics = self._train_epoch(
                ddp_model,
                optimizer,
                rollouts,
                config
            )
            
            # Sync gradients across GPUs
            dist.barrier()
            
            # Log from rank 0
            if rank == 0:
                print(f"Epoch {epoch}: Loss={metrics['loss']:.4f}, "
                      f"Reward={metrics['mean_reward']:.2f}")
                if data_queue:
                    data_queue.put({
                        'epoch': epoch,
                        'rank': rank,
                        'metrics': metrics
                    })
        
        self.cleanup()
        
    def _collect_rollouts(
        self,
        rank: int,
        model,
        env_fn,
        num_envs: int,
        config: Dict
    ) -> Dict:
        """Collect rollouts from parallel environments"""
        model.eval()
        
        # Create vectorized environments for this GPU
        try:
            envs = [env_fn() for _ in range(num_envs)]
            states = [env.reset() for env in envs]
        except Exception as e:
            print(f"⚠️  Warning on GPU {rank}: Could not create environments: {e}")
            # Return empty rollouts
            return {
                'states': [],
                'actions': [],
                'rewards': [],
                'dones': [],
                'values': []
            }
        
        rollouts = {
            'states': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'values': []
        }
        
        steps_per_rollout = config.get("steps_per_rollout", 512)
        
        with torch.no_grad():
            for step in range(steps_per_rollout):
                # Batch states
                state_batch = torch.FloatTensor(states).to(rank)
                if self.use_bf16:
                    state_batch = state_batch.to(torch.bfloat16)
                
                # Get actions from policy
                actions, values, log_probs = model(state_batch)
                
                # Step environments
                new_states = []
                for i, (env, action) in enumerate(zip(envs, actions)):
                    next_state, reward, done, info = env.step(action.cpu().float().numpy())
                    
                    rollouts['states'].append(states[i])
                    rollouts['actions'].append(action.cpu().float().numpy())
                    rollouts['rewards'].append(reward)
                    rollouts['dones'].append(done)
                    rollouts['values'].append(values[i].cpu().float().numpy())
                    
                    if done:
                        next_state = env.reset()
                    new_states.append(next_state)
                
                states = new_states
        
        return rollouts
    
    def _train_epoch(
        self,
        model,
        optimizer,
        rollouts: Dict,
        config: Dict
    ) -> Dict:
        """Train model on collected rollouts"""
        model.train()
        
        # Convert rollouts to tensors
        states = torch.FloatTensor(rollouts['states']).to(model.device)
        actions = torch.FloatTensor(rollouts['actions']).to(model.device)
        rewards = torch.FloatTensor(rollouts['rewards']).to(model.device)
        values = torch.FloatTensor(rollouts['values']).to(model.device)
        
        if self.use_bf16:
            states = states.to(torch.bfloat16)
            values = values.to(torch.bfloat16)
        
        # Compute advantages (GAE)
        advantages = self._compute_gae(rewards, values, config)
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training batches
        batch_size = config.get("batch_size", 256)
        num_batches = len(states) // batch_size
        total_loss = 0
        
        for _ in range(config.get("ppo_epochs", 4)):
            indices = torch.randperm(len(states))
            
            for i in range(num_batches):
                batch_indices = indices[i * batch_size:(i + 1) * batch_size]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Forward pass
                _, new_values, new_log_probs = model(batch_states)
                
                # PPO loss
                ratio = torch.exp(new_log_probs - model.get_log_probs(batch_actions))
                clip_param = config.get("clip_param", 0.2)
                
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = 0.5 * (new_values - batch_returns).pow(2).mean()
                
                loss = policy_loss + 0.5 * value_loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                
                total_loss += loss.item()
        
        return {
            'loss': total_loss / (num_batches * config.get("ppo_epochs", 4)),
            'mean_reward': rewards.mean().item(),
            'std_reward': rewards.std().item()
        }
    
    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        config: Dict
    ) -> torch.Tensor:
        """Compute Generalized Advantage Estimation"""
        gamma = config.get("gamma", 0.99)
        gae_lambda = config.get("gae_lambda", 0.95)
        
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value - values[t]
            gae = delta + gamma * gae_lambda * gae
            advantages[t] = gae
        
        return advantages
    
    def launch(
        self,
        config: Dict,
        model_class,
        env_fn
    ):
        """Launch distributed training"""
        print(f"🚀 Launching distributed training on {self.world_size} GPUs")
        print(f"📊 Total environments: {self.total_envs}")
        print(f"⚡ BF16 precision: {self.use_bf16}")
        
        # Create queue for metrics
        ctx = mp.get_context('spawn')
        data_queue = ctx.Queue()
        
        # Spawn workers
        mp.spawn(
            self.train_worker,
            args=(self.world_size, config, model_class, env_fn, data_queue),
            nprocs=self.world_size,
            join=True
        )
        
        # Collect metrics
        all_metrics = []
        while not data_queue.empty():
            all_metrics.append(data_queue.get())
        
        return all_metrics


def check_gpu_availability() -> Dict:
    """Check GPU availability and configuration"""
    if not torch.cuda.is_available():
        return {
            'available': False,
            'count': 0,
            'error': 'CUDA not available'
        }
    
    gpu_count = torch.cuda.device_count()
    gpus = []
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        gpus.append({
            'id': i,
            'name': props.name,
            'memory_gb': props.total_memory / 1024**3,
            'compute_capability': f"{props.major}.{props.minor}"
        })
    
    return {
        'available': True,
        'count': gpu_count,
        'gpus': gpus,
        'nccl_available': dist.is_nccl_available()
    }


if __name__ == "__main__":
    # Test GPU setup
    info = check_gpu_availability()
    print(json.dumps(info, indent=2))
