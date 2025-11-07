"""
Distributed RL Training System - 8 GPUs with DDP
Supports: 8000 parallel environments, BF16 precision, Transformer policies
"""
import os
import socket
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
        envs_per_gpu: int = 16  # Balanced for CPU/GPU pipeline
    ):
        self.requested_world_size = world_size
        self.backend = backend
        self.use_bf16 = use_bf16
        self.envs_per_gpu = envs_per_gpu
        self.is_cuda_available = torch.cuda.is_available()

        if self.is_cuda_available:
            max_devices = torch.cuda.device_count()
            if world_size > max_devices:
                print(
                    f"⚠️  Requested world_size {world_size} exceeds available CUDA devices ({max_devices}). "
                    "Clamping to available devices."
                )
            self.world_size = max(1, min(world_size, max_devices))
        else:
            if backend == "nccl":
                self.backend = "gloo"
            self.world_size = 1
            self.use_bf16 = False

        self.total_envs = self.world_size * self.envs_per_gpu
        
    def setup(self, rank: int, world_size: int):
        """Initialize distributed process group"""
        # MASTER_ADDR and MASTER_PORT should already be set by launch()
        import datetime
        timeout = datetime.timedelta(minutes=10)
        dist.init_process_group(
            self.backend,
            rank=rank,
            world_size=world_size,
            timeout=timeout
        )
        if self.is_cuda_available:
            torch.cuda.set_device(rank)
    
    @staticmethod
    def _find_free_port() -> int:
        """Find a free port for distributed training"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port
        
    def cleanup(self):
        """Clean up distributed process group"""
        if dist.is_initialized():
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
        try:
            # Enable TF32 for H100 performance boost
            if self.is_cuda_available:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            device = torch.device("cuda", rank) if self.is_cuda_available else torch.device("cpu")
            device_label = f"GPU {rank}" if self.is_cuda_available else "CPU"

            print(f"🚀 Starting worker on {device_label} ({rank}/{world_size})")
            print(f"[{device_label}] Setting up distributed process group...")
            self.setup(rank, world_size)
            print(f"[{device_label}] Process group initialized!")

            # Set GPU and enable optimizations
            if self.is_cuda_available:
                torch.cuda.set_device(rank)
                torch.cuda.empty_cache()  # Clear any stale memory

            # Create model on this GPU
            print(f"[{device_label}] Creating model...")
            try:
                model = model_class(config).to(device)
            except TypeError:
                # Fallback if model_class doesn't accept config
                model = model_class(
                    state_dim=config.get('state_dim', 50),
                    action_dim=config.get('action_dim', 3)
                ).to(device)
            print(f"[{device_label}] Model created and moved to {device}!")

            if self.use_bf16 and device.type == "cuda":
                model = model.to(torch.bfloat16)

            # Wrap with DDP
            if self.is_cuda_available:
                ddp_model = DDP(model, device_ids=[rank])
            else:
                ddp_model = DDP(model)

            # Create optimizer with fused=True for H100 optimization
            optimizer_kwargs = {
                "lr": config.get("learning_rate", 3e-4),
                "weight_decay": config.get("weight_decay", 1e-5)
            }
            if self.is_cuda_available:
                optimizer_kwargs["fused"] = True  # H100 optimization

            try:
                optimizer = torch.optim.AdamW(
                    ddp_model.parameters(),
                    **optimizer_kwargs
                )
            except TypeError:
                optimizer_kwargs.pop("fused", None)
                optimizer = torch.optim.AdamW(
                    ddp_model.parameters(),
                    **optimizer_kwargs
                )
            
            # Training loop for this worker
            print(f"[{device_label}] Starting training loop for {config.get('epochs', 10)} epochs...")
            for epoch in range(config.get("epochs", 10)):
                print(f"[{device_label}] Epoch {epoch+1}: Collecting rollouts...")
                # Generate rollouts on this GPU
                rollouts = self._collect_rollouts(
                    rank,
                    ddp_model,
                    env_fn,
                    self.envs_per_gpu,
                    config
                )
                print(f"[{device_label}] Epoch {epoch+1}: Collected {len(rollouts['rewards'])} steps")
                
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
        
        except Exception as e:
            print(f"❌ Worker {rank} error: {e}")
            import traceback
            traceback.print_exc()
        finally:
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
        device = next(model.parameters()).device
        device_label = f"GPU {rank}" if device.type == "cuda" else "CPU"

        # Create vectorized environments for this GPU
        print(f"[{device_label}] Creating {num_envs} environments...")
        try:
            envs = [env_fn() for _ in range(num_envs)]
            print(f"[{device_label}] Environments created, resetting...")
            states = [env.reset() for env in envs]
            print(f"[{device_label}] Environments ready!")
        except Exception as e:
            print(f"⚠️  Warning on {device_label}: Could not create environments: {e}")
            import traceback
            traceback.print_exc()
            # Return empty rollouts
            return {
                'states': [],
                'actions': [],
                'rewards': [],
                'dones': [],
                'values': [],
                'log_probs': []
            }
        
        rollouts = {
            'states': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'values': [],
            'log_probs': []  # Store old log probs for PPO
        }
        
        steps_per_rollout = config.get("steps_per_rollout", 8192)  # Much larger rollouts
        
        with torch.no_grad():
            for step in range(steps_per_rollout):
                # Batch states - keep everything on GPU for speed
                state_batch = torch.as_tensor(
                    np.array(states),
                    dtype=torch.float32,
                    device=device
                )
                if self.use_bf16 and device.type == "cuda":
                    state_batch = state_batch.to(torch.bfloat16)
                
                # Get actions from policy
                actions, values, log_probs = model(state_batch)
                
                # Step environments - vectorize for efficiency
                new_states = []
                actions_cpu = actions.cpu().float().numpy()
                for i, (env, action) in enumerate(zip(envs, actions_cpu)):
                    next_state, reward, done, info = env.step(action)
                    
                    rollouts['states'].append(states[i])
                    rollouts['actions'].append(action)  # action is already numpy
                    rollouts['rewards'].append(float(reward))  # Ensure scalar
                    rollouts['dones'].append(float(done))  # Ensure scalar
                    rollouts['values'].append(values[i].cpu().float().item())  # Use .item() for scalar
                    rollouts['log_probs'].append(log_probs[i].cpu().float().item())  # Use .item() for scalar
                    
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
        
        # Check if rollouts are empty
        if not rollouts['states'] or len(rollouts['states']) == 0:
            print("⚠️  Warning: Empty rollouts, skipping training epoch")
            return {
                'loss': 0.0,
                'mean_reward': 0.0,
                'std_reward': 0.0
            }
        
        # Convert rollouts to tensors
        # Get device from model (handle DDP wrapping)
        device = next(model.parameters()).device
        
        # Convert lists to numpy arrays first, then to tensors for efficiency
        states = torch.as_tensor(np.array(rollouts['states']), dtype=torch.float32, device=device)
        actions = torch.as_tensor(np.array(rollouts['actions']), dtype=torch.float32, device=device)
        rewards = torch.as_tensor(rollouts['rewards'], dtype=torch.float32, device=device)
        values = torch.as_tensor(rollouts['values'], dtype=torch.float32, device=device)
        old_log_probs = torch.as_tensor(rollouts['log_probs'], dtype=torch.float32, device=device)

        if self.use_bf16 and device.type == "cuda":
            states = states.to(torch.bfloat16)
            values = values.to(torch.bfloat16)
        
        # Compute advantages (GAE)
        advantages = self._compute_gae(rewards, values, config)
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training batches - MUCH LARGER for H100
        batch_size = config.get("batch_size", 32768)  # 128x larger batches
        num_batches = max(1, len(states) // batch_size)
        total_loss = 0
        
        for _ in range(config.get("ppo_epochs", 4)):
            indices = torch.randperm(len(states))
            
            for i in range(num_batches):
                batch_indices = indices[i * batch_size:(i + 1) * batch_size]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                # Forward pass - get new log probs for old actions
                # Access the underlying module if wrapped in DDP
                policy_module = model.module if hasattr(model, 'module') else model
                
                # Get new values and log probs for the old actions
                _, new_values, _ = model(batch_states)
                new_log_probs = policy_module.get_log_probs(batch_states, batch_actions)
                
                # PPO loss
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
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
        self.total_envs = self.world_size * self.envs_per_gpu
        device_label = "GPUs" if self.is_cuda_available else "CPU workers"
        if not self.is_cuda_available:
            print("⚠️  CUDA not available. Falling back to single-process CPU training with Gloo backend.")

        print(f"🚀 Launching distributed training on {self.world_size} {device_label}")
        print(f"📊 Total environments: {self.total_envs}")
        print(f"⚡ BF16 precision: {self.use_bf16}")

        # Set master address and port ONCE before spawning workers
        os.environ['MASTER_ADDR'] = 'localhost'
        if 'MASTER_PORT' not in os.environ:
            master_port = self._find_free_port()
            os.environ['MASTER_PORT'] = str(master_port)
            print(f"📡 Using MASTER_PORT: {master_port}")
        
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
