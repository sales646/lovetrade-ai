#!/usr/bin/env python3
"""Simple GPU Training - BC + PPO"""

import argparse
import os
from typing import Optional, Tuple

from supabase_logger import SupabaseLogger

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint_sequential
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

import sys
from progress import PhaseProgress

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimized GPU training")
    parser.add_argument("--epochs", type=int, default=7500, help="Number of PPO epochs")
    parser.add_argument("--bc-epochs", type=int, default=2000, help="Number of BC epochs")
    parser.add_argument("--bc-episodes", type=int, default=20000, help="Expert episodes for BC pretraining")
    parser.add_argument("--bc-steps", type=int, default=256, help="Steps per expert episode")
    parser.add_argument("--bc-batch-size", type=int, default=8192, help="Global BC batch size")
    parser.add_argument("--bc-workers", type=int, default=0, help="Data loader workers for BC stage")
    parser.add_argument("--ppo-rollout-steps", type=int, default=2048, help="Rollout steps per epoch")
    parser.add_argument("--ppo-mini-batch", type=int, default=512, help="Mini-batch size for PPO updates")
    parser.add_argument("--ppo-workers", type=int, default=0, help="Data loader workers for PPO stage")
    parser.add_argument("--rollout-envs", type=int, default=8, help="Vectorized environments for PPO rollouts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mixed-precision", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--compile-mode", default="reduce-overhead", help="torch.compile mode")
    parser.add_argument("--checkpoint-splits", type=int, default=3, help="Gradient checkpoint partitions")
    parser.add_argument(
        "--data-start",
        default="2010-01-01",
        help="Historical data start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--data-end",
        default="2024-12-31",
        help="Historical data end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--symbol-limit",
        type=int,
        default=7500,
        help="Maximum number of symbols to include in training",
    )
    parser.add_argument("--enable-cuda-graph", action="store_true", help="Capture CUDA graph for BC training")

    parser.add_argument(
        "--repeat-trainings",
        type=int,
        default=3,
        help="Number of end-to-end training cycles to run sequentially",
    )
    
    # Early stopping arguments
    parser.add_argument("--early-stopping-patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--validation-split", type=float, default=0.2, help="Validation data split ratio")
    parser.add_argument("--no-early-stopping", action="store_true", help="Disable early stopping")
    parser.add_argument("--min-delta", type=float, default=0.001, help="Minimum improvement for early stopping")
    parser.add_argument("--top-k-checkpoints", type=int, default=3, help="Keep top K checkpoints")
    parser.add_argument(
        "--supabase-log-interval",
        type=int,
        default=10,
        help="Epoch interval for logging metrics to Supabase",
    )
    parser.add_argument(
        "--disable-supabase-logging",
        action="store_true",
        help="Disable Supabase logging even if credentials are present",
    )

    return parser.parse_args()


def setup_distributed() -> Tuple[int, int, int, bool]:
    """Initialise distributed environment if launched with torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank, True

    return 0, 1, 0, False


def setup_gpu_environment(local_rank: int, rank: int) -> torch.device:
    if not torch.cuda.is_available():
        if rank == 0:
            print("❌ No GPU detected!")
        sys.exit(1)

    device = torch.device("cuda", local_rank)
    if rank == 0:
        gpu_name = torch.cuda.get_device_name(device)
        gpu_props = torch.cuda.get_device_properties(device)
        gpu_memory = gpu_props.total_memory / 1024 ** 3
        print("=" * 70)
        print("SIMPLE GPU TRAINING")
        print("=" * 70)
        print("\n[STEP 1/6] GPU Setup")
        print(f"✅ Using: {gpu_name} ({gpu_memory:.1f} GB)")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    return device


# ====================
# Early Stopping Class
# ====================

class EarlyStopping:
    """
    Advanced Early Stopping with:
    - Patience-based stopping on validation metric
    - Divergence detection (train vs val)
    - Top-K checkpoint tracking
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        top_k: int = 3,
        metric_name: str = "val_loss",
        mode: str = "min"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.top_k = top_k
        self.metric_name = metric_name
        self.mode = mode
        
        self.counter = 0
        self.best_metric = float('inf') if mode == "min" else float('-inf')
        self.best_epoch = 0
        self.best_checkpoint = None
        self.top_checkpoints = []  # List of (metric, epoch, path)
        
        self.train_metrics = []
        self.val_metrics = []
    
    def __call__(
        self,
        epoch: int,
        val_metric: float,
        train_metric: Optional[float] = None,
        checkpoint_path: Optional[str] = None
    ) -> bool:
        """
        Returns True if training should stop
        """
        
        # Track metrics for divergence detection
        self.val_metrics.append(val_metric)
        if train_metric is not None:
            self.train_metrics.append(train_metric)
        
        # Check if improved
        is_better = (
            (self.mode == "min" and val_metric < self.best_metric - self.min_delta) or
            (self.mode == "max" and val_metric > self.best_metric + self.min_delta)
        )
        
        if is_better:
            self.best_metric = val_metric
            self.best_epoch = epoch
            self.counter = 0
            
            if checkpoint_path:
                self.best_checkpoint = checkpoint_path
                # Add to top-K
                self.top_checkpoints.append((val_metric, epoch, checkpoint_path))
                # Sort and keep top K
                if self.mode == "min":
                    self.top_checkpoints.sort(key=lambda x: x[0])
                else:
                    self.top_checkpoints.sort(key=lambda x: x[0], reverse=True)
                self.top_checkpoints = self.top_checkpoints[:self.top_k]
        else:
            self.counter += 1
        
        # Divergence detection: if train-val gap is widening rapidly
        if len(self.train_metrics) >= 5 and len(self.val_metrics) >= 5:
            recent_train = np.mean(self.train_metrics[-5:])
            recent_val = np.mean(self.val_metrics[-5:])
            gap = abs(recent_val - recent_train)
            
            if self.mode == "min" and gap > 0.3:  # Val loss >> Train loss
                return True
        
        # Patience exceeded
        if self.counter >= self.patience:
            return True
        
        return False
    
    def get_summary(self) -> str:
        """Returns a summary of the early stopping state"""
        return (
            f"Best {self.metric_name}: {self.best_metric:.6f} at epoch {self.best_epoch}\n"
            f"Top-{self.top_k} checkpoints: {[(m, e) for m, e, _ in self.top_checkpoints]}"
        )


# ====================
# STEP 2: Load Data
# ====================
def load_data(rank: int, args: argparse.Namespace):
    if rank == 0:
        print("\n[STEP 2/6] Loading Data from S3")
    from s3_data_loader import S3DataLoader

    loader = S3DataLoader()

    if rank == 0:
        print("   Discovering symbols...")

    discovery_limit = max(args.symbol_limit * 2, args.symbol_limit + 50)
    symbols = loader.discover_all_symbols(max_symbols=discovery_limit)

    if rank == 0:
        print(f"✅ Found {len(symbols)} symbols")

    if not symbols:
        return loader, {}, []

    symbols_to_use = symbols[: min(args.symbol_limit, len(symbols))]

    if rank == 0:
        print(f"   Using {len(symbols_to_use)} symbols for training")
        print(
            f"   Loading market data ({args.data_start} → {args.data_end})..."
        )

    data_map = loader.load_multi_day_data(
        start_date=args.data_start,
        end_date=args.data_end,
        symbols=symbols_to_use,
    )
    if rank == 0:
        total_rows = sum(len(rows) for rows in data_map.values())
        print(f"✅ Loaded {total_rows:,} rows across {len(data_map)} symbols")

    return loader, data_map, symbols_to_use

# ====================
# STEP 3: Environment
# ====================
def create_environment(data_map, symbols, rank: int, args: argparse.Namespace):
    if rank == 0:
        print("\n[STEP 3/6] Creating Trading Environment")
    from trading_environment import TradingEnvironment

    env_kwargs = dict(symbols=symbols, external_data=data_map, enable_multi_market=True)
    env = TradingEnvironment(**env_kwargs)
    if rank == 0:
        print("✅ Environment ready")
        print(f"   Symbols: {len(symbols)} | Date range: {args.data_start} → {args.data_end}")
    return env, env_kwargs

# ====================
# STEP 4: Neural Network
# ====================

class SimplePolicy(nn.Module):
    """Simple but effective policy network"""

    def __init__(self, state_dim=52, action_dim=3, hidden=512, checkpoint_splits: int = 3):
        super().__init__()
        self.feature_layers = nn.ModuleList(
            [
                nn.Linear(state_dim, hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            ]
        )
        self.policy_head = nn.Linear(hidden, action_dim)
        self.value_head = nn.Linear(hidden, 1)
        self.checkpoint_splits = max(1, checkpoint_splits)

    def forward_features(self, x):
        return checkpoint_sequential(list(self.feature_layers), self.checkpoint_splits, x)

    def forward(self, x):
        features = self.forward_features(x)
        action_logits = self.policy_head(features)
        value = self.value_head(features)
        return action_logits, value


def build_model(device: torch.device, args: argparse.Namespace, world_size: int, rank: int, progress: PhaseProgress):
    if rank == 0:
        print("\n[STEP 4/6] Creating Neural Network")
        print("   Initializing policy network...")
    
    policy = SimplePolicy(checkpoint_splits=args.checkpoint_splits).to(device)

    if rank == 0:
        print("   Initializing optimizer...")
    
    optimizer_kwargs = dict(lr=0.0003, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    try:
        optimizer = optim.AdamW(policy.parameters(), fused=True, **optimizer_kwargs)
    except TypeError:
        optimizer = optim.AdamW(policy.parameters(), **optimizer_kwargs)

    # Make torch.compile optional and less aggressive
    if args.compile_mode != "none":
        if rank == 0:
            print(f"   Compiling model (mode: {args.compile_mode})...")
            print("   ⏳ This may take 5-15 minutes on first run, please wait...")
        
        with progress.track("Model Compilation", total=100, unit="%") as pbar:
            if rank == 0:
                pbar.update(10)
            policy = torch.compile(policy, mode=args.compile_mode, fullgraph=False)
            if rank == 0:
                pbar.update(90)
        
        if rank == 0:
            print("   ✅ Compilation complete")
    else:
        if rank == 0:
            print("   ⏩ Skipping compilation (--compile-mode=none)")

    if world_size > 1:
        if rank == 0:
            print("   Wrapping with DistributedDataParallel...")
        policy = DDP(
            policy,
            device_ids=[device.index],
            output_device=device.index,
            gradient_as_bucket_view=True,
            broadcast_buffers=False,
            static_graph=True,
        )

    return policy, optimizer

def get_policy_state_dict(policy) -> dict:
    module = policy
    if isinstance(module, DDP):
        module = module.module
    if hasattr(module, "_orig_mod"):
        module = module._orig_mod
    return module.state_dict()


def collect_expert_demonstrations(env, args: argparse.Namespace, rank: int):
    if rank == 0:
        print("\n[STEP 5/6] Behavior Cloning (BC) Pretraining")
        print("   Collecting expert demonstrations...")

    states, actions = [], []
    for _ in range(args.bc_episodes):
        state = env.reset()
        for _ in range(args.bc_steps):
            price_change = state[1]  # Assume this is price change
            if price_change > 0.01:
                action = 2  # Buy
            elif price_change < -0.01:
                action = 0  # Sell
            else:
                action = 1  # Hold

            states.append(state)
            actions.append(action)

            next_state, _, done, _ = env.step(action)
            if done:
                break
            state = next_state

    states_np = np.asarray(states, dtype=np.float32)
    actions_np = np.asarray(actions, dtype=np.int64)

    if rank == 0:
        print(f"✅ Collected {len(states_np)} transitions")

    # Split into train/validation
    val_split_idx = int((1.0 - args.validation_split) * len(states_np))
    
    train_states = states_np[:val_split_idx]
    train_actions = actions_np[:val_split_idx]
    val_states = states_np[val_split_idx:]
    val_actions = actions_np[val_split_idx:]
    
    if rank == 0:
        print(f"   Train: {len(train_states)} | Validation: {len(val_states)}")

    train_dataset = TensorDataset(
        torch.from_numpy(train_states),
        torch.from_numpy(train_actions)
    )
    val_dataset = TensorDataset(
        torch.from_numpy(val_states),
        torch.from_numpy(val_actions)
    )

    return train_dataset, val_dataset


def make_dataloader(
    dataset: TensorDataset,
    args: argparse.Namespace,
    world_size: int,
    rank: int,
    use_cuda_graph: bool,
) -> Tuple[DataLoader, Optional[DistributedSampler]]:
    sampler: Optional[DistributedSampler] = None
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)

    effective_batch = max(1, args.bc_batch_size // world_size)
    if args.bc_workers > 0:
        num_workers = args.bc_workers
    else:
        num_workers = min(8, os.cpu_count() or 1)
        if len(dataset) < num_workers:
            num_workers = 0

    loader_kwargs = dict(
        batch_size=effective_batch,
        sampler=sampler,
        shuffle=sampler is None,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        num_workers=num_workers,
        drop_last=use_cuda_graph,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4

    loader = DataLoader(dataset, **loader_kwargs)

    return loader, sampler


def bc_training(
    policy,
    optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_sampler: Optional[DistributedSampler],
    val_sampler: Optional[DistributedSampler],
    device: torch.device,
    args: argparse.Namespace,
    rank: int,
    use_cuda_graph: bool,
    world_size: int,
    supabase_logger: Optional[SupabaseLogger],
    cycle_index: int,
    log_interval: int,
):
    ce_loss = nn.CrossEntropyLoss()
    mp_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision == "fp16")

    bc_graph = torch.cuda.CUDAGraph() if use_cuda_graph else None
    static_states = static_actions = None
    
    # Initialize early stopping
    early_stopping = None
    if not args.no_early_stopping:
        early_stopping = EarlyStopping(
            patience=args.early_stopping_patience,
            min_delta=args.min_delta,
            top_k=args.top_k_checkpoints,
            metric_name="val_acc",
            mode="max"
        )

    if rank == 0:
        print("   Training on demonstrations...")
        if early_stopping:
            print(f"   Early stopping enabled (patience={args.early_stopping_patience})")

    best_val_acc = float("-inf")

    for epoch in range(args.bc_epochs):
        # Training phase
        policy.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        running_loss = 0.0
        running_acc = 0.0
        count = 0

        for states_batch, actions_batch in train_loader:
            states_batch = states_batch.to(device, non_blocking=True)
            actions_batch = actions_batch.to(device, non_blocking=True)

            if bc_graph is not None and static_states is None:
                static_states = torch.empty_like(states_batch)
                static_actions = torch.empty_like(actions_batch)
                torch.cuda.synchronize()
                with torch.cuda.graph(bc_graph):
                    optimizer.zero_grad(set_to_none=True)
                    logits, _ = policy(static_states)
                    loss = ce_loss(logits, static_actions)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                    optimizer.step()

            if bc_graph is not None:
                static_states.copy_(states_batch)
                static_actions.copy_(actions_batch)
                bc_graph.replay()
                with torch.no_grad():
                    logits, _ = policy(states_batch)
                    loss = ce_loss(logits, actions_batch)
            else:
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(dtype=mp_dtype):
                    logits, _ = policy(states_batch)
                    loss = ce_loss(logits, actions_batch)
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                    optimizer.step()

            preds = torch.argmax(logits, dim=1)
            acc = (preds == actions_batch).float().mean()

            running_loss += loss.detach()
            running_acc += acc.detach()
            count += 1

        avg_loss = running_loss / max(1, count)
        avg_acc = running_acc / max(1, count)

        # Validation phase
        policy.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_count = 0

        with torch.no_grad():
            for val_states, val_actions in val_loader:
                val_states = val_states.to(device, non_blocking=True)
                val_actions = val_actions.to(device, non_blocking=True)
                logits, _ = policy(val_states)
                loss = ce_loss(logits, val_actions)
                preds = torch.argmax(logits, dim=1)
                acc = (preds == val_actions).float().mean()

                val_loss += loss.detach()
                val_acc += acc.detach()
                val_count += 1

        avg_val_loss = val_loss / max(1, val_count)
        avg_val_acc = val_acc / max(1, val_count)

        if rank == 0 and ((epoch + 1) % 5 == 0 or epoch + 1 == args.bc_epochs):
            print(
                "   Epoch {}/{} | Loss: {:.4f} | Acc: {:.2%} | Val Loss: {:.4f} | Val Acc: {:.2%}".format(
                    epoch + 1,
                    args.bc_epochs,
                    avg_loss.item(),
                    avg_acc.item(),
                    avg_val_loss.item(),
                    avg_val_acc.item(),
                )
            )

        if supabase_logger and rank == 0 and (
            (epoch + 1) % max(1, log_interval) == 0 or epoch + 1 == args.bc_epochs
        ):
            supabase_logger.log_metrics(
                "bc_train",
                step=epoch + 1,
                metrics={
                    "cycle": cycle_index + 1,
                    "loss": float(avg_loss.item()),
                    "accuracy": float(avg_acc.item()),
                },
            )
            supabase_logger.log_metrics(
                "bc_val",
                step=epoch + 1,
                metrics={
                    "cycle": cycle_index + 1,
                    "loss": float(avg_val_loss.item()),
                    "accuracy": float(avg_val_acc.item()),
                },
            )

        stop_training = False

        if avg_val_acc.item() > best_val_acc:
            best_val_acc = avg_val_acc.item()

        if early_stopping and rank == 0:
            checkpoint_path = f"checkpoints/bc_cycle{cycle_index + 1}_epoch{epoch + 1}.pt"
            should_stop = early_stopping(
                epoch=epoch,
                val_metric=avg_val_acc.item(),
                train_metric=avg_acc.item(),
                checkpoint_path=checkpoint_path,
            )

            if early_stopping.counter == 0:
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(get_policy_state_dict(policy), checkpoint_path)

            if should_stop:
                print(f"   🛑 Early stopping triggered at epoch {epoch + 1}")
                print(f"   {early_stopping.get_summary()}")
                stop_training = True

        if world_size > 1:
            stop_tensor = torch.tensor(1 if stop_training else 0, device=device)
            dist.broadcast(stop_tensor, src=0)
            stop_training = bool(stop_tensor.item())

        if stop_training:
            break

    if rank == 0:
        print("✅ BC pretraining complete")

    return best_val_acc


class VectorizedEnvRunner:
    def __init__(self, env_kwargs: dict, num_envs: int):
        from trading_environment import TradingEnvironment

        self.num_envs = max(1, num_envs)
        self.envs = [TradingEnvironment(**env_kwargs) for _ in range(self.num_envs)]
        self.current_states: Optional[np.ndarray] = None

    def reset(self) -> np.ndarray:
        self.current_states = np.stack([env.reset() for env in self.envs])
        return self.current_states

    def step(self, actions: np.ndarray):
        next_states = []
        rewards = []
        dones = []
        infos = []

        for env, action in zip(self.envs, actions):
            next_state, reward, done, info = env.step(int(action))
            if done:
                next_state = env.reset()
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        self.current_states = np.stack(next_states)
        return self.current_states, np.asarray(rewards, dtype=np.float32), np.asarray(dones), infos


def collect_rollout(env_runner, policy, steps: int, device: torch.device, mp_dtype, rank: int):
    states_list, actions_list, rewards_list = [], [], []

    if isinstance(env_runner, VectorizedEnvRunner):
        states = env_runner.reset() if env_runner.current_states is None else env_runner.current_states
        batch_size = env_runner.num_envs
    else:
        state = env_runner.reset()
        batch_size = 1

    collected = 0
    while collected < steps:
        if isinstance(env_runner, VectorizedEnvRunner):
            state_tensor = torch.as_tensor(states, dtype=torch.float32, device=device)
        else:
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=mp_dtype):
            logits, _ = policy(state_tensor)
            probs = torch.softmax(logits, dim=-1)
            sampled_actions = torch.multinomial(probs, 1).squeeze(1)

        actions_cpu = sampled_actions.detach().cpu().numpy()

        if isinstance(env_runner, VectorizedEnvRunner):
            next_states, rewards, dones, _ = env_runner.step(actions_cpu)
            states_list.append(state_tensor.detach().cpu())
            actions_list.append(sampled_actions.detach().cpu())
            rewards_list.append(torch.from_numpy(rewards))
            states = next_states
        else:
            next_state, reward, done, _ = env_runner.step(int(actions_cpu[0]))
            states_list.append(state_tensor.detach().cpu())
            actions_list.append(sampled_actions.detach().cpu())
            rewards_list.append(torch.tensor([reward], dtype=torch.float32))
            state = next_state if not done else env_runner.reset()

        collected += batch_size

    states_batch = torch.cat(states_list, dim=0)[:steps]
    actions_batch = torch.cat(actions_list, dim=0)[:steps]
    rewards_batch = torch.cat(rewards_list, dim=0)[:steps]

    return states_batch, actions_batch, rewards_batch


def ppo_training(
    env_runner,
    policy,
    optimizer,
    device: torch.device,
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    supabase_logger: Optional[SupabaseLogger],
    cycle_index: int,
    log_interval: int,
):
    if rank == 0:
        print("\n[STEP 6/6] PPO Reinforcement Learning")
        print(f"   Training for {args.epochs} epochs...")
        print("-" * 70)

    mp_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision == "fp16")
    best_reward = torch.tensor(float("-inf"), device=device)
    
    # Initialize early stopping for PPO
    early_stopping = None
    if not args.no_early_stopping:
        early_stopping = EarlyStopping(
            patience=args.early_stopping_patience,
            min_delta=args.min_delta,
            top_k=args.top_k_checkpoints,
            metric_name="avg_reward",
            mode="max"
        )
        if rank == 0:
            print(f"   Early stopping enabled (patience={args.early_stopping_patience})")

    rollout_steps_per_rank = max(1, args.ppo_rollout_steps // world_size)

    for epoch in range(args.epochs):
        states_batch_cpu, actions_batch_cpu, rewards_batch_cpu = collect_rollout(
            env_runner, policy, rollout_steps_per_rank, device, mp_dtype, rank
        )

        dataset = TensorDataset(states_batch_cpu, actions_batch_cpu, rewards_batch_cpu)
        sampler = (
            DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
            if world_size > 1
            else None
        )
        mini_batch = max(1, args.ppo_mini_batch // world_size)
        if args.ppo_workers > 0:
            num_workers = args.ppo_workers
        else:
            num_workers = min(8, os.cpu_count() or 1)
            if len(dataset) < num_workers:
                num_workers = 0
        loader_kwargs = dict(
            batch_size=mini_batch,
            sampler=sampler,
            shuffle=sampler is None,
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            drop_last=False,
        )
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = 2

        loader = DataLoader(dataset, **loader_kwargs)

        if sampler is not None:
            sampler.set_epoch(epoch)

        epoch_rewards = []

        for states_cpu, actions_cpu, rewards_cpu in loader:
            states = states_cpu.to(device, non_blocking=True)
            actions = actions_cpu.to(device, non_blocking=True)
            rewards = rewards_cpu.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=mp_dtype):
                logits, values = policy(states)
                values = values.squeeze(-1)

                log_probs = torch.log_softmax(logits, dim=-1)
                action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
                advantages = rewards - values.detach()
                actor_loss = -(action_log_probs * advantages).mean()
                critic_loss = nn.MSELoss()(values, rewards)
                loss = actor_loss + 0.5 * critic_loss

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()

            epoch_rewards.append(rewards.detach())

        if epoch_rewards:
            rewards_tensor = torch.cat(epoch_rewards)
            avg_reward = rewards_tensor.mean().to(device)
        else:
            avg_reward = torch.tensor(0.0, device=device)

        if world_size > 1:
            dist.all_reduce(avg_reward, op=dist.ReduceOp.AVG)

        if avg_reward > best_reward:
            best_reward = avg_reward
            if rank == 0:
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(
                    get_policy_state_dict(policy),
                    f"checkpoints/best_policy_cycle{cycle_index + 1}.pt",
                )
                torch.save(get_policy_state_dict(policy), "checkpoints/best_policy.pt")

        if supabase_logger and rank == 0 and (
            (epoch + 1) % max(1, log_interval) == 0 or epoch + 1 == args.epochs
        ):
            supabase_logger.log_metrics(
                "ppo",
                step=epoch + 1,
                metrics={
                    "cycle": cycle_index + 1,
                    "avg_reward": float(avg_reward.item()),
                },
            )

        if rank == 0 and (epoch % 5 == 0 or epoch == args.epochs - 1):
            print(f"Epoch {epoch + 1:3d}/{args.epochs} | Reward: {avg_reward.item():7.4f}")

        # Early stopping check for PPO
        stop_training = False
        if early_stopping and rank == 0:
            checkpoint_path = f"checkpoints/ppo_cycle{cycle_index + 1}_epoch_{epoch + 1}.pt"

            should_stop = early_stopping(
                epoch=epoch,
                val_metric=avg_reward.item(),
                checkpoint_path=checkpoint_path
            )

            if early_stopping.counter == 0:  # New best model
                torch.save(get_policy_state_dict(policy), checkpoint_path)
                print(f"   ✓ New best reward: {avg_reward.item():.4f}")

            if should_stop:
                print(f"   🛑 Early stopping triggered at epoch {epoch+1}")
                print(f"   {early_stopping.get_summary()}")
                stop_training = True

        if world_size > 1:
            stop_tensor = torch.tensor(1 if stop_training else 0, device=device)
            dist.broadcast(stop_tensor, src=0)
            stop_training = bool(stop_tensor.item())

        if stop_training:
            break

    if rank == 0:
        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETE!")
        print("=" * 70)
        print(f"\nBest average reward: {best_reward.item():.4f}")
        print("Model saved: checkpoints/best_policy.pt")
        print("\nNext steps:")
        print("1. Check training dashboard for results")
        print("2. Increase epochs and data range for better performance")
        print("3. Add more symbols and longer time periods")

    return best_reward.item()


def cleanup_distributed(enabled: bool):
    if enabled:
        dist.destroy_process_group()


def main():
    args = parse_args()
    rank, world_size, local_rank, distributed_enabled = setup_distributed()
    device = setup_gpu_environment(local_rank, rank)

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    progress = PhaseProgress()

    _, data_map, symbols = load_data(rank, args)
    env, env_kwargs = create_environment(data_map, symbols, rank, args)

    policy, optimizer = build_model(device, args, world_size, rank, progress)

    param_count = sum(p.numel() for p in policy.parameters())
    if rank == 0:
        print(f"✅ Model on GPU ({param_count:,} parameters)")

    supabase_logger = SupabaseLogger(enabled=(rank == 0 and not args.disable_supabase_logging))
    supabase_enabled = getattr(supabase_logger, "_enabled", False)
    run_status = "completed"
    run_id = None

    if supabase_enabled:
        run_config = {
            "epochs": args.epochs,
            "bc_epochs": args.bc_epochs,
            "symbol_limit": args.symbol_limit,
            "repeat_trainings": args.repeat_trainings,
            "data_start": args.data_start,
            "data_end": args.data_end,
            "symbols_loaded": len(symbols),
        }
        run_id = supabase_logger.start_run(run_config)
        if rank == 0:
            print(f"✅ Supabase logging enabled (run_id: {run_id})")
    elif rank == 0:
        print("⚠️  Supabase logging disabled (missing credentials or --disable-supabase-logging)")

    try:
        for cycle_index in range(max(1, args.repeat_trainings)):
            if rank == 0:
                print(
                    f"\n🔁 Training cycle {cycle_index + 1}/{max(1, args.repeat_trainings)}"
                )

            train_dataset, val_dataset = collect_expert_demonstrations(env, args, rank)
            use_cuda_graph = (
                args.enable_cuda_graph
                and args.mixed_precision == "bf16"
                and torch.cuda.is_available()
            )
            train_loader, train_sampler = make_dataloader(
                train_dataset, args, world_size, rank, use_cuda_graph
            )
            val_loader, val_sampler = make_dataloader(
                val_dataset, args, world_size, rank, False
            )
            best_val_acc = bc_training(
                policy,
                optimizer,
                train_loader,
                val_loader,
                train_sampler,
                val_sampler,
                device,
                args,
                rank,
                use_cuda_graph,
                world_size,
                supabase_logger if supabase_enabled else None,
                cycle_index,
                args.supabase_log_interval,
            )

            if supabase_enabled and rank == 0:
                supabase_logger.log_metrics(
                    "bc_summary",
                    step=cycle_index + 1,
                    metrics={
                        "cycle": cycle_index + 1,
                        "best_val_acc": float(best_val_acc),
                    },
                )

            env_runner = VectorizedEnvRunner(env_kwargs, args.rollout_envs)
            best_reward = ppo_training(
                env_runner,
                policy,
                optimizer,
                device,
                args,
                rank,
                world_size,
                supabase_logger if supabase_enabled else None,
                cycle_index,
                args.supabase_log_interval,
            )

            if supabase_enabled and rank == 0:
                supabase_logger.log_metrics(
                    "ppo_summary",
                    step=cycle_index + 1,
                    metrics={
                        "cycle": cycle_index + 1,
                        "best_avg_reward": float(best_reward),
                    },
                )

    except Exception:
        run_status = "failed"
        raise
    finally:
        if supabase_enabled:
            supabase_logger.finalize_run(run_status)
            supabase_logger.close()

    progress.close_all()
    cleanup_distributed(distributed_enabled)


if __name__ == "__main__":
    main()
