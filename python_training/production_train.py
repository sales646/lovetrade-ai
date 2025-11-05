#!/usr/bin/env python3
"""Production training script using full-history cached data and stable RL."""

from __future__ import annotations

import math
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from dotenv import load_dotenv
from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from full_history_config import FullHistoryConfig
from full_history_data import FullHistoryDataManager
from offline_dataset import BehaviorCloningTrainer, ExpertDatasetBuilder
from progress import PhaseProgress
from supabase_logger import SupabaseLogger
from trading_environment import create_trading_env
from transformer_policy import TransformerPolicy

load_dotenv()


@dataclass
class TrainingMetrics:
    seed: int
    best_epoch: int
    best_sortino: float
    checkpoint: Path
    train: Dict[str, float]
    val: Dict[str, float]
    test: Dict[str, float]


class ProductionTrainer:
    """Train RL agent on real market data with actual outcomes."""

    def __init__(self, config: Dict):
        self.config = config
        self.settings = self._build_history_config(config.get("full_history", {}))
        self.data_manager = FullHistoryDataManager(
            self.settings,
            stocks=self._default_stock_tickers(),
            crypto=self._default_crypto_tickers(),
        )

        self.rank, self.world_size = self._setup_distributed()
        self.device = self._resolve_device()
        self.training_dir = self._resolve_training_dir()
        self.progress = PhaseProgress()
        self.supabase = SupabaseLogger()

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.training_seeds: Sequence[int] = self.config.get("seeds", [0, 1, 2])
        self.epochs: int = self.config.get("epochs", 10)
        self.train_steps_per_epoch: int = self.config.get("steps_per_epoch", 64)
        self.grad_accum_steps: int = self.config.get("grad_accum_steps", 1)
        self.grad_clip: float = self.config.get("grad_clip", 1.0)
        self.entropy_coef: float = self.config.get("entropy_coef", 0.01)
        self.value_coef: float = self.config.get("value_coef", 0.5)
        self.discount_gamma: float = self.config.get("gamma", 0.99)
        self.eval_windows: int = self.config.get("eval_windows", 64)
        self.patience: int = self.config.get("patience", 12)
        ppo_cfg = self.config.get("ppo", {})
        self.clip_range: float = ppo_cfg.get("clip_range", 0.2)
        self.kl_coef: float = ppo_cfg.get("kl_coef", 1.0)
        self.kl_target: Optional[float] = ppo_cfg.get("kl_target", 0.01)
        self.kl_adapt_rate: float = ppo_cfg.get("kl_adapt_rate", 1.5)

        self.market_data: Dict[str, pd.DataFrame] = {}
        self.env_manifest = pd.DataFrame()
        self.env_specs: Dict[str, List[Dict]] = {}
        self.policy: Optional[TransformerPolicy] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[LambdaLR] = None
        self.bc_metrics: Dict[str, float] = {}
        self.bc_checkpoint: Optional[Path] = None
        self.bc_dataset = None
        self.bc_reference: Optional[TransformerPolicy] = None
        self.bc_state_dict: Optional[Dict[str, torch.Tensor]] = None

        if self.rank == 0:
            print(f"🖥️  Device: {self.device}")
            if torch.cuda.is_available():
                print(f"   GPU: {torch.cuda.get_device_name(0)}")
                print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _build_history_config(self, overrides: Dict) -> FullHistoryConfig:
        cfg = FullHistoryConfig()
        for key, value in overrides.items():
            setattr(cfg, key, value)
        return cfg

    def _default_stock_tickers(self) -> List[str]:
        return [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD",
            "SPY", "QQQ", "IWM", "DIA",
            "JPM", "BAC", "GS", "WFC",
            "XOM", "CVX", "COP",
            "UNH", "JNJ", "PFE",
        ]

    def _default_crypto_tickers(self) -> List[str]:
        return [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
            "ADAUSDT", "DOGEUSDT", "MATICUSDT", "DOTUSDT", "AVAXUSDT",
        ]

    def _setup_distributed(self) -> Sequence[int]:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        if world_size > 1 and not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
        return rank, world_size

    def _resolve_device(self) -> torch.device:
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            torch.cuda.set_device(local_rank)
            return torch.device(f"cuda:{local_rank}")
        return torch.device("cpu")

    def _resolve_training_dir(self) -> Path:
        directory = Path(self.config.get("training_dir", self.data_manager.cache_dir / "training"))
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    def prepare_data(self) -> None:
        market_data, manifest = self.data_manager.prepare(
            force_full_prep=self.config.get("force_full_prep"),
            progress=self.progress,
        )
        self.market_data = market_data
        self.env_manifest = manifest
        self.env_specs = {
            split: df.reset_index(drop=True).to_dict("records")
            for split, df in manifest.groupby("split")
        }
        self.train_specs = self.env_specs.get("TRAIN", [])
        self.val_specs = self.env_specs.get("VAL", [])
        self.test_specs = self.env_specs.get("TEST", [])

        self.bc_dataset = self._build_offline_dataset()

        if not self.train_specs:
            raise RuntimeError("No training window specs available")

        if self.rank == 0:
            split_counts = manifest["split"].value_counts().to_dict()
            print(
                f"Prepared {len(self.market_data)} symbols | "
                f"Windows: train={split_counts.get('TRAIN', 0)}, "
                f"val={split_counts.get('VAL', 0)}, test={split_counts.get('TEST', 0)}"
            )
            print("✅ PREP DONE — starting training now")

    def _build_offline_dataset(self):
        train_range = self.settings.resolved_range(self.settings.train_range)
        val_range = self.settings.resolved_range(self.settings.val_range)
        test_range = self.settings.resolved_range(self.settings.test_range)
        builder = ExpertDatasetBuilder(train_range, val_range, test_range)
        return builder.build(self.market_data)

    def _policy_kwargs(self) -> Dict[str, int]:
        policy_cfg = self.config.get("policy", {})
        feature_dim = int(self.bc_dataset.train.features.shape[1])
        return {
            "state_dim": feature_dim,
            "action_dim": policy_cfg.get("action_dim", 3),
            "size_dim": policy_cfg.get("size_dim", 5),
            "d_model": policy_cfg.get("d_model", 256),
            "nhead": policy_cfg.get("num_heads", 8),
            "num_layers": policy_cfg.get("num_layers", 4),
            "dim_feedforward": policy_cfg.get("dim_feedforward", 1024),
            "dropout": policy_cfg.get("dropout", 0.1),
        }

    def _create_policy(self) -> TransformerPolicy:
        kwargs = self._policy_kwargs()
        policy = TransformerPolicy(config=kwargs)
        return policy.to(self.device)

    def _run_behavioral_cloning(self) -> None:
        if self.bc_dataset is None:
            raise RuntimeError("BC dataset not prepared")

        self.policy = self._create_policy()

        if self.rank == 0:
            print("🚀 TRAINING START (BC)")

        bc_cfg = self.config.get("bc", {})
        epochs = bc_cfg.get("epochs", 20)
        batch_size = bc_cfg.get("batch_size", 256)
        patience = bc_cfg.get("patience", 6)

        trainer = BehaviorCloningTrainer(
            policy=self.policy,
            feature_dim=self.bc_dataset.train.features.shape[1],
            action_classes=bc_cfg.get("action_dim", 3),
            size_classes=bc_cfg.get("size_dim", 5),
            lr=bc_cfg.get("lr", 3e-4),
            weight_decay=bc_cfg.get("weight_decay", 0.01),
            dropout=bc_cfg.get("dropout", 0.15),
            label_smoothing=bc_cfg.get("label_smoothing", 0.05),
            device=self.device,
        )

        self.progress.start("BC TRAIN", total=epochs, unit="epoch")

        def callback(epoch: int, metrics: Dict[str, float]) -> None:
            if self.rank == 0:
                self.progress.update("BC TRAIN")
                payload = {
                    "bc_loss": metrics.get("loss", 0.0),
                    "bc_acc": metrics.get("accuracy", 0.0),
                    "bc_f1": metrics.get("macro_f1", 0.0),
                    "bc_ece": metrics.get("ece", 0.0),
                }
                self.supabase.log_metrics("val", epoch, payload)

        final_metrics = trainer.fit(
            self.bc_dataset,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            progress_callback=callback,
        )
        self.progress.close("BC TRAIN")
        self.bc_metrics = final_metrics

        if self.rank == 0:
            self.bc_checkpoint = self.training_dir / "behavioral_cloning.pt"
            torch.save(self.policy.state_dict(), self.bc_checkpoint)

        reference = self._create_policy()
        reference.load_state_dict(self.policy.state_dict())
        reference.eval()
        for param in reference.parameters():
            param.requires_grad_(False)
        self.bc_reference = reference
        self.bc_state_dict = {k: v.detach().cpu() for k, v in self.policy.state_dict().items()}

    # ------------------------------------------------------------------
    # Environment helpers
    # ------------------------------------------------------------------
    def _create_env_from_spec(self, spec: Dict) -> object:
        symbol = spec["symbol"]
        window_df = self.market_data[symbol].iloc[spec["start_idx"]:spec["end_idx"]]
        env_kwargs = self.config.get("env_config", {}).copy()
        return create_trading_env(
            symbols=[symbol],
            phase=spec["split"].lower(),
            external_data={symbol: window_df.to_dict("records")},
            **env_kwargs,
        )

    @staticmethod
    def _reset_env(env, phase: str):
        result = env.reset(phase=phase.lower())
        if isinstance(result, tuple) and len(result) == 2:
            return result
        return result, {}

    @staticmethod
    def _step_env(env, action: int):
        result = env.step(action)
        if isinstance(result, tuple) and len(result) == 5:
            return result
        obs, reward, done, info = result
        return obs, reward, done, False, info

    # ------------------------------------------------------------------
    # Policy setup
    # ------------------------------------------------------------------
    def initialize_policy(self, seed: int) -> None:
        self.policy = self._create_policy()
        if self.bc_state_dict:
            self.policy.load_state_dict(self.bc_state_dict)

        if self.world_size > 1:
            self.policy = DDP(self.policy, device_ids=[self.device.index], find_unused_parameters=False)

        updates_per_epoch = math.ceil(min(self.train_steps_per_epoch, len(self.train_specs)) / self.grad_accum_steps)
        total_steps = max(1, updates_per_epoch * self.epochs)
        self.optimizer = AdamW(
            self.policy.parameters(),
            lr=self.config.get("learning_rate", 3e-4),
            weight_decay=self.config.get("weight_decay", 0.01),
        )
        self.scheduler = self._build_scheduler(self.optimizer, total_steps)

    def _build_scheduler(self, optimizer: torch.optim.Optimizer, total_steps: int) -> LambdaLR:
        warmup_steps = max(1, int(0.1 * total_steps))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return LambdaLR(optimizer, lr_lambda)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self) -> None:
        run_config = {
            "start": self.settings.start,
            "end": self.settings.end,
            "seeds": list(self.training_seeds),
            "max_stocks": self.config.get("MAX_STOCKS"),
            "max_crypto": self.config.get("MAX_CRYPTO"),
        }
        if self.rank == 0:
            self.supabase.start_run(run_config)

        self.prepare_data()
        self._run_behavioral_cloning()

        metrics: List[TrainingMetrics] = []
        total_epochs = self.epochs * max(1, len(self.training_seeds))
        self.progress.start("PPO TRAIN", total=total_epochs, unit="epoch")

        for idx, seed in enumerate(self.training_seeds):
            if self.rank == 0:
                print(f"\n===== PPO seed {seed} ({idx + 1}/{len(self.training_seeds)}) =====")
            self._set_seed(seed)
            self.initialize_policy(seed)
            seed_metrics = self._train_single_seed(seed)
            metrics.append(seed_metrics)

        self.progress.close("PPO TRAIN")

        if self.rank == 0:
            aggregated = self._aggregate_metrics(metrics)
            print(
                f"🏁 TRAIN DONE — best Sortino: {aggregated['best_sortino']:.2f} "
                f"@ epoch {aggregated['best_epoch']} — checkpoint: {aggregated['checkpoint']}"
            )
            self._print_summary_table(aggregated)
            self.supabase.finalize_run("completed")

        self.supabase.close()

    def _train_single_seed(self, seed: int) -> TrainingMetrics:
        assert self.policy is not None
        scaler = torch.cuda.amp.GradScaler(enabled=self.device.type == "cuda")
        best_sortino = float("-inf")
        best_epoch = -1
        best_path = None
        patience_counter = 0
        first_step_reported = False

        steps_per_epoch = min(self.train_steps_per_epoch, len(self.train_specs))
        total_steps = max(1, steps_per_epoch * self.epochs)

        for epoch in range(self.epochs):
            epoch_returns: List[float] = []
            epoch_losses: List[float] = []
            self.optimizer.zero_grad(set_to_none=True)

            for step in range(steps_per_epoch):
                spec = random.choice(self.train_specs)
                episode = self._collect_episode(spec, train=True)
                loss, episode_return = self._compute_loss(episode, scaler)
                epoch_returns.append(episode_return)
                epoch_losses.append(loss.item())

                if (step + 1) % self.grad_accum_steps == 0:
                    scaler.unscale_(self.optimizer)
                    clip_grad_norm_(self.policy.parameters(), self.grad_clip)
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()

                    if not first_step_reported and self.rank == 0:
                        train_metrics = self._compute_metrics_from_returns(epoch_returns)
                        print(
                            "🚀 TRAINING START (PPO) | "
                            f"loss={np.mean(epoch_losses):.4f} "
                            f"sharpe={train_metrics['sharpe']:.3f} "
                            f"sortino={train_metrics['sortino']:.3f} "
                            f"hit_rate={train_metrics['hit_rate']:.3f}"
                        )
                        first_step_reported = True

            remainder = steps_per_epoch % self.grad_accum_steps
            if remainder != 0:
                scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.policy.parameters(), self.grad_clip)
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()

            train_metrics = self._compute_metrics_from_returns(epoch_returns)
            val_metrics = self._evaluate_split(self.val_specs, deterministic=True)
            test_metrics = self._evaluate_split(self.test_specs, deterministic=True)

            if self.rank == 0:
                self.supabase.log_metrics(
                    "train",
                    epoch,
                    {
                        "loss": float(np.mean(epoch_losses)),
                        **train_metrics,
                    },
                )
                self.supabase.log_metrics("val", epoch, val_metrics)
                self.supabase.log_metrics("test", epoch, test_metrics)

            self.progress.update("PPO TRAIN")

            if val_metrics["sortino"] > best_sortino:
                best_sortino = val_metrics["sortino"]
                best_epoch = epoch
                patience_counter = 0
                best_path = self._save_checkpoint(seed, epoch)
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                if self.rank == 0:
                    print(f"Early stopping triggered at epoch {epoch}")
                remaining_epochs = self.epochs - epoch - 1
                if remaining_epochs > 0:
                    self.progress.update("PPO TRAIN", remaining_epochs)
                break

        if best_path is None:
            best_path = self._save_checkpoint(seed, best_epoch if best_epoch >= 0 else 0)

        self._load_checkpoint(best_path)
        final_train = self._evaluate_split(self.train_specs, deterministic=True)
        final_val = self._evaluate_split(self.val_specs, deterministic=True)
        final_test = self._evaluate_split(self.test_specs, deterministic=True)

        return TrainingMetrics(
            seed=seed,
            best_epoch=best_epoch,
            best_sortino=best_sortino,
            checkpoint=best_path,
            train=final_train,
            val=final_val,
            test=final_test,
        )

    # ------------------------------------------------------------------
    # Episodes and losses
    # ------------------------------------------------------------------
    def _collect_episode(self, spec: Dict, train: bool) -> Dict:
        env = self._create_env_from_spec(spec)
        obs, _ = self._reset_env(env, phase=spec["split"])
        done = False

        states: List[torch.Tensor] = []
        actions: List[torch.Tensor] = []
        sizes: List[torch.Tensor] = []
        log_probs: List[torch.Tensor] = []
        values: List[torch.Tensor] = []
        rewards: List[float] = []
        entropies: List[torch.Tensor] = []

        autocast_enabled = self.device.type == "cuda"
        grad_context = nullcontext() if train else torch.no_grad()
        module = self.policy.module if isinstance(self.policy, DDP) else self.policy

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with grad_context:
                with torch.cuda.amp.autocast(enabled=autocast_enabled):
                    action_tensor, size_tensor, value, log_prob, entropy, _ = module(obs_tensor)

            action = action_tensor.squeeze(0)
            size = size_tensor.squeeze(0)
            log_prob = log_prob.squeeze(0)
            entropy = entropy.squeeze(0)
            value = value.squeeze(0)

            next_obs, reward, terminated, truncated, _ = self._step_env(env, int(action.item()))

            states.append(obs_tensor.squeeze(0))
            actions.append(action)
            sizes.append(size)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(float(reward))
            entropies.append(entropy)

            obs = next_obs
            done = bool(terminated or truncated)

        return {
            "states": states,
            "actions": actions,
            "sizes": sizes,
            "log_probs": log_probs,
            "values": values,
            "rewards": rewards,
            "entropies": entropies,
        }

    def _compute_loss(self, episode: Dict, scaler: torch.cuda.amp.GradScaler) -> torch.Tensor:
        rewards = episode["rewards"]
        returns = self._discount_rewards(rewards)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        states = torch.stack(episode["states"])
        actions = torch.stack(episode["actions"]).long()
        sizes = torch.stack(episode["sizes"]).long()
        old_log_probs = torch.stack(episode["log_probs"]).detach()
        old_values = torch.stack(episode["values"])

        advantages = returns_tensor - old_values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        module = self.policy.module if isinstance(self.policy, DDP) else self.policy
        new_log_probs, entropies, values, _ = module.evaluate_actions(states, actions, sizes)

        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
        surrogate1 = ratio * advantages
        surrogate2 = clipped_ratio * advantages
        policy_loss = -torch.min(surrogate1, surrogate2).mean()

        value_loss = 0.5 * (returns_tensor - values).pow(2).mean()
        entropy_loss = -entropies.mean()

        if self.bc_reference is not None:
            with torch.no_grad():
                bc_log_probs, _, _, _ = self.bc_reference.evaluate_actions(states, actions, sizes)
            kl = (new_log_probs - bc_log_probs).mean()
        else:
            kl = torch.tensor(0.0, device=self.device)

        loss = (
            policy_loss
            + self.value_coef * value_loss
            + self.entropy_coef * entropy_loss
            + self.kl_coef * kl
        )

        scaled_loss = loss / self.grad_accum_steps
        scaler.scale(scaled_loss).backward()
        episode_return = sum(rewards)

        if self.rank == 0 and self.kl_target is not None:
            kl_value = float(abs(kl.detach()))
            if kl_value > self.kl_target * 1.5:
                self.kl_coef *= self.kl_adapt_rate
            elif kl_value < self.kl_target / 1.5:
                self.kl_coef /= self.kl_adapt_rate

        return loss.detach(), episode_return

    def _discount_rewards(self, rewards: List[float]) -> List[float]:
        discounted = []
        running = 0.0
        for reward in reversed(rewards):
            running = reward + self.discount_gamma * running
            discounted.append(running)
        return list(reversed(discounted))

    # ------------------------------------------------------------------
    # Evaluation and metrics
    # ------------------------------------------------------------------
    def _evaluate_split(self, specs: List[Dict], deterministic: bool) -> Dict[str, float]:
        if not specs:
            return {"sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0, "hit_rate": 0.0, "profit_factor": 0.0}

        returns: List[float] = []
        limit = min(self.eval_windows, len(specs))
        for spec in random.sample(specs, limit):
            episode = self._collect_episode(spec, train=not deterministic)
            returns.append(sum(episode["rewards"]))

        return self._compute_metrics_from_returns(returns)

    def _compute_metrics_from_returns(self, returns: List[float]) -> Dict[str, float]:
        if not returns:
            return {"sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0, "hit_rate": 0.0, "profit_factor": 0.0}

        returns_array = np.array(returns, dtype=np.float32)
        mean = returns_array.mean()
        std = returns_array.std() + 1e-8
        sharpe = (mean / std) * math.sqrt(252)

        downside = returns_array[returns_array < 0]
        downside_std = downside.std() + 1e-8 if downside.size > 0 else std
        sortino = (mean / downside_std) * math.sqrt(252)

        cumulative = np.cumsum(returns_array)
        max_drawdown = float(np.max(np.maximum.accumulate(cumulative) - cumulative)) if cumulative.size else 0.0

        hit_rate = float(np.mean(returns_array > 0)) if returns_array.size else 0.0
        positive = returns_array[returns_array > 0].sum()
        negative = -returns_array[returns_array < 0].sum()
        profit_factor = float(positive / (negative + 1e-8))

        return {
            "sharpe": float(sharpe),
            "sortino": float(sortino),
            "max_drawdown": max_drawdown,
            "hit_rate": hit_rate,
            "profit_factor": profit_factor,
        }

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def _save_checkpoint(self, seed: int, epoch: int) -> Path:
        module = self.policy.module if isinstance(self.policy, DDP) else self.policy
        checkpoint_dir = self.training_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / f"seed{seed}_epoch{epoch}_policy.pt"
        torch.save({"model_state": module.state_dict()}, path)
        return path

    def _load_checkpoint(self, path: Path) -> None:
        module = self.policy.module if isinstance(self.policy, DDP) else self.policy
        state = torch.load(path, map_location=self.device)
        module.load_state_dict(state["model_state"])

    # ------------------------------------------------------------------
    # Aggregation and reporting
    # ------------------------------------------------------------------
    def _aggregate_metrics(self, metrics: List[TrainingMetrics]) -> Dict[str, float]:
        best_entry = max(metrics, key=lambda m: m.best_sortino if math.isfinite(m.best_sortino) else -1e9)
        summary = {
            "best_sortino": best_entry.best_sortino,
            "best_epoch": best_entry.best_epoch,
            "checkpoint": str(best_entry.checkpoint),
        }

        for split in ["train", "val", "test"]:
            values = np.array([getattr(m, split)["sortino"] for m in metrics], dtype=np.float32)
            summary[f"{split}_sortino_mean"] = float(values.mean())
            summary[f"{split}_sortino_std"] = float(values.std())

        summary["table"] = {
            split.upper(): getattr(best_entry, split)
            for split in ["train", "val", "test"]
        }
        return summary

    def _print_summary_table(self, aggregated: Dict[str, float]) -> None:
        table = aggregated["table"]
        print("\nSummary per split:")
        print(f"{'Split':<8} {'Sharpe':>8} {'Sortino':>10} {'Max DD':>10} {'Hit Rate':>10} {'ProfitF':>10}")
        for split, metrics in table.items():
            print(
                f"{split:<8} "
                f"{metrics['sharpe']:>8.3f} "
                f"{metrics['sortino']:>10.3f} "
                f"{metrics['max_drawdown']:>10.3f} "
                f"{metrics['hit_rate']:>10.3f} "
                f"{metrics['profit_factor']:>10.3f}"
            )
        print("\nSortino (mean ± std) across seeds:")
        for split in ["train", "val", "test"]:
            mean = aggregated.get(f"{split}_sortino_mean", 0.0)
            std = aggregated.get(f"{split}_sortino_std", 0.0)
            print(f"  {split.upper()}: {mean:.3f} ± {std:.3f}")


def main() -> None:
    config: Dict = {}
    trainer = ProductionTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
