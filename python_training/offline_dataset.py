"""Offline dataset construction and behavioral cloning trainer."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .expert_strategies import StrategyOutput, generate_expert_labels


@dataclass
class DatasetSplit:
    features: torch.Tensor
    action_labels: torch.Tensor
    size_labels: torch.Tensor
    stop_labels: torch.Tensor
    take_labels: torch.Tensor


@dataclass
class OfflineDataset:
    train: DatasetSplit
    val: DatasetSplit
    test: DatasetSplit
    feature_columns: List[str]
    stats: Dict[str, float]


class ExpertTrajectoryDataset(Dataset):
    def __init__(
        self,
        features: torch.Tensor,
        action_labels: torch.Tensor,
        size_labels: torch.Tensor,
        stop_labels: torch.Tensor,
        take_labels: torch.Tensor,
    ) -> None:
        self.features = features.float()
        self.action_labels = action_labels.long()
        self.size_labels = size_labels.long()
        self.stop_labels = stop_labels.long()
        self.take_labels = take_labels.long()

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int):
        return (
            self.features[idx],
            self.action_labels[idx],
            self.size_labels[idx],
            self.stop_labels[idx],
            self.take_labels[idx],
        )


class ExpertDatasetBuilder:
    """Construct an offline dataset from expert strategies and feature frames."""

    def __init__(
        self,
        train_range: Tuple[pd.Timestamp, pd.Timestamp],
        val_range: Tuple[pd.Timestamp, pd.Timestamp],
        test_range: Tuple[pd.Timestamp, pd.Timestamp],
        news_window_minutes: int = 60,
    ) -> None:
        self.train_range = train_range
        self.val_range = val_range
        self.test_range = test_range
        self.news_window = pd.Timedelta(minutes=news_window_minutes)

    def build(self, market_data: Dict[str, pd.DataFrame]) -> OfflineDataset:
        frames: List[pd.DataFrame] = []
        for symbol, df in market_data.items():
            working = df.copy()
            working["symbol"] = symbol
            frames.append(working)

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.sort_values("timestamp")
        combined = self._inject_news_features(combined)

        expert = generate_expert_labels(combined)
        combined["action_label"] = expert.action
        combined["size_label"] = expert.size_bucket
        combined["stop_label"] = expert.stop_bucket
        combined["take_label"] = expert.take_bucket

        combined = self._balance_actions(combined)

        feature_columns = [
            col
            for col in combined.columns
            if col
            not in {
                "timestamp",
                "symbol",
                "action_label",
                "size_label",
                "stop_label",
                "take_label",
            }
            and combined[col].dtype.kind in "if"
        ]

        stats = {
            f"mean_{col}": float(combined[col].mean())
            for col in feature_columns
        }
        stats.update({
            f"std_{col}": float(combined[col].std() or 1.0)
            for col in feature_columns
        })

        combined = self._normalize(combined, feature_columns, stats)

        train_mask = (combined["timestamp"] >= self.train_range[0]) & (
            combined["timestamp"] <= self.train_range[1]
        )
        val_mask = (combined["timestamp"] >= self.val_range[0]) & (
            combined["timestamp"] <= self.val_range[1]
        )
        test_mask = (combined["timestamp"] >= self.test_range[0]) & (
            combined["timestamp"] <= self.test_range[1]
        )

        splits = {
            "train": combined[train_mask],
            "val": combined[val_mask],
            "test": combined[test_mask],
        }

        dataset_splits = {
            name: self._to_split(df, feature_columns)
            for name, df in splits.items()
        }

        return OfflineDataset(
            train=dataset_splits["train"],
            val=dataset_splits["val"],
            test=dataset_splits["test"],
            feature_columns=feature_columns,
            stats=stats,
        )

    def _inject_news_features(self, df: pd.DataFrame) -> pd.DataFrame:
        news_cols = [
            "news_count",
            "news_sent_mean",
            "news_sent_max",
            "news_sent_min",
            "news_headline_len_mean",
        ]
        for col in news_cols:
            if col not in df.columns:
                df[col] = 0.0
        return df

    def _balance_actions(self, df: pd.DataFrame) -> pd.DataFrame:
        counts = df["action_label"].value_counts()
        if counts.empty:
            return df
        min_count = counts.min()
        balanced_frames = []
        rng = np.random.default_rng(42)
        for action, action_df in df.groupby("action_label"):
            take = min(len(action_df), int(min_count * 1.5))
            if take <= 0:
                continue
            indices = rng.choice(len(action_df), size=take, replace=False)
            balanced_frames.append(action_df.iloc[indices])
        return pd.concat(balanced_frames).sample(frac=1.0, random_state=42)

    def _normalize(
        self,
        df: pd.DataFrame,
        feature_columns: Sequence[str],
        stats: Dict[str, float],
    ) -> pd.DataFrame:
        normalized = df.copy()
        for col in feature_columns:
            mean = stats[f"mean_{col}"]
            std = stats[f"std_{col}"] or 1.0
            normalized[col] = (normalized[col] - mean) / std
        return normalized

    def _to_split(self, df: pd.DataFrame, feature_columns: Sequence[str]) -> DatasetSplit:
        features = torch.tensor(df[feature_columns].to_numpy(dtype=np.float32))
        actions = torch.tensor(df["action_label"].to_numpy(dtype=np.int64))
        sizes = torch.tensor(df["size_label"].to_numpy(dtype=np.int64))
        stops = torch.tensor(df["stop_label"].to_numpy(dtype=np.int64))
        takes = torch.tensor(df["take_label"].to_numpy(dtype=np.int64))
        return DatasetSplit(features, actions, sizes, stops, takes)


class BehaviorCloningTrainer:
    def __init__(
        self,
        policy: nn.Module,
        feature_dim: int,
        action_classes: int = 3,
        size_classes: int = 5,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        dropout: float = 0.15,
        label_smoothing: float = 0.05,
        device: Optional[torch.device] = None,
    ) -> None:
        self.policy = policy
        self.feature_dim = feature_dim
        self.action_classes = action_classes
        self.size_classes = size_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.label_smoothing = label_smoothing
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def fit(
        self,
        dataset: OfflineDataset,
        batch_size: int = 256,
        epochs: int = 30,
        patience: int = 6,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, float]:
        train_loader = DataLoader(
            ExpertTrajectoryDataset(
                dataset.train.features,
                dataset.train.action_labels,
                dataset.train.size_labels,
                dataset.train.stop_labels,
                dataset.train.take_labels,
            ),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )
        val_loader = DataLoader(
            ExpertTrajectoryDataset(
                dataset.val.features,
                dataset.val.action_labels,
                dataset.val.size_labels,
                dataset.val.stop_labels,
                dataset.val.take_labels,
            ),
            batch_size=batch_size,
            shuffle=False,
        )

        best_loss = float("inf")
        best_state = None
        patience_counter = 0
        metrics: Dict[str, float] = {}

        for epoch in range(epochs):
            self.policy.train()
            total_loss = 0.0
            total_action_correct = 0
            total_samples = 0
            for batch in train_loader:
                features, action_labels, size_labels, stop_labels, take_labels = [
                    b.to(self.device) for b in batch
                ]
                logits = self.policy.get_action_logits(features)
                size_logits = self.policy.get_size_logits(features)

                action_loss = self._cross_entropy(logits, action_labels)
                size_loss = self._cross_entropy(size_logits, size_labels)
                loss = action_loss + 0.5 * size_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item() * features.size(0)
                preds = logits.argmax(dim=-1)
                total_action_correct += (preds == action_labels).sum().item()
                total_samples += features.size(0)

            train_loss = total_loss / max(1, total_samples)
            train_acc = total_action_correct / max(1, total_samples)

            val_metrics = self.evaluate(val_loader)
            metrics = {
                "train_loss": train_loss,
                "train_acc": train_acc,
                **val_metrics,
            }

            if progress_callback:
                progress_callback(epoch + 1, metrics)

            if val_metrics["loss"] + 1e-5 < best_loss:
                best_loss = val_metrics["loss"]
                best_state = {k: v.detach().cpu() for k, v in self.policy.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_state:
            self.policy.load_state_dict(best_state)

        return metrics

    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.policy.eval()
        total_loss = 0.0
        total_samples = 0
        total_action_correct = 0
        confusion = np.zeros((self.action_classes, self.action_classes))
        with torch.no_grad():
            for batch in loader:
                features, action_labels, size_labels, _, _ = [b.to(self.device) for b in batch]
                logits = self.policy.get_action_logits(features)
                size_logits = self.policy.get_size_logits(features)
                action_loss = self._cross_entropy(logits, action_labels)
                size_loss = self._cross_entropy(size_logits, size_labels)
                loss = action_loss + 0.5 * size_loss

                total_loss += loss.item() * features.size(0)
                total_samples += features.size(0)
                preds = logits.argmax(dim=-1)
                total_action_correct += (preds == action_labels).sum().item()

                for true, pred in zip(action_labels.cpu().numpy(), preds.cpu().numpy()):
                    confusion[true, pred] += 1

        macro_f1 = self._macro_f1(confusion)
        ece = self._expected_calibration_error(loader)

        return {
            "loss": total_loss / max(1, total_samples),
            "accuracy": total_action_correct / max(1, total_samples),
            "macro_f1": macro_f1,
            "ece": ece,
        }

    def _expected_calibration_error(self, loader: DataLoader, bins: int = 10) -> float:
        confidences: List[float] = []
        accuracies: List[float] = []
        with torch.no_grad():
            for batch in loader:
                features, action_labels, *_ = [b.to(self.device) for b in batch]
                logits = self.policy.get_action_logits(features)
                probs = logits.softmax(dim=-1)
                conf, pred = torch.max(probs, dim=-1)
                correct = pred == action_labels
                confidences.extend(conf.cpu().tolist())
                accuracies.extend(correct.cpu().numpy().astype(float))
        if not confidences:
            return 0.0
        confidences = np.array(confidences)
        accuracies = np.array(accuracies)
        bin_edges = np.linspace(0.0, 1.0, bins + 1)
        ece = 0.0
        for i in range(bins):
            mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
            if not np.any(mask):
                continue
            bin_conf = confidences[mask].mean()
            bin_acc = accuracies[mask].mean()
            ece += np.abs(bin_conf - bin_acc) * mask.mean()
        return float(ece)

    def _cross_entropy(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing <= 0:
            return nn.functional.cross_entropy(logits, targets)
        num_classes = logits.size(-1)
        smoothing = self.label_smoothing / (num_classes - 1)
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(smoothing)
            true_dist.scatter_(1, targets.unsqueeze(1), 1 - self.label_smoothing)
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        return -(true_dist * log_probs).sum(dim=-1).mean()

    def _macro_f1(self, confusion: np.ndarray) -> float:
        f1_scores = []
        for i in range(confusion.shape[0]):
            tp = confusion[i, i]
            fp = confusion[:, i].sum() - tp
            fn = confusion[i, :].sum() - tp
            denom = (2 * tp + fp + fn)
            f1_scores.append(0.0 if denom == 0 else (2 * tp) / denom)
        return float(np.mean(f1_scores))


__all__ = [
    "OfflineDataset",
    "ExpertDatasetBuilder",
    "BehaviorCloningTrainer",
]
