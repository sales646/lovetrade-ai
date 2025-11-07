"""Transformer-based policy networks for trading agents."""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        *,
        max_seq_len: Optional[int] = None,
    ) -> None:
        super().__init__()
        if max_seq_len is not None:
            max_len = max_seq_len
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerPolicy(nn.Module):
    """Hierarchical discrete policy with shared transformer encoder."""

    def __init__(
        self,
        config: Optional[Dict] = None,
        state_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
        size_dim: Optional[int] = None,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
    ) -> None:
        super().__init__()

        if config is not None:
            state_dim = config.get("state_dim", state_dim or 64)
            action_dim = config.get("action_dim", action_dim or 3)
            size_dim = config.get("size_dim", size_dim or 5)
            d_model = config.get("d_model", d_model)
            nhead = config.get("nhead", nhead)
            num_layers = config.get("num_layers", num_layers)
            dim_feedforward = config.get("dim_feedforward", dim_feedforward)
            dropout = config.get("dropout", dropout)
            max_seq_len = config.get("max_seq_len", max_seq_len)

        if state_dim is None or action_dim is None:
            raise ValueError("state_dim and action_dim must be provided")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.size_dim = size_dim or 5

        self.input_embedding = nn.Linear(state_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len=max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.backbone = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.LayerNorm(dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_feedforward // 2),
            nn.LayerNorm(dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        head_dim = dim_feedforward // 2
        self.action_head = nn.Linear(head_dim, self.action_dim)
        self.size_head = nn.Linear(head_dim, self.size_dim)
        self.risk_head = nn.Linear(head_dim, 2)
        self.value_head = nn.Linear(head_dim, 1)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.LayerNorm)):
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def _encode(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 2:
            state = state.unsqueeze(1)
        x = self.input_embedding(state)
        x = self.pos_encoder(x)
        encoded = self.transformer(x)
        return encoded[:, -1, :]

    def _shared_head(self, encoded: torch.Tensor) -> torch.Tensor:
        return self.backbone(encoded)

    def forward(
        self, state: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self._encode(state)
        core = self._shared_head(encoded)
        action_logits = self.action_head(core)
        size_logits = self.size_head(core)
        risk_params = self.risk_head(core)
        value = self.value_head(core).squeeze(-1)

        action_dist = torch.distributions.Categorical(logits=action_logits)
        size_dist = torch.distributions.Categorical(logits=size_logits)

        action = action_dist.sample()
        size = size_dist.sample()
        log_prob = action_dist.log_prob(action) + size_dist.log_prob(size)
        entropy = action_dist.entropy() + size_dist.entropy()

        return action, size, value, log_prob, entropy, risk_params

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        encoded = self._encode(state)
        core = self._shared_head(encoded)
        logits = self.action_head(core)
        if deterministic:
            return logits.argmax(dim=-1)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.sample()

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        encoded = self._encode(state)
        core = self._shared_head(encoded)
        return self.value_head(core).squeeze(-1)

    def get_action_logits(self, state: torch.Tensor) -> torch.Tensor:
        encoded = self._encode(state)
        core = self._shared_head(encoded)
        return self.action_head(core)

    def get_size_logits(self, state: torch.Tensor) -> torch.Tensor:
        encoded = self._encode(state)
        core = self._shared_head(encoded)
        return self.size_head(core)

    def evaluate_actions(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self._encode(state)
        core = self._shared_head(encoded)
        action_logits = self.action_head(core)
        value = self.value_head(core).squeeze(-1)

        action_dist = torch.distributions.Categorical(logits=action_logits)
        log_prob = action_dist.log_prob(action)
        entropy = action_dist.entropy()
        return log_prob, entropy, value


class LightweightTransformerPolicy(TransformerPolicy):
    def __init__(self, config: Optional[Dict] = None, **kwargs) -> None:
        defaults = {
            "d_model": 128,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 512,
        }
        merged_config = {**defaults, **(config or {})}
        merged_config.update(kwargs)
        super().__init__(config=merged_config)


class HybridTransformerMLP(nn.Module):
    def __init__(
        self,
        temporal_dim: int,
        static_dim: int,
        action_dim: int,
        size_dim: int = 5,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 3,
        mlp_hidden: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.temporal_encoder = TransformerPolicy(
            state_dim=temporal_dim,
            action_dim=d_model,
            size_dim=size_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.LayerNorm(mlp_hidden // 2),
            nn.GELU(),
        )
        combined_dim = d_model + mlp_hidden // 2
        self.head = nn.Sequential(
            nn.Linear(combined_dim, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.action_head = nn.Linear(mlp_hidden, action_dim)
        self.size_head = nn.Linear(mlp_hidden, size_dim)
        self.value_head = nn.Linear(mlp_hidden, 1)

    def forward(
        self, temporal_state: torch.Tensor, static_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.temporal_encoder._encode(temporal_state)
        static = self.static_encoder(static_state)
        combined = torch.cat([encoded, static], dim=-1)
        core = self.head(combined)
        action_logits = self.action_head(core)
        size_logits = self.size_head(core)
        value = self.value_head(core).squeeze(-1)
        action_dist = torch.distributions.Categorical(logits=action_logits)
        size_dist = torch.distributions.Categorical(logits=size_logits)
        action = action_dist.sample()
        size = size_dist.sample()
        log_prob = action_dist.log_prob(action) + size_dist.log_prob(size)
        return action, size, value, log_prob


__all__ = [
    "TransformerPolicy",
    "LightweightTransformerPolicy",
    "HybridTransformerMLP",
]
