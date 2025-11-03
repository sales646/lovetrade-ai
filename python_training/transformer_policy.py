"""
Transformer-based Policy Network for Trading
Better at capturing temporal patterns than MLPs
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for time series"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TransformerPolicy(nn.Module):
    """
    Transformer-based policy for trading decisions
    
    Features:
    - Multi-head attention to capture market relationships
    - Positional encoding for time-series data
    - Separate actor and critic heads
    - Layer normalization and dropout for stability
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 100
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_model = d_model
        
        # Input embedding
        self.input_embedding = nn.Linear(state_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.LayerNorm(dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_feedforward // 2),
            nn.LayerNorm(dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, action_dim)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.LayerNorm(dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_feedforward // 2),
            nn.LayerNorm(dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 1)
        )
        
        # Action distribution parameters
        self.action_log_std = nn.Parameter(torch.zeros(action_dim))
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with Xavier uniform"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        state: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            state: (batch_size, seq_len, state_dim) or (batch_size, state_dim)
            mask: Optional attention mask
        
        Returns:
            action: Sampled action
            value: State value estimate
            log_prob: Log probability of action
        """
        # Handle single timestep input
        if state.dim() == 2:
            state = state.unsqueeze(1)  # Add sequence dimension
        
        # Embed input
        x = self.input_embedding(state)
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x, mask=mask)
        
        # Use last timestep for action/value
        x = x[:, -1, :]  # (batch_size, d_model)
        
        # Actor: compute action distribution
        action_mean = self.actor(x)
        action_std = torch.exp(self.action_log_std)
        
        # Sample action
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        # Critic: compute value
        value = self.critic(x).squeeze(-1)
        
        return action, value, log_prob
    
    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> torch.Tensor:
        """Get action without value estimate (for inference)"""
        with torch.no_grad():
            if state.dim() == 2:
                state = state.unsqueeze(1)
            
            x = self.input_embedding(state)
            x = self.pos_encoder(x)
            x = self.transformer_encoder(x)
            x = x[:, -1, :]
            
            action_mean = self.actor(x)
            
            if deterministic:
                return action_mean
            else:
                action_std = torch.exp(self.action_log_std)
                dist = torch.distributions.Normal(action_mean, action_std)
                return dist.sample()
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Get value estimate without action"""
        with torch.no_grad():
            if state.dim() == 2:
                state = state.unsqueeze(1)
            
            x = self.input_embedding(state)
            x = self.pos_encoder(x)
            x = self.transformer_encoder(x)
            x = x[:, -1, :]
            
            return self.critic(x).squeeze(-1)
    
    def get_log_probs(self, action: torch.Tensor) -> torch.Tensor:
        """Get log probabilities for given actions"""
        # This is a simplified version - in practice, you'd need to recompute
        # the action distribution
        action_std = torch.exp(self.action_log_std)
        dist = torch.distributions.Normal(0, action_std)  # Centered at 0
        return dist.log_prob(action).sum(dim=-1)


class LightweightTransformerPolicy(TransformerPolicy):
    """
    Lighter version of Transformer policy for faster training
    Fewer layers and smaller hidden dims
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 50
    ):
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_len=max_seq_len
        )


class HybridTransformerMLP(nn.Module):
    """
    Hybrid model: Transformer for temporal features + MLP for static features
    Best of both worlds
    """
    
    def __init__(
        self,
        temporal_dim: int,
        static_dim: int,
        action_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 3,
        mlp_hidden: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Transformer for temporal features (price, volume, indicators over time)
        self.temporal_encoder = TransformerPolicy(
            state_dim=temporal_dim,
            action_dim=d_model,  # Output embedding
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # MLP for static features (account state, positions, time of day)
        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.LayerNorm(mlp_hidden // 2),
            nn.GELU()
        )
        
        # Combined decision layers
        combined_dim = d_model + mlp_hidden // 2
        
        self.actor = nn.Sequential(
            nn.Linear(combined_dim, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, action_dim)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(combined_dim, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1)
        )
        
        self.action_log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(
        self,
        temporal_state: torch.Tensor,
        static_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with temporal and static inputs"""
        # Encode temporal features
        temporal_features = self.temporal_encoder.transformer_encoder(
            self.temporal_encoder.input_embedding(temporal_state)
        )[:, -1, :]
        
        # Encode static features
        static_features = self.static_encoder(static_state)
        
        # Combine
        combined = torch.cat([temporal_features, static_features], dim=-1)
        
        # Actor
        action_mean = self.actor(combined)
        action_std = torch.exp(self.action_log_std)
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        # Critic
        value = self.critic(combined).squeeze(-1)
        
        return action, value, log_prob
