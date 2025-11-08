"""
Model architectures for the HRL system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import numpy as np

class CustomActor(nn.Module):
    """Custom actor network for layer 1 and 2."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [512, 512, 256],
        activation: nn.Module = nn.ReLU
    ):
        super().__init__()
        
        # Build network layers
        layers = []
        input_dim = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(activation())
            input_dim = h
        
        self.net = nn.Sequential(*layers)
        self.mu_head = nn.Linear(input_dim, action_dim)
        self.log_std_head = nn.Linear(input_dim, action_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network."""
        x = self.net(x)
        mu = self.mu_head(x)
        log_std = self.log_std_head(x).clamp(-20, 2)
        std = torch.exp(log_std)
        return mu, std

    def get_action(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample an action from the policy."""
        mu, std = self.forward(x)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        action = torch.tanh(action)  # constrain |x|<1
        return action, dist.log_prob(action).sum(dim=-1)

class CustomCritic(nn.Module):
    """Custom critic network for layer 1 and 2."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [512, 512, 256],
        activation: nn.Module = nn.ReLU
    ):
        super().__init__()
        
        # Build network layers
        layers = []
        input_dim = obs_dim + action_dim
        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(activation())
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = torch.cat([obs, action], dim=-1)
        value = self.net(x)
        return value

class Layer3ActorCritic(nn.Module):
    """Combined actor-critic network with attention for layer 3."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        n_heads: int = 4
    ):
        super().__init__()
        
        # Feature extraction
        self.input_proj = nn.Linear(obs_dim, hidden_dim)
        self.res_block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.res_block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            batch_first=True
        )

        # Actor head
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_logstd = nn.Linear(hidden_dim, action_dim)

        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the network."""
        x = F.relu(self.input_proj(obs))
        x = self.norm1(x + self.res_block1(x))
        x = self.norm2(x + self.res_block2(x))
        x = self.dropout(x)

        # Apply attention
        x_attn = x.unsqueeze(1)
        attn_out, _ = self.attention(x_attn, x_attn, x_attn)
        attn_out = attn_out.squeeze(1)

        # Actor outputs
        mean = self.actor_mean(attn_out)
        log_std = torch.clamp(self.actor_logstd(attn_out), -20, 2)
        std = torch.exp(log_std)

        # Critic output
        value = self.critic(attn_out)
        
        return mean, std, value

    def get_action(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample an action from the policy."""
        mean, std, _ = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        action = torch.tanh(action)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob