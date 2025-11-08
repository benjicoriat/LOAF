"""
Custom PyTorch-based RL agents for trading layers 2 and 3.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
from typing import Tuple


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        
        self.d_k = d_model // num_heads
        self.scale = self.d_k ** -0.5
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                mask: torch.Tensor = None) -> torch.Tensor:
        batch_size = q.size(0)
        
        # Linear transformations and reshape
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        out = self.out_linear(out)
        
        return out


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        attn_out = self.attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class CustomActorCritic(nn.Module):
    """
    Actor-Critic network with attention mechanism for Layer 2.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int,
                 num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        
        # Actor network (portfolio weights)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic network (state value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.state_encoder(state)
        x = x.unsqueeze(1)  # Add sequence dimension
        x = self.attention(x, x, x)
        x = x.squeeze(1)
        
        # Get action distribution parameters
        action_mean = self.actor(x)
        action_std = self.log_std.exp()
        
        # Get state value
        value = self.critic(x)
        
        return action_mean, action_std, value
        
    def train(self, env, num_episodes: int, seed: int = None):
        """Train the agent using PPO."""
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        
        for episode in range(num_episodes):
            state, _ = env.reset(seed=seed)
            done = False
            episode_reward = 0
            
            while not done:
                # Convert state to tensor
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                
                # Get action from policy
                with torch.no_grad():
                    action_mean, action_std, _ = self(state_tensor)
                    dist = Normal(action_mean, action_std)
                    action = dist.sample()
                    action = action.squeeze().numpy()
                
                # Take action in environment
                next_state, reward, done, _, _ = env.step(action)
                episode_reward += reward
                
                # Store transition
                self.memory.append((state, action, reward, next_state, done))
                
                # Update policy if memory is full
                if len(self.memory) >= self.batch_size:
                    self._update_policy(optimizer)
                
                state = next_state
            
            if episode % 10 == 0:
                print(f"Episode {episode}, Reward: {episode_reward:.2f}")
                
    def evaluate(self, env) -> dict:
        """Evaluate the trained agent."""
        state, _ = env.reset()
        done = False
        total_reward = 0
        actions_taken = []
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_mean, _, _ = self(state_tensor)
                action = action_mean.squeeze().numpy()
            
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            actions_taken.append(action)
            state = next_state
        
        return {
            'total_reward': total_reward,
            'actions': np.array(actions_taken)
        }


class MetaActorCritic(nn.Module):
    """
    Meta-learning Actor-Critic network with transformer architecture for Layer 3.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int,
                 num_encoder_layers: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Transformer encoder
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, hidden_dim * 4, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Actor network (portfolio adjustment factors)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Output in [-1, 1]
        )
        
        # Critic network (state value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.state_encoder(state)
        x = x.unsqueeze(1)  # Add sequence dimension
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        x = x.squeeze(1)
        
        # Get action distribution parameters
        action_mean = self.actor(x)
        action_std = self.log_std.exp()
        
        # Get state value
        value = self.critic(x)
        
        return action_mean, action_std, value
        
    def meta_train(self, env, num_tasks: int, num_episodes_per_task: int, seed: int = None):
        """Train the agent using MAML-style meta-learning."""
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        
        for task in range(num_tasks):
            # Sample new task (market regime)
            env.reset(seed=seed)
            
            # Store original parameters
            original_params = {name: param.clone() for name, param in self.named_parameters()}
            
            # Inner loop (task adaptation)
            for episode in range(num_episodes_per_task):
                state, _ = env.reset(seed=seed)
                done = False
                episode_reward = 0
                
                while not done:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    
                    # Get action from policy
                    with torch.no_grad():
                        action_mean, action_std, _ = self(state_tensor)
                        dist = Normal(action_mean, action_std)
                        action = dist.sample()
                        action = action.squeeze().numpy()
                    
                    # Take action in environment
                    next_state, reward, done, _, _ = env.step(action)
                    episode_reward += reward
                    
                    # Store transition and update policy
                    self.memory.append((state, action, reward, next_state, done))
                    if len(self.memory) >= self.batch_size:
                        self._update_policy(optimizer)
                    
                    state = next_state
            
            # Outer loop (meta-update)
            meta_loss = self._compute_meta_loss()
            optimizer.zero_grad()
            meta_loss.backward()
            optimizer.step()
            
            # Restore original parameters
            for name, param in self.named_parameters():
                param.data = original_params[name]
            
            if task % 5 == 0:
                print(f"Task {task}, Average Reward: {episode_reward/num_episodes_per_task:.2f}")
    
    def meta_evaluate(self, env) -> dict:
        """Evaluate the meta-learned policy."""
        state, _ = env.reset()
        done = False
        total_reward = 0
        actions_taken = []
        
        # Quick adaptation phase
        self._adapt_to_current_task(env)
        
        # Evaluation phase
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_mean, _, _ = self(state_tensor)
                action = action_mean.squeeze().numpy()
            
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            actions_taken.append(action)
            state = next_state
        
        return {
            'total_reward': total_reward,
            'actions': np.array(actions_taken)
        }
        
    def _adapt_to_current_task(self, env, num_steps: int = 100):
        """Quick adaptation to current market regime."""
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        
        state, _ = env.reset()
        for _ in range(num_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_mean, action_std, value = self(state_tensor)
            
            dist = Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            next_state, reward, done, _, _ = env.step(action.squeeze().numpy())
            
            # Compute losses
            value_loss = F.mse_loss(value, torch.tensor([reward]))
            policy_loss = -log_prob.mean() * (reward - value.item())
            
            # Update policy
            total_loss = value_loss + policy_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            if done:
                state, _ = env.reset()
            else:
                state = next_state