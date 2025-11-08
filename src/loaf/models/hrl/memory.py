"""
Memory buffer for experience replay in the HRL system.
"""

import numpy as np
import torch
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class Experience:
    """Single experience tuple."""
    observation: np.ndarray
    action: np.ndarray
    reward: float
    next_observation: np.ndarray
    done: bool

class ReplayBuffer:
    """
    Buffer for storing and sampling experience tuples.
    
    Attributes:
        max_size: Maximum number of experiences to store
    """
    
    def __init__(self, max_size: int = 1000000):
        self.max_size = max_size
        self.obs: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.next_obs: List[np.ndarray] = []
        self.done: List[bool] = []

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool
    ):
        """Add a new experience to the buffer."""
        if len(self.obs) >= self.max_size:
            self.obs.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_obs.pop(0)
            self.done.pop(0)
            
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_obs.append(next_obs)
        self.done.append(done)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample a batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of tensors containing batched experiences
        """
        idx = np.random.choice(len(self.obs), batch_size, replace=False)
        
        return (
            torch.tensor(np.array([self.obs[i] for i in idx]), dtype=torch.float32),
            torch.tensor(np.array([self.actions[i] for i in idx]), dtype=torch.float32),
            torch.tensor(np.array([self.rewards[i] for i in idx]), dtype=torch.float32).unsqueeze(-1),
            torch.tensor(np.array([self.next_obs[i] for i in idx]), dtype=torch.float32),
            torch.tensor(np.array([self.done[i] for i in idx]), dtype=torch.float32).unsqueeze(-1)
        )

    def __len__(self) -> int:
        """Get current size of buffer."""
        return len(self.obs)