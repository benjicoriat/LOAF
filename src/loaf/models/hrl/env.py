"""
Environment definition for the HRL system.
"""

import gym
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
from gym import spaces
from ...models.hrl.rewards import options_reward_pipeline

class OptionsTradingEnv(gym.Env):
    """
    Custom Gym environment for options trading with multiple tickers.
    
    Attributes:
        observation_space: Normalized quantitative vectors
        action_space: Allocation vector with constraints |x|<1, sum=1
        reward: Output from options_reward_pipeline
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(
        self,
        all_tickers: list,
        obs_df: pd.DataFrame,
        price_data: pd.DataFrame,
        train_dates: list,
        num_days: int = 10,
        risk_free_rate: float = 0.05
    ):
        super(OptionsTradingEnv, self).__init__()
        
        self.all_tickers = all_tickers
        self.price_data = price_data
        self.obs_df = obs_df
        self.train_dates = train_dates
        self.num_days = num_days
        self.risk_free_rate = risk_free_rate
        
        self.n_tickers = len(all_tickers)
        self.current_step = 0
        
        # Define spaces
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_tickers,),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(self.n_tickers, self.obs_df.shape[1]),
            dtype=np.float32
        )
    
    def reset(self) -> np.ndarray:
        """Reset the environment to initial state."""
        self.current_step = 0
        return self._get_obs()
    
    def _get_obs(self) -> np.ndarray:
        """Get current observation."""
        date = self.train_dates[self.current_step]
        obs = self.obs_df.loc[date].values
        return obs.astype(np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Action vector for ticker allocations
            
        Returns:
            Tuple containing:
            - observation
            - reward
            - done flag
            - info dictionary
        """
        # Enforce constraints
        action = np.clip(action, -1 + 1e-6, 1 - 1e-6)
        if np.any(action < 0):
            # Allow negative allocations for puts
            sum_abs = np.sum(np.abs(action))
            if sum_abs > 0:
                action = action / sum_abs
        else:
            action = action / (np.sum(action) + 1e-8)
        
        # Get current date and compute reward
        date = self.train_dates[self.current_step]
        reward, pnl, wins, losses, partial_losses = options_reward_pipeline(
            date_t=date,
            action_vector=action,
            all_tickers=self.all_tickers,
            num_days=self.num_days,
            price_data=self.price_data,
            risk_free_rate=self.risk_free_rate
        )
        
        # Update state
        self.current_step += 1
        done = self.current_step >= len(self.train_dates)
        
        # Get next observation
        obs = self._get_obs() if not done else np.zeros_like(self._get_obs())
        
        # Additional info
        info = {
            'pnl': pnl,
            'wins': wins,
            'losses': losses,
            'partial_losses': partial_losses
        }
        
        return obs, reward, done, info