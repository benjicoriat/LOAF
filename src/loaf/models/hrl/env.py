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

    Observation: a 1-D vector with one numeric observation per ticker (e.g. normalized price).
    Action: allocation vector with values in [-1, 1]. Positive values = calls, negative = puts.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        all_tickers: list,
        obs_df: pd.DataFrame,
        price_data: pd.DataFrame,
        train_dates,
        num_days: int = 10,
        risk_free_rate: float = 0.05,
    ):
        super(OptionsTradingEnv, self).__init__()

        # Ensure train_dates is a DatetimeIndex
        train_dates = pd.DatetimeIndex(train_dates)

        self.all_tickers = all_tickers
        self.price_data = price_data

        # Align observations to training dates and keep only those dates
        common_idx = obs_df.index.intersection(train_dates)
        self.obs_df = obs_df.reindex(common_idx).fillna(method="ffill").fillna(0)
        self.train_dates = self.obs_df.index

        self.num_days = num_days
        self.risk_free_rate = risk_free_rate

        self.n_tickers = len(all_tickers)
        self.current_step = 0

        # Validate data
        if len(self.train_dates) == 0:
            raise ValueError("No valid training dates available after alignment with observations")
        missing = [t for t in self.all_tickers if t not in self.obs_df.columns]
        if missing:
            raise ValueError(f"Missing ticker columns in obs_df: {missing}")

        # Action: one value per ticker
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_tickers,), dtype=np.float32)

        # Observation: flattened vector (one scalar per ticker)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_tickers,), dtype=np.float32)

    def reset(self) -> np.ndarray:
        """Reset the environment to the initial state and return first observation."""
        self.current_step = 0
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        """Return the observation for the current step as a 1-D numpy array."""
        try:
            date = self.train_dates[self.current_step]
            obs = np.array([self.obs_df.loc[date, t] for t in self.all_tickers], dtype=np.float32)
        except Exception:
            obs = np.zeros(self.n_tickers, dtype=np.float32)
        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Apply action, compute reward, and return (obs, reward, done, info)."""
        # Clip and normalize action to represent exposures
        action = np.clip(np.array(action, dtype=float), -1.0, 1.0)

        sum_pos = np.sum(action[action > 0])
        sum_neg = np.abs(np.sum(action[action < 0]))
        total_exposure = max(1e-6, sum_pos + sum_neg)
        action = action / total_exposure

        # Compute reward using reward pipeline
        date = self.train_dates[self.current_step]
        reward, pnl, wins, losses, partial_losses = options_reward_pipeline(
            date_t=date,
            action_vector=action,
            all_tickers=self.all_tickers,
            num_days=self.num_days,
            price_data=self.price_data,
            risk_free_rate=self.risk_free_rate,
        )

        # Advance step
        self.current_step += 1
        done = self.current_step >= len(self.train_dates)

        obs = self._get_obs() if not done else np.zeros(self.n_tickers, dtype=np.float32)

        info = {"pnl": pnl, "wins": wins, "losses": losses, "partial_losses": partial_losses}
        return obs, float(reward), bool(done), info