"""
Custom Gym environments for trading layers 2 and 3.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import torch


class Layer2TradingEnv(gym.Env):
    """
    Layer 2 Trading Environment combining quantitative and NLP signals.
    """
    def __init__(self, data, dates, tickers):
        super().__init__()
        self.data = data
        self.dates = dates
        self.tickers = tickers
        self.current_step = 0
        
        # Define observation space (combined quant + NLP features)
        num_features = len(tickers) * 2  # Price features + NLP features per ticker
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_features,),
            dtype=np.float32
        )
        
        # Define action space (portfolio weights)
        self.action_space = spaces.Box(
            low=0,
            high=1,
            shape=(len(tickers),),
            dtype=np.float32
        )

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        # Ensure action sums to 1 (valid portfolio weights)
        action = np.array(action)
        action = action / action.sum()
        
        # Calculate returns
        returns = self._calculate_returns()
        reward = np.dot(action, returns)
        
        # Move to next timestep
        self.current_step += 1
        done = self.current_step >= len(self.dates) - 1
        
        obs = self._get_observation()
        return obs, reward, done, False, {}

    def _get_observation(self):
        """Combine price and NLP features into observation."""
        date = self.dates[self.current_step]
        price_features = []
        nlp_features = []
        
        for ticker in self.tickers:
            # Get price features
            price = self.data.loc[date, ticker]
            returns = self.data.loc[:date, ticker].pct_change().dropna()
            vol = returns.std()
            price_features.extend([price, vol])
            
            # Get NLP features (assuming stored in data)
            sentiment = self.data.loc[date, f"{ticker}_sentiment"]
            attention = self.data.loc[date, f"{ticker}_attention"]
            nlp_features.extend([sentiment, attention])
            
        obs = np.concatenate([price_features, nlp_features])
        return obs.astype(np.float32)

    def _calculate_returns(self):
        """Calculate asset returns for current timestep."""
        current_date = self.dates[self.current_step]
        next_date = self.dates[self.current_step + 1]
        
        returns = []
        for ticker in self.tickers:
            current_price = self.data.loc[current_date, ticker]
            next_price = self.data.loc[next_date, ticker]
            ret = (next_price - current_price) / current_price
            returns.append(ret)
            
        return np.array(returns, dtype=np.float32)


class Layer3MetaLearningEnv(gym.Env):
    """
    Layer 3 Meta-Learning Environment for adaptive trading strategies.
    """
    def __init__(self, data, dates, tickers, layer2_results):
        super().__init__()
        self.data = data
        self.dates = dates
        self.tickers = tickers
        self.layer2_results = layer2_results
        self.current_step = 0
        self.current_task = 0
        
        # Define observation space (market state + L2 agent state)
        num_market_features = len(tickers) * 3  # Price, volume, volatility
        num_l2_features = len(tickers)  # L2 agent portfolio weights
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_market_features + num_l2_features,),
            dtype=np.float32
        )
        
        # Define action space (portfolio adjustment factors)
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(len(tickers),),
            dtype=np.float32
        )

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        # Randomly select new market regime/task
        self.current_task = np.random.randint(0, self.num_tasks)
        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        # Get L2 portfolio weights
        l2_weights = self.layer2_results[self.current_step]
        
        # Adjust L2 weights using L3 actions
        adjusted_weights = l2_weights * (1 + action)
        adjusted_weights = np.clip(adjusted_weights, 0, None)
        adjusted_weights = adjusted_weights / adjusted_weights.sum()
        
        # Calculate returns using adjusted weights
        returns = self._calculate_returns()
        reward = np.dot(adjusted_weights, returns)
        
        # Add meta-learning reward bonus for adaptation
        adaptation_bonus = self._calculate_adaptation_bonus()
        reward += adaptation_bonus
        
        # Move to next timestep
        self.current_step += 1
        done = self.current_step >= len(self.dates) - 1
        
        obs = self._get_observation()
        return obs, reward, done, False, {}

    def _get_observation(self):
        """Combine market state and L2 agent state into observation."""
        date = self.dates[self.current_step]
        
        # Market features
        market_features = []
        for ticker in self.tickers:
            price = self.data.loc[date, ticker]
            volume = self.data.loc[date, f"{ticker}_volume"]
            returns = self.data.loc[:date, ticker].pct_change().dropna()
            vol = returns.std()
            market_features.extend([price, volume, vol])
        
        # L2 agent state
        l2_weights = self.layer2_results[self.current_step]
        
        obs = np.concatenate([market_features, l2_weights])
        return obs.astype(np.float32)

    def _calculate_returns(self):
        """Calculate asset returns for current timestep."""
        current_date = self.dates[self.current_step]
        next_date = self.dates[self.current_step + 1]
        
        returns = []
        for ticker in self.tickers:
            current_price = self.data.loc[current_date, ticker]
            next_price = self.data.loc[next_date, ticker]
            ret = (next_price - current_price) / current_price
            returns.append(ret)
            
        return np.array(returns, dtype=np.float32)

    def _calculate_adaptation_bonus(self):
        """Calculate bonus reward for successful adaptation to market regime."""
        # Measure how well the agent adapts to the current market regime
        regime_indicator = self._identify_market_regime()
        adaptation_score = self._evaluate_adaptation(regime_indicator)
        return adaptation_score * 0.1  # Scale the bonus

    def _identify_market_regime(self):
        """Identify current market regime using price patterns."""
        window = 20  # Look at last 20 days
        start_idx = max(0, self.current_step - window)
        regime_features = []
        
        for ticker in self.tickers:
            prices = self.data.loc[self.dates[start_idx:self.current_step], ticker]
            returns = prices.pct_change().dropna()
            vol = returns.std()
            trend = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
            regime_features.extend([vol, trend])
            
        return np.array(regime_features)

    def _evaluate_adaptation(self, regime_indicator):
        """Evaluate how well the agent has adapted to the current regime."""
        # Compare agent's actions with optimal actions for the regime
        optimal_weights = self._get_optimal_weights(regime_indicator)
        l2_weights = self.layer2_results[self.current_step]
        adaptation_score = -np.mean(np.abs(l2_weights - optimal_weights))
        return adaptation_score

    def _get_optimal_weights(self, regime_indicator):
        """Get optimal portfolio weights for current regime."""
        # Simple regime-based portfolio optimization
        volatilities = regime_indicator[::2]  # Every other feature is volatility
        trends = regime_indicator[1::2]  # Every other feature is trend
        
        # Favor low volatility in high vol regime
        if np.mean(volatilities) > 0.02:  # High vol regime
            weights = 1 / (volatilities + 1e-6)
        else:  # Trend following in low vol regime
            weights = np.maximum(trends, 0)
            
        # Ensure valid probability distribution
        weights = np.maximum(weights, 0)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones_like(weights) / len(weights)
            
        return weights