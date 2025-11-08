"""
Reward functions for the HRL system.
"""

import numpy as np
from scipy.stats import norm
from typing import Tuple, Dict, Any

def black_scholes_price(
    spot: float,
    strike: float,
    time_to_exp: float,
    vol: float,
    r: float,
    option_type: str = "call"
) -> float:
    """
    Computes the Black-Scholes option price.
    
    Args:
        spot: Spot price (S)
        strike: Strike price (K)
        time_to_exp: Time to expiration in years
        vol: Volatility (sigma)
        r: Risk-free rate
        option_type: "call" or "put"
        
    Returns:
        float: Option premium
    """
    if time_to_exp <= 0 or vol <= 0:
        return max(0.0, (spot - strike) if option_type == "call" else (strike - spot))

    try:
        d1 = (np.log(spot / strike) + (r + 0.5 * vol ** 2) * time_to_exp) / (vol * np.sqrt(time_to_exp))
        d2 = d1 - vol * np.sqrt(time_to_exp)
        
        if option_type == "call":
            price = spot * norm.cdf(d1) - strike * np.exp(-r * time_to_exp) * norm.cdf(d2)
        else:
            price = strike * np.exp(-r * time_to_exp) * norm.cdf(-d2) - spot * norm.cdf(-d1)
            
        return price
    except Exception:
        return 0.0

def options_reward_pipeline(
    date_t: str,
    action_vector: np.ndarray,
    all_tickers: list,
    num_days: int,
    price_data: Dict[str, np.ndarray],
    risk_free_rate: float = 0.05
) -> Tuple[float, float, int, int, int]:
    """
    Computes the reward for an options allocation on a given date.
    
    Args:
        date_t: Target date
        action_vector: Array of allocations (-1 to 1)
        all_tickers: List of ticker symbols
        num_days: Number of days for option expiry
        price_data: Dictionary of price data per ticker
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Tuple containing:
        - reward: Combined reward metric
        - PnL: Profit and Loss
        - num_wins: Number of winning trades
        - num_loss: Number of total losses
        - num_partial_loss: Number of partial losses
    """
    # Validate and normalize action vector if needed
    action_vector = np.array(action_vector, dtype=float)
    
    # Ensure no individual action exceeds bounds
    if np.any(np.abs(action_vector) > 1.0):
        action_vector = np.clip(action_vector, -1.0, 1.0)
        
    # Normalize by total exposure
    sum_pos = np.sum(action_vector[action_vector > 0])
    sum_neg = np.abs(np.sum(action_vector[action_vector < 0]))
    total_exposure = max(1e-6, sum_pos + sum_neg)
    action_vector = action_vector / total_exposure  # This ensures sum of absolute values = 1

    # Ensure date exists
    if date_t not in price_data.index:
        raise KeyError(f"Target date {date_t} not in price_data")
    date_idx = price_data.index.get_loc(date_t)
    if date_idx == 0:
        raise ValueError("Cannot price options on first available date")

    # Precompute volatility
    vol_cache = {}
    hist_window = 20
    for ticker in all_tickers:
        try:
            returns = np.log(price_data[ticker].pct_change() + 1)
            vol = returns.rolling(hist_window).std().iloc[date_idx - 1]
            vol_cache[ticker] = vol if not np.isnan(vol) else 0.2
        except Exception:
            vol_cache[ticker] = 0.2

    # Process each ticker
    payouts = []
    num_wins = 0
    num_loss = 0
    num_partial_loss = 0

    for i, ticker in enumerate(all_tickers):
        alloc = action_vector[i]
        option_type = "call" if alloc >= 0 else "put"
        weight = abs(alloc)

        try:
            spot_prev = price_data[ticker].iloc[date_idx - 1]
            spot_t = price_data[ticker].iloc[date_idx]
            vol = vol_cache[ticker]
            time_to_exp = num_days / 252.0  # convert to years

            # Price option
            premium = black_scholes_price(spot_prev, spot_prev, time_to_exp, vol, risk_free_rate, option_type)

            # Compute payout
            if option_type == "call":
                payout_raw = max(0, spot_t - spot_prev) - premium
            else:
                payout_raw = max(0, spot_prev - spot_t) - premium

            payout_weighted = payout_raw * weight
            payouts.append(payout_weighted)

            # Track metrics
            if payout_weighted > 0:
                num_wins += 1
            elif np.isclose(payout_weighted, -premium * weight):
                num_loss += 1
            elif payout_weighted < 0:
                num_partial_loss += 1

        except Exception as e:
            payouts.append(0.0)
            continue

    # Calculate final metrics
    PnL = np.sum(payouts)
    reward = PnL + num_wins - 2 * num_loss

    return reward, PnL, num_wins, num_loss, num_partial_loss