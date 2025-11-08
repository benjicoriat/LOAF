"""
Performance metrics and analytics utilities.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats


class PerformanceAnalytics:
    """
    Comprehensive performance analytics for trading strategies.
    """
    @staticmethod
    def calculate_returns(prices: pd.Series) -> pd.Series:
        """Calculate returns from price series."""
        return prices.pct_change()
    
    @staticmethod
    def calculate_log_returns(prices: pd.Series) -> pd.Series:
        """Calculate logarithmic returns from price series."""
        return np.log(prices / prices.shift(1))
    
    @staticmethod
    def calculate_cumulative_returns(returns: pd.Series) -> pd.Series:
        """Calculate cumulative returns."""
        return (1 + returns).cumprod() - 1
    
    @staticmethod
    def calculate_annualized_return(returns: pd.Series) -> float:
        """Calculate annualized return."""
        total_return = PerformanceAnalytics.calculate_cumulative_returns(returns).iloc[-1]
        years = len(returns) / 252  # Assuming 252 trading days per year
        return (1 + total_return) ** (1 / years) - 1
    
    @staticmethod
    def calculate_volatility(returns: pd.Series, annualized: bool = True) -> float:
        """Calculate return volatility."""
        vol = returns.std()
        if annualized:
            vol *= np.sqrt(252)  # Annualize
        return vol
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        excess_returns = returns - risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / returns.std()
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio."""
        excess_returns = returns - risk_free_rate / 252
        downside_returns = returns[returns < 0]
        downside_std = np.sqrt(np.mean(downside_returns ** 2))
        return np.sqrt(252) * excess_returns.mean() / downside_std
    
    @staticmethod
    def calculate_max_drawdown(returns: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
        """Calculate maximum drawdown and its timeframe."""
        cum_returns = PerformanceAnalytics.calculate_cumulative_returns(returns)
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns - rolling_max
        max_drawdown = drawdowns.min()
        end_idx = drawdowns.idxmin()
        peak_idx = rolling_max.loc[:end_idx].idxmax()
        return max_drawdown, peak_idx, end_idx
    
    @staticmethod
    def calculate_calmar_ratio(returns: pd.Series) -> float:
        """Calculate Calmar ratio."""
        ann_return = PerformanceAnalytics.calculate_annualized_return(returns)
        max_drawdown = abs(PerformanceAnalytics.calculate_max_drawdown(returns)[0])
        return ann_return / max_drawdown
    
    @staticmethod
    def calculate_omega_ratio(returns: pd.Series, threshold: float = 0) -> float:
        """Calculate Omega ratio."""
        excess_returns = returns - threshold
        positive_returns = excess_returns[excess_returns > 0].sum()
        negative_returns = abs(excess_returns[excess_returns < 0].sum())
        return positive_returns / negative_returns if negative_returns != 0 else np.inf
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Value at Risk."""
        return np.percentile(returns, (1 - confidence) * 100)
    
    @staticmethod
    def calculate_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        var = PerformanceAnalytics.calculate_var(returns, confidence)
        return returns[returns <= var].mean()
    
    @staticmethod
    def calculate_beta(returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate beta relative to market."""
        covar = np.cov(returns, market_returns)[0][1]
        market_var = np.var(market_returns)
        return covar / market_var
    
    @staticmethod
    def calculate_alpha(returns: pd.Series, market_returns: pd.Series, 
                       risk_free_rate: float = 0.02) -> float:
        """Calculate Jensen's Alpha."""
        beta = PerformanceAnalytics.calculate_beta(returns, market_returns)
        excess_return = PerformanceAnalytics.calculate_annualized_return(returns) - risk_free_rate
        market_premium = PerformanceAnalytics.calculate_annualized_return(market_returns) - risk_free_rate
        return excess_return - beta * market_premium
    
    @staticmethod
    def calculate_information_ratio(returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate Information Ratio."""
        active_returns = returns - benchmark_returns
        return np.sqrt(252) * active_returns.mean() / active_returns.std()
    
    @staticmethod
    def calculate_win_rate(returns: pd.Series) -> float:
        """Calculate win rate."""
        total_trades = len(returns)
        winning_trades = len(returns[returns > 0])
        return winning_trades / total_trades if total_trades > 0 else 0
    
    @staticmethod
    def calculate_profit_factor(returns: pd.Series) -> float:
        """Calculate profit factor."""
        positive_returns = returns[returns > 0].sum()
        negative_returns = abs(returns[returns < 0].sum())
        return positive_returns / negative_returns if negative_returns != 0 else np.inf
    
    @staticmethod
    def calculate_autocorrelation(returns: pd.Series, lag: int = 1) -> float:
        """Calculate return autocorrelation."""
        return returns.autocorr(lag)
    
    @staticmethod
    def calculate_skewness(returns: pd.Series) -> float:
        """Calculate return skewness."""
        return returns.skew()
    
    @staticmethod
    def calculate_kurtosis(returns: pd.Series) -> float:
        """Calculate return kurtosis."""
        return returns.kurtosis()
    
    @staticmethod
    def calculate_rolling_metrics(returns: pd.Series, window: int = 30) -> pd.DataFrame:
        """Calculate rolling performance metrics."""
        rolling_ret = returns.rolling(window=window).mean() * 252  # Annualized
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
        rolling_sharpe = rolling_ret / rolling_vol
        
        return pd.DataFrame({
            'Returns': rolling_ret,
            'Volatility': rolling_vol,
            'Sharpe': rolling_sharpe
        })
    
    @staticmethod
    def generate_performance_summary(returns: pd.Series, 
                                  benchmark_returns: Optional[pd.Series] = None,
                                  risk_free_rate: float = 0.02) -> Dict[str, float]:
        """Generate comprehensive performance summary."""
        metrics = {
            'Total Return': PerformanceAnalytics.calculate_cumulative_returns(returns).iloc[-1],
            'Annualized Return': PerformanceAnalytics.calculate_annualized_return(returns),
            'Annualized Volatility': PerformanceAnalytics.calculate_volatility(returns),
            'Sharpe Ratio': PerformanceAnalytics.calculate_sharpe_ratio(returns, risk_free_rate),
            'Sortino Ratio': PerformanceAnalytics.calculate_sortino_ratio(returns, risk_free_rate),
            'Max Drawdown': PerformanceAnalytics.calculate_max_drawdown(returns)[0],
            'Calmar Ratio': PerformanceAnalytics.calculate_calmar_ratio(returns),
            'Win Rate': PerformanceAnalytics.calculate_win_rate(returns),
            'Profit Factor': PerformanceAnalytics.calculate_profit_factor(returns),
            'VaR (95%)': PerformanceAnalytics.calculate_var(returns),
            'CVaR (95%)': PerformanceAnalytics.calculate_cvar(returns),
            'Skewness': PerformanceAnalytics.calculate_skewness(returns),
            'Kurtosis': PerformanceAnalytics.calculate_kurtosis(returns)
        }
        
        if benchmark_returns is not None:
            metrics.update({
                'Beta': PerformanceAnalytics.calculate_beta(returns, benchmark_returns),
                'Alpha': PerformanceAnalytics.calculate_alpha(returns, benchmark_returns, risk_free_rate),
                'Information Ratio': PerformanceAnalytics.calculate_information_ratio(returns, benchmark_returns)
            })
            
        return metrics