"""
Risk management utilities.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats


class RiskManager:
    """
    Comprehensive risk management system.
    """
    def __init__(self, max_position_size: float = 0.2,
                 max_sector_exposure: float = 0.4,
                 stop_loss_threshold: float = 0.02,
                 var_confidence: float = 0.95,
                 leverage_limit: float = 1.0):
        self.max_position_size = max_position_size
        self.max_sector_exposure = max_sector_exposure
        self.stop_loss_threshold = stop_loss_threshold
        self.var_confidence = var_confidence
        self.leverage_limit = leverage_limit
        
    def calculate_position_risks(self, positions: Dict[str, float],
                               prices: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate risk metrics for each position."""
        returns = prices.pct_change().dropna()
        position_risks = {}
        
        for ticker, quantity in positions.items():
            if ticker in returns.columns:
                position_returns = returns[ticker]
                position_value = quantity * prices[ticker].iloc[-1]
                
                risk_metrics = {
                    'volatility': position_returns.std() * np.sqrt(252),
                    'var': self.calculate_var(position_returns, self.var_confidence),
                    'expected_shortfall': self.calculate_expected_shortfall(position_returns, self.var_confidence),
                    'position_size': position_value,
                    'drawdown': self.calculate_drawdown(position_returns)[-1]
                }
                
                position_risks[ticker] = risk_metrics
                
        return position_risks
    
    def check_risk_limits(self, positions: Dict[str, float],
                         sector_mapping: Dict[str, str],
                         total_portfolio_value: float) -> Dict[str, List[str]]:
        """Check if positions violate risk limits."""
        violations = {
            'position_size': [],
            'sector_exposure': [],
            'leverage': []
        }
        
        # Check position size limits
        for ticker, value in positions.items():
            position_weight = value / total_portfolio_value
            if position_weight > self.max_position_size:
                violations['position_size'].append(ticker)
                
        # Check sector exposure limits
        sector_exposure = {}
        for ticker, value in positions.items():
            sector = sector_mapping.get(ticker, 'Unknown')
            sector_exposure[sector] = sector_exposure.get(sector, 0) + value / total_portfolio_value
            
        for sector, exposure in sector_exposure.items():
            if exposure > self.max_sector_exposure:
                violations['sector_exposure'].append(sector)
                
        # Check leverage
        total_exposure = sum(abs(value) for value in positions.values())
        if total_exposure / total_portfolio_value > self.leverage_limit:
            violations['leverage'].append('Portfolio')
            
        return violations
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Value at Risk."""
        return np.percentile(returns, (1 - confidence) * 100)
    
    @staticmethod
    def calculate_expected_shortfall(returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Expected Shortfall (CVaR)."""
        var = RiskManager.calculate_var(returns, confidence)
        return returns[returns <= var].mean()
    
    @staticmethod
    def calculate_drawdown(returns: pd.Series) -> pd.Series:
        """Calculate drawdown series."""
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        return drawdowns
    
    def calculate_portfolio_var(self, positions: Dict[str, float],
                              returns: pd.DataFrame,
                              correlation_matrix: Optional[pd.DataFrame] = None) -> float:
        """Calculate portfolio VaR considering correlations."""
        if correlation_matrix is None:
            correlation_matrix = returns.corr()
            
        position_weights = pd.Series(positions)
        portfolio_std = np.sqrt(
            position_weights.dot(correlation_matrix).dot(position_weights)
        )
        
        z_score = stats.norm.ppf(1 - self.var_confidence)
        portfolio_var = portfolio_std * z_score
        
        return portfolio_var
    
    def generate_risk_report(self, positions: Dict[str, float],
                           prices: pd.DataFrame,
                           sector_mapping: Dict[str, str],
                           total_portfolio_value: float) -> Dict[str, Dict]:
        """Generate comprehensive risk report."""
        position_risks = self.calculate_position_risks(positions, prices)
        risk_violations = self.check_risk_limits(positions, sector_mapping, total_portfolio_value)
        returns = prices.pct_change().dropna()
        
        portfolio_metrics = {
            'total_risk': {
                'portfolio_var': self.calculate_portfolio_var(positions, returns),
                'portfolio_volatility': returns.std() * np.sqrt(252),
                'max_drawdown': self.calculate_drawdown(returns.sum(axis=1))[-1]
            },
            'position_risks': position_risks,
            'risk_violations': risk_violations,
            'concentration_risk': {
                'position_concentration': self._calculate_herfindahl_index(positions),
                'sector_concentration': self._calculate_sector_concentration(positions, sector_mapping)
            }
        }
        
        return portfolio_metrics
    
    @staticmethod
    def _calculate_herfindahl_index(positions: Dict[str, float]) -> float:
        """Calculate Herfindahl Index for concentration risk."""
        total = sum(abs(v) for v in positions.values())
        weights = [abs(v)/total for v in positions.values()]
        return sum(w*w for w in weights)
    
    @staticmethod
    def _calculate_sector_concentration(positions: Dict[str, float],
                                     sector_mapping: Dict[str, str]) -> float:
        """Calculate sector concentration using Herfindahl Index."""
        sector_exposure = {}
        total = sum(abs(v) for v in positions.values())
        
        for ticker, value in positions.items():
            sector = sector_mapping.get(ticker, 'Unknown')
            sector_exposure[sector] = sector_exposure.get(sector, 0) + abs(value)
            
        weights = [v/total for v in sector_exposure.values()]
        return sum(w*w for w in weights)
    
    def adjust_positions_for_risk(self, positions: Dict[str, float],
                                prices: pd.DataFrame,
                                sector_mapping: Dict[str, str],
                                total_portfolio_value: float) -> Dict[str, float]:
        """Adjust positions to comply with risk limits."""
        adjusted_positions = positions.copy()
        
        # Check and adjust position sizes
        for ticker, value in positions.items():
            position_weight = value / total_portfolio_value
            if position_weight > self.max_position_size:
                adjusted_positions[ticker] = self.max_position_size * total_portfolio_value
                
        # Check and adjust sector exposures
        sector_exposure = {}
        for ticker, value in adjusted_positions.items():
            sector = sector_mapping.get(ticker, 'Unknown')
            sector_exposure[sector] = sector_exposure.get(sector, 0) + value / total_portfolio_value
            
        for sector, exposure in sector_exposure.items():
            if exposure > self.max_sector_exposure:
                scale_factor = self.max_sector_exposure / exposure
                for ticker, value in adjusted_positions.items():
                    if sector_mapping.get(ticker) == sector:
                        adjusted_positions[ticker] *= scale_factor
                        
        # Adjust for leverage
        total_exposure = sum(abs(v) for v in adjusted_positions.values())
        if total_exposure / total_portfolio_value > self.leverage_limit:
            scale_factor = self.leverage_limit * total_portfolio_value / total_exposure
            adjusted_positions = {k: v * scale_factor for k, v in adjusted_positions.items()}
            
        return adjusted_positions