"""
Portfolio management and trade execution utilities.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Trade:
    """Represents a single trade."""
    timestamp: datetime
    ticker: str
    action: str  # 'BUY' or 'SELL'
    quantity: float
    price: float
    transaction_cost: float = 0.0
    
    @property
    def value(self) -> float:
        """Calculate trade value."""
        return self.quantity * self.price
    
    @property
    def total_cost(self) -> float:
        """Calculate total cost including transaction costs."""
        return self.value + self.transaction_cost


@dataclass
class Position:
    """Represents a position in a single asset."""
    ticker: str
    quantity: float
    avg_price: float
    last_price: float
    
    @property
    def value(self) -> float:
        """Calculate position value."""
        return self.quantity * self.last_price
    
    @property
    def pnl(self) -> float:
        """Calculate unrealized P&L."""
        return self.quantity * (self.last_price - self.avg_price)
    
    @property
    def pnl_percentage(self) -> float:
        """Calculate percentage P&L."""
        return (self.last_price / self.avg_price - 1) * 100


class PortfolioManager:
    """
    Portfolio management system with position tracking and risk management.
    """
    def __init__(self, initial_cash: float = 1000000.0, transaction_cost: float = 0.001):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.transaction_cost = transaction_cost
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.history: List[Dict] = []
        
    def update_prices(self, prices: Dict[str, float]):
        """Update last prices for all positions."""
        for ticker, price in prices.items():
            if ticker in self.positions:
                self.positions[ticker].last_price = price
                
        # Record portfolio state
        self._record_state()
        
    def execute_trade(self, timestamp: datetime, ticker: str, action: str,
                     quantity: float, price: float) -> Optional[Trade]:
        """Execute a trade and update positions."""
        if quantity <= 0:
            return None
            
        transaction_cost = self.transaction_cost * quantity * price
        total_cost = quantity * price + transaction_cost
        
        if action == 'BUY':
            if total_cost > self.cash:
                return None  # Insufficient funds
                
            self.cash -= total_cost
            if ticker in self.positions:
                # Update existing position
                pos = self.positions[ticker]
                total_quantity = pos.quantity + quantity
                pos.avg_price = (pos.avg_price * pos.quantity + price * quantity) / total_quantity
                pos.quantity = total_quantity
            else:
                # Create new position
                self.positions[ticker] = Position(ticker, quantity, price, price)
                
        elif action == 'SELL':
            if ticker not in self.positions or self.positions[ticker].quantity < quantity:
                return None  # Insufficient position
                
            pos = self.positions[ticker]
            pos.quantity -= quantity
            self.cash += (quantity * price - transaction_cost)
            
            if pos.quantity == 0:
                del self.positions[ticker]
                
        # Record the trade
        trade = Trade(timestamp, ticker, action, quantity, price, transaction_cost)
        self.trades.append(trade)
        
        # Record portfolio state
        self._record_state()
        
        return trade
    
    def rebalance_portfolio(self, timestamp: datetime, target_weights: Dict[str, float],
                          prices: Dict[str, float]) -> List[Trade]:
        """Rebalance portfolio to target weights."""
        if not np.isclose(sum(target_weights.values()), 1.0):
            raise ValueError("Target weights must sum to 1")
            
        # Update prices
        self.update_prices(prices)
        
        # Calculate current weights
        total_value = self.total_portfolio_value
        current_weights = {
            ticker: pos.value / total_value 
            for ticker, pos in self.positions.items()
        }
        
        # Calculate required trades
        trades: List[Trade] = []
        for ticker, target_weight in target_weights.items():
            current_weight = current_weights.get(ticker, 0.0)
            weight_diff = target_weight - current_weight
            
            if abs(weight_diff) < 0.0001:  # Small difference threshold
                continue
                
            trade_value = weight_diff * total_value
            quantity = abs(trade_value) / prices[ticker]
            action = 'BUY' if weight_diff > 0 else 'SELL'
            
            trade = self.execute_trade(timestamp, ticker, action, quantity, prices[ticker])
            if trade:
                trades.append(trade)
                
        return trades
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate portfolio metrics."""
        total_value = self.total_portfolio_value
        unrealized_pnl = sum(pos.pnl for pos in self.positions.values())
        realized_pnl = sum(
            trade.price * trade.quantity if trade.action == 'SELL'
            else -trade.price * trade.quantity
            for trade in self.trades
        )
        
        return {
            'total_value': total_value,
            'cash': self.cash,
            'unrealized_pnl': unrealized_pnl,
            'realized_pnl': realized_pnl,
            'total_pnl': unrealized_pnl + realized_pnl,
            'return_pct': (total_value / self.initial_cash - 1) * 100
        }
    
    @property
    def total_portfolio_value(self) -> float:
        """Calculate total portfolio value."""
        return self.cash + sum(pos.value for pos in self.positions.values())
    
    def get_position_weights(self) -> Dict[str, float]:
        """Get current position weights."""
        total_value = self.total_portfolio_value
        return {
            ticker: pos.value / total_value
            for ticker, pos in self.positions.items()
        }
    
    def get_portfolio_history(self) -> pd.DataFrame:
        """Get portfolio history as DataFrame."""
        return pd.DataFrame(self.history)
    
    def _record_state(self):
        """Record current portfolio state."""
        metrics = self.calculate_metrics()
        weights = self.get_position_weights()
        
        state = {
            'timestamp': datetime.now(),
            'total_value': metrics['total_value'],
            'cash': self.cash,
            'positions': {
                ticker: {
                    'quantity': pos.quantity,
                    'value': pos.value,
                    'pnl': pos.pnl,
                    'weight': weights.get(ticker, 0.0)
                }
                for ticker, pos in self.positions.items()
            }
        }
        
        self.history.append(state)