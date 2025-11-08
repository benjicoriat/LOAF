"""
Unit tests for LOAF trading system.
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loaf.utils.performance import PerformanceAnalytics
from loaf.utils.logging import MetricsLogger
from loaf.utils.portfolio import PortfolioManager
from loaf.utils.risk import RiskManager
from loaf.utils.evaluation import ModelEvaluator


class TestPerformanceAnalytics(unittest.TestCase):
    """Test performance analytics utilities."""
    
    def setUp(self):
        """Set up test data."""
        self.dates = pd.date_range(start='2020-01-01', end='2021-12-31', freq='B')
        self.returns = pd.Series(np.random.normal(0.001, 0.02, len(self.dates)), 
                               index=self.dates)
        self.analytics = PerformanceAnalytics()
        
    def test_returns_calculations(self):
        """Test basic returns calculations."""
        prices = pd.Series(100 * (1 + self.returns).cumprod())
        calc_returns = self.analytics.calculate_returns(prices)
        # Skip the first element which will be NaN
        np.testing.assert_array_almost_equal(calc_returns[1:], self.returns[1:])
        
    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        sharpe = self.analytics.calculate_sharpe_ratio(self.returns)
        self.assertIsInstance(sharpe, float)
        
    def test_max_drawdown(self):
        """Test maximum drawdown calculation."""
        drawdown, peak, valley = self.analytics.calculate_max_drawdown(self.returns)
        self.assertIsInstance(drawdown, float)
        self.assertLessEqual(drawdown, 0)
        self.assertIsInstance(peak, pd.Timestamp)
        self.assertIsInstance(valley, pd.Timestamp)


class TestLogging(unittest.TestCase):
    """Test logging functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.logger = MetricsLogger("./test_logs", "test_experiment")
        self.test_data = pd.DataFrame({
            'value': np.random.randn(100),
            'timestamp': pd.date_range(start='2020-01-01', periods=100)
        })
        
    def test_market_data_logging(self):
        """Test market data logging."""
        self.logger.log_market_data(self.test_data, "test_market_data")
        
    def test_metrics_logging(self):
        """Test metrics logging."""
        metrics = {'loss': 0.5, 'accuracy': 0.85}
        self.logger.log_training_metrics(metrics, step=1)


class TestPortfolioManagement(unittest.TestCase):
    """Test portfolio management functionality."""
    
    def setUp(self):
        """Set up test portfolio."""
        self.portfolio = PortfolioManager(initial_cash=100000.0)
        self.test_prices = {
            'AAPL': 150.0,
            'GOOGL': 2800.0,
            'MSFT': 300.0
        }
        
    def test_trade_execution(self):
        """Test trade execution."""
        # Execute buy trade
        trade = self.portfolio.execute_trade(
            timestamp=datetime.now(),
            ticker='AAPL',
            action='BUY',
            quantity=10,
            price=150.0
        )
        self.assertIsNotNone(trade)
        self.assertEqual(trade.ticker, 'AAPL')
        
        # Check portfolio value
        self.portfolio.update_prices(self.test_prices)
        self.assertGreater(self.portfolio.total_portfolio_value, 0)


class TestRiskManagement(unittest.TestCase):
    """Test risk management functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.risk_manager = RiskManager()
        self.test_returns = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, 100),
            'GOOGL': np.random.normal(0.001, 0.02, 100),
            'MSFT': np.random.normal(0.001, 0.02, 100)
        })
        
    def test_var_calculation(self):
        """Test Value at Risk calculation."""
        var = self.risk_manager.calculate_var(self.test_returns['AAPL'])
        self.assertIsInstance(var, float)
        self.assertLess(var, 0)  # VaR should be negative
        
    def test_risk_limits(self):
        """Test risk limit checks."""
        positions = {
            'AAPL': 50000,
            'GOOGL': 30000,
            'MSFT': 20000
        }
        sector_mapping = {
            'AAPL': 'Technology',
            'GOOGL': 'Technology',
            'MSFT': 'Technology'
        }
        violations = self.risk_manager.check_risk_limits(
            positions, sector_mapping, 100000.0)
        self.assertIsInstance(violations, dict)


class TestModelEvaluation(unittest.TestCase):
    """Test model evaluation functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.evaluator = ModelEvaluator(MetricsLogger("./test_logs", "test_eval"))
        self.y_true = np.random.randn(100)
        self.y_pred = self.y_true + np.random.normal(0, 0.1, 100)
        
    def test_regression_metrics(self):
        """Test regression evaluation metrics."""
        metrics = self.evaluator.evaluate_predictions(
            self.y_true, self.y_pred, task_type='regression')
        self.assertIn('mse', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('r2', metrics)
        
    def test_trading_performance(self):
        """Test trading performance evaluation."""
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        metrics = self.evaluator.evaluate_trading_performance(returns)
        self.assertIsInstance(metrics, dict)


if __name__ == '__main__':
    unittest.main()