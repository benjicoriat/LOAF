"""
Unit tests for market data functionality.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
from loaf.data.market_data import MarketDataFetcher
from loaf.config.config import MarketConfig

class TestMarketData(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.config = MarketConfig.default()
        self.fetcher = MarketDataFetcher(self.config)
        self.test_data_dir = Path("test_data")
        self.test_data_dir.mkdir(exist_ok=True)

    def tearDown(self):
        """Clean up after tests."""
        import shutil
        shutil.rmtree(self.test_data_dir)

    def test_volatility_calculation(self):
        """Test volatility calculation."""
        # Create test data
        dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='B')
        close_prices = pd.Series(
            data=np.array([100.0, 101.0, 99.0, 102.0, 98.0]),
            index=dates[:5]
        )
        
        vol = self.fetcher.compute_daily_volatility(close_prices, window=3)
        
        # Basic checks
        self.assertEqual(len(vol), len(close_prices))
        self.assertTrue(all(vol >= 0))  # Volatility should be non-negative

    def test_sharpe_calculation(self):
        """Test Sharpe ratio calculation."""
        dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='B')
        close_prices = pd.Series(
            data=np.array([100.0, 101.0, 102.0, 103.0, 104.0]),
            index=dates[:5]
        )
        
        sharpe = self.fetcher.compute_sharpe_ratio(close_prices, window=3)
        
        self.assertEqual(len(sharpe), len(close_prices))

    def test_data_download(self):
        """Test market data download functionality."""
        # Use a small set of tickers for testing
        self.config.tickers = {
            "Test": ["SPY", "AAPL"]  # Use well-known tickers for testing
        }
        self.config.date_ranges = {
            "Download": {
                "start": "2023-01-01",
                "end": "2023-01-05"
            }
        }
        
        paths = self.fetcher.download_data(save_dir=str(self.test_data_dir))
        
        # Check that files were created
        expected_files = ['open.csv', 'high.csv', 'low.csv', 'close.csv',
                         'volume.csv', 'volatility.csv', 'sharpe.csv']
        
        for file in expected_files:
            self.assertTrue(
                (self.test_data_dir / file).exists(),
                f"Expected file {file} not found"
            )

if __name__ == '__main__':
    unittest.main()