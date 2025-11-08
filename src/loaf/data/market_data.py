"""
Data fetching and preprocessing module.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import os
import warnings
from typing import Dict, List
from ..config.config import MarketConfig

class MarketDataFetcher:
    """Fetches and processes market data."""

    def __init__(self, config: MarketConfig):
        self.config = config
        self.data: Dict[str, pd.DataFrame] = {}

    @staticmethod
    def compute_daily_volatility(close_series: pd.Series, window: int = 20) -> pd.Series:
        """Compute rolling daily volatility (log returns)."""
        log_returns = np.log(close_series / close_series.shift(1))
        vol = log_returns.rolling(window=window).std().fillna(method='bfill')
        return vol

    @staticmethod
    def compute_sharpe_ratio(close_series: pd.Series, window: int = 10) -> pd.Series:
        """Compute rolling 10-day Sharpe ratio proxy."""
        log_returns = np.log(close_series / close_series.shift(1))
        mean_return = log_returns.rolling(window=window).mean()
        std_return = log_returns.rolling(window=window).std()
        sharpe = (mean_return / std_return).fillna(method='bfill')
        return sharpe

    def download_data(self, save_dir: str = "./data") -> Dict[str, str]:
        """Downloads and processes market data for all tickers."""
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=DeprecationWarning)

        os.makedirs(save_dir, exist_ok=True)
        
        all_tickers = [ticker for group in self.config.tickers.values() for ticker in group]
        start_date = self.config.date_ranges["Download"]["start"]
        end_date = self.config.date_ranges["Download"]["end"]

        # Download data
        raw_data = {}
        for ticker in all_tickers:
            try:
                print(f"Downloading {ticker}...")
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if data.empty:
                    print(f"⚠️ No data for {ticker}, skipping...")
                    continue
                raw_data[ticker] = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            except Exception as e:
                print(f"⚠️ Failed to download {ticker}: {e}")
                continue

        # Process data
        paths = self._process_and_save_data(raw_data, save_dir)
        return paths

    def _process_and_save_data(self, raw_data: Dict[str, pd.DataFrame], save_dir: str) -> Dict[str, str]:
        """Process raw data and save to CSV files."""
        common_index = pd.date_range(
            start=self.config.date_ranges["Download"]["start"],
            end=self.config.date_ranges["Download"]["end"],
            freq='B'
        )

        def build_df(col_name: str) -> pd.DataFrame:
            return pd.DataFrame(
                {t: d[col_name].squeeze() for t, d in raw_data.items()},
                index=common_index
            ).interpolate().bfill()

        # Build DataFrames
        dfs = {
            'open': build_df('Open'),
            'high': build_df('High'),
            'low': build_df('Low'),
            'close': build_df('Close'),
            'volume': build_df('Volume').fillna(0)
        }

        # Add derived metrics
        dfs['normalized_close'] = dfs['close'] / dfs['close'].iloc[0]
        dfs['volatility'] = dfs['close'].apply(self.compute_daily_volatility)
        dfs['sharpe'] = dfs['close'].apply(self.compute_sharpe_ratio)

        # Save files
        paths = {}
        for name, df in dfs.items():
            path = os.path.join(save_dir, f"{name}.csv")
            df.to_csv(path)
            paths[name] = path

        return paths