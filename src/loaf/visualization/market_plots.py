"""
Market visualization module.
"""

import matplotlib.pyplot as plt
import pandas as pd
import os
from typing import Dict, List

class MarketVisualizer:
    """Handles market data visualization."""

    def __init__(self, base_folder_plots: str = "./plots"):
        self.base_folder_plots = base_folder_plots
        os.makedirs(base_folder_plots, exist_ok=True)

    def save_market_plots(
        self,
        ticker: str,
        data: Dict[str, pd.Series],
        ma_window_vol: int = 20,
        ma_window_volum: int = 20,
        smooth_window: int = 10
    ) -> List[str]:
        """
        Creates and saves comprehensive market plots for a single ticker.
        
        Args:
            ticker: Ticker symbol
            data: Dictionary containing Series for OHLCV, volatility, etc.
            ma_window_vol: Moving average window for volatility
            ma_window_volum: Moving average window for volume
            smooth_window: Smoothing window for curves
        
        Returns:
            List of saved plot file paths
        """
        saved_plots = []

        # Smooth data
        smooth = lambda s: s.rolling(smooth_window, min_periods=1).mean()
        
        smoothed_data = {
            'close': smooth(data['close']),
            'normalized_close': smooth(data['normalized_close']),
            'open': smooth(data['open']),
            'high': smooth(data['high']),
            'low': smooth(data['low']),
            'volume_ma': data['volume'].rolling(ma_window_volum, min_periods=1).mean(),
            'volatility_ma': data['volatility'].rolling(ma_window_vol, min_periods=1).mean(),
            'sharpe_ma': data['sharpe'].rolling(ma_window_vol, min_periods=1).mean()
        }

        # Figure 1: Linear Plots
        fig1, axes = plt.subplots(5, 1, figsize=(16, 28), sharex=True)
        fig1.suptitle(f"{ticker} Market Data Visualization", fontsize=18)

        # OHLC
        axes[0].plot(smoothed_data['open'], label='Open', color='blue', alpha=0.7)
        axes[0].plot(smoothed_data['high'], label='High', color='green', alpha=0.7)
        axes[0].plot(smoothed_data['low'], label='Low', color='red', alpha=0.7)
        axes[0].plot(smoothed_data['close'], label='Close', color='black', alpha=0.8)
        axes[0].fill_between(
            smoothed_data['high'].index,
            smoothed_data['low'],
            smoothed_data['high'],
            color='gray',
            alpha=0.1
        )
        axes[0].set_title("OHLC Prices (Smoothed)")
        axes[0].legend()
        axes[0].grid(True)

        # Normalized Close
        axes[1].plot(smoothed_data['normalized_close'], color='purple')
        axes[1].set_title("Normalized Close (Smoothed)")
        axes[1].grid(True)

        # Volume
        axes[2].bar(data['volume'].index, data['volume'], color='lightblue', alpha=0.5, label='Volume')
        axes[2].plot(smoothed_data['volume_ma'], color='blue', label=f'{ma_window_volum}-day MA')
        axes[2].set_title("Volume with Moving Average")
        axes[2].legend()
        axes[2].grid(True)

        # Volatility
        axes[3].plot(data['volatility'], color='orange', alpha=0.4, label='Volatility')
        axes[3].plot(smoothed_data['volatility_ma'], color='red', label=f'{ma_window_vol}-day MA')
        axes[3].set_title("Volatility (Smoothed MA)")
        axes[3].legend()
        axes[3].grid(True)

        # Sharpe
        axes[4].plot(data['sharpe'], color='green', alpha=0.5, label='Sharpe')
        axes[4].plot(smoothed_data['sharpe_ma'], color='darkgreen', label=f'{ma_window_vol}-day MA')
        axes[4].set_title("10-day Sharpe Ratio (Smoothed)")
        axes[4].legend()
        axes[4].grid(True)

        # Save Figure 1
        fig1.tight_layout(rect=[0, 0, 1, 0.96])
        fig_path1 = os.path.join(self.base_folder_plots, f"{ticker}_linear_plots.png")
        fig1.savefig(fig_path1)
        plt.close(fig1)
        saved_plots.append(fig_path1)

        # Figure 2: Log + High-Low Spread
        fig2, ax2 = plt.subplots(2, 1, figsize=(16,12), sharex=True)

        # Log evolution
        ax2[0].plot(smoothed_data['close'], label='Close', color='black')
        ax2[0].plot(smoothed_data['normalized_close'], label='Normalized Close', color='purple')
        ax2[0].set_yscale('log')
        ax2[0].set_title("Close & Normalized Close (Log Scale)")
        ax2[0].legend()
        ax2[0].grid(True)

        # High-Low spread
        ax2[1].plot(smoothed_data['high'] - smoothed_data['low'], color='gray')
        ax2[1].set_title("High-Low Spread (Smoothed)")
        ax2[1].grid(True)

        # Save Figure 2
        fig2.tight_layout()
        fig_path2 = os.path.join(self.base_folder_plots, f"{ticker}_log_hl_plots.png")
        fig2.savefig(fig_path2)
        plt.close(fig2)
        saved_plots.append(fig_path2)

        return saved_plots