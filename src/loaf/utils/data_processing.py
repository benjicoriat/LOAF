"""
Utility functions for data processing and analysis.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict

def get_10day_intervals(dates_index: pd.DatetimeIndex) -> List[Tuple[str, str]]:
    """
    Splits dates into 10-trading-day intervals.

    Args:
        dates_index: DatetimeIndex of trading days

    Returns:
        List of (start_date, end_date) tuples for each 10-trading-day interval
    """
    intervals = []
    for i in range(0, len(dates_index), 10):
        start = dates_index[i]
        end = dates_index[min(i + 9, len(dates_index) - 1)]
        intervals.append((start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))
    return intervals

def aggregate_observations(
    data: Dict[str, pd.DataFrame],
    tickers: List[str],
    interval_days: int = 10
) -> pd.DataFrame:
    """
    Aggregates market data over intervals, stacking ticker metrics into 1D vectors.
    
    Args:
        data: Dictionary of DataFrames (open, close, etc.)
        tickers: List of ticker symbols
        interval_days: Number of trading days per interval
    
    Returns:
        DataFrame with columns = intervals, rows = ticker × metrics
    """
    all_dates = data['close'].index
    final_data = {}

    # Process intervals
    for i in range(0, len(all_dates), interval_days):
        start = all_dates[i]
        end = all_dates[min(i + interval_days - 1, len(all_dates) - 1)]
        period_label = f"{start.date()}_{end.date()}"
        period_vector = []

        for ticker in tickers:
            # Calculate metrics for period
            metrics = [
                data['open'].loc[start:end, ticker].iloc[0],     # First open
                data['close'].loc[start:end, ticker].iloc[-1],   # Last close
                data['high'].loc[start:end, ticker].max(),       # Period high
                data['low'].loc[start:end, ticker].min(),        # Period low
                data['volatility'].loc[start:end, ticker].mean(),# Avg volatility
                data['sharpe'].loc[start:end, ticker].mean(),    # Avg Sharpe
                data['volume'].loc[start:end, ticker].sum()      # Total volume
            ]
            period_vector.extend(metrics)

        final_data[period_label] = period_vector

    # Create index labels
    metric_names = ['Open', 'Close', 'High', 'Low', 'AvgVolatility', 'AvgSharpe', 'SumVolume']
    index_labels = [
        f"{ticker}_{metric}"
        for ticker in tickers
        for metric in metric_names
    ]

    return pd.DataFrame(final_data, index=index_labels)

def normalize_observation_vector(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes each row of observation matrix by its first column value.
    
    Args:
        df: DataFrame to normalize
    
    Returns:
        Normalized DataFrame
    """
    normalized_df = df.copy()
    first_col = df.iloc[:, 0]
    
    # Avoid divide-by-zero
    first_col_replaced = first_col.replace(0, pd.NA)
    normalized_df = df.div(first_col_replaced, axis=0)
    
    # Force first column to be 1.0
    normalized_df.iloc[:, 0] = 1.0
    
    return normalized_df