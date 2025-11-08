def setup_environment_and_api():
    import subprocess
    import sys
    import importlib

    # ---- Step 1: Package Installation ----
    required_packages = ["yfinance", "stable-baselines3"]
    for pkg in required_packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

    # ---- Step 2: API Key Setup (Commented for Security) ----
    api_key ="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"

        # ---- Step 3: Dynamic Imports ----
    import_list = [
        ("yfinance", "yf"),
        ("sb3", None),
    ]

    imported_modules = {}
    for module_name, alias in import_list:
        try:
            module = importlib.import_module(module_name)
            if alias:
                globals()[alias] = module
                imported_modules[module_name] = alias
            else:
                globals()[module_name] = module
                imported_modules[module_name] = module_name
        except Exception as e:
            imported_modules[module_name] = f"Failed to import ({e})"

    # ---- Step 4: Summary ----
    imported_summary = ", ".join([
        f"{name} as {alias}" if not alias.startswith("Failed") else alias
        for name, alias in imported_modules.items()
    ])

    summary = (
        "==========================================\n"
        "⚙️ ENVIRONMENT & API CONFIGURATION SUMMARY\n"
        "==========================================\n"
        f"Installed Packages : {', '.join(required_packages)}\n"
        f"Imported Libraries  : {imported_summary}\n"
        f"API Key Source      : {api_key}\n"
    )

    return summary, api_key

# Imports and pips
env_summary, api_key = setup_environment_and_api()
print(env_summary)

# ==========================================
# 📊 Market, Model & Data Pipeline Configuration Function
# ==========================================

def setup_market_and_models():

    # ---- Market Universe (25 tickers total) ----
    tickers = {
        "Equity Indices": ["SPY", "DIA", "QQQ", "IWM", "^FCHI", "^GDAXI", "^FTSE",
                           "^N225", "^STOXX50E", "^HSI"],
        "REITs": ["VNQ", "SCHH", "IYR"],
        "Commodities": ["GLD", "SLV", "USO", "UNG", "DBA", "PALL", "CPER"],
        "Cryptocurrencies": ["BTC-USD", "ETH-USD"],
        "Bonds/Defensive": ["TLT", "IEF", "SHY"]
    }

    # Flatten tickers for iteration
    all_tickers = [ticker for group in tickers.values() for ticker in group]

    # ---- Date Ranges for Data Pipeline ----
    date_ranges = {
        "Download": {"start": "2023-01-01", "end": "2023-03-31"},
        "Training": {"start": "2023-01-01", "end": "2023-01-31"},
        "Testing":  {"start": "2023-01-31", "end": "2023-02-31"}
    }

    # ---- Prompt for Homologous Terms ----
    prompt = ("For the following {ticker}, please generate a list of exactly 10 homologous terms, for example NVDA: ['Microprocessors', 'Nvidia', 'Jensen Huang', etc.], as you can see, only TICKER SPECIFIC INFO, nothing generic, the output should be strictly this and nothing more. the only output I want is the list, python style, NOTHING ELSE, don't answer anything else or even introduce it, just the list.")

    # ---- Model Layers ----
    model_layer_1 = ["PPO", "SAC", "TD3"]
    model_layer_2 = []
    model_layer_3 = []

    # ---- Build Extended Summary ----
    summary_lines = [
        "==========================================",
        "📈 MARKET & MODEL CONFIGURATION SUMMARY",
        "==========================================",
        f"Total Tickers: {len(all_tickers)}",
    ]

    for group_name, group_tickers in tickers.items():
        summary_lines.append(f"{group_name} ({len(group_tickers)}): {', '.join(group_tickers)}")

    summary_lines.extend([
        "",
        "Model Layers:",
        f"  Layer 1: {', '.join(model_layer_1) if model_layer_1 else 'None'}",
        f"  Layer 2: {', '.join(model_layer_2) if model_layer_2 else 'None'}",
        f"  Layer 3: {', '.join(model_layer_3) if model_layer_3 else 'None'}",
        "",
        "Date Ranges:",
        f"  Download: {date_ranges['Download']['start']} → {date_ranges['Download']['end']}",
        f"  Training: {date_ranges['Training']['start']} → {date_ranges['Training']['end']}",
        f"  Testing : {date_ranges['Testing']['start']} → {date_ranges['Testing']['end']}",
        "",
        "Prompt for Homologous Terms:",
        f"  {prompt}"
    ])

    summary = "\n".join(summary_lines)

    # ---- Return All Relevant Objects ----
    return summary, all_tickers, model_layer_1, model_layer_2, model_layer_3, date_ranges, prompt


# Configuration of constants
summary, all_tickers, layer1, layer2, layer3, date_ranges, prompt = setup_market_and_models()
print(summary)
# ==========================================
# 📂 Folder Structure Creation Function
# ==========================================

import os

def create_folder_structure(base_dir=".", layers=["layer_1", "layer_2", "layer_3"]):

    folder_structure = {}

    # ---- 1. Download folders ----
    download_subfolders = ["plots", "time_series", "stats"]
    download_paths = {}
    for sub in download_subfolders:
        path = os.path.join(base_dir, "download", sub)
        os.makedirs(path, exist_ok=True)
        download_paths[sub] = path
    folder_structure["download"] = download_paths

    # ---- 2. Layer folders ----
    for layer in layers:
        layer_subfolders = ["plots", "time_series", "stats"]
        layer_paths = {}
        for sub in layer_subfolders:
            path = os.path.join(base_dir, layer, sub)
            os.makedirs(path, exist_ok=True)
            layer_paths[sub] = path
        folder_structure[layer] = layer_paths

    # ---- 3. Final folders ----
    final_subfolders = ["plots", "time_series"]
    final_paths = {}
    for sub in final_subfolders:
        path = os.path.join(base_dir, "Final", sub)
        os.makedirs(path, exist_ok=True)
        final_paths[sub] = path
    folder_structure["Final"] = final_paths

    return folder_structure

# Directory Organisation
folders = create_folder_structure(base_dir="./my_project")
print("Folder structure created successfully:")
for k, v in folders.items():
    print(f"{k}: {v}")
    import pandas as pd
import numpy as np
import yfinance as yf
import os
import warnings

def compute_daily_volatility(close_series, window=20):
    """Compute rolling daily volatility (log returns)"""
    log_returns = np.log(close_series / close_series.shift(1))
    vol = log_returns.rolling(window=window).std().fillna(method='bfill')
    return vol

def compute_sharpe_ratio(close_series, window=10):
    """Compute rolling 10-day Sharpe ratio proxy"""
    log_returns = np.log(close_series / close_series.shift(1))
    mean_return = log_returns.rolling(window=window).mean()
    std_return = log_returns.rolling(window=window).std()
    sharpe = mean_return / std_return
    sharpe = sharpe.fillna(method='bfill')
    return sharpe

def download_and_clean_data(all_tickers, start_date_download, end_date_download, download_folder="download"):
    """
    Downloads Open, High, Low, Close, Volume for a list of tickers,
    cleans, normalizes, computes volatility and Sharpe ratio, and saves CSVs.
    """
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=DeprecationWarning)

    os.makedirs(download_folder, exist_ok=True)

import pandas as pd
import numpy as np
import yfinance as yf
import os
import warnings

def compute_daily_volatility(close_series, window=20):
    log_returns = np.log(close_series / close_series.shift(1))
    vol = log_returns.rolling(window=window).std().fillna(method='bfill')
    return vol

def compute_sharpe_ratio(close_series, window=10):
    log_returns = np.log(close_series / close_series.shift(1))
    mean_return = log_returns.rolling(window=window).mean()
    std_return = log_returns.rolling(window=window).std()
    sharpe = mean_return / std_return
    sharpe = sharpe.fillna(method='bfill')
    return sharpe

def download_and_clean_data(all_tickers, start_date_download, end_date_download, download_folder="download"):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=DeprecationWarning)

    os.makedirs(download_folder, exist_ok=True)

    raw_data = {}
    for ticker in all_tickers:
        try:
            print(f"Downloading {ticker}...")
            data = yf.download(ticker, start=start_date_download, end=end_date_download, progress=False, auto_adjust=False)
            if data.empty:
                print(f"⚠️ No data for {ticker}, skipping...")
                continue

            # Only weekdays
            data = data[data.index.dayofweek < 5]

            # Ensure all columns are Series
            for col in ['Open','High','Low','Close','Volume']:
                if isinstance(data[col], pd.DataFrame):
                    data[col] = data[col].squeeze()

            raw_data[ticker] = data[['Open','High','Low','Close','Volume']]

        except Exception as e:
            print(f"⚠️ Failed to download {ticker}: {e}")
            continue

    # Align all tickers to common business day index
    common_index = pd.date_range(start=start_date_download, end=end_date_download, freq='B')

    def build_df(col_name):
        return pd.DataFrame({t: d[col_name].squeeze() for t, d in raw_data.items()}, index=common_index).interpolate().bfill()

    open_df = build_df('Open')
    high_df = build_df('High')
    low_df = build_df('Low')
    close_df = build_df('Close')
    volume_df = build_df('Volume').fillna(0)

    # Normalized close
    normalized_close_df = close_df / close_df.iloc[0]

    # Volatility
    vol_df = close_df.apply(compute_daily_volatility)

    # Sharpe
    sharpe_df = close_df.apply(compute_sharpe_ratio)

    # Save CSVs
    paths = {}
    paths['open'] = os.path.join(download_folder, "open.csv")
    paths['high'] = os.path.join(download_folder, "high.csv")
    paths['low'] = os.path.join(download_folder, "low.csv")
    paths['close'] = os.path.join(download_folder, "close.csv")
    paths['normalized_close'] = os.path.join(download_folder, "normalized_close.csv")
    paths['volume'] = os.path.join(download_folder, "volume.csv")
    paths['volatility'] = os.path.join(download_folder, "volatility.csv")
    paths['sharpe'] = os.path.join(download_folder, "sharpe.csv")

    open_df.to_csv(paths['open'])
    high_df.to_csv(paths['high'])
    low_df.to_csv(paths['low'])
    close_df.to_csv(paths['close'])
    normalized_close_df.to_csv(paths['normalized_close'])
    volume_df.to_csv(paths['volume'])
    vol_df.to_csv(paths['volatility'])
    sharpe_df.to_csv(paths['sharpe'])

    print("\n✅ Data download, cleaning, and saving complete!")
    for k, v in paths.items():
        print(f"{k}: {v}")

    return paths

# ---- Usage ----
paths = download_and_clean_data(
    all_tickers=all_tickers,
    start_date_download=date_ranges['Download']['start'],
    end_date_download=date_ranges['Download']['end'],
    download_folder="./my_project/download/time_series"
)
import os
import pandas as pd
import requests
import time
import json
from ast import literal_eval

def fetch_homologous_terms(all_tickers, prompt, api_key, download_folder="download", pause=0.5):
    """
    Fetch homologous terms for each ticker using Groq API and save to CSV.
    Each element of the list is stored in a separate column, with proper headers.
    """
    os.makedirs(download_folder, exist_ok=True)
    results = []

    max_terms = 0  # Track the maximum number of terms returned

    for ticker in all_tickers:
        current_prompt = prompt.replace("{ticker}", ticker)
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [
                        {"role": "user", "content": current_prompt}
                    ],
                    "max_tokens": 100,
                    "temperature": 0.7
                }
            )
            response.raise_for_status()
            data = response.json()

            # Extract text output
            output_text = data['choices'][0]['message']['content'].strip()

            # Parse as list
            try:
                homologous_list = json.loads(output_text)
                if not isinstance(homologous_list, list):
                    homologous_list = [output_text]
            except json.JSONDecodeError:
                try:
                    homologous_list = literal_eval(output_text)
                    if not isinstance(homologous_list, list):
                        homologous_list = [output_text]
                except (ValueError, SyntaxError):
                    homologous_list = [output_text]

            results.append([ticker] + homologous_list)
            max_terms = max(max_terms, len(homologous_list))
            print(f"✓ Processed {ticker}")

        except Exception as e:
            print(f"⚠️ Failed for {ticker}: {e}")
            results.append([ticker])
            max_terms = max(max_terms, 0)

        time.sleep(pause)

    # Build headers dynamically
    headers = ["Ticker"] + [f"Term{i+1}" for i in range(max_terms)]

    # Pad rows so all have the same length
    padded_results = [row + [""]*(len(headers)-len(row)) for row in results]

    df = pd.DataFrame(padded_results, columns=headers)

    # Save CSV
    csv_path = os.path.join(download_folder, "homologous.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Homologous terms saved to {csv_path}")
    return df

df_homologous = fetch_homologous_terms(
     all_tickers=all_tickers,
     prompt=prompt,
     api_key=api_key,
     download_folder="./my_project/download"
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def save_market_plots(
    ticker,
    base_folder_ts="./my_project/download/time_series/",
    base_folder_plots="./my_project/download/plots/",
    ma_window_vol=20,
    ma_window_volum=20,
    smooth_window=10
):
    """
    Saves comprehensive market plots for a single ticker.

    Reads CSVs from ./my_project/download/time_series/
    Saves plots into ./my_project/download/plots/

    Args:
        ticker (str): Ticker symbol to plot.
        base_folder_ts (str): Folder containing CSVs.
        base_folder_plots (str): Folder to save plots.
        ma_window_vol (int): Moving average window for volatility.
        ma_window_volum (int): Moving average window for volume.
        smooth_window (int): Smoothing window for curves.
    """

    # Ensure folder exists
    os.makedirs(base_folder_plots, exist_ok=True)

    # ---- Load data ----
    def load_csv(name):
        path = os.path.join(base_folder_ts, f"{name}.csv")
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df[ticker]

    close = load_csv("close")
    normalized_close = load_csv("normalized_close")
    open_ = load_csv("open")
    high = load_csv("high")
    low = load_csv("low")
    volume = load_csv("volume")
    volatility = load_csv("volatility")
    sharpe = load_csv("sharpe")

    # ---- Smooth data ----
    def smooth(series, window):
        return series.rolling(window, min_periods=1).mean()

    close_smooth = smooth(close, smooth_window)
    normalized_close_smooth = smooth(normalized_close, smooth_window)
    open_smooth = smooth(open_, smooth_window)
    high_smooth = smooth(high, smooth_window)
    low_smooth = smooth(low, smooth_window)
    volume_ma = volume.rolling(ma_window_volum, min_periods=1).mean()
    volatility_ma = volatility.rolling(ma_window_vol, min_periods=1).mean()
    sharpe_ma = sharpe.rolling(ma_window_vol, min_periods=1).mean()

    # ---- Figure 1: Linear Plots ----
    fig, axes = plt.subplots(5, 1, figsize=(16, 28), sharex=True)
    fig.suptitle(f"{ticker} Market Data Visualization", fontsize=18)

    # OHLC
    axes[0].plot(open_smooth, label='Open', color='blue', alpha=0.7)
    axes[0].plot(high_smooth, label='High', color='green', alpha=0.7)
    axes[0].plot(low_smooth, label='Low', color='red', alpha=0.7)
    axes[0].plot(close_smooth, label='Close', color='black', alpha=0.8)
    axes[0].fill_between(high_smooth.index, low_smooth, high_smooth, color='gray', alpha=0.1)
    axes[0].set_title("OHLC Prices (Smoothed)")
    axes[0].legend()
    axes[0].grid(True)

    # Normalized Close
    axes[1].plot(normalized_close_smooth, color='purple')
    axes[1].set_title("Normalized Close (Smoothed)")
    axes[1].grid(True)

    # Volume
    axes[2].bar(volume.index, volume, color='lightblue', alpha=0.5, label='Volume')
    axes[2].plot(volume_ma, color='blue', label=f'{ma_window_volum}-day MA')
    axes[2].set_title("Volume with Moving Average")
    axes[2].legend()
    axes[2].grid(True)

    # Volatility
    axes[3].plot(volatility, color='orange', alpha=0.4, label='Volatility')
    axes[3].plot(volatility_ma, color='red', label=f'{ma_window_vol}-day MA')
    axes[3].set_title("Volatility (Smoothed MA)")
    axes[3].legend()
    axes[3].grid(True)

    # Sharpe
    axes[4].plot(sharpe, color='green', alpha=0.5, label='Sharpe')
    axes[4].plot(sharpe_ma, color='darkgreen', label=f'{ma_window_vol}-day MA')
    axes[4].set_title("10-day Sharpe Ratio (Smoothed)")
    axes[4].legend()
    axes[4].grid(True)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path1 = os.path.join(base_folder_plots, f"{ticker}_linear_plots.png")
    fig.savefig(fig_path1)
    plt.close(fig)

    # ---- Figure 2: Log + High-Low Spread ----
    fig2, ax2 = plt.subplots(2, 1, figsize=(16,12), sharex=True)

    # Log evolution
    ax2[0].plot(close_smooth, label='Close', color='black')
    ax2[0].plot(normalized_close_smooth, label='Normalized Close', color='purple')
    ax2[0].set_yscale('log')
    ax2[0].set_title("Close & Normalized Close (Log Scale)")
    ax2[0].legend()
    ax2[0].grid(True)

    # High-Low spread
    ax2[1].plot(high_smooth - low_smooth, color='gray')
    ax2[1].set_title("High-Low Spread (Smoothed)")
    ax2[1].grid(True)

    fig2.tight_layout()
    fig_path2 = os.path.join(base_folder_plots, f"{ticker}_log_hl_plots.png")
    fig2.savefig(fig_path2)
    plt.close(fig2)

    print(f"✅ Saved plots for {ticker}:\n  {fig_path1}\n  {fig_path2}")


# ---- Run for all tickers ----
for ticker in all_tickers:
    save_market_plots(
        ticker,
        base_folder_ts="./my_project/download/time_series/",
        base_folder_plots="./my_project/download/plots/",
        ma_window_vol=20,
        ma_window_volum=30,
        smooth_window=10
    )

import pandas as pd

def get_10day_intervals(csv_file):
    """
    Splits the CSV dates into 10-trading-day intervals.

    Args:
        csv_file (str): Path to CSV (assumes index is date).

    Returns:
        list of tuples: [(start_date, end_date), ...] for each 10-trading-day interval
    """
    # Load the CSV
    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)

    # Get all dates (business days)
    all_dates = df.index
    intervals = []

    # Step through in chunks of 10
    for i in range(0, len(all_dates), 10):
        start = all_dates[i]
        # Ensure we don't go out of bounds
        end = all_dates[min(i + 9, len(all_dates) - 1)]
        intervals.append((start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))

    return intervals
close_csv = "./my_project/download/time_series/close.csv"
intervals_10day = get_10day_intervals(close_csv)
print(intervals_10day[:5])  # print first 5 intervals

import pandas as pd
import os

def aggregate_observations(all_tickers, download_folder="./my_project/download/",
                           output_folder="./my_project/layer_1/", interval_days=10):
    """
    Aggregates market data over intervals, stacking ticker metrics into 1D vectors per period.
    Each column is a 10-trading-day interval, each row = ticker × metrics.
    Metrics: open (first), close (last), high (max), low (min), avg volatility, avg sharpe, sum volume.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Load CSVs
    open_df = pd.read_csv(f"{download_folder}/open.csv", index_col=0, parse_dates=True)
    close_df = pd.read_csv(f"{download_folder}/close.csv", index_col=0, parse_dates=True)
    high_df = pd.read_csv(f"{download_folder}/high.csv", index_col=0, parse_dates=True)
    low_df = pd.read_csv(f"{download_folder}/low.csv", index_col=0, parse_dates=True)
    volume_df = pd.read_csv(f"{download_folder}/volume.csv", index_col=0, parse_dates=True)
    volatility_df = pd.read_csv(f"{download_folder}/volatility.csv", index_col=0, parse_dates=True)
    sharpe_df = pd.read_csv(f"{download_folder}/sharpe.csv", index_col=0, parse_dates=True)

    all_dates = close_df.index
    n_tickers = len(all_tickers)
    intervals = []

    for i in range(0, len(all_dates), interval_days):
        start = all_dates[i]
        end = all_dates[min(i + interval_days - 1, len(all_dates) - 1)]
        intervals.append((start, end))

    # Prepare final DataFrame: rows = n_tickers * n_metrics, columns = intervals
    final_data = {}

    for start, end in intervals:
        period_label = f"{start.date()}_{end.date()}"
        period_vector = []

        for ticker in all_tickers:
            o = open_df.loc[start:end, ticker].iloc[0]
            c = close_df.loc[start:end, ticker].iloc[-1]
            h = high_df.loc[start:end, ticker].max()
            l = low_df.loc[start:end, ticker].min()
            avg_vol = volatility_df.loc[start:end, ticker].mean()
            avg_sharpe = sharpe_df.loc[start:end, ticker].mean()
            sum_vol = volume_df.loc[start:end, ticker].sum()

            # Append metrics in order
            period_vector.extend([o, c, h, l, avg_vol, avg_sharpe, sum_vol])

        final_data[period_label] = period_vector

    # Index labels = ticker × metrics
    metric_names = ['Open', 'Close', 'High', 'Low', 'AvgVolatility', 'AvgSharpe', 'SumVolume']
    index_labels = []
    for ticker in all_tickers:
        for metric in metric_names:
            index_labels.append(f"{ticker}_{metric}")

    final_df = pd.DataFrame(final_data, index=index_labels)

    # Save CSV
    output_path = os.path.join(output_folder, "observation_quant.csv")
    final_df.to_csv(output_path)
    print(f"✅ Observation quant CSV saved to {output_path}")
    return final_df
df_obs = aggregate_observations(
    all_tickers=all_tickers,
    download_folder="./my_project/download/time_series",
    output_folder="./my_project/layer_1/",
    interval_days=10
)

import pandas as pd
import os

def normalize_observation_vector(
    input_csv="./my_project/layer_1/observation_quant.csv",
    output_csv="./my_project/layer_1/normalized_obs_vect.csv"
):
    """
    Normalizes each row of observation_quant.csv by its first column value.
    The first column becomes all 1’s. Handles division by zero safely.

    Args:
        input_csv (str): Path to the input observation_quant.csv.
        output_csv (str): Path to save normalized file.
    """

    # Ensure folder exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Load data
    df = pd.read_csv(input_csv, index_col=0)

    # Copy to avoid overwriting
    normalized_df = df.copy()

    # Normalize each row by its first column value
    first_col = df.iloc[:, 0]

    # Avoid divide-by-zero issues
    first_col_replaced = first_col.replace(0, pd.NA)
    normalized_df = df.div(first_col_replaced, axis=0)

    # Ensure first column = 1 explicitly
    normalized_df.iloc[:, 0] = 1.0

    # Save result
    normalized_df.to_csv(output_csv)

    print(f"✅ Normalized observation vector saved to: {output_csv}")
    return normalized_df


# ---- Usage ----
normalized_obs_vect = normalize_observation_vector(
    input_csv="./my_project/layer_1/observation_quant.csv",
    output_csv="./my_project/layer_1/normalized_obs_vect.csv"
)

import os
import pandas as pd
from urllib.parse import quote_plus
from datetime import datetime

def get_10day_intervals(close_csv):
    """Return list of (start_date, end_date) tuples every 10 trading days."""
    df = pd.read_csv(close_csv, index_col=0, parse_dates=True)
    dates = df.index.sort_values()
    intervals = []
    for i in range(0, len(dates), 10):
        start = dates[i]
        end = dates[min(i + 9, len(dates) - 1)]
        intervals.append((start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))
    return intervals

def generate_ticker_links(
    homologous_csv="./my_project/download/homologous.csv",
    close_csv="./my_project/download/close.csv",
    output_folder="./my_project/layer_1/links/"
):
    os.makedirs(output_folder, exist_ok=True)

    # --- Load homologous file ---
    df_homo = pd.read_csv(homologous_csv)
    df_homo.columns = [c.strip() for c in df_homo.columns]

    # Identify columns
    ticker_col = "Ticker"
    term_cols = [c for c in df_homo.columns if c.lower().startswith("term")]

    # --- Load date intervals ---
    intervals = get_10day_intervals(close_csv)

    print(f"✅ Found {len(intervals)} intervals and {len(df_homo)} tickers.")

    # --- Iterate tickers ---
    for _, row in df_homo.iterrows():
        ticker = row[ticker_col]
        terms = [str(row[c]) for c in term_cols if pd.notna(row[c])]

        # Prepare dataframe for this ticker
        df_links = pd.DataFrame(index=[start for start, _ in intervals], columns=terms)

        for (start, end) in intervals:
            for term in terms:
                query = quote_plus(term)
                link = f"https://www.google.com/search?q={query}&tbs=cdr:1,cd_min:{start},cd_max:{end}"
                df_links.loc[start, term] = link

        # Save CSV
        out_path = os.path.join(output_folder, f"{ticker}_links.csv")
        df_links.to_csv(out_path)
        print(f"💾 Saved {out_path}")

# ---- Example usage ----
generate_ticker_links(
    homologous_csv="./my_project/download/homologous.csv",
    close_csv="./my_project/download/time_series/close.csv",
    output_folder="./my_project/layer_1/links/"
)
# Note: required packages should be installed in the environment; removed inline pip install
import os
import gc
import time
import glob
import torch
import pandas as pd
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from urllib.parse import urlparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json

# -------------------------------
# Config
# -------------------------------
INPUT_FOLDER = "./my_project/layer_1/links/"
OUTPUT_FOLDER = "./my_project/layer_1/links_nlp/"
ERROR_LOG = os.path.join(OUTPUT_FOLDER, "error_log.txt")
CHECKPOINT_FILE = os.path.join(OUTPUT_FOLDER, "checkpoint.json")
SCRAPED_CACHE_FILE = os.path.join(OUTPUT_FOLDER, "scraped_cache.json")
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
REQUEST_TIMEOUT = 10
REQUEST_DELAY = 1.0  # Delay between requests (seconds)
MAX_WORDS = 500  # Increased from 200 to get more content
DEFAULT_SCORE = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8  # Process sentiment in batches

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -------------------------------
# Initialize FinBERT model
# -------------------------------
print(f"Loading FinBERT model on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
model.to(DEVICE)
model.eval()

# CRITICAL FIX: Get and verify actual label mapping
id2label = model.config.id2label
print(f"FinBERT label mapping: {id2label}")

# Determine correct indices based on actual model config
label_to_idx = {v.lower(): k for k, v in id2label.items()}
POS_IDX = label_to_idx.get('positive', 0)
NEG_IDX = label_to_idx.get('negative', 1)
NEUTRAL_IDX = label_to_idx.get('neutral', 2)

print(f"Using indices: positive={POS_IDX}, negative={NEG_IDX}, neutral={NEUTRAL_IDX}")

# -------------------------------
# Utilities
# -------------------------------
def is_valid_url(url):
    """Validate URL format and scheme."""
    if not isinstance(url, str) or not url.startswith("http"):
        return False
    try:
        parsed = urlparse(url)
        return parsed.scheme in ['http', 'https'] and bool(parsed.netloc)
    except:
        return False

def load_scraped_cache():
    """Load cache of previously scraped URLs."""
    if os.path.exists(SCRAPED_CACHE_FILE):
        try:
            with open(SCRAPED_CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_scraped_cache(cache):
    """Save cache of scraped URLs."""
    with open(SCRAPED_CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2)

def scrape_text(url, session, cache):
    """
    Scrape first MAX_WORDS of text from a URL.
    Uses cache to avoid re-scraping same URLs.
    Returns string (empty string if failed).
    """
    # Check cache first
    if url in cache:
        return cache[url]
    
    try:
        headers = {"User-Agent": USER_AGENT}
        r = session.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # Remove script/style/nav/footer elements
        for tag in soup(["script", "style", "noscript", "nav", "header", "footer"]):
            tag.extract()

        # Try to find main content areas first
        main_content = soup.find(['main', 'article', 'div'], class_=lambda x: x and any(
            word in str(x).lower() for word in ['content', 'main', 'article', 'body']
        ))
        
        if main_content:
            text = main_content.get_text(separator=" ", strip=True)
        else:
            text = soup.get_text(separator=" ", strip=True)
        
        words = text.split()
        scraped = " ".join(words[:MAX_WORDS])
        
        # Cache the result
        cache[url] = scraped
        
        time.sleep(REQUEST_DELAY)  # Rate limiting
        return scraped
    except Exception as e:
        with open(ERROR_LOG, "a", encoding='utf-8') as f:
            f.write(f"URL scraping failed: {url} | Error: {str(e)}\n")
        cache[url] = ""  # Cache failures too
        return ""

def compute_sentiment_batch(texts):
    """
    Compute FinBERT sentiment scores for a batch of texts.
    Returns list of floats normalized to [0,1].
    
    FIXED: Now uses correct label indices from model config.
    """
    """
    Compute sentiment probabilities for a batch of texts.

    Returns a list of dicts for each input text in order with keys:
      {'pos': float, 'neg': float, 'neu': float, 'net': float}
    Empty or failed texts will have pos/neg/neu = DEFAULT_SCORE, net = 0.
    """
    results = []
    if not texts:
        return results

    try:
        valid_texts = [(i, t) for i, t in enumerate(texts) if isinstance(t, str) and t.strip()]
        indices, text_list = (zip(*valid_texts) if valid_texts else ([], []))

        if valid_texts:
            # Tokenize batch
            inputs = tokenizer(
                list(text_list),
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()

            # For each valid text, capture prob vector
            prob_map = {}
            for idx, p in zip(indices, probs):
                # Ensure ordering: POS_IDX, NEG_IDX, NEUTRAL_IDX
                p_pos = float(p[POS_IDX])
                p_neg = float(p[NEG_IDX])
                p_neu = float(p[NEUTRAL_IDX])
                net = p_pos - p_neg
                prob_map[idx] = {"pos": p_pos, "neg": p_neg, "neu": p_neu, "net": net}
        else:
            prob_map = {}

        # Build results keeping original order
        for i, txt in enumerate(texts):
            if i in prob_map:
                results.append(prob_map[i])
            else:
                # Blank or failed text
                results.append({"pos": DEFAULT_SCORE, "neg": DEFAULT_SCORE, "neu": DEFAULT_SCORE, "net": 0.0})

    except Exception as e:
        with open(ERROR_LOG, "a", encoding="utf-8") as f:
            f.write(f"Batch sentiment computation failed | Error: {str(e)}\n")
        # On failure, return default dicts
        results = [{"pos": DEFAULT_SCORE, "neg": DEFAULT_SCORE, "neu": DEFAULT_SCORE, "net": 0.0} for _ in texts]

    return results

def load_checkpoint():
    """Load processing checkpoint."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_checkpoint(checkpoint):
    """Save processing checkpoint."""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f)

# -------------------------------
# Main Processing
# -------------------------------
def process_all_csv(input_folder=INPUT_FOLDER, output_folder=OUTPUT_FOLDER):
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    print(f"Found {len(csv_files)} CSV files.")
    
    if not csv_files:
        print("No CSV files found!")
        return
    
    checkpoint = load_checkpoint()
    scraped_cache = load_scraped_cache()
    
    # Create session with retry logic
    session = requests.Session()
    retries = Retry(
        total=3, 
        backoff_factor=0.5, 
        status_forcelist=[429, 500, 502, 503, 504]
    )
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))

    for csv_file in tqdm(csv_files, desc="📁 Processing Files", position=0, leave=True):
        basename = os.path.basename(csv_file)
        
        # Skip if already processed
        if checkpoint.get(basename) == 'completed':
            tqdm.write(f"⏭️  Skipping already processed: {basename}")
            continue
        
        try:
            # Load CSV
            df = pd.read_csv(csv_file, index_col=0)
            
            if df.empty:
                tqdm.write(f"⚠️  Empty CSV: {basename}")
                continue
            
            output_df = pd.DataFrame(index=df.index, columns=df.columns)
            
            # Collect all URLs for processing
            url_batch = []
            url_positions = []
            
            total_cells = len(df.index) * len(df.columns)
            with tqdm(total=total_cells, desc=f"  📊 Collecting URLs", position=1, leave=False) as pbar:
                for row_idx in df.index:
                    for col in df.columns:
                        try:
                            url = df.at[row_idx, col]
                            if is_valid_url(url):
                                first_url = url.split()[0]  # Take first URL if multiple
                                url_batch.append(first_url)
                                url_positions.append((row_idx, col))
                            else:
                                output_df.at[row_idx, col] = DEFAULT_SCORE
                        except:
                            output_df.at[row_idx, col] = DEFAULT_SCORE
                        pbar.update(1)
            
            tqdm.write(f"  Found {len(url_batch)} valid URLs in {basename}")
            
            # Scrape all URLs (with caching)
            texts = []
            for url in tqdm(url_batch, desc=f"  🌐 Scraping URLs", position=1, leave=False):
                text = scrape_text(url, session, scraped_cache)
                texts.append(text)

            # Save cache periodically
            save_scraped_cache(scraped_cache)

            # Compute sentiments in batches (returns list of dicts with pos/neg/neu/net)
            all_scores = []
            num_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
            with tqdm(total=num_batches, desc=f"  🧠 Computing Sentiment", position=1, leave=False) as pbar:
                for i in range(0, len(texts), BATCH_SIZE):
                    batch = texts[i:i+BATCH_SIZE]
                    scores = compute_sentiment_batch(batch)
                    all_scores.extend(scores)
                    pbar.update(1)

            # Fill output dataframe (legacy cell-level scores: store net score)
            for (row_idx, col), score in zip(url_positions, all_scores):
                # store net score for backward compatibility
                output_df.at[row_idx, col] = score.get('net', 0.0) if isinstance(score, dict) else score

            # Aggregate per-period features for this ticker
            # period -> list of indices into url_batch/texts/all_scores
            period_to_indices = {}
            for idx, (row_idx, col) in enumerate(url_positions):
                period_to_indices.setdefault(row_idx, []).append(idx)

            feature_rows = []
            for period in df.index:
                idxs = period_to_indices.get(period, [])
                # select only indices with non-empty scraped text
                valid_idxs = [i for i in idxs if i < len(texts) and isinstance(texts[i], str) and texts[i].strip()]

                if not valid_idxs:
                    # no articles -> default values
                    p_pos = DEFAULT_SCORE
                    p_neg = DEFAULT_SCORE
                    p_neu = DEFAULT_SCORE
                    net_mean = p_pos - p_neg
                    N = 0
                    sigma = 0.0
                else:
                    pos_vals = [all_scores[i]['pos'] for i in valid_idxs]
                    neg_vals = [all_scores[i]['neg'] for i in valid_idxs]
                    neu_vals = [all_scores[i]['neu'] for i in valid_idxs]
                    net_vals = [all_scores[i]['net'] for i in valid_idxs]

                    p_pos = float(sum(pos_vals) / len(pos_vals))
                    p_neg = float(sum(neg_vals) / len(neg_vals))
                    p_neu = float(sum(neu_vals) / len(neu_vals))
                    net_mean = float(sum(net_vals) / len(net_vals))
                    N = len(valid_idxs)
                    sigma = float(pd.Series(net_vals).std(ddof=0)) if len(net_vals) > 1 else 0.0

                feature_rows.append({
                    'period': period,
                    'bar_s': net_mean,
                    'p_pos': p_pos,
                    'p_neg': p_neg,
                    'p_neu': p_neu,
                    'N': N,
                    'sigma': sigma
                })

            ticker_features = pd.DataFrame(feature_rows).set_index('period')

            # Save per-ticker features
            agg_dir = os.path.join(output_folder, 'aggregated')
            os.makedirs(agg_dir, exist_ok=True)
            agg_path = os.path.join(agg_dir, f"{os.path.splitext(basename)[0]}_sentiment_features.csv")
            ticker_features.to_csv(agg_path, index=True)
            tqdm.write(f"💾 Saved per-ticker features: {agg_path}")

            # Save legacy processed CSV (net scores per term)
            output_path = os.path.join(output_folder, basename)
            output_df.to_csv(output_path, index=True)
            tqdm.write(f"✅ Saved: {output_path}")
            
            # Update checkpoint
            checkpoint[basename] = 'completed'
            save_checkpoint(checkpoint)

        except Exception as e_file:
            with open(ERROR_LOG, "a", encoding='utf-8') as f:
                f.write(f"File processing failed: {csv_file} | Error={str(e_file)}\n")
            tqdm.write(f"❌ Error processing {basename}: {str(e_file)}")

        # Memory management
        del df, output_df
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        time.sleep(0.5)

    # Final cache save
    save_scraped_cache(scraped_cache)
    
    print("\n🎉 All files processed!")
    print(f"📦 Total URLs cached: {len(scraped_cache)}")
    
    # Clean up checkpoint file
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":

  
    process_all_csv()

    import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
from scipy.stats import norm

# -----------------------------
# Black-Scholes Pricing Function
# -----------------------------
def black_scholes_price(spot, strike, time_to_exp, vol, r, option_type="call"):
    """
    Computes the Black-Scholes option price.
    Args:
        spot (float): Spot price (S)
        strike (float): Strike price (K)
        time_to_exp (float): Time to expiration in years
        vol (float): Volatility (sigma)
        r (float): Risk-free rate
        option_type (str): "call" or "put"
    Returns:
        float: option premium
    """
    if time_to_exp <= 0 or vol <= 0:
        return max(0.0, (spot - strike) if option_type=="call" else (strike - spot))

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

# -----------------------------
# Reward Pipeline Function
# -----------------------------
def options_reward_pipeline(date_t, action_vector, all_tickers, num_days, price_data, risk_free_rate=0.05):
    """
    Computes the reward for an options allocation on a given date.
    """
    # -----------------------------
    # Data & Action Validation
    # -----------------------------
    action_vector = np.array(action_vector, dtype=float)
    if not np.isclose(action_vector.sum(), 1.0, atol=1e-4):
        raise ValueError(f"Action vector sum != 1: {action_vector.sum()}")
    if np.any(np.abs(action_vector) >= 1.0):
        raise ValueError("All action_vector elements must be < 1 in absolute value")

    # Ensure date exists
    if date_t not in price_data.index:
        raise KeyError(f"Target date {date_t} not in price_data")
    date_idx = price_data.index.get_loc(date_t)
    if date_idx == 0:
        raise ValueError("Cannot price options on first available date")

    # -----------------------------
    # Precompute volatility for all tickers
    # -----------------------------
    vol_cache = {}
    hist_window = 20  # days for volatility
    for ticker in all_tickers:
        try:
            returns = np.log(price_data[ticker].pct_change() + 1)
            vol_cache[ticker] = returns.rolling(hist_window).std().iloc[date_idx - 1]
            if np.isnan(vol_cache[ticker]):
                vol_cache[ticker] = 0.2  # fallback default volatility
        except Exception:
            vol_cache[ticker] = 0.2

    # -----------------------------
    # Iterate tickers
    # -----------------------------
    payouts = []
    num_wins = 0
    num_loss = 0
    num_partial_loss = 0

    for i, ticker in enumerate(tqdm(all_tickers, desc=f"Processing tickers for {date_t}")):
        alloc = action_vector[i]
        option_type = "call" if alloc >= 0 else "put"
        weight = abs(alloc)

        try:
            spot_prev = price_data[ticker].iloc[date_idx - 1]
            spot_t = price_data[ticker].iloc[date_idx]
            vol = vol_cache[ticker]
            time_to_exp = num_days / 252.0  # convert trading days to years

            # Price the option
            premium = black_scholes_price(spot_prev, spot_prev, time_to_exp, vol, risk_free_rate, option_type)

            # Compute payout at expiry
            if option_type == "call":
                payout_raw = max(0, spot_t - spot_prev) - premium
            else:
                payout_raw = max(0, spot_prev - spot_t) - premium

            payout_weighted = payout_raw * weight
            payouts.append(payout_weighted)

            # Count metrics
            if payout_weighted > 0:
                num_wins += 1
            elif np.isclose(payout_weighted, -premium * weight):
                num_loss += 1
            elif payout_weighted < 0:
                num_partial_loss += 1

        except Exception as e:
            payouts.append(0.0)
            with open("options_reward_errors.log", "a") as f:
                f.write(f"Ticker {ticker} failed: {str(e)}\n")
            continue

    # -----------------------------
    # Aggregate metrics
    # -----------------------------
    PnL = np.sum(payouts)
    reward = PnL + num_wins - 2 * num_loss

    # -----------------------------
    # Cleanup
    # -----------------------------
    del payouts
    gc.collect()

    return reward, PnL, num_wins, num_loss, num_partial_loss

    import numpy as np
import pandas as pd
import gym
from gym import spaces
import torch
from stable_baselines3 import PPO, TD3, SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from tqdm import tqdm
import os
import gc

# -----------------------------
# 1. Environment Definition
# -----------------------------
class OptionsTradingEnv(gym.Env):
    """
    Custom Gym environment for options trading with multiple tickers.
    Observation: normalized quantitative vectors
    Action: allocation vector with constraints |x|<1, sum=1
    Reward: options_reward_pipeline output
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, all_tickers, obs_df, price_data, train_dates, num_days=10, risk_free_rate=0.05):
        super(OptionsTradingEnv, self).__init__()
        self.all_tickers = all_tickers
        self.price_data = price_data
        self.obs_df = obs_df
        self.train_dates = train_dates
        self.num_days = num_days
        self.risk_free_rate = risk_free_rate
        
        self.n_tickers = len(all_tickers)
        self.current_step = 0
        
        # Action space: allocation vector [-1,1], sum constraint enforced in step
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_tickers,), dtype=np.float32)
        # Observation space: normalized quantitative observation per ticker
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.n_tickers, self.obs_df.shape[1]), dtype=np.float32)
        
    def reset(self):
        self.current_step = 0
        return self._get_obs()
    
    def _get_obs(self):
        date = self.train_dates[self.current_step]
        obs = self.obs_df.loc[date].values
        return obs.astype(np.float32)
    
    def step(self, action):
        # Enforce sum constraint: normalize to sum 1
        action = np.clip(action, -1 + 1e-6, 1 - 1e-6)
        if np.any(action < 0):
            # allow negative allocations for puts
            sum_abs = np.sum(np.abs(action))
            if sum_abs > 0:
                action = action / sum_abs
        else:
            action = action / (np.sum(action) + 1e-8)
        
        date = self.train_dates[self.current_step]
        reward, _, _, _, _ = options_reward_pipeline(date, action, self.all_tickers, self.num_days, self.price_data, risk_free_rate=self.risk_free_rate)
        
        self.current_step += 1
        done = self.current_step >= len(self.train_dates)
        obs = self._get_obs() if not done else np.zeros_like(self._get_obs())
        
        info = {}
        return obs, reward, done, info

# -----------------------------
# 2. Training Pipeline
# -----------------------------
def train_agents(seeds=[1,2,3,4,5], agents_list=["PPO","TD3","SAC"], obs_df=None, price_data=None, train_dates=None, all_tickers=None, num_days=10):
    results = {}
    save_base = "./my_project/layer_1/rl_models/"
    os.makedirs(save_base, exist_ok=True)
    
    for seed in seeds:
        print(f"\n==== Training Seed {seed} ====")
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        env = OptionsTradingEnv(all_tickers=all_tickers, obs_df=obs_df, price_data=price_data, train_dates=train_dates, num_days=num_days)
        
        for agent_name in agents_list:
            print(f"\nTraining {agent_name} for seed {seed}...")
            model_save_path = os.path.join(save_base, f"{agent_name}_seed{seed}")
            
            if agent_name == "PPO":
                model = PPO("MlpPolicy", env, verbose=1, seed=seed)
            elif agent_name == "TD3":
                model = TD3("MlpPolicy", env, verbose=1, seed=seed)
            elif agent_name == "SAC":
                model = SAC("MlpPolicy", env, verbose=1, seed=seed)
            else:
                continue
            
            # Checkpoint callback every 5000 steps
            checkpoint_cb = CheckpointCallback(save_freq=500, save_path=model_save_path, name_prefix="rl_model")
            model.learn(total_timesteps=2000, callback=checkpoint_cb)
            
            # Save final model
            model.save(os.path.join(model_save_path, f"{agent_name}_final_seed{seed}"))
            results[(seed, agent_name)] = model
            
            # Cleanup
            del model
            gc.collect()
            torch.cuda.empty_cache()
        
        del env
        gc.collect()
    
    return results


trained_models = train_agents(
    seeds=[1,2,3,4,5],
    agents_list=["PPO","TD3","SAC"],
    obs_df=normalized_obs_vect,
    price_data=close_df,  # assuming close_df has your historical prices
    train_dates=train_dates,  # list of dates for training
    all_tickers=all_tickers,
    num_days=10
)

import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import gc

def backtest_rl_models(trained_models, normalized_obs_vect, price_data, test_dates, all_tickers, num_days=10, save_folder="./my_project/layer_1/results"):
    """
    Backtests RL agents on testing period and saves results.
    
    Args:
        trained_models (dict): {(seed, agent_name): model}
        normalized_obs_vect (pd.DataFrame): observation vectors (dates x features)
        price_data (pd.DataFrame): historical prices (dates x tickers)
        test_dates (list): list of dates for testing
        all_tickers (list): list of tickers
        num_days (int): option horizon
        save_folder (str): folder to save CSV results
    """
    os.makedirs(save_folder, exist_ok=True)
    results_all = []

    for (seed, agent_name), model in trained_models.items():
        print(f"\n=== Backtesting {agent_name}, Seed {seed} ===")
        
        # Create environment for testing
        env = OptionsTradingEnv(all_tickers=all_tickers,
                                obs_df=normalized_obs_vect,
                                price_data=price_data,
                                train_dates=test_dates,  # test dates
                                num_days=num_days)
        
        obs = env.reset()
        done = False
        step = 0

        pnl_list = []
        wins_list = []
        losses_list = []
        partial_losses_list = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            # Compute step-level metrics using your reward function
            _, pnl, num_wins, num_loss, num_partial_loss = options_reward_pipeline(
                date_t=test_dates[step],
                action_vector=action,
                all_tickers=all_tickers,
                num_days=num_days,
                price_data=price_data
            )
            
            pnl_list.append(pnl)
            wins_list.append(num_wins)
            losses_list.append(num_loss)
            partial_losses_list.append(num_partial_loss)
            
            step += 1

        # Save metrics in a DataFrame
        df_result = pd.DataFrame({
            "Date": test_dates,
            "PnL": pnl_list,
            "Num_Wins": wins_list,
            "Num_Losses": losses_list,
            "Num_Partial_Losses": partial_losses_list
        })
        df_result["Agent"] = agent_name
        df_result["Seed"] = seed
        
        # Save CSV
        csv_path = os.path.join(save_folder, f"{agent_name}_seed{seed}_backtest.csv")
        df_result.to_csv(csv_path, index=False)
        print(f"Saved backtest CSV: {csv_path}")

        # Save cumulative PnL plot
        plt.figure(figsize=(12,6))
        plt.plot(pd.to_datetime(test_dates), np.cumsum(pnl_list), label="Cumulative PnL")
        plt.title(f"{agent_name} Seed {seed} - Cumulative PnL")
        plt.xlabel("Date")
        plt.ylabel("Cumulative PnL")
        plt.grid(True)
        plot_path = os.path.join(save_folder, f"{agent_name}_seed{seed}_cumulative_pnl.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot: {plot_path}")
        
        results_all.append(df_result)

        # Cleanup
        del env
        gc.collect()
        torch.cuda.empty_cache()

    # Optionally, aggregate all results in one CSV
    df_all = pd.concat(results_all, ignore_index=True)
    aggregate_csv = os.path.join(save_folder, "all_backtests.csv")
    df_all.to_csv(aggregate_csv, index=False)
    print(f"\n✅ All backtests saved to {aggregate_csv}")

    return df_all

 backtest_results = backtest_rl_models(
     trained_models=trained_models,
     normalized_obs_vect=normalized_obs_vect,
     price_data=close_df,
     test_dates=test_dates,
     all_tickers=all_tickers,
     num_days=10,
     save_folder="./my_project/layer_1/results"
 )

 import pandas as pd
import numpy as np

def build_nlp_observation_df(sentiment_csv_folder, volatility_csv, tickers, dates):
    """
    Builds a normalized NLP-based observation DataFrame.
    
    Args:
        sentiment_csv_folder (str): folder with per-ticker sentiment CSVs
        volatility_csv (str): volatility CSV file path
        tickers (list): list of tickers
        dates (list): list of dates for simulation
    
    Returns:
        pd.DataFrame: obs_df with shape (len(dates), n_tickers*6)
    """
    # Load volatility
    vol_df = pd.read_csv(volatility_csv, index_col=0, parse_dates=True)

    obs_data = []

    for date in dates:
        row_features = []
        for ticker in tickers:
            # Load sentiment for ticker
            try:
                df = pd.read_csv(f"{sentiment_csv_folder}/{ticker}_sentiment.csv", index_col=0, parse_dates=True)
                df_date = df.loc[date]
                avg_score = df_date['avg_score']
                num_articles = df_date['num_articles']
                num_pos = df_date['num_positive']
                num_neg = df_date['num_negative']
                num_neu = df_date['num_neutral']
            except:
                # Default in case of missing data
                avg_score = 0.5
                num_articles = 0
                num_pos = 0
                num_neg = 0
                num_neu = 0
            
            # Volatility
            try:
                vol = vol_df.loc[date, ticker]
            except:
                vol = 0.0

            row_features.extend([avg_score, vol, num_articles, num_pos, num_neg, num_neu])
        obs_data.append(row_features)
    
    obs_df = pd.DataFrame(obs_data, index=dates)
    return obs_df
from stable_baselines3 import PPO, TD3, SAC

def train_nlp_rl_agents(obs_df, price_data, tickers, train_dates, seeds=[1,2,3,4,5], num_days=10):
    trained_models = {}
    
    for seed in seeds:
        for agent_name, agent_class in zip(['PPO','TD3','SAC'], [PPO, TD3, SAC]):
            env = OptionsTradingEnv(
                all_tickers=tickers,
                obs_df=obs_df.loc[train_dates],
                price_data=price_data.loc[train_dates],
                train_dates=train_dates,
                num_days=num_days
            )
            env = DummyVecEnv([lambda: env])
            model = agent_class('MlpPolicy', env, verbose=1, seed=seed)
            model.learn(total_timesteps=20000)
            
            trained_models[(seed, agent_name)] = model
            del env
            import gc; gc.collect()
            import torch; torch.cuda.empty_cache()
    
    return trained_models
def backtest_nlp_rl_models(trained_models, obs_df, price_data, test_dates, tickers, num_days=10, save_folder="./my_project/layer_1/results"):
    """
    Backtests RL agents trained on NLP-based observations.
    """
    os.makedirs(save_folder, exist_ok=True)
    results_all = []

    for (seed, agent_name), model in trained_models.items():
        print(f"\n=== Backtesting {agent_name}, Seed {seed} ===")
        
        env = OptionsTradingEnv(
            all_tickers=tickers,
            obs_df=obs_df.loc[test_dates],
            price_data=price_data.loc[test_dates],
            train_dates=test_dates,
            num_days=num_days
        )
        obs = env.reset()
        done = False
        step = 0

        pnl_list, wins_list, losses_list, partial_losses_list = [], [], [], []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            # Compute reward metrics
            _, pnl, num_wins, num_loss, num_partial_loss = options_reward_pipeline(
                date_t=test_dates[step],
                action_vector=action,
                all_tickers=tickers,
                num_days=num_days,
                price_data=price_data
            )
            
            pnl_list.append(pnl)
            wins_list.append(num_wins)
            losses_list.append(num_loss)
            partial_losses_list.append(num_partial_loss)
            step += 1

        # Save CSV
        df_result = pd.DataFrame({
            "Date": test_dates,
            "PnL": pnl_list,
            "Num_Wins": wins_list,
            "Num_Losses": losses_list,
            "Num_Partial_Losses": partial_losses_list
        })
        df_result["Agent"] = agent_name
        df_result["Seed"] = seed

        csv_path = os.path.join(save_folder, f"{agent_name}_seed{seed}_backtest_nlp.csv")
        df_result.to_csv(csv_path, index=False)
        results_all.append(df_result)
        del env; gc.collect(); torch.cuda.empty_cache()

    # Aggregate all
    df_all = pd.concat(results_all, ignore_index=True)
    df_all.to_csv(os.path.join(save_folder, "all_backtests_nlp.csv"), index=False)
    return df_all
import pandas as pd
import os

def concatenate_rl_actions(first_pipeline_csvs, second_pipeline_csvs, output_folder="./my_project/layer_2/time_series"):
    """
    Concatenates action vectors from multiple RL models for each period.

    Args:
        first_pipeline_csvs (list of str): Paths to per-model CSVs from the first pipeline (quant).
        second_pipeline_csvs (list of str): Paths to per-model CSVs from the second pipeline (NLP).
        output_folder (str): Folder where concatenated CSVs will be saved.

    Output:
        - observation_quant.csv: Concatenated action vectors from the first pipeline
        - observation_nlp.csv: Concatenated action vectors from the second pipeline
    """
    os.makedirs(output_folder, exist_ok=True)

    def concat_actions(csv_list):
        """
        Reads multiple CSVs and concatenates the action columns per period.
        """
        dfs = []
        for csv_path in csv_list:
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            # Assume action columns start after first column
            action_cols = df.columns[df.columns.str.contains("Action|action|a")]
            dfs.append(df[action_cols])
        # Concatenate horizontally (axis=1)
        concatenated_df = pd.concat(dfs, axis=1)
        return concatenated_df

    # ---- First pipeline: Quant ----
    obs_quant = concat_actions(first_pipeline_csvs)
    obs_quant_path = os.path.join(output_folder, "observation_quant.csv")
    obs_quant.to_csv(obs_quant_path)
    print(f"✅ Saved concatenated quant observations: {obs_quant_path}")

    # ---- Second pipeline: NLP ----
    obs_nlp = concat_actions(second_pipeline_csvs)
    obs_nlp_path = os.path.join(output_folder, "observation_nlp.csv")
    obs_nlp.to_csv(obs_nlp_path)
    print(f"✅ Saved concatenated NLP observations: {obs_nlp_path}")

    return obs_quant_path, obs_nlp_path
# List of CSVs produced by RL models (first pipeline)
first_pipeline_csvs = [
    "./my_project/layer_1/results/PPO_seed1_backtest.csv",
    "./my_project/layer_1/results/TD3_seed1_backtest.csv",
    "./my_project/layer_1/results/SAC_seed1_backtest.csv",
    # Add other seeds/models as needed
]

# List of CSVs produced by RL models (second pipeline: NLP)
second_pipeline_csvs = [
    "./my_project/layer_1/results/PPO_seed1_backtest_nlp.csv",
    "./my_project/layer_1/results/TD3_seed1_backtest_nlp.csv",
    "./my_project/layer_1/results/SAC_seed1_backtest_nlp.csv",
]

obs_quant_path, obs_nlp_path = concatenate_rl_actions(
    first_pipeline_csvs,
    second_pipeline_csvs,
    output_folder="./my_project/layer_2/time_series"
)
import pandas as pd
import os

def normalize_observations(input_csv, output_csv=None):
    """
    Normalizes a concatenated observation CSV so that each row is divided
    by its first column value. The first column becomes all 1s.

    Args:
        input_csv (str): Path to the input CSV.
        output_csv (str, optional): Path to save the normalized CSV. 
                                    If None, overwrites input_csv.

    Returns:
        pd.DataFrame: Normalized observation dataframe.
    """
    df = pd.read_csv(input_csv, index_col=0, parse_dates=True)

    # Avoid division by zero
    first_col = df.iloc[:, 0].replace(0, 1e-8)

    # Normalize row-wise
    normalized_df = df.div(first_col, axis=0)

    if output_csv is None:
        output_csv = input_csv

    # Save normalized CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    normalized_df.to_csv(output_csv)
    print(f"✅ Saved normalized observations to: {output_csv}")

    return normalized_df
# Normalize concatenated quant observations
normalized_quant_path = "./my_project/layer_2/time_series/observation_quant_normalized.csv"
normalize_observations(obs_quant_path, normalized_quant_path)

# Normalize concatenated NLP observations
normalized_nlp_path = "./my_project/layer_2/time_series/observation_nlp_normalized.csv"
normalize_observations(obs_nlp_path, normalized_nlp_path)
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.distributions import Normal
import gc

# ===========================
# Custom PyTorch Actor-Critic Agent
# ===========================
class CustomActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims=[512, 512, 256], activation=nn.ReLU):
        super().__init__()
        layers = []
        input_dim = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(activation())
            input_dim = h
        self.net = nn.Sequential(*layers)
        self.mu_head = nn.Linear(input_dim, action_dim)
        self.log_std_head = nn.Linear(input_dim, action_dim)

    def forward(self, x):
        x = self.net(x)
        mu = self.mu_head(x)
        log_std = self.log_std_head(x).clamp(-20, 2)  # prevent extreme std
        std = torch.exp(log_std)
        return mu, std

    def get_action(self, x):
        mu, std = self.forward(x)
        dist = Normal(mu, std)
        action = dist.sample()
        action = torch.tanh(action)  # constrain |x|<1
        return action, dist.log_prob(action).sum(dim=-1)

class CustomCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims=[512, 512, 256], activation=nn.ReLU):
        super().__init__()
        layers = []
        input_dim = obs_dim + action_dim
        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(activation())
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        value = self.net(x)
        return value

# ===========================
# Replay Buffer
# ===========================
class ReplayBuffer:
    def __init__(self, max_size=1000000):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.next_obs = []
        self.done = []
        self.max_size = max_size

    def add(self, obs, action, reward, next_obs, done):
        if len(self.obs) >= self.max_size:
            self.obs.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_obs.pop(0)
            self.done.pop(0)
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_obs.append(next_obs)
        self.done.append(done)

    def sample(self, batch_size):
        idx = np.random.choice(len(self.obs), batch_size, replace=False)
        return (
            torch.tensor(np.array([self.obs[i] for i in idx]), dtype=torch.float32),
            torch.tensor(np.array([self.actions[i] for i in idx]), dtype=torch.float32),
            torch.tensor(np.array([self.rewards[i] for i in idx]), dtype=torch.float32).unsqueeze(-1),
            torch.tensor(np.array([self.next_obs[i] for i in idx]), dtype=torch.float32),
            torch.tensor(np.array([self.done[i] for i in idx]), dtype=torch.float32).unsqueeze(-1)
        )

# ===========================
# Custom RL Pipeline
# ===========================
def train_layer2_rl(obs_csv, reward_func, all_tickers, seeds=[1,2,3,4,5],
                    n_epochs=2000, batch_size=64, gamma=0.99,
                    lr_actor=3e-4, lr_critic=3e-4, output_folder="./my_project/layer_2/results"):
    
    os.makedirs(output_folder, exist_ok=True)
    
    obs_df = pd.read_csv(obs_csv, index_col=0)
    obs_array = obs_df.values  # shape: [time_steps, obs_dim]
    obs_dim = obs_array.shape[1]
    action_dim = len(all_tickers)

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        actor = CustomActor(obs_dim, action_dim)
        critic = CustomCritic(obs_dim, action_dim)
        actor_optimizer = optim.Adam(actor.parameters(), lr=lr_actor)
        critic_optimizer = optim.Adam(critic.parameters(), lr=lr_critic)

        buffer = ReplayBuffer(max_size=50000)
        rewards_history = []

        # Simulate environment with reward_func
        for t in tqdm(range(len(obs_array)-1), desc=f"Seed {seed} Training"):
            obs_t = torch.tensor(obs_array[t], dtype=torch.float32).unsqueeze(0)
            
            # Actor selects action
            action, log_prob = actor.get_action(obs_t)
            action_np = action.detach().numpy().flatten()
            
            # Compute reward using your existing reward function
            reward, _, _, _, _ = reward_func(
                date_t=obs_df.index[t+1],
                action_vector=action_np,
                all_tickers=all_tickers
            )
            
            # Get next observation
            next_obs_t = torch.tensor(obs_array[t+1], dtype=torch.float32).unsqueeze(0)
            done = 0.0 if t < len(obs_array)-2 else 1.0
            
            buffer.add(obs_t.squeeze(0).numpy(), action_np, reward, next_obs_t.squeeze(0).numpy(), done)
            
            rewards_history.append(reward)
            
            # --- Sample from buffer ---
            if len(buffer.obs) >= batch_size:
                obs_b, act_b, rew_b, next_obs_b, done_b = buffer.sample(batch_size)
                
                # Critic update
                with torch.no_grad():
                    next_action, _ = actor.get_action(next_obs_b)
                    target_value = rew_b + gamma * critic(next_obs_b, next_action) * (1 - done_b)
                value = critic(obs_b, act_b)
                critic_loss = nn.MSELoss()(value, target_value)
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()
                
                # Actor update (policy gradient)
                action_pred, log_prob_pred = actor.get_action(obs_b)
                actor_loss = -critic(obs_b, action_pred).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
            
            if t % 500 == 0 and t > 0:
                print(f"Step {t}, Avg reward last 500 steps: {np.mean(rewards_history[-500:]):.4f}")
        
        # Save actor and critic models
        actor_path = os.path.join(output_folder, f"actor_seed_{seed}.pt")
        critic_path = os.path.join(output_folder, f"critic_seed_{seed}.pt")
        torch.save(actor.state_dict(), actor_path)
        torch.save(critic.state_dict(), critic_path)
        
        # Save rewards history
        rewards_path = os.path.join(output_folder, f"rewards_seed_{seed}.csv")
        pd.DataFrame(rewards_history, columns=['reward']).to_csv(rewards_path)
        
        print(f"✅ Seed {seed} training completed and saved to {output_folder}")
        gc.collect()
        torch.cuda.empty_cache()

import matplotlib.pyplot as plt

def backtest_layer2_rl(obs_csv, reward_func, all_tickers, actor_paths, output_folder="./my_project/layer_2/results"):
    """
    Backtest trained Layer 2 RL agents.

    Args:
        obs_csv (str): Path to observation CSV (quant or NLP pipeline).
        reward_func (callable): Reward function with signature reward_func(date_t, action_vector, all_tickers)
        all_tickers (list): List of tickers
        actor_paths (list): List of trained actor model paths (per seed)
        output_folder (str): Folder to save metrics and plots
    """
    os.makedirs(output_folder, exist_ok=True)
    
    obs_df = pd.read_csv(obs_csv, index_col=0)
    obs_array = obs_df.values
    time_steps, obs_dim = obs_array.shape
    action_dim = len(all_tickers)

    metrics = {}

    for actor_path in actor_paths:
        seed_name = os.path.basename(actor_path).split("_seed_")[-1].split(".")[0]
        print(f"\n=== Backtesting Seed {seed_name} ===")
        
        # Load actor
        actor = CustomActor(obs_dim, action_dim)
        actor.load_state_dict(torch.load(actor_path))
        actor.eval()
        
        pnl_list, num_wins_list, num_loss_list, num_partial_loss_list = [], [], [], []

        for t in tqdm(range(time_steps-1), desc=f"Backtesting Seed {seed_name}"):
            obs_t = torch.tensor(obs_array[t], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action, _ = actor.get_action(obs_t)
                action_np = action.squeeze(0).numpy()
            
            # Compute reward metrics
            reward, pnl, num_wins, num_loss, num_partial_loss = reward_func(
                date_t=obs_df.index[t+1],
                action_vector=action_np,
                all_tickers=all_tickers
            )

            pnl_list.append(pnl)
            num_wins_list.append(num_wins)
            num_loss_list.append(num_loss)
            num_partial_loss_list.append(num_partial_loss)

        # Save metrics CSV
        metrics_df = pd.DataFrame({
            'PnL': pnl_list,
            'Num_Wins': num_wins_list,
            'Num_Loss': num_loss_list,
            'Num_Partial_Loss': num_partial_loss_list
        }, index=obs_df.index[1:])

        metrics_path = os.path.join(output_folder, f"metrics_seed_{seed_name}.csv")
        metrics_df.to_csv(metrics_path)
        print(f"Metrics saved: {metrics_path}")

        # --- Plotting ---
        fig, ax = plt.subplots(4,1, figsize=(16,18), sharex=True)
        fig.suptitle(f"Backtest Seed {seed_name}", fontsize=16)

        ax[0].plot(metrics_df['PnL'].cumsum(), color='blue', label='Cumulative PnL')
        ax[0].set_title('Cumulative PnL')
        ax[0].legend()
        ax[0].grid(True)

        ax[1].plot(metrics_df['Num_Wins'], color='green', label='Num Wins')
        ax[1].set_title('Number of Wins')
        ax[1].legend()
        ax[1].grid(True)

        ax[2].plot(metrics_df['Num_Loss'], color='red', label='Num Total Losses')
        ax[2].set_title('Number of Total Losses')
        ax[2].legend()
        ax[2].grid(True)

        ax[3].plot(metrics_df['Num_Partial_Loss'], color='orange', label='Num Partial Losses')
        ax[3].set_title('Number of Partial Losses')
        ax[3].legend()
        ax[3].grid(True)

        plt.tight_layout(rect=[0,0,1,0.96])
        plot_path = os.path.join(output_folder, f"backtest_seed_{seed_name}.png")
        fig.savefig(plot_path)
        plt.close(fig)
        print(f"Backtest plot saved: {plot_path}")

        # Store in dictionary
        metrics[seed_name] = metrics_df

        # Memory cleanup
        gc.collect()
        torch.cuda.empty_cache()

    return metrics
actor_paths = [
    "./my_project/layer_2/results/actor_seed_1.pt",
    "./my_project/layer_2/results/actor_seed_2.pt",
    "./my_project/layer_2/results/actor_seed_3.pt",
    "./my_project/layer_2/results/actor_seed_4.pt",
    "./my_project/layer_2/results/actor_seed_5.pt"
]

metrics = backtest_layer2_rl(
    obs_csv="./my_project/layer_2/time_series/observation_quant_normalized.csv",
    reward_func=reward,  # your existing reward function
    all_tickers=all_tickers,
    actor_paths=actor_paths,
    output_folder="./my_project/layer_2/results"
)
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np

# --- Step 1: Concatenate Layer 2 Actions ---
def concatenate_layer2_actions(actor_paths, obs_csv, all_tickers, output_csv="./my_project/layer_3/time_series/observation_actions_layer3.csv"):
    """
    Generate Layer 3 observation by concatenating Layer 2 agents' actions.

    Args:
        actor_paths (list): Paths to Layer 2 trained actors
        obs_csv (str): Observation CSV used by Layer 2 agents
        all_tickers (list): List of tickers
        output_csv (str): Path to save concatenated actions
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    obs_df = pd.read_csv(obs_csv, index_col=0)
    time_steps = len(obs_df)
    
    concatenated_actions = []

    for t in range(time_steps-1):
        actions_t = []
        obs_t = torch.tensor(obs_df.iloc[t].values, dtype=torch.float32).unsqueeze(0)
        
        for actor_path in actor_paths:
            actor = CustomActor(obs_dim=obs_t.shape[1], action_dim=len(all_tickers))
            actor.load_state_dict(torch.load(actor_path))
            actor.eval()
            with torch.no_grad():
                action, _ = actor.get_action(obs_t)
                actions_t.append(action.squeeze(0).numpy())
            del actor
            torch.cuda.empty_cache()
        
        concatenated_actions.append(np.concatenate(actions_t))
    
    concatenated_df = pd.DataFrame(concatenated_actions, index=obs_df.index[1:])
    concatenated_df.to_csv(output_csv)
    print(f"✅ Layer 3 observation CSV saved: {output_csv}")
    return concatenated_df
class Layer3ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=512, n_heads=4):
        super(Layer3ActorCritic, self).__init__()
        
        # Shared feature extractor
        self.input_proj = nn.Linear(obs_dim, hidden_dim)
        self.res_block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.res_block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

        # Multi-head attention for cross-ticker correlation
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, batch_first=True)

        # Actor head
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_logstd = nn.Linear(hidden_dim, action_dim)

        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        x = F.relu(self.input_proj(obs))
        x = self.norm1(x + self.res_block1(x))
        x = self.norm2(x + self.res_block2(x))
        x = self.dropout(x)

        # Reshape for attention: (batch, seq_len=1, features)
        x_attn = x.unsqueeze(1)
        attn_out, _ = self.attention(x_attn, x_attn, x_attn)
        attn_out = attn_out.squeeze(1)

        # Actor
        mean = self.actor_mean(attn_out)
        log_std = torch.clamp(self.actor_logstd(attn_out), -20, 2)  # stable range
        std = torch.exp(log_std)

        # Critic
        value = self.critic(attn_out)
        return mean, std, value

    def get_action(self, obs):
        mean, std, _ = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        action = torch.tanh(action)  # ensure |x| < 1
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob


        
def train_layer3(obs_csv, reward_func, all_tickers, seeds=[1,2,3,4,5], output_folder="./my_project/layer_3/results"):
    os.makedirs(output_folder, exist_ok=True)
    obs_df = pd.read_csv(obs_csv, index_col=0)
    obs_array = obs_df.values
    obs_dim = obs_array.shape[1]
    action_dim = len(all_tickers)

    for seed in seeds:
        torch.manual_seed(seed)
        agent = Layer3ActorCritic(obs_dim, action_dim)
        optimizer = torch.optim.Adam(agent.parameters(), lr=1e-4)
        torch.save(agent.state_dict(), os.path.join(output_folder, f"layer3_actor_seed_{seed}.pt"))
        print(f"✅ Saved Layer 3 actor seed {seed}")
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc

def backtest_layer3(obs_csv, all_tickers, reward_func, actor_paths, download_folder="./my_project/layer_3"):
    """
    Full backtest for Layer 3 agents.
    
    Args:
        obs_csv (str): CSV file of Layer 3 observations (concatenated Layer 2 actions)
        all_tickers (list): List of tickers
        reward_func (callable): Function reward(date_index, action_vector) -> reward
        actor_paths (list): Paths to trained Layer 3 actor models (per seed)
        download_folder (str): Base folder for saving results and plots
    """
    os.makedirs(os.path.join(download_folder, "results"), exist_ok=True)
    os.makedirs(os.path.join(download_folder, "plots"), exist_ok=True)

    obs_df = pd.read_csv(obs_csv, index_col=0)
    dates = obs_df.index
    obs_array = obs_df.values

    for actor_path in actor_paths:
        seed = int(actor_path.split("_seed_")[-1].split(".")[0])
        print(f"--- Backtesting Layer 3, Seed {seed} ---")

        # Load agent
        obs_dim = obs_array.shape[1]
        action_dim = len(all_tickers)
        agent = Layer3ActorCritic(obs_dim, action_dim)
        agent.load_state_dict(torch.load(actor_path))
        agent.eval()

        rewards_list = []
        pnl_list = []
        wins_list = []
        losses_list = []
        partial_losses_list = []

        for t in range(len(obs_array)-1):
            obs_t = torch.tensor(obs_array[t], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action, _ = agent.get_action(obs_t)
            action_np = action.squeeze(0).numpy()

            # Compute reward for this step
            reward, pnl, num_wins, num_loss, num_partial_loss = reward_func(date_index=t+1, action_vector=action_np)
            rewards_list.append(reward)
            pnl_list.append(pnl)
            wins_list.append(num_wins)
            losses_list.append(num_loss)
            partial_losses_list.append(num_partial_loss)

        # Save CSV
        results_df = pd.DataFrame({
            "Date": dates[1:],
            "Reward": rewards_list,
            "PnL": pnl_list,
            "Num_Wins": wins_list,
            "Num_Losses": losses_list,
            "Num_Partial_Loss": partial_losses_list
        }).set_index("Date")
        result_csv = os.path.join(download_folder, "results", f"layer3_backtest_seed_{seed}.csv")
        results_df.to_csv(result_csv)
        print(f"✅ Saved backtest CSV: {result_csv}")

        # Plot cumulative metrics
        cum_pnl = np.cumsum(pnl_list)
        cum_reward = np.cumsum(rewards_list)
        cum_wins = np.cumsum(wins_list)
        cum_losses = np.cumsum(losses_list)

        plt.figure(figsize=(16,6))
        plt.plot(dates[1:], cum_pnl, label="Cumulative PnL", color='blue')
        plt.plot(dates[1:], cum_reward, label="Cumulative Reward", color='green')
        plt.plot(dates[1:], cum_wins, label="Cumulative Wins", color='orange')
        plt.plot(dates[1:], cum_losses, label="Cumulative Losses", color='red')
        plt.title(f"Layer 3 Backtest Seed {seed}")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Metrics")
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(download_folder, "plots", f"layer3_backtest_seed_{seed}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"✅ Saved plot: {plot_path}")

        # Cleanup
        del agent, obs_t, action
        torch.cuda.empty_cache()
        gc.collect()

    print("\n✅ Layer 3 backtesting complete for all seeds!")
actor_paths = [
    "./my_project/layer_3/results/layer3_actor_seed_1.pt",
    "./my_project/layer_3/results/layer3_actor_seed_2.pt",
    "./my_project/layer_3/results/layer3_actor_seed_3.pt",
    "./my_project/layer_3/results/layer3_actor_seed_4.pt",
    "./my_project/layer_3/results/layer3_actor_seed_5.pt"
]

backtest_layer3(
    obs_csv="./my_project/layer_3/time_series/observation_actions_layer3.csv",
    all_tickers=all_tickers,
    reward_func=reward,  # previously defined reward function
    actor_paths=actor_paths,
    download_folder="./my_project/layer_3"
)
