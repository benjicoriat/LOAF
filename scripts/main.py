"""
Main entry point for the LOAF trading system.
Executes the complete pipeline in order:
1. Setup and data processing
2. Layer 1 (Quantitative)
3. Layer 1 (NLP)
4. Layer 2 (Combined)
5. Layer 3 (Meta-learning)
"""

# ==========================================
# 1. Imports
# ==========================================

# Core imports
import os
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
import json
import gc

# LOAF imports
from loaf.config.config import MarketConfig, NLPConfig, SystemConfig
from loaf.data.market_data import MarketDataFetcher
from loaf.data.nlp_processing import SentimentAnalyzer, WebScraper
from loaf.visualization.market_plots import MarketVisualizer
from loaf.models.hrl.pipeline import HRLPipeline
from loaf.utils.data_processing import normalize_observation_vector

# ==========================================
# 2. Constants and Configuration
# ==========================================
BASE_DIR = "./loaf_data"
API_KEY = "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"

# Date ranges
DATES = {
    "Download": {"start": "2022-01-01", "end": datetime.now().strftime("%Y-%m-%d")},
    "Training": {"start": "2023-01-01", "end": "2023-02-28"},  # With enough historical data
    "Testing": {"start": "2023-03-01", "end": "2023-03-31"}
}

# Market universe
TICKERS = {
    "Equity Indices": ["SPY", "DIA", "QQQ", "IWM", "^FCHI", "^GDAXI", "^FTSE",
                      "^N225", "^STOXX50E", "^HSI"],
    "REITs": ["VNQ", "SCHH", "IYR"],
    "Commodities": ["GLD", "SLV", "USO", "UNG", "DBA", "PALL", "CPER"],
    "Cryptocurrencies": ["BTC-USD", "ETH-USD"],
    "Bonds/Defensive": ["TLT", "IEF", "SHY"]
}

# Flatten tickers
ALL_TICKERS = [ticker for group in TICKERS.values() for ticker in group]

# ==========================================
# 3. Directory Setup
# ==========================================
def setup_directories():
    """Create project directory structure."""
    directories = [
        f"{BASE_DIR}/data/quantitative/raw",
        f"{BASE_DIR}/data/quantitative/processed",
        f"{BASE_DIR}/data/nlp/raw",
        f"{BASE_DIR}/data/nlp/processed",
        f"{BASE_DIR}/models/layer1/quant",
        f"{BASE_DIR}/models/layer1/nlp",
        f"{BASE_DIR}/models/layer2",
        f"{BASE_DIR}/models/layer3",
        f"{BASE_DIR}/results/layer1/quant",
        f"{BASE_DIR}/results/layer1/nlp",
        f"{BASE_DIR}/results/layer2",
        f"{BASE_DIR}/results/layer3",
        f"{BASE_DIR}/plots/layer1/quant",
        f"{BASE_DIR}/plots/layer1/nlp",
        f"{BASE_DIR}/plots/layer2",
        f"{BASE_DIR}/plots/layer3"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

# ==========================================
# 4. Download and Process Quantitative Data
# ==========================================
def process_quantitative_data():
    """Download and process market data."""
    market_config = MarketConfig(tickers=TICKERS, date_ranges=DATES)
    fetcher = MarketDataFetcher(market_config)
    
    print("\nDownloading market data...")
    data_paths = fetcher.download_data(
        save_dir=f"{BASE_DIR}/data/quantitative/raw"
    )
    
    print("\nCreating market visualizations...")
    visualizer = MarketVisualizer(base_folder_plots=f"{BASE_DIR}/plots/layer1/quant")
    
    for ticker in ALL_TICKERS:
        visualizer.save_market_plots(ticker, {
            'close': pd.read_csv(data_paths['close'])[ticker],
            'open': pd.read_csv(data_paths['open'])[ticker],
            'high': pd.read_csv(data_paths['high'])[ticker],
            'low': pd.read_csv(data_paths['low'])[ticker],
            'volume': pd.read_csv(data_paths['volume'])[ticker],
            'volatility': pd.read_csv(data_paths['volatility'])[ticker],
            'sharpe': pd.read_csv(data_paths['sharpe'])[ticker],
            'normalized_close': pd.read_csv(data_paths['normalized_close'])[ticker]
        })
    
    return data_paths

# ==========================================
# 5. Download and Process NLP Data
# ==========================================
def process_nlp_data():
    """Process NLP and sentiment data."""
    nlp_config = NLPConfig()
    sentiment_analyzer = SentimentAnalyzer(nlp_config)
    scraper = WebScraper(nlp_config)
    
    # Process NLP data for each ticker
    nlp_data = {}
    for ticker in ALL_TICKERS:
        print(f"\nProcessing NLP data for {ticker}...")
        
        # Scrape and analyze news/social media data
        urls = [f"https://finance.yahoo.com/quote/{ticker}"]  # Add more sources as needed
        texts = []
        
        for url in urls:
            if scraper.is_valid_url(url):
                text = scraper.scrape_text(url)
                texts.append(text)
        
        # Compute sentiment scores
        if texts:
            sentiments = sentiment_analyzer.compute_sentiment_batch(texts)
            nlp_data[ticker] = np.mean(sentiments)
        else:
            nlp_data[ticker] = 0.5  # neutral sentiment if no data
    
    # Save NLP data
    nlp_df = pd.DataFrame.from_dict(nlp_data, orient='index', columns=['sentiment'])
    nlp_path = f"{BASE_DIR}/data/nlp/processed/sentiment_scores.csv"
    nlp_df.to_csv(nlp_path)
    
    return nlp_path

# ==========================================
# 6. Layer 1 Quantitative Pipeline
# ==========================================
def run_layer1_quant(data_paths):
    """Run Layer 1 quantitative models."""
    print("\nRunning Layer 1 Quantitative Pipeline...")
    
    # Load and prepare data
    close_data = pd.read_csv(data_paths['close'], index_col=0, parse_dates=True)
    normalized_data = pd.read_csv(data_paths['normalized_close'], index_col=0, parse_dates=True)
    
    # Filter out missing data
    common_idx = close_data.dropna(how='any').index
    close_data = close_data.loc[common_idx]
    normalized_data = normalized_data.loc[common_idx]
    
    # Split dates, ensuring data availability
    train_start = pd.Timestamp(DATES['Training']['start'])
    train_end = pd.Timestamp(DATES['Training']['end'])
    test_start = pd.Timestamp(DATES['Testing']['start'])
    test_end = pd.Timestamp(DATES['Testing']['end'])
    
    train_dates = pd.date_range(train_start, train_end, freq='B').intersection(common_idx)
    test_dates = pd.date_range(test_start, test_end, freq='B').intersection(common_idx)
    
    # Initialize and run pipeline
    pipeline = HRLPipeline(
        base_dir=f"{BASE_DIR}/models/layer1/quant",
        seeds=[1],  # Single seed as requested
        agents_list=["PPO", "TD3", "SAC"]
    )
    
    # Train and backtest
    pipeline.train_layer1(normalized_data, close_data, train_dates, ALL_TICKERS)
    results = pipeline.backtest_layer1(normalized_data, close_data, test_dates, ALL_TICKERS)
    
    return results

# ==========================================
# 7. Layer 1 NLP Pipeline
# ==========================================
def run_layer1_nlp(nlp_path, data_paths):
    """Run Layer 1 NLP models."""
    print("\nRunning Layer 1 NLP Pipeline...")
    
    # Load data
    nlp_data = pd.read_csv(nlp_path, index_col=0)
    close_data = pd.read_csv(data_paths['close'], index_col=0, parse_dates=True)
    
    # Split dates
    train_dates = pd.date_range(DATES['Training']['start'], DATES['Training']['end'], freq='B')
    test_dates = pd.date_range(DATES['Testing']['start'], DATES['Testing']['end'], freq='B')
    
    # Initialize and run pipeline
    pipeline = HRLPipeline(
        base_dir=f"{BASE_DIR}/models/layer1/nlp",
        seeds=[1],  # Single seed
        agents_list=["PPO"]  # Simplified for NLP
    )
    
    # Train and backtest
    pipeline.train_layer1(nlp_data, close_data, train_dates, ALL_TICKERS)
    results = pipeline.backtest_layer1(nlp_data, close_data, test_dates, ALL_TICKERS)
    
    return results

# ==========================================
# 8. Layer 2 Combined Pipeline
# ==========================================
def run_layer2(l1_quant_results, l1_nlp_results, data_paths):
    """Run Layer 2 combined models."""
    print("\nRunning Layer 2 Pipeline...")
    
    # Load data
    close_data = pd.read_csv(data_paths['close'], index_col=0, parse_dates=True)
    
    # Split dates
    train_dates = pd.date_range(DATES['Training']['start'], DATES['Training']['end'], freq='B')
    test_dates = pd.date_range(DATES['Testing']['start'], DATES['Testing']['end'], freq='B')
    
    # Combine Layer 1 results
    combined_obs = pd.concat([l1_quant_results, l1_nlp_results], axis=1)
    
    # Initialize custom Layer 2 environment and agent
    from loaf.models.custom_envs import Layer2TradingEnv
    from loaf.models.custom_agents import CustomActorCritic
    
    # Create environment
    env = Layer2TradingEnv(
        data=close_data,
        dates=train_dates,
        tickers=ALL_TICKERS
    )
    
    # Initialize custom PyTorch agent with attention mechanism
    agent = CustomActorCritic(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        hidden_dim=256,
        num_heads=4,
        dropout=0.1,
        learning_rate=3e-4
    )
    
    # Train and evaluate
    agent.train(env, num_episodes=1000, seed=1)
    results = agent.evaluate(env)
    
    # Save model
    torch.save(agent.state_dict(), f"{BASE_DIR}/models/layer2/model.pth")
    
    return results

# ==========================================
# 9. Layer 3 Meta-Learning Pipeline
# ==========================================
def run_layer3(l2_results, data_paths):
    """Run Layer 3 meta-learning models."""
    print("\nRunning Layer 3 Pipeline...")
    
    # Load data
    close_data = pd.read_csv(data_paths['close'], index_col=0, parse_dates=True)
    
    # Split dates
    train_dates = pd.date_range(DATES['Training']['start'], DATES['Training']['end'], freq='B')
    test_dates = pd.date_range(DATES['Testing']['start'], DATES['Testing']['end'], freq='B')
    
    # Initialize custom Layer 3 meta-learning environment and agent
    from loaf.models.custom_envs import Layer3MetaLearningEnv
    from loaf.models.custom_agents import MetaActorCritic
    
    # Create meta-learning environment
    env = Layer3MetaLearningEnv(
        data=close_data,
        dates=train_dates,
        tickers=ALL_TICKERS,
        layer2_results=l2_results
    )
    
    # Initialize custom meta-learning agent with transformer architecture
    agent = MetaActorCritic(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        hidden_dim=512,
        num_encoder_layers=3,
        num_heads=8,
        dropout=0.1,
        learning_rate=1e-4
    )
    
    # Train and evaluate with meta-learning approach
    agent.meta_train(env, num_tasks=50, num_episodes_per_task=100, seed=1)
    results = agent.meta_evaluate(env)
    
    # Save model
    torch.save(agent.state_dict(), f"{BASE_DIR}/models/layer3/model.pth")
    
    return results

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    # 1. Setup
    print("Setting up project structure...")
    setup_directories()
    
    # 2. Data Processing
    print("\nProcessing market data...")
    data_paths = process_quantitative_data()
    nlp_path = process_nlp_data()
    
    # 3. Layer 1 - Quantitative
    l1_quant_results = run_layer1_quant(data_paths)
    
    # 4. Layer 1 - NLP
    l1_nlp_results = run_layer1_nlp(nlp_path, data_paths)
    
    # 5. Layer 2 - Combined
    l2_results = run_layer2(l1_quant_results, l1_nlp_results, data_paths)
    
    # 6. Layer 3 - Meta-Learning
    l3_results = run_layer3(l2_results, data_paths)
    
    print("\nPipeline execution complete!")