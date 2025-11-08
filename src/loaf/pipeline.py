"""
Main system orchestration and pipeline.
"""

import os
from typing import Dict, Any

from .config.config import SystemConfig, MarketConfig, NLPConfig
from .data.market_data import MarketDataFetcher
from .data.nlp_processing import SentimentAnalyzer, WebScraper
from .visualization.market_plots import MarketVisualizer
from .utils.data_processing import aggregate_observations, normalize_observation_vector

class LOAF:
    """Main LOAF system class."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize LOAF system.
        
        Args:
            config: Optional configuration dictionary
        """
        # Set up configuration
        self.config = SystemConfig(
            market=MarketConfig.default(),
            nlp=NLPConfig(),
            base_dir=config.get('base_dir', './data') if config else './data'
        )

        # Initialize components
        self.market_data = MarketDataFetcher(self.config.market)
        self.sentiment = SentimentAnalyzer(self.config.nlp)
        self.scraper = WebScraper(self.config.nlp)
        self.visualizer = MarketVisualizer(os.path.join(self.config.base_dir, 'plots'))

    def run_market_pipeline(self):
        """Run the market data pipeline."""
        # Create directory structure
        os.makedirs(self.config.base_dir, exist_ok=True)
        
        # Download and process market data
        print("Downloading market data...")
        data_paths = self.market_data.download_data(
            save_dir=os.path.join(self.config.base_dir, 'market_data')
        )
        
        # Create visualizations
        print("\nGenerating visualizations...")
        all_tickers = [t for group in self.config.market.tickers.values() for t in group]
        for ticker in all_tickers:
            plots = self.visualizer.save_market_plots(ticker, {
                'close': pd.read_csv(data_paths['close'])[ticker],
                'open': pd.read_csv(data_paths['open'])[ticker],
                'high': pd.read_csv(data_paths['high'])[ticker],
                'low': pd.read_csv(data_paths['low'])[ticker],
                'volume': pd.read_csv(data_paths['volume'])[ticker],
                'volatility': pd.read_csv(data_paths['volatility'])[ticker],
                'sharpe': pd.read_csv(data_paths['sharpe'])[ticker],
                'normalized_close': pd.read_csv(data_paths['normalized_close'])[ticker]
            })
            print(f"Created plots for {ticker}: {plots}")
        
        print("\nMarket pipeline complete!")
        return data_paths

    def run_nlp_pipeline(self, urls_file: str):
        """
        Run the NLP analysis pipeline.
        
        Args:
            urls_file: Path to CSV file containing URLs to analyze
        """
        print("Starting NLP pipeline...")
        
        # Load URLs
        df_urls = pd.read_csv(urls_file)
        
        # Scrape text
        texts = []
        for url in df_urls['url']:
            if self.scraper.is_valid_url(url):
                text = self.scraper.scrape_text(url)
                texts.append(text)
            else:
                texts.append("")
        
        # Analyze sentiment in batches
        sentiments = self.sentiment.compute_sentiment_batch(texts)
        
        # Save results
        df_urls['sentiment'] = sentiments
        output_path = os.path.join(self.config.base_dir, 'nlp_results.csv')
        df_urls.to_csv(output_path, index=False)
        
        print(f"\nNLP pipeline complete! Results saved to {output_path}")
        return output_path