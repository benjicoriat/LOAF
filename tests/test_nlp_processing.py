"""
Unit tests for NLP processing functionality.
"""

import unittest
import pandas as pd
import torch
from pathlib import Path
from loaf.data.nlp_processing import SentimentAnalyzer, WebScraper
from loaf.config.config import NLPConfig

class TestNLPProcessing(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.config = NLPConfig()
        self.sentiment_analyzer = SentimentAnalyzer(self.config)
        self.web_scraper = WebScraper(self.config)
        
    def test_sentiment_analysis(self):
        """Test sentiment analysis functionality."""
        test_texts = [
            "The company reported strong earnings growth.",
            "The stock price fell sharply after poor results.",
            "Markets remained stable throughout the day."
        ]
        
        scores = self.sentiment_analyzer.compute_sentiment_batch(test_texts)
        
        self.assertEqual(len(scores), len(test_texts))
        self.assertTrue(all(0 <= score <= 1 for score in scores))
        
        # Positive text should have higher score than negative
        pos_idx = 0  # Index of positive text
        neg_idx = 1  # Index of negative text
        self.assertGreater(scores[pos_idx], scores[neg_idx])

    def test_url_validation(self):
        """Test URL validation."""
        valid_urls = [
            "https://www.example.com",
            "http://example.com/path",
            "https://finance.yahoo.com"
        ]
        
        invalid_urls = [
            "not_a_url",
            "ftp://example.com",
            "",
            None,
            "http:/invalid.com"
        ]
        
        for url in valid_urls:
            self.assertTrue(
                self.web_scraper.is_valid_url(url),
                f"URL {url} should be valid"
            )
            
        for url in invalid_urls:
            self.assertFalse(
                self.web_scraper.is_valid_url(url),
                f"URL {url} should be invalid"
            )

    def test_text_scraping(self):
        """Test web scraping functionality."""
        # Note: This is a basic test with a well-known URL
        test_url = "https://example.com"
        
        text = self.web_scraper.scrape_text(test_url)
        
        self.assertIsInstance(text, str)
        self.assertTrue(len(text) > 0)
        self.assertLessEqual(
            len(text.split()),
            self.config.max_words,
            "Scraped text exceeds max_words limit"
        )

    def test_scraper_caching(self):
        """Test web scraper caching functionality."""
        test_url = "https://example.com"
        
        # First request
        text1 = self.web_scraper.scrape_text(test_url)
        
        # Second request (should use cache)
        text2 = self.web_scraper.scrape_text(test_url)
        
        self.assertEqual(text1, text2)
        self.assertIn(test_url, self.web_scraper.cache)

if __name__ == '__main__':
    unittest.main()