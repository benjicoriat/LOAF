"""
NLP data processing module.
"""

import os
import gc
import time
import json
import torch
import pandas as pd
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from urllib.parse import urlparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import List, Dict, Tuple
from ..config.config import NLPConfig

class SentimentAnalyzer:
    """Handles sentiment analysis using FinBERT."""

    def __init__(self, config: NLPConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._setup_model()

    def _setup_model(self):
        """Initialize FinBERT model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.model.to(self.device)
        self.model.eval()

        # Get label mapping
        self.id2label = self.model.config.id2label
        self.label_to_idx = {v.lower(): k for k, v in self.id2label.items()}
        
        self.POS_IDX = self.label_to_idx.get('positive', 0)
        self.NEG_IDX = self.label_to_idx.get('negative', 1)
        self.NEUTRAL_IDX = self.label_to_idx.get('neutral', 2)

    def compute_sentiment_batch(self, texts: List[str]) -> List[float]:
        """Compute sentiment scores for a batch of texts."""
        if not texts:
            return []
        
        DEFAULT_SCORE = 0.5
        results = []

        try:
            # Filter empty texts
            valid_texts = [(i, t) for i, t in enumerate(texts) if t.strip()]
            if not valid_texts:
                return [DEFAULT_SCORE] * len(texts)
            
            indices, text_list = zip(*valid_texts)
            
            # Tokenize
            inputs = self.tokenizer(
                list(text_list),
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Compute sentiment
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                scores = probs[:, self.POS_IDX] - probs[:, self.NEG_IDX]
                normalized_scores = ((scores + 1) / 2).cpu().tolist()
            
            # Map back to original indices
            result_map = {idx: score for idx, score in zip(indices, normalized_scores)}
            results = [result_map.get(i, DEFAULT_SCORE) for i in range(len(texts))]
            
        except Exception as e:
            print(f"Batch sentiment computation failed: {e}")
            results = [DEFAULT_SCORE] * len(texts)
        
        return results

class WebScraper:
    """Handles web scraping with caching."""

    def __init__(self, config: NLPConfig):
        self.config = config
        self.session = self._setup_session()
        self.cache = {}

    def _setup_session(self) -> requests.Session:
        """Set up requests session with retry logic."""
        session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        session.mount('http://', HTTPAdapter(max_retries=retries))
        session.mount('https://', HTTPAdapter(max_retries=retries))
        return session

    def is_valid_url(self, url: str) -> bool:
        """Validate URL format and scheme."""
        if not isinstance(url, str) or not url.startswith("http"):
            return False
        try:
            parsed = urlparse(url)
            return parsed.scheme in ['http', 'https'] and bool(parsed.netloc)
        except:
            return False

    def scrape_text(self, url: str) -> str:
        """Scrape and process text from a URL with caching."""
        if url in self.cache:
            return self.cache[url]
        
        try:
            headers = {"User-Agent": self.config.user_agent}
            r = self.session.get(url, headers=headers, timeout=self.config.request_timeout)
            r.raise_for_status()
            
            soup = BeautifulSoup(r.text, "html.parser")
            
            # Remove unwanted elements
            for tag in soup(["script", "style", "nav", "header", "footer"]):
                tag.extract()
            
            # Try to find main content
            main_content = soup.find(['main', 'article', 'div'], class_=lambda x: x and any(
                word in str(x).lower() for word in ['content', 'main', 'article', 'body']
            ))
            
            if main_content:
                text = main_content.get_text(separator=" ", strip=True)
            else:
                text = soup.get_text(separator=" ", strip=True)
            
            # Limit words
            words = text.split()
            scraped = " ".join(words[:self.config.max_words])
            
            self.cache[url] = scraped
            time.sleep(self.config.request_delay)
            
            return scraped
            
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")
            self.cache[url] = ""
            return ""