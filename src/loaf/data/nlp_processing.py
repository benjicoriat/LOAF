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
import glob

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
        """Compute sentiment net scores for a batch of texts (normalized to [0,1]).

        This is a convenience wrapper that returns the normalized net score for each
        text by calling `compute_sentiment_probs` and extracting the 'net' value.
        """
        if not texts:
            return []

        probs_list = self.compute_sentiment_probs(texts)
        # Extract net scores and normalize from [-1,1] to [0,1]
        normalized = [((p['net'] + 1) / 2) if isinstance(p, dict) else 0.5 for p in probs_list]
        return normalized

    def compute_sentiment_probs(self, texts: List[str]) -> List[Dict[str, float]]:
        """Compute sentiment probability vectors for a batch of texts.

        Returns a list of dicts with keys: 'pos','neg','neu','net' in the same order as inputs.
        Blank or failed texts yield DEFAULT_SCORE probabilities and net=0.
        """
        DEFAULT_SCORE = 0.5
        results: List[Dict[str, float]] = []
        if not texts:
            return results

        try:
            valid_texts = [(i, t) for i, t in enumerate(texts) if isinstance(t, str) and t.strip()]
            indices, text_list = (zip(*valid_texts) if valid_texts else ([], []))

            prob_map = {}
            if valid_texts:
                inputs = self.tokenizer(
                    list(text_list),
                    truncation=True,
                    max_length=512,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=1).cpu().numpy()

                for idx, p in zip(indices, probs):
                    p_pos = float(p[self.POS_IDX])
                    p_neg = float(p[self.NEG_IDX])
                    p_neu = float(p[self.NEUTRAL_IDX])
                    net = p_pos - p_neg
                    prob_map[idx] = {"pos": p_pos, "neg": p_neg, "neu": p_neu, "net": net}

            for i, txt in enumerate(texts):
                if i in prob_map:
                    results.append(prob_map[i])
                else:
                    results.append({"pos": DEFAULT_SCORE, "neg": DEFAULT_SCORE, "neu": DEFAULT_SCORE, "net": 0.0})

        except Exception as e:
            print(f"Batch sentiment probability computation failed: {e}")
            results = [{"pos": DEFAULT_SCORE, "neg": DEFAULT_SCORE, "neu": DEFAULT_SCORE, "net": 0.0} for _ in texts]

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


def aggregate_term_sentiments(input_folder: str, output_mean: str, output_vol: str, default_score: float = 0.5):
    """
    Aggregate term-level sentiment CSVs into ticker-level per-period mean and volatility.

    - Expects CSV files in `input_folder` where each file corresponds to one ticker and
      has index = period start date (parseable) and columns = terms. Values are sentiment
      scores (floats) or empty strings.
    - Produces two CSVs:
      * `output_mean`: rows = periods (datetime), columns = tickers, values = mean sentiment
      * `output_vol`: rows = periods (datetime), columns = tickers, values = std dev across terms

    This function is robust to missing/empty cells. Periods with no valid terms get `default_score`.
    Returns (mean_df, vol_df).
    """
    files = glob.glob(os.path.join(input_folder, "*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {input_folder}")

    mean_frames = {}
    vol_frames = {}

    for path in files:
        ticker = os.path.basename(path).replace("_links.csv", "")
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
        except Exception:
            # Fallback: read without parsing then try to coerce index
            df = pd.read_csv(path, index_col=0)
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                # if index cannot be parsed, skip file
                continue

        # Coerce non-numeric to NaN
        df_numeric = df.apply(pd.to_numeric, errors='coerce')

        # Compute mean and std across terms for each period (row-wise)
        mean_series = df_numeric.mean(axis=1, skipna=True)
        vol_series = df_numeric.std(axis=1, skipna=True)

        # Where all terms were NaN, fill with default_score and vol=0
        all_nan_mask = mean_series.isna()
        if all_nan_mask.any():
            mean_series.loc[all_nan_mask] = default_score
            vol_series.loc[all_nan_mask] = 0.0

        mean_frames[ticker] = mean_series
        vol_frames[ticker] = vol_series

    # Combine into DataFrames (union of all dates)
    mean_df = pd.DataFrame(mean_frames).sort_index()
    vol_df = pd.DataFrame(vol_frames).sort_index()

    # Save outputs
    out_dir_mean = os.path.dirname(output_mean) or "."
    out_dir_vol = os.path.dirname(output_vol) or "."
    os.makedirs(out_dir_mean, exist_ok=True)
    os.makedirs(out_dir_vol, exist_ok=True)
    mean_df.to_csv(output_mean, index=True)
    vol_df.to_csv(output_vol, index=True)

    return mean_df, vol_df