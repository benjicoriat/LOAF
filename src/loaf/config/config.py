"""
Configuration management for the LOAF system.
"""

from dataclasses import dataclass
from typing import Dict, List
from datetime import datetime

"""
Configuration management for the LOAF system.
"""

from dataclasses import dataclass
from typing import Dict, List
from datetime import datetime


@dataclass
class MarketConfig:
    """Market universe configuration."""
    tickers: Dict[str, List[str]] = None
    date_ranges: Dict[str, Dict[str, str]] = None
    api_key: str = None

    @classmethod
    def default(cls):
        return cls(
            tickers={
                "Equity Indices": ["SPY", "DIA", "QQQ", "IWM", "^FCHI", "^GDAXI", "^FTSE",
                                 "^N225", "^STOXX50E", "^HSI"],
                "REITs": ["VNQ", "SCHH", "IYR"],
                "Commodities": ["GLD", "SLV", "USO", "UNG", "DBA", "PALL", "CPER"],
                "Cryptocurrencies": ["BTC-USD", "ETH-USD"],
                "Bonds/Defensive": ["TLT", "IEF", "SHY"]
            },
            date_ranges={
                "Download": {"start": "2017-01-01", "end": datetime.now().strftime("%Y-%m-%d")},
                "Training": {"start": "2017-01-01", "end": "2019-12-31"},
                "Testing": {"start": "2020-01-01", "end": "2023-12-31"}
            }
        )


@dataclass
class NLPConfig:
    """NLP processing configuration."""
    max_words: int = 500
    request_timeout: int = 10
    request_delay: float = 1.0
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    batch_size: int = 8


@dataclass
class SystemConfig:
    """Global system configuration."""
    market: MarketConfig
    nlp: NLPConfig
    base_dir: str = "./data"