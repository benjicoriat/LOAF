"""
Model interfaces and implementations.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any

class MarketModel(ABC):
    """Abstract base class for market models."""
    
    @abstractmethod
    def train(self, data: pd.DataFrame, **kwargs):
        """Train the model on market data."""
        pass
    
    @abstractmethod
    def predict(self, data: pd.DataFrame) -> Any:
        """Make predictions using the model."""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save model to disk."""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Load model from disk."""
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        pass
    
    @abstractmethod
    def set_params(self, params: Dict[str, Any]):
        """Set model parameters."""
        pass