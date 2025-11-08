"""
Main pipeline for the hierarchical reinforcement learning system.
"""

import os
from typing import Dict, List
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from .training import train_layer1, train_layer2, train_layer3
from .backtesting import backtest_layer, backtest_layer3
from .data_utils import (
    aggregate_layer1_actions,
    aggregate_layer2_actions,
    normalize_aggregated_actions
)

class HRLPipeline:
    """
    Hierarchical Reinforcement Learning Pipeline for options trading.
    
    Implements a 3-layer system:
    1. Layer 1: Base models on quantitative features
    2. Layer 2: Models on aggregated Layer 1 actions
    3. Layer 3: Attention-based model on Layer 2 actions
    """
    
    def __init__(
        self,
        base_dir: str = "./hrl_pipeline",
        seeds: List[int] = [1,2,3,4,5],
        agents_list: List[str] = ["PPO", "TD3", "SAC"],
        num_days: int = 10
    ):
        """
        Initialize the pipeline.
        
        Args:
            base_dir: Base directory for all pipeline data
            seeds: List of random seeds
            agents_list: List of agent types for Layer 1
            num_days: Option expiry horizon
        """
        self.base_dir = base_dir
        self.seeds = seeds
        self.agents_list = agents_list
        self.num_days = num_days
        
        # Create directories
        self.dirs = {
            'layer1': {
                'models': os.path.join(base_dir, 'layer1/models'),
                'results': os.path.join(base_dir, 'layer1/results'),
                'data': os.path.join(base_dir, 'layer1/data')
            },
            'layer2': {
                'models': os.path.join(base_dir, 'layer2/models'),
                'results': os.path.join(base_dir, 'layer2/results'),
                'data': os.path.join(base_dir, 'layer2/data')
            },
            'layer3': {
                'models': os.path.join(base_dir, 'layer3/models'),
                'results': os.path.join(base_dir, 'layer3/results'),
                'data': os.path.join(base_dir, 'layer3/data')
            }
        }
        
        for layer_dirs in self.dirs.values():
            for d in layer_dirs.values():
                os.makedirs(d, exist_ok=True)
        
        self.models = {
            'layer1': None,
            'layer2': None,
            'layer3': None
        }
    
    def train_layer1(
        self,
        obs_df: pd.DataFrame,
        price_data: pd.DataFrame,
        train_dates: List[str],
        all_tickers: List[str]
    ) -> Dict:
        """
        Train Layer 1 models on quantitative features.
        """
        print("\n=== Training Layer 1 Models ===")
        
        self.models['layer1'] = train_layer1(
            obs_df=obs_df,
            price_data=price_data,
            train_dates=train_dates,
            all_tickers=all_tickers,
            seeds=self.seeds,
            agents_list=self.agents_list,
            num_days=self.num_days,
            save_dir=self.dirs['layer1']['models']
        )
        
        return self.models['layer1']
    
    def backtest_layer1(
        self,
        obs_df: pd.DataFrame,
        price_data: pd.DataFrame,
        test_dates: List[str],
        all_tickers: List[str]
    ) -> pd.DataFrame:
        """
        Backtest Layer 1 models and aggregate actions.
        """
        print("\n=== Backtesting Layer 1 Models ===")
        
        results = backtest_layer(
            models=self.models['layer1'],
            obs_df=obs_df,
            price_data=price_data,
            test_dates=test_dates,
            all_tickers=all_tickers,
            num_days=self.num_days,
            save_dir=self.dirs['layer1']['results'],
            layer=1
        )
        
        # Aggregate actions for Layer 2
        aggregated_path = aggregate_layer1_actions(
            results_dir=self.dirs['layer1']['results'],
            test_dates=test_dates,
            output_dir=self.dirs['layer2']['data']
        )
        
        # Normalize actions
        self.layer2_obs = normalize_aggregated_actions(
            input_path=aggregated_path,
            output_dir=self.dirs['layer2']['data']
        )
        
        return results
    
    def train_layer2(
        self,
        price_data: pd.DataFrame,
        train_dates: List[str],
        all_tickers: List[str]
    ) -> Dict:
        """
        Train Layer 2 models on aggregated Layer 1 actions.
        """
        print("\n=== Training Layer 2 Models ===")
        
        self.models['layer2'] = train_layer2(
            obs_df=self.layer2_obs,
            price_data=price_data,
            train_dates=train_dates,
            all_tickers=all_tickers,
            seeds=self.seeds,
            num_days=self.num_days,
            save_dir=self.dirs['layer2']['models']
        )
        
        return self.models['layer2']
    
    def backtest_layer2(
        self,
        price_data: pd.DataFrame,
        test_dates: List[str],
        all_tickers: List[str]
    ) -> pd.DataFrame:
        """
        Backtest Layer 2 models and aggregate actions.
        """
        print("\n=== Backtesting Layer 2 Models ===")
        
        results = backtest_layer(
            models=self.models['layer2'],
            obs_df=self.layer2_obs,
            price_data=price_data,
            test_dates=test_dates,
            all_tickers=all_tickers,
            num_days=self.num_days,
            save_dir=self.dirs['layer2']['results'],
            layer=2
        )
        
        # Aggregate actions for Layer 3
        aggregated_path = aggregate_layer2_actions(
            results_dir=self.dirs['layer2']['results'],
            test_dates=test_dates,
            output_dir=self.dirs['layer3']['data']
        )
        
        # Normalize actions
        self.layer3_obs = normalize_aggregated_actions(
            input_path=aggregated_path,
            output_dir=self.dirs['layer3']['data']
        )
        
        return results
    
    def train_layer3(
        self,
        price_data: pd.DataFrame,
        train_dates: List[str],
        all_tickers: List[str]
    ) -> Dict:
        """
        Train Layer 3 models on aggregated Layer 2 actions.
        """
        print("\n=== Training Layer 3 Models ===")
        
        self.models['layer3'] = train_layer3(
            obs_df=self.layer3_obs,
            price_data=price_data,
            train_dates=train_dates,
            all_tickers=all_tickers,
            seeds=self.seeds,
            num_days=self.num_days,
            save_dir=self.dirs['layer3']['models']
        )
        
        return self.models['layer3']
    
    def backtest_layer3(
        self,
        price_data: pd.DataFrame,
        test_dates: List[str],
        all_tickers: List[str]
    ) -> pd.DataFrame:
        """
        Backtest Layer 3 models.
        """
        print("\n=== Backtesting Layer 3 Models ===")
        
        results = backtest_layer3(
            models=self.models['layer3'],
            obs_df=self.layer3_obs,
            price_data=price_data,
            test_dates=test_dates,
            all_tickers=all_tickers,
            num_days=self.num_days,
            save_dir=self.dirs['layer3']['results']
        )
        
        return results
    
    def run_full_pipeline(
        self,
        obs_df: pd.DataFrame,
        price_data: pd.DataFrame,
        train_dates: List[str],
        test_dates: List[str],
        all_tickers: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """
        Run the complete HRL pipeline.
        
        Args:
            obs_df: Initial observation DataFrame
            price_data: Price data DataFrame
            train_dates: List of training dates
            test_dates: List of test dates
            all_tickers: List of ticker symbols
            
        Returns:
            Dictionary of results for each layer
        """
        results = {}
        
        # Layer 1
        self.train_layer1(obs_df, price_data, train_dates, all_tickers)
        results['layer1'] = self.backtest_layer1(obs_df, price_data, test_dates, all_tickers)
        
        # Layer 2
        self.train_layer2(price_data, train_dates, all_tickers)
        results['layer2'] = self.backtest_layer2(price_data, test_dates, all_tickers)
        
        # Layer 3
        self.train_layer3(price_data, train_dates, all_tickers)
        results['layer3'] = self.backtest_layer3(price_data, test_dates, all_tickers)
        
        return results