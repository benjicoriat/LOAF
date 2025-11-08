"""
Utils for HRL pipeline data aggregation and processing.
"""

import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict

def aggregate_layer1_actions(
    results_dir: str,
    test_dates: List[str],
    output_dir: str = "./layer2/data"
) -> str:
    """
    Aggregate actions from layer 1 models for layer 2 input.
    
    Args:
        results_dir: Directory containing layer 1 backtest results
        test_dates: List of test dates
        output_dir: Directory to save aggregated actions
        
    Returns:
        Path to aggregated actions CSV
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all CSV files
    all_dfs = []
    for file in os.listdir(results_dir):
        if file.endswith("_backtest.csv"):
            df = pd.read_csv(os.path.join(results_dir, file))
            df['File'] = file
            all_dfs.append(df)
    
    # Combine all actions
    combined_df = pd.concat(all_dfs)
    
    # Pivot to get actions as features
    action_cols = [col for col in combined_df.columns if 'action' in col.lower()]
    pivoted = combined_df.pivot(
        index='Date',
        columns=['Agent', 'Seed'],
        values=action_cols
    )
    
    # Flatten column names
    pivoted.columns = [f"action_{agent}_{seed}_{col}" for (agent, seed, col) in pivoted.columns]
    
    # Save
    output_path = os.path.join(output_dir, "layer1_actions.csv")
    pivoted.to_csv(output_path)
    print(f"✅ Saved aggregated layer 1 actions to {output_path}")
    
    return output_path

def aggregate_layer2_actions(
    results_dir: str,
    test_dates: List[str],
    output_dir: str = "./layer3/data"
) -> str:
    """
    Aggregate actions from layer 2 models for layer 3 input.
    
    Args:
        results_dir: Directory containing layer 2 backtest results
        test_dates: List of test dates
        output_dir: Directory to save aggregated actions
        
    Returns:
        Path to aggregated actions CSV
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all CSV files
    all_dfs = []
    for file in os.listdir(results_dir):
        if file.endswith("_backtest.csv"):
            df = pd.read_csv(os.path.join(results_dir, file))
            df['File'] = file
            all_dfs.append(df)
    
    # Combine all actions
    combined_df = pd.concat(all_dfs)
    
    # Pivot to get actions as features
    action_cols = [col for col in combined_df.columns if 'action' in col.lower()]
    pivoted = combined_df.pivot(
        index='Date',
        columns='Seed',
        values=action_cols
    )
    
    # Flatten column names
    pivoted.columns = [f"action_seed{seed}_{col}" for (seed, col) in pivoted.columns]
    
    # Save
    output_path = os.path.join(output_dir, "layer2_actions.csv")
    pivoted.to_csv(output_path)
    print(f"✅ Saved aggregated layer 2 actions to {output_path}")
    
    return output_path

def normalize_aggregated_actions(
    input_path: str,
    output_dir: str = None
) -> pd.DataFrame:
    """
    Normalize aggregated actions by first column value.
    
    Args:
        input_path: Path to input CSV
        output_dir: Optional directory to save normalized actions
        
    Returns:
        DataFrame with normalized actions
    """
    df = pd.read_csv(input_path, index_col=0)
    
    # Normalize by first column
    first_col = df.iloc[:, 0].replace(0, 1e-8)
    normalized_df = df.div(first_col, axis=0)
    normalized_df.iloc[:, 0] = 1.0
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "normalized_actions.csv")
        normalized_df.to_csv(output_path)
        print(f"✅ Saved normalized actions to {output_path}")
    
    return normalized_df