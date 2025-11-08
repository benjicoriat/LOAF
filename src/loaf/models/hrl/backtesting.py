"""
Backtesting utilities for the HRL system.
"""

import os
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Any
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc

from .networks import CustomActor, CustomCritic, Layer3ActorCritic
from .env import OptionsTradingEnv

def backtest_layer(
    models: Dict[str, Any],
    obs_df: pd.DataFrame,
    price_data: pd.DataFrame,
    test_dates: List[str],
    all_tickers: List[str],
    num_days: int = 10,
    save_dir: str = "./results",
    layer: int = 1
) -> pd.DataFrame:
    """
    Backtest layer 1 or 2 models.
    
    Args:
        models: Dictionary of trained models
        obs_df: DataFrame with observation vectors
        price_data: DataFrame with price data
        test_dates: List of test dates
        all_tickers: List of ticker symbols
        num_days: Option expiry horizon
        save_dir: Directory to save results
        layer: Layer number (1 or 2)
        
    Returns:
        DataFrame with backtest results
    """
    os.makedirs(save_dir, exist_ok=True)
    results_all = []
    
    for (seed, agent_type), model_dict in models.items():
        print(f"\n=== Backtesting Layer {layer}, {agent_type}, Seed {seed} ===")
        
        actor = model_dict['actor']
        actor.eval()
        
        # Create environment
        env = OptionsTradingEnv(
            all_tickers=all_tickers,
            obs_df=obs_df,
            price_data=price_data,
            train_dates=test_dates,
            num_days=num_days
        )
        
        obs = env.reset()
        done = False
        step = 0
        
        pnl_list = []
        wins_list = []
        losses_list = []
        partial_losses_list = []
        
        while not done:
            # Get action
            with torch.no_grad():
                action, _ = actor.get_action(torch.FloatTensor(obs).unsqueeze(0))
                action = action.squeeze(0).numpy()
            
            # Take step
            obs, reward, done, info = env.step(action)
            
            # Store metrics
            pnl_list.append(info['pnl'])
            wins_list.append(info['wins'])
            losses_list.append(info['losses'])
            partial_losses_list.append(info['partial_losses'])
            
            step += 1
        
        # Create results DataFrame
        df_result = pd.DataFrame({
            "Date": test_dates,
            "PnL": pnl_list,
            "Num_Wins": wins_list,
            "Num_Losses": losses_list,
            "Num_Partial_Losses": partial_losses_list
        })
        df_result["Agent"] = agent_type
        df_result["Seed"] = seed
        
        # Save results
        csv_path = os.path.join(save_dir, f"layer{layer}_{agent_type}_seed{seed}_backtest.csv")
        df_result.to_csv(csv_path, index=False)
        print(f"Saved backtest CSV: {csv_path}")
        
        # Plot results
        fig, axes = plt.subplots(4, 1, figsize=(16, 20), sharex=True)
        fig.suptitle(f"Layer {layer} - {agent_type} Seed {seed} Backtest", fontsize=16)
        
        # Cumulative PnL
        axes[0].plot(pd.to_datetime(test_dates), np.cumsum(pnl_list), label="Cumulative PnL")
        axes[0].set_title("Cumulative PnL")
        axes[0].grid(True)
        axes[0].legend()
        
        # Wins
        axes[1].plot(pd.to_datetime(test_dates), wins_list, color='green', label="Wins")
        axes[1].set_title("Number of Winning Trades")
        axes[1].grid(True)
        axes[1].legend()
        
        # Losses
        axes[2].plot(pd.to_datetime(test_dates), losses_list, color='red', label="Losses")
        axes[2].set_title("Number of Total Losses")
        axes[2].grid(True)
        axes[2].legend()
        
        # Partial Losses
        axes[3].plot(pd.to_datetime(test_dates), partial_losses_list, color='orange', label="Partial Losses")
        axes[3].set_title("Number of Partial Losses")
        axes[3].grid(True)
        axes[3].legend()
        
        plt.tight_layout()
        plot_path = os.path.join(save_dir, f"layer{layer}_{agent_type}_seed{seed}_backtest.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot: {plot_path}")
        
        results_all.append(df_result)
        
        # Cleanup
        del env, actor
        gc.collect()
        torch.cuda.empty_cache()
    
    # Combine all results
    df_all = pd.concat(results_all, ignore_index=True)
    all_results_path = os.path.join(save_dir, f"layer{layer}_all_backtests.csv")
    df_all.to_csv(all_results_path, index=False)
    print(f"\n✅ All backtests saved to {all_results_path}")
    
    return df_all

def backtest_layer3(
    models: Dict[int, Layer3ActorCritic],
    obs_df: pd.DataFrame,
    price_data: pd.DataFrame,
    test_dates: List[str],
    all_tickers: List[str],
    num_days: int = 10,
    save_dir: str = "./results/layer3"
) -> pd.DataFrame:
    """
    Backtest layer 3 models.
    
    Args:
        models: Dictionary of trained Layer3ActorCritic models
        obs_df: DataFrame with observation vectors
        price_data: DataFrame with price data
        test_dates: List of test dates
        all_tickers: List of ticker symbols
        num_days: Option expiry horizon
        save_dir: Directory to save results
        
    Returns:
        DataFrame with backtest results
    """
    os.makedirs(save_dir, exist_ok=True)
    results_all = []
    
    for seed, agent in models.items():
        print(f"\n=== Backtesting Layer 3, Seed {seed} ===")
        
        agent.eval()
        
        # Create environment
        env = OptionsTradingEnv(
            all_tickers=all_tickers,
            obs_df=obs_df,
            price_data=price_data,
            train_dates=test_dates,
            num_days=num_days
        )
        
        obs = env.reset()
        done = False
        step = 0
        
        pnl_list = []
        wins_list = []
        losses_list = []
        partial_losses_list = []
        
        while not done:
            # Get action
            with torch.no_grad():
                action, _ = agent.get_action(torch.FloatTensor(obs).unsqueeze(0))
                action = action.squeeze(0).numpy()
            
            # Take step
            obs, reward, done, info = env.step(action)
            
            # Store metrics
            pnl_list.append(info['pnl'])
            wins_list.append(info['wins'])
            losses_list.append(info['losses'])
            partial_losses_list.append(info['partial_losses'])
            
            step += 1
        
        # Create results DataFrame
        df_result = pd.DataFrame({
            "Date": test_dates,
            "PnL": pnl_list,
            "Num_Wins": wins_list,
            "Num_Losses": losses_list,
            "Num_Partial_Losses": partial_losses_list
        })
        df_result["Seed"] = seed
        
        # Save results
        csv_path = os.path.join(save_dir, f"layer3_seed{seed}_backtest.csv")
        df_result.to_csv(csv_path, index=False)
        print(f"Saved backtest CSV: {csv_path}")
        
        # Plot results
        fig, axes = plt.subplots(4, 1, figsize=(16, 20), sharex=True)
        fig.suptitle(f"Layer 3 - Seed {seed} Backtest", fontsize=16)
        
        # Plot metrics (similar to layer 1/2)
        axes[0].plot(pd.to_datetime(test_dates), np.cumsum(pnl_list), label="Cumulative PnL")
        axes[0].set_title("Cumulative PnL")
        axes[0].grid(True)
        axes[0].legend()
        
        axes[1].plot(pd.to_datetime(test_dates), wins_list, color='green', label="Wins")
        axes[1].set_title("Number of Winning Trades")
        axes[1].grid(True)
        axes[1].legend()
        
        axes[2].plot(pd.to_datetime(test_dates), losses_list, color='red', label="Losses")
        axes[2].set_title("Number of Total Losses")
        axes[2].grid(True)
        axes[2].legend()
        
        axes[3].plot(pd.to_datetime(test_dates), partial_losses_list, color='orange', label="Partial Losses")
        axes[3].set_title("Number of Partial Losses")
        axes[3].grid(True)
        axes[3].legend()
        
        plt.tight_layout()
        plot_path = os.path.join(save_dir, f"layer3_seed{seed}_backtest.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot: {plot_path}")
        
        results_all.append(df_result)
        
        # Cleanup
        del env
        gc.collect()
        torch.cuda.empty_cache()
    
    # Combine all results
    df_all = pd.concat(results_all, ignore_index=True)
    all_results_path = os.path.join(save_dir, "layer3_all_backtests.csv")
    df_all.to_csv(all_results_path, index=False)
    print(f"\n✅ All backtests saved to {all_results_path}")
    
    return df_all