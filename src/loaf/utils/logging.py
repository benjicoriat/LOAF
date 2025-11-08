"""
Core logging utilities for the LOAF trading system.
"""
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import logging
from typing import Dict, List, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns


class MetricsLogger:
    """
    Comprehensive logging system for all metrics and data points.
    """
    def __init__(self, base_dir: str, experiment_name: str = None):
        self.base_dir = base_dir
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(base_dir, "logs", self.experiment_name)
        self.metrics_dir = os.path.join(self.log_dir, "metrics")
        self.plots_dir = os.path.join(self.log_dir, "plots")
        self.data_dir = os.path.join(self.log_dir, "data")
        
        # Create directories
        for directory in [self.log_dir, self.metrics_dir, self.plots_dir, self.data_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Set up logging
        self._setup_logging()
        
        # Initialize metrics storage
        self.metrics = {
            "training": {},
            "evaluation": {},
            "market_data": {},
            "portfolio": [],
            "performance": []
        }
        
    def _setup_logging(self):
        """Configure logging system."""
        log_file = os.path.join(self.log_dir, "experiment.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(self.experiment_name)
        
    def log_market_data(self, data: pd.DataFrame, name: str):
        """Log market data with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(self.data_dir, f"{name}_{timestamp}.csv")
        data.to_csv(file_path)
        self.logger.info(f"Saved market data: {name} to {file_path}")
        
    def log_portfolio_state(self, state: Dict[str, Any], timestamp: str):
        """Log portfolio state and actions."""
        if "portfolio" not in self.metrics:
            self.metrics["portfolio"] = []
        
        self.metrics["portfolio"].append({
            "timestamp": timestamp,
            "state": state
        })
        
        # Save immediately to CSV
        df = pd.DataFrame(self.metrics["portfolio"])
        df.to_csv(os.path.join(self.metrics_dir, "portfolio_states.csv"))
        
    def log_training_metrics(self, metrics: Dict[str, float], step: int):
        """Log training metrics for each step."""
        for metric_name, value in metrics.items():
            if metric_name not in self.metrics["training"]:
                self.metrics["training"][metric_name] = []
            self.metrics["training"][metric_name].append({
                "step": step,
                "value": value
            })
        
        # Save training metrics
        self._save_metrics_to_csv("training")
        
    def log_evaluation_metrics(self, metrics: Dict[str, float], step: int):
        """Log evaluation metrics."""
        for metric_name, value in metrics.items():
            if metric_name not in self.metrics["evaluation"]:
                self.metrics["evaluation"][metric_name] = []
            self.metrics["evaluation"][metric_name].append({
                "step": step,
                "value": value
            })
        
        # Save evaluation metrics
        self._save_metrics_to_csv("evaluation")
        
    def log_performance_metrics(self, metrics: Dict[str, float], timestamp: str):
        """Log performance metrics with timestamp."""
        if "performance" not in self.metrics:
            self.metrics["performance"] = []
            
        metrics["timestamp"] = timestamp
        self.metrics["performance"].append(metrics)
        
        # Save performance metrics
        df = pd.DataFrame(self.metrics["performance"])
        df.to_csv(os.path.join(self.metrics_dir, "performance_metrics.csv"))
        
    def plot_metric(self, metric_name: str, metric_type: str = "training"):
        """Generate and save plot for a specific metric."""
        if metric_name not in self.metrics[metric_type]:
            self.logger.warning(f"Metric {metric_name} not found in {metric_type} metrics")
            return
            
        data = pd.DataFrame(self.metrics[metric_type][metric_name])
        
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=data, x="step", y="value")
        plt.title(f"{metric_type.capitalize()} - {metric_name}")
        plt.xlabel("Step")
        plt.ylabel(metric_name)
        
        # Save plot
        plot_path = os.path.join(self.plots_dir, f"{metric_type}_{metric_name}.png")
        plt.savefig(plot_path)
        plt.close()
        
    def plot_portfolio_performance(self, returns: pd.Series, benchmark_returns: pd.Series = None):
        """Plot cumulative portfolio performance vs benchmark."""
        cum_returns = (1 + returns).cumprod()
        
        plt.figure(figsize=(12, 6))
        plt.plot(cum_returns.index, cum_returns.values, label="Portfolio")
        
        if benchmark_returns is not None:
            cum_benchmark = (1 + benchmark_returns).cumprod()
            plt.plot(cum_benchmark.index, cum_benchmark.values, label="Benchmark")
            
        plt.title("Cumulative Performance")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plot_path = os.path.join(self.plots_dir, "cumulative_performance.png")
        plt.savefig(plot_path)
        plt.close()
        
    def plot_rolling_metrics(self, returns: pd.Series, window: int = 30):
        """Plot rolling Sharpe ratio, volatility, and returns."""
        rolling_ret = returns.rolling(window=window).mean() * 252  # Annualized
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
        rolling_sharpe = rolling_ret / rolling_vol
        
        # Create subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))
        
        # Plot rolling returns
        axes[0].plot(rolling_ret.index, rolling_ret.values)
        axes[0].set_title(f"{window}-day Rolling Returns (Annualized)")
        axes[0].grid(True)
        
        # Plot rolling volatility
        axes[1].plot(rolling_vol.index, rolling_vol.values)
        axes[1].set_title(f"{window}-day Rolling Volatility (Annualized)")
        axes[1].grid(True)
        
        # Plot rolling Sharpe ratio
        axes[2].plot(rolling_sharpe.index, rolling_sharpe.values)
        axes[2].set_title(f"{window}-day Rolling Sharpe Ratio")
        axes[2].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.plots_dir, "rolling_metrics.png")
        plt.savefig(plot_path)
        plt.close()
        
    def _save_metrics_to_csv(self, metric_type: str):
        """Save metrics to CSV files."""
        for metric_name, values in self.metrics[metric_type].items():
            df = pd.DataFrame(values)
            file_path = os.path.join(self.metrics_dir, f"{metric_type}_{metric_name}.csv")
            df.to_csv(file_path, index=False)
            
    def save_experiment_config(self, config: Dict[str, Any]):
        """Save experiment configuration."""
        config_path = os.path.join(self.log_dir, "experiment_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
            
    def log_error(self, error: Exception, context: str = ""):
        """Log errors with context."""
        self.logger.error(f"{context}: {str(error)}", exc_info=True)
        
    def log_warning(self, message: str):
        """Log warning messages."""
        self.logger.warning(message)
        
    def log_info(self, message: str):
        """Log informational messages."""
        self.logger.info(message)