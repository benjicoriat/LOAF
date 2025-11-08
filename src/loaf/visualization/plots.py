"""
Visualization utilities for market data and model performance.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from ..utils.performance import PerformanceAnalytics
import mplfinance as mpf


class Visualizer:
    """
    Base class for visualization utilities.
    """
    def __init__(self, style: str = 'seaborn'):
        plt.style.use(style)
        self.performance_analytics = PerformanceAnalytics()
        
    def save_plot(self, path: str):
        """Save current plot to file."""
        plt.savefig(path, bbox_inches='tight', dpi=300)
        plt.close()


class MarketVisualizer(Visualizer):
    """
    Visualization utilities for market data.
    """
    def plot_price_history(self, data: pd.DataFrame, ticker: str, save_path: Optional[str] = None):
        """Plot price history with volume."""
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.03, subplot_titles=(f'{ticker} Price', 'Volume'),
                           row_heights=[0.7, 0.3])
        
        # Price candlestick
        fig.add_trace(
            go.Candlestick(x=data.index,
                          open=data['Open'],
                          high=data['High'],
                          low=data['Low'],
                          close=data['Close'],
                          name='OHLC'),
            row=1, col=1
        )
        
        # Volume bars
        colors = ['red' if row['Close'] < row['Open'] else 'green' 
                 for i, row in data.iterrows()]
        fig.add_trace(
            go.Bar(x=data.index, y=data['Volume'],
                  marker_color=colors,
                  name='Volume'),
            row=2, col=1
        )
        
        fig.update_layout(
            title=f'{ticker} Price History',
            yaxis_title='Price',
            yaxis2_title='Volume',
            xaxis_rangeslider_visible=False
        )
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()
            
    def plot_returns_distribution(self, returns: pd.Series, ticker: str, 
                                save_path: Optional[str] = None):
        """Plot returns distribution with normal curve."""
        plt.figure(figsize=(10, 6))
        
        # Plot histogram
        sns.histplot(returns, kde=True, stat='density')
        
        # Add normal distribution
        mu = returns.mean()
        sigma = returns.std()
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
        plt.plot(x, 1/(sigma * np.sqrt(2 * np.pi)) * 
                np.exp(-(x - mu)**2 / (2 * sigma**2)), 
                'r-', lw=2, label='Normal Distribution')
        
        plt.title(f'{ticker} Returns Distribution')
        plt.xlabel('Returns')
        plt.ylabel('Density')
        plt.legend()
        
        if save_path:
            self.save_plot(save_path)
        else:
            plt.show()
            
    def plot_rolling_metrics(self, returns: pd.Series, ticker: str, window: int = 30,
                           save_path: Optional[str] = None):
        """Plot rolling metrics (returns, volatility, Sharpe ratio)."""
        metrics = self.performance_analytics.calculate_rolling_metrics(returns, window)
        
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                           subplot_titles=('Rolling Returns', 'Rolling Volatility', 
                                         'Rolling Sharpe Ratio'))
        
        fig.add_trace(
            go.Scatter(x=metrics.index, y=metrics['Returns'],
                      name='Returns'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=metrics.index, y=metrics['Volatility'],
                      name='Volatility'),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=metrics.index, y=metrics['Sharpe'],
                      name='Sharpe Ratio'),
            row=3, col=1
        )
        
        fig.update_layout(
            title=f'{ticker} Rolling Metrics (Window: {window} days)',
            height=900
        )
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()
            
    def plot_correlation_matrix(self, returns: pd.DataFrame, save_path: Optional[str] = None):
        """Plot correlation matrix of returns."""
        plt.figure(figsize=(12, 8))
        
        corr = returns.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        sns.heatmap(corr, mask=mask, center=0, cmap='coolwarm',
                   annot=True, fmt='.2f', square=True)
        
        plt.title('Asset Returns Correlation Matrix')
        
        if save_path:
            self.save_plot(save_path)
        else:
            plt.show()
            
    def plot_drawdown(self, returns: pd.Series, ticker: str, save_path: Optional[str] = None):
        """Plot drawdown chart."""
        drawdown = self.performance_analytics.calculate_drawdown(returns)
        
        plt.figure(figsize=(12, 6))
        plt.plot(drawdown.index, drawdown.values * 100)
        plt.fill_between(drawdown.index, drawdown.values * 100, 0, alpha=0.3)
        
        plt.title(f'{ticker} Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        
        if save_path:
            self.save_plot(save_path)
        else:
            plt.show()


class PortfolioVisualizer(Visualizer):
    """
    Visualization utilities for portfolio analysis.
    """
    def plot_portfolio_composition(self, weights: Dict[str, float], 
                                 save_path: Optional[str] = None):
        """Plot portfolio composition as pie chart."""
        plt.figure(figsize=(10, 8))
        
        plt.pie(weights.values(), labels=weights.keys(), autopct='%1.1f%%',
                startangle=90)
        plt.title('Portfolio Composition')
        
        if save_path:
            self.save_plot(save_path)
        else:
            plt.show()
            
    def plot_portfolio_performance(self, portfolio_returns: pd.Series, 
                                 benchmark_returns: Optional[pd.Series] = None,
                                 save_path: Optional[str] = None):
        """Plot cumulative portfolio performance vs benchmark."""
        portfolio_cum = (1 + portfolio_returns).cumprod()
        
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_cum.index, portfolio_cum.values, 
                label='Portfolio', linewidth=2)
        
        if benchmark_returns is not None:
            benchmark_cum = (1 + benchmark_returns).cumprod()
            plt.plot(benchmark_cum.index, benchmark_cum.values,
                    label='Benchmark', linewidth=2, alpha=0.7)
            
        plt.title('Cumulative Performance')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            self.save_plot(save_path)
        else:
            plt.show()
            
    def plot_rolling_beta(self, portfolio_returns: pd.Series, 
                         market_returns: pd.Series, window: int = 60,
                         save_path: Optional[str] = None):
        """Plot rolling beta to market."""
        rolling_beta = pd.Series(index=portfolio_returns.index[window:])
        
        for i in range(window, len(portfolio_returns)):
            port_window = portfolio_returns.iloc[i-window:i]
            market_window = market_returns.iloc[i-window:i]
            rolling_beta.iloc[i-window] = self.performance_analytics.calculate_beta(
                port_window, market_window)
            
        plt.figure(figsize=(12, 6))
        plt.plot(rolling_beta.index, rolling_beta.values)
        plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)
        
        plt.title(f'Rolling Beta (Window: {window} days)')
        plt.xlabel('Date')
        plt.ylabel('Beta')
        plt.grid(True)
        
        if save_path:
            self.save_plot(save_path)
        else:
            plt.show()


class ModelVisualizer(Visualizer):
    """
    Visualization utilities for model performance and analysis.
    """
    def plot_training_history(self, history: Dict[str, List[float]], 
                            save_path: Optional[str] = None):
        """Plot training metrics history."""
        plt.figure(figsize=(12, 6))
        
        for metric, values in history.items():
            plt.plot(values, label=metric)
            
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Metric Value')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            self.save_plot(save_path)
        else:
            plt.show()
            
    def plot_prediction_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray,
                                dates: pd.DatetimeIndex, save_path: Optional[str] = None):
        """Plot model predictions vs actual values."""
        plt.figure(figsize=(12, 6))
        
        plt.plot(dates, y_true, label='Actual', alpha=0.7)
        plt.plot(dates, y_pred, label='Predicted', alpha=0.7)
        
        plt.title('Model Predictions vs Actual Values')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            self.save_plot(save_path)
        else:
            plt.show()
            
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            labels: List[str], save_path: Optional[str] = None):
        """Plot confusion matrix for classification tasks."""
        cm = pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted'])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        
        plt.title('Confusion Matrix')
        
        if save_path:
            self.save_plot(save_path)
        else:
            plt.show()
            
    def plot_feature_importance(self, feature_importance: pd.Series,
                              save_path: Optional[str] = None):
        """Plot feature importance."""
        plt.figure(figsize=(12, 6))
        
        feature_importance.sort_values(ascending=True).plot(kind='barh')
        
        plt.title('Feature Importance')
        plt.xlabel('Importance Score')
        
        if save_path:
            self.save_plot(save_path)
        else:
            plt.show()