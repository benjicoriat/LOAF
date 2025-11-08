"""
Model evaluation and metrics tracking.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc
)
from ..utils.performance import PerformanceAnalytics
from ..utils.logging import MetricsLogger


class ModelEvaluator:
    """
    Comprehensive model evaluation system.
    """
    def __init__(self, logger: MetricsLogger):
        self.logger = logger
        self.performance_analytics = PerformanceAnalytics()
        
    def evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray,
                           task_type: str = 'regression') -> Dict[str, float]:
        """Evaluate model predictions."""
        metrics = {}
        
        if task_type == 'regression':
            # Regression metrics
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2'] = r2_score(y_true, y_pred)
            
            # Additional regression metrics
            metrics['mape'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            metrics['smape'] = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100
            
        elif task_type == 'classification':
            # Classification metrics
            metrics['accuracy'] = np.mean(y_true == y_pred)
            report = classification_report(y_true, y_pred, output_dict=True)
            metrics.update({f"{k}_precision": v['precision'] 
                          for k, v in report.items() if k not in ['accuracy', 'macro avg', 'weighted avg']})
            metrics.update({f"{k}_recall": v['recall']
                          for k, v in report.items() if k not in ['accuracy', 'macro avg', 'weighted avg']})
            metrics.update({f"{k}_f1": v['f1-score']
                          for k, v in report.items() if k not in ['accuracy', 'macro avg', 'weighted avg']})
            
        # Log metrics
        self.logger.log_evaluation_metrics(metrics, step=0)
        return metrics
    
    def evaluate_trading_performance(self, returns: pd.Series, 
                                  benchmark_returns: Optional[pd.Series] = None,
                                  risk_free_rate: float = 0.02) -> Dict[str, float]:
        """Evaluate trading strategy performance."""
        metrics = self.performance_analytics.generate_performance_summary(
            returns, benchmark_returns, risk_free_rate)
        
        # Log performance metrics
        self.logger.log_performance_metrics(metrics, timestamp=pd.Timestamp.now().isoformat())
        return metrics
    
    def evaluate_portfolio_attribution(self, returns: pd.DataFrame, 
                                    weights: pd.DataFrame) -> pd.DataFrame:
        """Perform portfolio performance attribution analysis."""
        # Calculate asset contributions
        asset_contribution = returns * weights.shift(1)
        
        attribution = pd.DataFrame({
            'total_return': returns.mean() * 252,  # Annualized return
            'contribution': asset_contribution.mean() * 252,  # Annualized contribution
            'weight': weights.mean(),  # Average weight
            'volatility': returns.std() * np.sqrt(252)  # Annualized volatility
        })
        
        # Calculate contribution to risk
        cov_matrix = returns.cov() * 252  # Annualized covariance
        portfolio_risk = np.sqrt(weights.mean().dot(cov_matrix).dot(weights.mean()))
        
        for asset in returns.columns:
            asset_weight = weights[asset].mean()
            asset_risk_contribution = (
                asset_weight * (cov_matrix[asset].dot(weights.mean())) / portfolio_risk
            )
            attribution.loc[asset, 'risk_contribution'] = asset_risk_contribution
            
        # Log attribution analysis
        self.logger.log_market_data(attribution, "portfolio_attribution")
        return attribution
    
    def evaluate_risk_adjusted_metrics(self, returns: pd.Series,
                                    risk_free_rate: float = 0.02) -> Dict[str, float]:
        """Calculate risk-adjusted performance metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['annualized_return'] = self.performance_analytics.calculate_annualized_return(returns)
        metrics['annualized_volatility'] = self.performance_analytics.calculate_volatility(returns)
        metrics['sharpe_ratio'] = self.performance_analytics.calculate_sharpe_ratio(returns, risk_free_rate)
        metrics['sortino_ratio'] = self.performance_analytics.calculate_sortino_ratio(returns, risk_free_rate)
        
        # Drawdown metrics
        max_dd, peak, valley = self.performance_analytics.calculate_max_drawdown(returns)
        metrics['max_drawdown'] = max_dd
        metrics['max_drawdown_duration'] = (valley - peak).days
        metrics['calmar_ratio'] = self.performance_analytics.calculate_calmar_ratio(returns)
        
        # Additional risk metrics
        metrics['var_95'] = self.performance_analytics.calculate_var(returns)
        metrics['cvar_95'] = self.performance_analytics.calculate_cvar(returns)
        metrics['omega_ratio'] = self.performance_analytics.calculate_omega_ratio(returns)
        
        # Trading metrics
        metrics['win_rate'] = self.performance_analytics.calculate_win_rate(returns)
        metrics['profit_factor'] = self.performance_analytics.calculate_profit_factor(returns)
        
        # Statistical metrics
        metrics['skewness'] = self.performance_analytics.calculate_skewness(returns)
        metrics['kurtosis'] = self.performance_analytics.calculate_kurtosis(returns)
        
        # Log risk-adjusted metrics
        self.logger.log_performance_metrics(metrics, timestamp=pd.Timestamp.now().isoformat())
        return metrics
    
    def evaluate_model_robustness(self, model_predictions: List[np.ndarray],
                                y_true: np.ndarray) -> Dict[str, float]:
        """Evaluate model robustness across multiple runs."""
        predictions = np.array(model_predictions)
        
        metrics = {
            'mean_prediction': predictions.mean(axis=0),
            'std_prediction': predictions.std(axis=0),
            'prediction_interval_95': np.percentile(predictions, [2.5, 97.5], axis=0)
        }
        
        # Calculate consistency metrics
        metrics['prediction_consistency'] = np.mean(np.std(predictions, axis=0))
        metrics['directional_consistency'] = np.mean(
            np.sign(predictions - predictions.mean(axis=0)).std(axis=0)
        )
        
        # Calculate error metrics for mean prediction
        mean_pred = metrics['mean_prediction']
        metrics['mean_rmse'] = np.sqrt(mean_squared_error(y_true, mean_pred))
        metrics['mean_mae'] = mean_absolute_error(y_true, mean_pred)
        
        # Log robustness metrics
        self.logger.log_evaluation_metrics(metrics, step=0)
        return metrics
    
    def evaluate_feature_importance(self, feature_importance: np.ndarray,
                                 feature_names: List[str]) -> pd.Series:
        """Evaluate and rank feature importance."""
        importance = pd.Series(feature_importance, index=feature_names)
        importance = importance.sort_values(ascending=False)
        
        # Log feature importance
        self.logger.log_market_data(importance.to_frame('importance'), "feature_importance")
        return importance
    
    def evaluate_online_performance(self, predictions: pd.Series,
                                 actuals: pd.Series,
                                 window: int = 60) -> pd.DataFrame:
        """Evaluate online learning performance."""
        online_metrics = pd.DataFrame(index=predictions.index)
        
        # Calculate rolling metrics
        online_metrics['rmse'] = np.sqrt(
            (predictions - actuals).rolling(window).apply(lambda x: np.mean(x**2))
        )
        online_metrics['mae'] = (predictions - actuals).abs().rolling(window).mean()
        online_metrics['bias'] = (predictions - actuals).rolling(window).mean()
        
        # Calculate directional accuracy
        pred_dir = predictions.diff().fillna(0)
        actual_dir = actuals.diff().fillna(0)
        online_metrics['directional_accuracy'] = (
            (pred_dir * actual_dir > 0).rolling(window).mean()
        )
        
        # Log online metrics
        self.logger.log_market_data(online_metrics, "online_performance")
        return online_metrics