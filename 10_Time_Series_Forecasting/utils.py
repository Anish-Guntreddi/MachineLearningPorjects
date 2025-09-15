"""
Utility functions for Time Series Forecasting
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple, Optional, Union
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings('ignore')


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 10, mode: str = 'min', delta: float = 0.0):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'min':
            self.is_better = lambda x, y: x < y - delta
        else:
            self.is_better = lambda x, y: x > y + delta
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def calculate_metrics(
    predictions: np.ndarray,
    targets: np.ndarray
) -> Dict[str, float]:
    """
    Calculate forecasting metrics
    
    Args:
        predictions: Predicted values
        targets: True values
    
    Returns:
        Dictionary of metrics
    """
    # Flatten arrays if needed
    if predictions.ndim > 2:
        predictions = predictions.reshape(-1, predictions.shape[-1])
    if targets.ndim > 2:
        targets = targets.reshape(-1, targets.shape[-1])
    
    metrics = {}
    
    # Basic metrics
    metrics['mae'] = mean_absolute_error(targets, predictions)
    metrics['mse'] = mean_squared_error(targets, predictions)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['r2'] = r2_score(targets, predictions)
    
    # MAPE (Mean Absolute Percentage Error)
    mask = targets != 0
    if mask.any():
        mape = np.mean(np.abs((targets[mask] - predictions[mask]) / targets[mask])) * 100
        metrics['mape'] = mape
    else:
        metrics['mape'] = 0.0
    
    # sMAPE (Symmetric MAPE)
    denominator = (np.abs(targets) + np.abs(predictions)) / 2
    mask = denominator != 0
    if mask.any():
        smape = np.mean(np.abs(targets[mask] - predictions[mask]) / denominator[mask]) * 100
        metrics['smape'] = smape
    else:
        metrics['smape'] = 0.0
    
    return metrics


def plot_predictions(
    targets: np.ndarray,
    predictions: np.ndarray,
    n_samples: int = 5,
    save_path: Optional[str] = None
):
    """Plot predictions vs targets"""
    n_samples = min(n_samples, len(targets))
    
    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 3 * n_samples))
    if n_samples == 1:
        axes = [axes]
    
    for i in range(n_samples):
        target = targets[i].squeeze()
        pred = predictions[i].squeeze()
        
        axes[i].plot(target, label='True', color='blue', linewidth=2)
        axes[i].plot(pred, label='Predicted', color='red', linewidth=2, alpha=0.7)
        axes[i].set_title(f'Sample {i+1}')
        axes[i].set_xlabel('Time Step')
        axes[i].set_ylabel('Value')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    val_maes: List[float],
    val_mapes: List[float],
    save_path: Optional[str] = None
):
    """Plot comprehensive training curves"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # MAE plot
    ax2.plot(epochs, val_maes, 'g-', label='Val MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.set_title('Validation MAE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # MAPE plot
    ax3.plot(epochs, val_mapes, 'm-', label='Val MAPE')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('MAPE (%)')
    ax3.set_title('Validation MAPE')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Combined metrics
    ax4_twin = ax4.twinx()
    ax4.plot(epochs, val_losses, 'r-', label='Loss')
    ax4_twin.plot(epochs, val_mapes, 'm-', label='MAPE')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss', color='r')
    ax4_twin.set_ylabel('MAPE (%)', color='m')
    ax4.set_title('All Validation Metrics')
    ax4.tick_params(axis='y', labelcolor='r')
    ax4_twin.tick_params(axis='y', labelcolor='m')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def plot_time_series(
    data: Union[pd.Series, np.ndarray],
    title: str = 'Time Series',
    save_path: Optional[str] = None
):
    """Plot time series data"""
    plt.figure(figsize=(14, 6))
    
    if isinstance(data, pd.Series):
        plt.plot(data.index, data.values, linewidth=1)
        plt.xlabel('Time')
    else:
        plt.plot(data, linewidth=1)
        plt.xlabel('Time Step')
    
    plt.ylabel('Value')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def plot_decomposition(
    data: pd.Series,
    period: Optional[int] = None,
    save_path: Optional[str] = None
):
    """Plot time series decomposition"""
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    if period is None:
        # Try to infer period
        if hasattr(data.index, 'freq'):
            if data.index.freq == 'D':
                period = 7  # Weekly seasonality
            elif data.index.freq == 'H':
                period = 24  # Daily seasonality
            else:
                period = 12  # Default
        else:
            period = 12
    
    decomposition = seasonal_decompose(data, model='additive', period=period)
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    
    data.plot(ax=axes[0], title='Original Time Series')
    axes[0].set_ylabel('Value')
    axes[0].grid(True, alpha=0.3)
    
    decomposition.trend.plot(ax=axes[1], title='Trend Component')
    axes[1].set_ylabel('Trend')
    axes[1].grid(True, alpha=0.3)
    
    decomposition.seasonal.plot(ax=axes[2], title='Seasonal Component')
    axes[2].set_ylabel('Seasonal')
    axes[2].grid(True, alpha=0.3)
    
    decomposition.resid.plot(ax=axes[3], title='Residual Component')
    axes[3].set_ylabel('Residual')
    axes[3].set_xlabel('Time')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def plot_acf_pacf(
    data: Union[pd.Series, np.ndarray],
    lags: int = 40,
    save_path: Optional[str] = None
):
    """Plot ACF and PACF"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # ACF
    acf_values = acf(data, nlags=lags)
    ax1.bar(range(len(acf_values)), acf_values)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.axhline(y=1.96/np.sqrt(len(data)), color='red', linestyle='--', linewidth=0.5)
    ax1.axhline(y=-1.96/np.sqrt(len(data)), color='red', linestyle='--', linewidth=0.5)
    ax1.set_xlabel('Lag')
    ax1.set_ylabel('ACF')
    ax1.set_title('Autocorrelation Function')
    ax1.grid(True, alpha=0.3)
    
    # PACF
    try:
        pacf_values = pacf(data, nlags=lags)
        ax2.bar(range(len(pacf_values)), pacf_values)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.axhline(y=1.96/np.sqrt(len(data)), color='red', linestyle='--', linewidth=0.5)
        ax2.axhline(y=-1.96/np.sqrt(len(data)), color='red', linestyle='--', linewidth=0.5)
        ax2.set_xlabel('Lag')
        ax2.set_ylabel('PACF')
        ax2.set_title('Partial Autocorrelation Function')
        ax2.grid(True, alpha=0.3)
    except:
        ax2.text(0.5, 0.5, 'PACF calculation failed', ha='center', va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def check_stationarity(
    data: Union[pd.Series, np.ndarray],
    significance_level: float = 0.05
) -> Dict:
    """
    Check stationarity using Augmented Dickey-Fuller test
    
    Args:
        data: Time series data
        significance_level: Significance level for test
    
    Returns:
        Dictionary with test results
    """
    result = adfuller(data, autolag='AIC')
    
    return {
        'adf_statistic': result[0],
        'p_value': result[1],
        'used_lag': result[2],
        'n_obs': result[3],
        'critical_values': result[4],
        'is_stationary': result[1] < significance_level
    }


def inverse_transform_predictions(
    predictions: np.ndarray,
    scaler: object
) -> np.ndarray:
    """
    Inverse transform scaled predictions
    
    Args:
        predictions: Scaled predictions
        scaler: Fitted scaler object
    
    Returns:
        Original scale predictions
    """
    original_shape = predictions.shape
    
    # Reshape for scaler
    if predictions.ndim == 3:
        # (batch, time, features)
        batch_size, time_steps, n_features = predictions.shape
        predictions = predictions.reshape(-1, n_features)
    elif predictions.ndim == 2:
        time_steps, n_features = predictions.shape
    else:
        n_features = 1
        predictions = predictions.reshape(-1, 1)
    
    # Inverse transform
    predictions = scaler.inverse_transform(predictions)
    
    # Reshape back
    predictions = predictions.reshape(original_shape)
    
    return predictions


def create_lagged_features(
    data: pd.DataFrame,
    target_col: str,
    lag_steps: List[int] = [1, 2, 3, 7, 14, 30]
) -> pd.DataFrame:
    """
    Create lagged features for time series
    
    Args:
        data: Input DataFrame
        target_col: Target column name
        lag_steps: List of lag steps
    
    Returns:
        DataFrame with lagged features
    """
    df = data.copy()
    
    for lag in lag_steps:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    
    # Rolling statistics
    for window in [7, 14, 30]:
        df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
        df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
    
    # Drop NaN rows
    df = df.dropna()
    
    return df


def calculate_prediction_intervals(
    predictions: np.ndarray,
    residuals: np.ndarray,
    confidence: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate prediction intervals
    
    Args:
        predictions: Point predictions
        residuals: Model residuals
        confidence: Confidence level
    
    Returns:
        Lower and upper bounds
    """
    from scipy import stats
    
    # Calculate standard deviation of residuals
    std_residuals = np.std(residuals)
    
    # Calculate z-score for confidence level
    z_score = stats.norm.ppf((1 + confidence) / 2)
    
    # Calculate intervals
    lower_bound = predictions - z_score * std_residuals
    upper_bound = predictions + z_score * std_residuals
    
    return lower_bound, upper_bound


def plot_forecast_with_intervals(
    historical: np.ndarray,
    forecast: np.ndarray,
    lower_bound: Optional[np.ndarray] = None,
    upper_bound: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
):
    """Plot forecast with prediction intervals"""
    plt.figure(figsize=(14, 6))
    
    n_historical = len(historical)
    n_forecast = len(forecast)
    
    # Plot historical data
    plt.plot(range(n_historical), historical, label='Historical', color='blue', linewidth=2)
    
    # Plot forecast
    forecast_range = range(n_historical - 1, n_historical + n_forecast - 1)
    plt.plot(forecast_range, np.concatenate([[historical[-1]], forecast[:-1]]), 
             label='Forecast', color='red', linewidth=2)
    
    # Plot prediction intervals if provided
    if lower_bound is not None and upper_bound is not None:
        plt.fill_between(
            forecast_range,
            np.concatenate([[historical[-1]], lower_bound[:-1]]),
            np.concatenate([[historical[-1]], upper_bound[:-1]]),
            alpha=0.3,
            color='red',
            label='95% Confidence Interval'
        )
    
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('Time Series Forecast')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def detect_outliers(
    data: np.ndarray,
    method: str = 'iqr',
    threshold: float = 1.5
) -> np.ndarray:
    """
    Detect outliers in time series
    
    Args:
        data: Time series data
        method: Detection method ('iqr', 'zscore', 'isolation_forest')
        threshold: Threshold for outlier detection
    
    Returns:
        Boolean array indicating outliers
    """
    if method == 'iqr':
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = (data < lower_bound) | (data > upper_bound)
        
    elif method == 'zscore':
        z_scores = np.abs((data - np.mean(data)) / np.std(data))
        outliers = z_scores > threshold
        
    elif method == 'isolation_forest':
        from sklearn.ensemble import IsolationForest
        clf = IsolationForest(contamination=0.1, random_state=42)
        outliers = clf.fit_predict(data.reshape(-1, 1)) == -1
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return outliers


if __name__ == "__main__":
    # Test utilities
    print("Testing time series utilities...")
    
    # Generate dummy data
    n_points = 100
    t = np.linspace(0, 10, n_points)
    data = np.sin(t) + np.random.randn(n_points) * 0.1
    
    # Test stationarity check
    stationarity_result = check_stationarity(data)
    print(f"Stationarity test p-value: {stationarity_result['p_value']:.4f}")
    print(f"Is stationary: {stationarity_result['is_stationary']}")
    
    # Test metrics calculation
    predictions = data + np.random.randn(n_points) * 0.1
    metrics = calculate_metrics(predictions.reshape(-1, 1), data.reshape(-1, 1))
    print(f"Metrics: {metrics}")
    
    # Test outlier detection
    outliers = detect_outliers(data, method='zscore', threshold=3)
    print(f"Number of outliers detected: {np.sum(outliers)}")
    
    # Test early stopping
    early_stop = EarlyStopping(patience=3, mode='min')
    scores = [0.5, 0.45, 0.46, 0.47, 0.48]
    for i, score in enumerate(scores):
        if early_stop(score):
            print(f"Early stopping at iteration {i+1}")
            break