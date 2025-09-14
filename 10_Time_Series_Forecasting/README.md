# Time Series Forecasting Project - M4 / ETT Datasets

## 1. Problem Definition & Use Case

**Problem:** Predict future values of sequential data points collected over time, accounting for trends, seasonality, cyclical patterns, and irregular fluctuations while handling multiple variables, different frequencies, and complex dependencies.

**Use Case:** Time series forecasting enables critical decision-making across industries:
- **Finance**: Stock price prediction, algorithmic trading, risk management
- **Supply Chain**: Demand forecasting, inventory optimization, logistics planning
- **Energy**: Load forecasting, renewable energy prediction, grid management
- **Healthcare**: Disease outbreak prediction, patient monitoring, resource allocation
- **Weather**: Climate modeling, disaster prediction, agricultural planning  
- **Manufacturing**: Predictive maintenance, quality control, production planning
- **Retail**: Sales forecasting, pricing optimization, customer behavior prediction
- **IoT/Sensors**: Anomaly detection, predictive maintenance, smart city applications

**Business Impact:** Accurate forecasting reduces inventory costs by 30%, improves demand planning accuracy by 25%, and enables proactive decision-making that saves companies millions annually through optimized operations.

## 2. Dataset Acquisition & Preprocessing

### Primary Datasets
- **M4 Competition**: 100,000 diverse time series across different domains and frequencies
  ```python
  import pandas as pd
  from datasetsforecast.m4 import M4
  
  # Load M4 dataset
  Y_df, X_df, S_df = M4.load(directory='./data/', group='Daily')
  ```
- **Electricity Transformer Temperature (ETT)**: Multivariate time series with hourly data
  ```python
  import pandas as pd
  
  # Load ETT dataset
  ett_h1 = pd.read_csv('https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv')
  ett_h1['date'] = pd.to_datetime(ett_h1['date'])
  ```
- **Weather Dataset**: Multi-location weather time series
  ```python
  weather_df = pd.read_csv('weather.csv', parse_dates=['date'])
  ```
- **Stock Market Data**: Financial time series with high frequency
  ```python
  import yfinance as yf
  
  # Download stock data
  stock_data = yf.download('AAPL', start='2010-01-01', end='2024-01-01')
  ```

### Data Schema
```python
{
    'timestamp': pd.Timestamp,     # Time index
    'unique_id': str,              # Series identifier  
    'target': float,               # Value to forecast
    'features': {                  # Additional features
        'feature_1': float,
        'feature_2': float,
        'categorical_1': str,
    },
    'frequency': str,              # Data frequency (D, H, M, etc.)
    'seasonality': int,            # Seasonal period length
    'horizon': int,                # Forecast horizon
}
```

### Preprocessing Pipeline
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesPreprocessor:
    def __init__(self, target_col='y', timestamp_col='ds', freq='D'):
        self.target_col = target_col
        self.timestamp_col = timestamp_col
        self.freq = freq
        self.scalers = {}
        self.outlier_bounds = {}
        
    def clean_and_validate(self, df):
        """Clean and validate time series data"""
        df = df.copy()
        
        # Convert timestamp column
        if self.timestamp_col in df.columns:
            df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])
            df = df.set_index(self.timestamp_col)
        
        # Sort by timestamp
        df = df.sort_index()
        
        # Check for missing timestamps and fill gaps
        full_range = pd.date_range(
            start=df.index.min(), 
            end=df.index.max(), 
            freq=self.freq
        )
        df = df.reindex(full_range)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Remove or fix outliers
        df = self.handle_outliers(df)
        
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values in time series"""
        # Linear interpolation for numerical columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isna().sum() > 0:
                # Use linear interpolation for small gaps
                df[col] = df[col].interpolate(method='linear', limit=5)
                
                # Forward fill remaining gaps
                df[col] = df[col].fillna(method='ffill', limit=3)
                
                # Backward fill if needed
                df[col] = df[col].fillna(method='bfill', limit=3)
        
        # Mode imputation for categorical columns
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_columns:
            if df[col].isna().sum() > 0:
                mode_value = df[col].mode()[0] if not df[col].mode().empty else 'unknown'
                df[col] = df[col].fillna(mode_value)
        
        return df
    
    def handle_outliers(self, df, method='iqr', threshold=3):
        """Detect and handle outliers"""
        for col in df.select_dtypes(include=[np.number]).columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                lower_bound = df[col].mean() - threshold * df[col].std()
                upper_bound = df[col].mean() + threshold * df[col].std()
            
            self.outlier_bounds[col] = (lower_bound, upper_bound)
            
            # Cap outliers instead of removing them to preserve temporal structure
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df
    
    def add_time_features(self, df):
        """Add time-based features"""
        time_features = pd.DataFrame(index=df.index)
        
        # Basic time features
        time_features['year'] = df.index.year
        time_features['month'] = df.index.month
        time_features['day'] = df.index.day
        time_features['dayofweek'] = df.index.dayofweek
        time_features['dayofyear'] = df.index.dayofyear
        time_features['quarter'] = df.index.quarter
        
        # Cyclical encoding for periodic features
        time_features['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
        time_features['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
        time_features['day_sin'] = np.sin(2 * np.pi * df.index.day / 31)
        time_features['day_cos'] = np.cos(2 * np.pi * df.index.day / 31)
        time_features['dayofweek_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        time_features['dayofweek_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        
        # Holiday indicators (simplified)
        time_features['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
        time_features['is_month_start'] = df.index.is_month_start.astype(int)
        time_features['is_month_end'] = df.index.is_month_end.astype(int)
        
        return pd.concat([df, time_features], axis=1)
    
    def add_lag_features(self, df, lags=[1, 2, 3, 7, 14, 30]):
        """Add lagged features"""
        for lag in lags:
            df[f'{self.target_col}_lag_{lag}'] = df[self.target_col].shift(lag)
        
        return df
    
    def add_rolling_features(self, df, windows=[3, 7, 14, 30]):
        """Add rolling window features"""
        for window in windows:
            # Rolling statistics
            df[f'{self.target_col}_rolling_mean_{window}'] = df[self.target_col].rolling(window).mean()
            df[f'{self.target_col}_rolling_std_{window}'] = df[self.target_col].rolling(window).std()
            df[f'{self.target_col}_rolling_min_{window}'] = df[self.target_col].rolling(window).min()
            df[f'{self.target_col}_rolling_max_{window}'] = df[self.target_col].rolling(window).max()
            
            # Exponentially weighted features
            df[f'{self.target_col}_ewm_{window}'] = df[self.target_col].ewm(span=window).mean()
        
        return df
    
    def detect_seasonality(self, df, max_period=365):
        """Detect seasonal patterns using autocorrelation"""
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        # Remove trend for seasonality detection
        try:
            decomposition = seasonal_decompose(
                df[self.target_col].dropna(), 
                model='additive', 
                period=min(max_period, len(df) // 2)
            )
            
            seasonal_strength = np.var(decomposition.seasonal) / np.var(decomposition.resid + decomposition.seasonal)
            return seasonal_strength > 0.6
            
        except Exception:
            return False
    
    def difference_series(self, df, periods=1):
        """Apply differencing to achieve stationarity"""
        df_diff = df.copy()
        df_diff[f'{self.target_col}_diff_{periods}'] = df[self.target_col].diff(periods=periods)
        return df_diff
    
    def normalize_features(self, df, method='standard'):
        """Normalize numerical features"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            
            # Fit and transform
            df_values = df[[col]].values
            df_scaled = scaler.fit_transform(df_values)
            df[f'{col}_normalized'] = df_scaled.flatten()
            
            # Store scaler for inverse transformation
            self.scalers[col] = scaler
        
        return df
    
    def prepare_sequences(self, df, seq_length=60, forecast_horizon=1, stride=1):
        """Prepare sequences for deep learning models"""
        sequences = []
        targets = []
        
        for i in range(0, len(df) - seq_length - forecast_horizon + 1, stride):
            # Input sequence
            seq = df.iloc[i:i + seq_length].values
            
            # Target (next forecast_horizon values)
            target = df[self.target_col].iloc[i + seq_length:i + seq_length + forecast_horizon].values
            
            if len(target) == forecast_horizon:  # Ensure complete target
                sequences.append(seq)
                targets.append(target)
        
        return np.array(sequences), np.array(targets)

# Preprocessing pipeline usage
def preprocess_pipeline(df, target_col='y', timestamp_col='ds'):
    """Complete preprocessing pipeline"""
    preprocessor = TimeSeriesPreprocessor(target_col, timestamp_col)
    
    # 1. Clean and validate
    df_clean = preprocessor.clean_and_validate(df)
    
    # 2. Add time features
    df_features = preprocessor.add_time_features(df_clean)
    
    # 3. Add lag features
    df_features = preprocessor.add_lag_features(df_features)
    
    # 4. Add rolling features
    df_features = preprocessor.add_rolling_features(df_features)
    
    # 5. Handle stationarity
    if not preprocessor.detect_seasonality(df_features):
        df_features = preprocessor.difference_series(df_features)
    
    # 6. Normalize features
    df_normalized = preprocessor.normalize_features(df_features)
    
    return df_normalized, preprocessor
```

### Feature Engineering
- **Temporal features**: Day of week, month, season, holidays
- **Lag features**: Previous values at different time steps
- **Rolling statistics**: Moving averages, standard deviations, min/max
- **Seasonal decomposition**: Trend, seasonal, and residual components
- **External variables**: Weather, economic indicators, events

## 3. Baseline Models

### ARIMA Models
```python
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

class ARIMAForecaster:
    def __init__(self, order=(1, 1, 1), seasonal_order=None):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None
        
    def check_stationarity(self, timeseries):
        """Check if time series is stationary using ADF test"""
        result = adfuller(timeseries.dropna())
        
        print('ADF Statistic:', result[0])
        print('p-value:', result[1])
        print('Critical Values:')
        for key, value in result[4].items():
            print(f'\t{key}: {value}')
        
        return result[1] <= 0.05  # p-value <= 0.05 means stationary
    
    def find_optimal_order(self, timeseries, max_p=5, max_d=2, max_q=5):
        """Find optimal ARIMA order using AIC/BIC"""
        from itertools import product
        
        best_aic = np.inf
        best_order = None
        best_bic = np.inf
        
        # Grid search over parameters
        for p, d, q in product(range(max_p+1), range(max_d+1), range(max_q+1)):
            try:
                model = ARIMA(timeseries, order=(p, d, q))
                fitted_model = model.fit()
                
                aic = fitted_model.aic
                bic = fitted_model.bic
                
                if aic < best_aic:
                    best_aic = aic
                    best_order = (p, d, q)
                    best_bic = bic
                    
            except Exception as e:
                continue
        
        print(f'Best ARIMA order: {best_order}')
        print(f'AIC: {best_aic:.2f}, BIC: {best_bic:.2f}')
        
        return best_order
    
    def fit(self, timeseries, auto_order=False):
        """Fit ARIMA model"""
        if auto_order:
            self.order = self.find_optimal_order(timeseries)
        
        # Fit model
        self.model = ARIMA(timeseries, order=self.order, seasonal_order=self.seasonal_order)
        self.fitted_model = self.model.fit()
        
        # Model diagnostics
        print(self.fitted_model.summary())
        
        return self.fitted_model
    
    def forecast(self, steps=1, confidence_interval=True):
        """Generate forecasts"""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before forecasting")
        
        # Generate forecast
        forecast_result = self.fitted_model.forecast(steps=steps)
        
        if confidence_interval:
            conf_int = self.fitted_model.get_forecast(steps=steps).conf_int()
            return forecast_result, conf_int
        
        return forecast_result
    
    def plot_diagnostics(self):
        """Plot model diagnostics"""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Residuals
        residuals = self.fitted_model.resid
        
        # Plot 1: Residuals
        axes[0, 0].plot(residuals)
        axes[0, 0].set_title('Residuals')
        
        # Plot 2: Residuals distribution
        axes[0, 1].hist(residuals, bins=30)
        axes[0, 1].set_title('Residuals Distribution')
        
        # Plot 3: ACF of residuals
        plot_acf(residuals.dropna(), ax=axes[1, 0], lags=20)
        axes[1, 0].set_title('ACF of Residuals')
        
        # Plot 4: PACF of residuals
        plot_pacf(residuals.dropna(), ax=axes[1, 1], lags=20)
        axes[1, 1].set_title('PACF of Residuals')
        
        plt.tight_layout()
        plt.show()

# Seasonal ARIMA (SARIMA)
class SARIMAForecaster(ARIMAForecaster):
    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
        super().__init__(order, seasonal_order)
    
    def find_optimal_seasonal_order(self, timeseries, season_period=12, max_P=2, max_D=1, max_Q=2):
        """Find optimal seasonal ARIMA parameters"""
        from itertools import product
        
        best_aic = np.inf
        best_seasonal_order = None
        
        for P, D, Q in product(range(max_P+1), range(max_D+1), range(max_Q+1)):
            try:
                seasonal_order = (P, D, Q, season_period)
                model = ARIMA(timeseries, order=self.order, seasonal_order=seasonal_order)
                fitted_model = model.fit()
                
                if fitted_model.aic < best_aic:
                    best_aic = fitted_model.aic
                    best_seasonal_order = seasonal_order
                    
            except Exception as e:
                continue
        
        print(f'Best seasonal order: {best_seasonal_order}')
        print(f'AIC: {best_aic:.2f}')
        
        return best_seasonal_order

# Example usage
def train_arima_model(df, target_col='y', forecast_steps=30):
    """Train and evaluate ARIMA model"""
    
    # Initialize forecaster
    arima = ARIMAForecaster()
    
    # Check stationarity
    is_stationary = arima.check_stationarity(df[target_col])
    print(f"Series is stationary: {is_stationary}")
    
    # Find optimal parameters
    optimal_order = arima.find_optimal_order(df[target_col])
    arima.order = optimal_order
    
    # Split data
    train_size = int(len(df) * 0.8)
    train_data = df[target_col][:train_size]
    test_data = df[target_col][train_size:]
    
    # Fit model
    fitted_model = arima.fit(train_data)
    
    # Generate forecasts
    forecasts, conf_int = arima.forecast(steps=len(test_data), confidence_interval=True)
    
    # Calculate metrics
    mse = np.mean((test_data.values - forecasts) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(test_data.values - forecasts))
    
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    return arima, forecasts, conf_int
```
**Expected Performance:** MAPE 10-20% on seasonal data, good interpretability

### Exponential Smoothing
```python
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing

class ExponentialSmoothingForecaster:
    def __init__(self, trend=None, seasonal=None, seasonal_periods=None):
        self.trend = trend  # None, 'add', 'mul'
        self.seasonal = seasonal  # None, 'add', 'mul'
        self.seasonal_periods = seasonal_periods
        self.model = None
        self.fitted_model = None
    
    def fit(self, timeseries):
        """Fit exponential smoothing model"""
        
        # Determine model type based on data characteristics
        if self.seasonal_periods and len(timeseries) >= 2 * self.seasonal_periods:
            # Use Holt-Winters method for seasonal data
            self.model = ExponentialSmoothing(
                timeseries,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods
            )
        else:
            # Use simpler exponential smoothing
            if self.trend:
                # Holt's method (double exponential smoothing)
                self.model = ExponentialSmoothing(timeseries, trend=self.trend)
            else:
                # Simple exponential smoothing
                self.model = ExponentialSmoothing(timeseries)
        
        # Fit the model
        self.fitted_model = self.model.fit(optimized=True, use_boxcox=False)
        
        print("Model fitted successfully")
        print(f"AIC: {self.fitted_model.aic:.2f}")
        print(f"BIC: {self.fitted_model.bic:.2f}")
        
        return self.fitted_model
    
    def forecast(self, steps=1):
        """Generate forecasts"""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before forecasting")
        
        forecast = self.fitted_model.forecast(steps=steps)
        return forecast
    
    def plot_components(self, figsize=(15, 8)):
        """Plot model components"""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        
        fig = self.fitted_model.plot_components(figsize=figsize)
        plt.tight_layout()
        plt.show()

# Auto ETS model selection
class AutoETSForecaster:
    def __init__(self):
        self.best_model = None
        self.best_params = None
        self.best_aic = np.inf
    
    def find_best_model(self, timeseries, seasonal_periods=None):
        """Automatically select best ETS model"""
        
        trend_options = [None, 'add', 'mul']
        seasonal_options = [None, 'add', 'mul'] if seasonal_periods else [None]
        
        results = []
        
        for trend in trend_options:
            for seasonal in seasonal_options:
                try:
                    model = ExponentialSmoothing(
                        timeseries,
                        trend=trend,
                        seasonal=seasonal,
                        seasonal_periods=seasonal_periods
                    )
                    fitted = model.fit(optimized=True)
                    
                    results.append({
                        'trend': trend,
                        'seasonal': seasonal,
                        'aic': fitted.aic,
                        'bic': fitted.bic,
                        'model': fitted
                    })
                    
                except Exception as e:
                    continue
        
        # Select best model based on AIC
        if results:
            best_result = min(results, key=lambda x: x['aic'])
            self.best_model = best_result['model']
            self.best_params = {
                'trend': best_result['trend'],
                'seasonal': best_result['seasonal']
            }
            self.best_aic = best_result['aic']
            
            print(f"Best model: Trend={self.best_params['trend']}, Seasonal={self.best_params['seasonal']}")
            print(f"AIC: {self.best_aic:.2f}")
            
            return self.best_model
        else:
            raise ValueError("No suitable model found")
    
    def forecast(self, steps=1):
        """Generate forecasts using best model"""
        if self.best_model is None:
            raise ValueError("Must find best model first")
        
        return self.best_model.forecast(steps=steps)

# Example usage
def train_exponential_smoothing(df, target_col='y', seasonal_periods=12):
    """Train exponential smoothing model"""
    
    # Split data
    train_size = int(len(df) * 0.8)
    train_data = df[target_col][:train_size]
    test_data = df[target_col][train_size:]
    
    # Auto model selection
    auto_ets = AutoETSForecaster()
    fitted_model = auto_ets.find_best_model(train_data, seasonal_periods=seasonal_periods)
    
    # Generate forecasts
    forecasts = auto_ets.forecast(steps=len(test_data))
    
    # Calculate metrics
    mse = np.mean((test_data.values - forecasts) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(test_data.values - forecasts))
    
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    return auto_ets, forecasts
```
**Expected Performance:** MAPE 8-15% on trending/seasonal data, fast training

## 4. Advanced/Stretch Models

### Deep Learning Approaches

1. **LSTM/GRU Networks**
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim=50, num_layers=2, output_dim=1, dropout=0.2):
        super(LSTMForecaster, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism (optional)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        
        # LSTM forward pass
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Apply attention (optional)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use last output for prediction
        final_out = attn_out[:, -1, :]
        
        # Final prediction
        prediction = self.fc(final_out)
        
        return prediction

class GRUForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim=50, num_layers=2, output_dim=1, dropout=0.2):
        super(GRUForecaster, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # GRU layers
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        
        # GRU forward pass
        gru_out, hn = self.gru(x, h0)
        
        # Use last output
        final_out = gru_out[:, -1, :]
        
        # Final prediction
        prediction = self.fc(final_out)
        
        return prediction

# Training function
class TimeSeriesTrainer:
    def __init__(self, model, learning_rate=0.001, weight_decay=1e-5):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, train_loader, val_loader, epochs=100):
        """Full training loop"""
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.scheduler.step(val_loss)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        return train_losses, val_losses
```

2. **Transformer-based Models**
```python
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=8, num_layers=6, 
                 dim_feedforward=256, dropout=0.1, output_dim=1):
        super(TimeSeriesTransformer, self).__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )
        
    def forward(self, src, src_mask=None):
        # Input projection and positional encoding
        src = self.input_projection(src) * np.sqrt(self.d_model)
        src = self.pos_encoding(src)
        
        # Transformer encoding
        encoded = self.transformer_encoder(src, src_mask)
        
        # Use last timestep for prediction
        output = self.output_projection(encoded[:, -1, :])
        
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)
```

3. **N-BEATS (Neural Basis Expansion Analysis)**
```python
class NBeatsBlock(nn.Module):
    def __init__(self, input_size, theta_size, basis_function, layers, layer_size):
        super().__init__()
        
        # Fully connected stack
        self.fc_stack = nn.ModuleList([nn.Linear(input_size, layer_size)] + 
                                     [nn.Linear(layer_size, layer_size) for _ in range(layers - 1)])
        
        # Basis parameters
        self.basis_function = basis_function
        self.backcast_linear = nn.Linear(layer_size, theta_size)
        self.forecast_linear = nn.Linear(layer_size, theta_size)
        
    def forward(self, x):
        # FC stack
        for layer in self.fc_stack:
            x = torch.relu(layer(x))
        
        # Generate basis parameters
        theta_backcast = self.backcast_linear(x)
        theta_forecast = self.forecast_linear(x)
        
        # Apply basis function
        backcast = self.basis_function(theta_backcast)
        forecast = self.basis_function(theta_forecast)
        
        return backcast, forecast

class NBeats(nn.Module):
    def __init__(self, input_size, output_size, stack_types=['trend', 'seasonality', 'generic'],
                 num_blocks=3, layers=4, layer_size=256):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.stacks = nn.ModuleList()
        
        for stack_type in stack_types:
            if stack_type == 'trend':
                basis_function = TrendBasis(input_size, output_size)
                theta_size = 4  # polynomial degree + 1
            elif stack_type == 'seasonality':
                basis_function = SeasonalityBasis(input_size, output_size)
                theta_size = input_size
            else:  # generic
                basis_function = GenericBasis(input_size, output_size)
                theta_size = input_size + output_size
            
            blocks = nn.ModuleList([
                NBeatsBlock(input_size, theta_size, basis_function, layers, layer_size)
                for _ in range(num_blocks)
            ])
            self.stacks.append(blocks)
    
    def forward(self, x):
        forecast = torch.zeros(x.size(0), self.output_size).to(x.device)
        
        for stack in self.stacks:
            for block in stack:
                backcast, block_forecast = block(x)
                x = x - backcast
                forecast = forecast + block_forecast
        
        return forecast

# Basis functions
class TrendBasis(nn.Module):
    def __init__(self, backcast_length, forecast_length, degree=3):
        super().__init__()
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.degree = degree
        
    def forward(self, theta):
        backcast_time = torch.linspace(0, 1, self.backcast_length).to(theta.device)
        forecast_time = torch.linspace(1, 1 + self.forecast_length/self.backcast_length, 
                                      self.forecast_length).to(theta.device)
        
        backcast = self.polynomial_basis(backcast_time, theta)
        forecast = self.polynomial_basis(forecast_time, theta)
        
        return backcast, forecast
    
    def polynomial_basis(self, time, theta):
        basis = torch.stack([time**i for i in range(self.degree + 1)], dim=1)
        return torch.sum(basis * theta.unsqueeze(-1), dim=1)
```

4. **Prophet-inspired Deep Learning Model**
```python
class DeepProphet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_seasonalities=3):
        super(DeepProphet, self).__init__()
        
        # Trend component
        self.trend_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Seasonal components
        self.seasonal_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(num_seasonalities)
        ])
        
        # Holiday/event component
        self.holiday_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Uncertainty estimation
        self.uncertainty_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Ensure positive uncertainty
        )
    
    def forward(self, x):
        # Decompose into components
        trend = self.trend_net(x)
        
        seasonal = torch.zeros_like(trend)
        for seasonal_net in self.seasonal_nets:
            seasonal += seasonal_net(x)
        
        holiday = self.holiday_net(x)
        uncertainty = self.uncertainty_net(x)
        
        # Combine components
        prediction = trend + seasonal + holiday
        
        return prediction, uncertainty
```

**Target Performance:** MAPE < 5% on complex seasonal patterns, handling multiple time series

## 5. Training Details

### Data Preparation
```python
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import TimeSeriesSplit

class TimeSeriesDataPreparation:
    def __init__(self, sequence_length=60, forecast_horizon=1, features=None):
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.features = features
        
    def create_sequences(self, df, target_col, feature_cols=None):
        """Create sequences for supervised learning"""
        sequences = []
        targets = []
        
        # Select features
        if feature_cols is None:
            feature_cols = [col for col in df.columns if col != target_col]
        
        # Create sequences
        for i in range(len(df) - self.sequence_length - self.forecast_horizon + 1):
            # Input sequence (features)
            seq_features = df[feature_cols].iloc[i:i + self.sequence_length].values
            
            # Target sequence
            target_seq = df[target_col].iloc[
                i + self.sequence_length:i + self.sequence_length + self.forecast_horizon
            ].values
            
            if len(target_seq) == self.forecast_horizon:
                sequences.append(seq_features)
                targets.append(target_seq)
        
        return np.array(sequences), np.array(targets)
    
    def train_val_test_split(self, sequences, targets, train_ratio=0.7, val_ratio=0.15):
        """Split data temporally"""
        n = len(sequences)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_seq, train_targets = sequences[:train_end], targets[:train_end]
        val_seq, val_targets = sequences[train_end:val_end], targets[train_end:val_end]
        test_seq, test_targets = sequences[val_end:], targets[val_end:]
        
        return (train_seq, train_targets), (val_seq, val_targets), (test_seq, test_targets)
    
    def create_dataloaders(self, train_data, val_data, test_data, batch_size=32):
        """Create PyTorch dataloaders"""
        train_dataset = TimeSeriesDataset(*train_data)
        val_dataset = TimeSeriesDataset(*val_data)
        test_dataset = TimeSeriesDataset(*test_data)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader

# Cross-validation for time series
class TimeSeriesCrossValidator:
    def __init__(self, n_splits=5, test_size=None):
        self.n_splits = n_splits
        self.test_size = test_size
        
    def split(self, df):
        """Time series cross-validation splits"""
        tscv = TimeSeriesSplit(n_splits=self.n_splits, test_size=self.test_size)
        
        splits = []
        for train_idx, test_idx in tscv.split(df):
            train_data = df.iloc[train_idx]
            test_data = df.iloc[test_idx]
            splits.append((train_data, test_data))
        
        return splits
```

### Training Configuration
```python
training_config = {
    # Model architecture
    'model_type': 'lstm',  # 'lstm', 'gru', 'transformer', 'nbeats'
    'input_dim': 10,
    'hidden_dim': 128,
    'num_layers': 3,
    'dropout': 0.2,
    
    # Data configuration
    'sequence_length': 60,
    'forecast_horizon': 1,
    'batch_size': 64,
    
    # Training parameters
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'num_epochs': 200,
    'early_stopping_patience': 20,
    'gradient_clip_norm': 1.0,
    
    # Regularization
    'dropout_rate': 0.2,
    'l1_lambda': 0.0,
    'l2_lambda': 1e-4,
    
    # Scheduler
    'scheduler_type': 'reduce_lr_on_plateau',
    'scheduler_patience': 10,
    'scheduler_factor': 0.5,
    
    # Loss function
    'loss_function': 'mse',  # 'mse', 'mae', 'quantile'
    'loss_weights': None,
}

class AdvancedTimeSeriesTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
        # Loss function
        if config['loss_function'] == 'mse':
            self.criterion = nn.MSELoss()
        elif config['loss_function'] == 'mae':
            self.criterion = nn.L1Loss()
        elif config['loss_function'] == 'quantile':
            self.criterion = QuantileLoss()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Scheduler
        if config['scheduler_type'] == 'reduce_lr_on_plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=config['scheduler_patience'],
                factor=config['scheduler_factor']
            )
        elif config['scheduler_type'] == 'cosine_annealing':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config['num_epochs']
            )
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Add regularization
            if self.config['l1_lambda'] > 0:
                l1_loss = sum(p.abs().sum() for p in self.model.parameters())
                loss += self.config['l1_lambda'] * l1_loss
            
            if self.config['l2_lambda'] > 0:
                l2_loss = sum((p**2).sum() for p in self.model.parameters())
                loss += self.config['l2_lambda'] * l2_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config['gradient_clip_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['gradient_clip_norm']
                )
            
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, train_loader, val_loader):
        """Full training loop with early stopping"""
        train_losses = []
        val_losses = []
        
        for epoch in range(self.config['num_epochs']):
            # Train
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            # Scheduler step
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pt')
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.config['early_stopping_patience']:
                print(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        return train_losses, val_losses

# Custom loss functions
class QuantileLoss(nn.Module):
    def __init__(self, quantile=0.5):
        super().__init__()
        self.quantile = quantile
    
    def forward(self, pred, target):
        error = target - pred
        return torch.mean(torch.max((self.quantile - 1) * error, self.quantile * error))

class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta
    
    def forward(self, pred, target):
        error = torch.abs(target - pred)
        quadratic = torch.clamp(error, max=self.delta)
        linear = error - quadratic
        return torch.mean(0.5 * quadratic**2 + self.delta * linear)
```

### Advanced Training Techniques
```python
# Multi-task learning for multiple time series
class MultiTaskTimeSeriesModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_series, shared_layers=2):
        super().__init__()
        
        # Shared encoder
        self.shared_encoder = nn.LSTM(input_dim, hidden_dim, shared_layers, batch_first=True)
        
        # Task-specific heads
        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(num_series)
        ])
        
    def forward(self, x, task_id):
        # Shared representation
        encoded, _ = self.shared_encoder(x)
        
        # Task-specific prediction
        prediction = self.task_heads[task_id](encoded[:, -1, :])
        
        return prediction

# Meta-learning for few-shot forecasting
class MAMLTimeSeriesForecaster:
    def __init__(self, model, meta_lr=0.001, inner_lr=0.01):
        self.model = model
        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)
        self.inner_lr = inner_lr
        
    def inner_loop_update(self, support_data, support_targets):
        """Inner loop adaptation"""
        # Clone model for inner loop
        adapted_model = copy.deepcopy(self.model)
        inner_optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.inner_lr)
        
        # Adapt on support set
        for _ in range(5):  # Few gradient steps
            predictions = adapted_model(support_data)
            loss = nn.MSELoss()(predictions, support_targets)
            
            inner_optimizer.zero_grad()
            loss.backward()
            inner_optimizer.step()
        
        return adapted_model
    
    def meta_update(self, tasks):
        """Meta-learning update"""
        meta_loss = 0
        
        for support_data, support_targets, query_data, query_targets in tasks:
            # Inner loop adaptation
            adapted_model = self.inner_loop_update(support_data, support_targets)
            
            # Evaluate on query set
            query_predictions = adapted_model(query_data)
            task_loss = nn.MSELoss()(query_predictions, query_targets)
            meta_loss += task_loss
        
        # Meta gradient update
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item()
```

## 6. Evaluation Metrics & Validation Strategy

### Core Metrics
```python
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

def mean_absolute_percentage_error(y_true, y_pred, epsilon=1e-8):
    """Mean Absolute Percentage Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    """Symmetric MAPE"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return 200 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def mean_absolute_scaled_error(y_true, y_pred, y_train):
    """Mean Absolute Scaled Error"""
    y_true, y_pred, y_train = np.array(y_true), np.array(y_pred), np.array(y_train)
    
    # Calculate naive forecast error (seasonal naive)
    naive_errors = np.abs(np.diff(y_train))
    mean_naive_error = np.mean(naive_errors)
    
    # Calculate MASE
    mae = mean_absolute_error(y_true, y_pred)
    mase = mae / mean_naive_error
    
    return mase

def directional_accuracy(y_true, y_pred):
    """Directional Accuracy"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Calculate directions
    true_direction = np.sign(np.diff(y_true))
    pred_direction = np.sign(np.diff(y_pred))
    
    # Calculate accuracy
    correct_directions = np.sum(true_direction == pred_direction)
    total_directions = len(true_direction)
    
    return correct_directions / total_directions if total_directions > 0 else 0

def theil_u_statistic(y_true, y_pred):
    """Theil's U Statistic"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Calculate U statistic
    numerator = np.sqrt(np.mean((y_pred - y_true) ** 2))
    denominator = np.sqrt(np.mean(y_true ** 2)) + np.sqrt(np.mean(y_pred ** 2))
    
    return numerator / denominator if denominator > 0 else np.inf

def coverage_probability(y_true, y_pred_lower, y_pred_upper):
    """Coverage Probability for prediction intervals"""
    y_true = np.array(y_true)
    y_pred_lower = np.array(y_pred_lower)
    y_pred_upper = np.array(y_pred_upper)
    
    # Check if true values fall within prediction intervals
    within_interval = (y_true >= y_pred_lower) & (y_true <= y_pred_upper)
    
    return np.mean(within_interval)

def pinball_loss(y_true, y_pred, quantile):
    """Pinball loss for quantile regression"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    error = y_true - y_pred
    return np.mean(np.maximum(quantile * error, (quantile - 1) * error))

class TimeSeriesEvaluator:
    def __init__(self):
        self.metrics = {}
        
    def evaluate_forecast(self, y_true, y_pred, y_train=None, prediction_intervals=None):
        """Comprehensive evaluation of time series forecasts"""
        results = {}
        
        # Basic metrics
        results['mse'] = mean_squared_error(y_true, y_pred)
        results['rmse'] = np.sqrt(results['mse'])
        results['mae'] = mean_absolute_error(y_true, y_pred)
        results['mape'] = mean_absolute_percentage_error(y_true, y_pred)
        results['smape'] = symmetric_mean_absolute_percentage_error(y_true, y_pred)
        
        # Scaled metrics
        if y_train is not None:
            results['mase'] = mean_absolute_scaled_error(y_true, y_pred, y_train)
        
        # Directional metrics
        results['directional_accuracy'] = directional_accuracy(y_true, y_pred)
        
        # Statistical metrics
        results['theil_u'] = theil_u_statistic(y_true, y_pred)
        
        # Interval metrics
        if prediction_intervals is not None:
            results['coverage_80'] = coverage_probability(
                y_true, prediction_intervals['lower_80'], prediction_intervals['upper_80']
            )
            results['coverage_95'] = coverage_probability(
                y_true, prediction_intervals['lower_95'], prediction_intervals['upper_95']
            )
        
        return results
    
    def evaluate_multiple_horizons(self, forecasts_dict, actuals_dict):
        """Evaluate forecasts at multiple horizons"""
        results = {}
        
        for horizon, forecasts in forecasts_dict.items():
            actuals = actuals_dict[horizon]
            horizon_results = self.evaluate_forecast(actuals, forecasts)
            results[f'horizon_{horizon}'] = horizon_results
        
        return results
    
    def statistical_significance_test(self, errors_1, errors_2, test='dm'):
        """Test statistical significance between two forecasting methods"""
        if test == 'dm':  # Diebold-Mariano test
            from scipy import stats
            
            # Calculate loss differential
            d = errors_1**2 - errors_2**2  # Assuming squared errors
            
            # Calculate test statistic
            d_mean = np.mean(d)
            d_var = np.var(d, ddof=1)
            
            if d_var > 0:
                dm_stat = d_mean / np.sqrt(d_var / len(d))
                p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
                
                return {
                    'statistic': dm_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
            else:
                return {'statistic': 0, 'p_value': 1.0, 'significant': False}
        
        return None

# M4 Competition style evaluation
class M4Evaluator:
    def __init__(self):
        self.frequency_weights = {
            'Yearly': 1.0,
            'Quarterly': 1.0, 
            'Monthly': 1.0,
            'Weekly': 1.0,
            'Daily': 1.0,
            'Hourly': 1.0
        }
    
    def evaluate_m4_style(self, forecasts_df, actuals_df, frequency_col='frequency'):
        """M4 Competition style evaluation"""
        results = {}
        
        # Group by frequency
        for freq in forecasts_df[frequency_col].unique():
            freq_mask = forecasts_df[frequency_col] == freq
            
            freq_forecasts = forecasts_df[freq_mask]
            freq_actuals = actuals_df[freq_mask]
            
            # Calculate metrics for this frequency
            freq_results = []
            for idx in freq_forecasts.index:
                series_id = freq_forecasts.loc[idx, 'series_id']
                forecast = freq_forecasts.loc[idx, 'forecast']
                actual = freq_actuals[freq_actuals['series_id'] == series_id]['actual'].values
                
                if len(actual) > 0 and len(forecast) > 0:
                    evaluator = TimeSeriesEvaluator()
                    series_results = evaluator.evaluate_forecast(actual, forecast)
                    freq_results.append(series_results)
            
            # Aggregate results
            if freq_results:
                results[freq] = {
                    metric: np.mean([r[metric] for r in freq_results])
                    for metric in freq_results[0].keys()
                }
        
        return results
```

### Validation Strategy
- **Time series cross-validation**: Expanding window or sliding window
- **Walk-forward validation**: Sequential out-of-sample testing
- **Backtesting**: Historical performance simulation
- **Seasonal validation**: Evaluate across different seasonal periods
- **Multi-horizon evaluation**: Performance at different forecast horizons

### Advanced Evaluation
```python
# Residual analysis
def analyze_residuals(residuals):
    """Analyze forecast residuals"""
    from scipy import stats
    import matplotlib.pyplot as plt
    
    results = {}
    
    # Basic statistics
    results['mean'] = np.mean(residuals)
    results['std'] = np.std(residuals)
    results['skewness'] = stats.skew(residuals)
    results['kurtosis'] = stats.kurtosis(residuals)
    
    # Normality test
    _, results['shapiro_p_value'] = stats.shapiro(residuals[:5000])  # Sample if too large
    results['normal_residuals'] = results['shapiro_p_value'] > 0.05
    
    # Autocorrelation test (Ljung-Box)
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb_results = acorr_ljungbox(residuals, lags=10, return_df=True)
    results['ljung_box_p_value'] = lb_results['lb_pvalue'].iloc[-1]
    results['no_autocorrelation'] = results['ljung_box_p_value'] > 0.05
    
    # Heteroscedasticity test
    from statsmodels.stats.diagnostic import het_breuschpagan
    try:
        # Need to create a simple regression for the test
        X = np.arange(len(residuals)).reshape(-1, 1)
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression().fit(X, residuals)
        fitted_values = lr.predict(X)
        
        _, results['breusch_pagan_p_value'], _, _ = het_breuschpagan(residuals, fitted_values)
        results['homoscedastic'] = results['breusch_pagan_p_value'] > 0.05
    except:
        results['breusch_pagan_p_value'] = None
        results['homoscedastic'] = None
    
    return results

# Forecast combination evaluation
def evaluate_forecast_combination(individual_forecasts, combination_weights, actuals):
    """Evaluate forecast combination methods"""
    # Calculate combined forecast
    combined_forecast = np.zeros_like(actuals)
    
    for i, (forecast, weight) in enumerate(zip(individual_forecasts, combination_weights)):
        combined_forecast += weight * np.array(forecast)
    
    # Evaluate individual and combined forecasts
    evaluator = TimeSeriesEvaluator()
    
    results = {
        'individual': [],
        'combined': evaluator.evaluate_forecast(actuals, combined_forecast)
    }
    
    # Individual forecasts
    for i, forecast in enumerate(individual_forecasts):
        individual_result = evaluator.evaluate_forecast(actuals, forecast)
        individual_result['weight'] = combination_weights[i]
        results['individual'].append(individual_result)
    
    return results

# Regime-specific evaluation
def evaluate_by_regime(y_true, y_pred, regime_indicator):
    """Evaluate forecasts by market/economic regime"""
    unique_regimes = np.unique(regime_indicator)
    results = {}
    
    evaluator = TimeSeriesEvaluator()
    
    for regime in unique_regimes:
        regime_mask = regime_indicator == regime
        
        if np.sum(regime_mask) > 0:
            regime_true = y_true[regime_mask]
            regime_pred = y_pred[regime_mask]
            
            results[f'regime_{regime}'] = evaluator.evaluate_forecast(regime_true, regime_pred)
            results[f'regime_{regime}']['sample_size'] = np.sum(regime_mask)
    
    return results
```

## 7. Experiment Tracking & Reproducibility

### Weights & Biases Integration
```python
import wandb
import matplotlib.pyplot as plt
import numpy as np

# Initialize experiment tracking
wandb.init(
    project="time-series-forecasting",
    config=training_config,
    tags=["lstm", "m4-dataset", "multivariate"]
)

class TimeSeriesLoggingCallback:
    def __init__(self, model, val_loader, test_data, scaler=None):
        self.model = model
        self.val_loader = val_loader
        self.test_data = test_data
        self.scaler = scaler
        
    def on_epoch_end(self, epoch, train_loss, val_loss):
        """Log metrics and visualizations"""
        metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': self.get_lr()
        }
        
        # Generate sample predictions every 10 epochs
        if epoch % 10 == 0:
            sample_predictions = self.generate_sample_predictions()
            metrics.update(sample_predictions)
            
            # Log forecast visualization
            self.log_forecast_plot(epoch)
        
        # Log advanced metrics every 25 epochs
        if epoch % 25 == 0:
            advanced_metrics = self.calculate_advanced_metrics()
            metrics.update(advanced_metrics)
        
        wandb.log(metrics)
    
    def generate_sample_predictions(self, n_samples=5):
        """Generate sample predictions for logging"""
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for i, (data, target) in enumerate(self.val_loader):
                if i >= n_samples:
                    break
                    
                output = self.model(data)
                
                # Convert to numpy and potentially inverse transform
                pred_np = output.cpu().numpy()
                actual_np = target.cpu().numpy()
                
                if self.scaler:
                    pred_np = self.scaler.inverse_transform(pred_np.reshape(-1, 1)).flatten()
                    actual_np = self.scaler.inverse_transform(actual_np.reshape(-1, 1)).flatten()
                
                predictions.extend(pred_np)
                actuals.extend(actual_np)
        
        # Calculate metrics
        evaluator = TimeSeriesEvaluator()
        sample_metrics = evaluator.evaluate_forecast(actuals, predictions)
        
        return {f'val_{k}': v for k, v in sample_metrics.items()}
    
    def log_forecast_plot(self, epoch):
        """Log forecast visualization"""
        self.model.eval()
        
        # Get a sample sequence
        with torch.no_grad():
            data, target = next(iter(self.val_loader))
            prediction = self.model(data[:1])  # Single sample
            
            # Convert to numpy
            data_np = data[0].cpu().numpy()
            target_np = target[0].cpu().numpy()
            pred_np = prediction[0].cpu().numpy()
            
            # Create plot
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot historical data (last part of sequence)
            hist_steps = min(50, len(data_np))
            ax.plot(range(-hist_steps, 0), data_np[-hist_steps:, 0], 
                   label='Historical', color='blue')
            
            # Plot actual and predicted future
            future_steps = len(target_np)
            ax.plot(range(0, future_steps), target_np, 
                   label='Actual', color='green', marker='o')
            ax.plot(range(0, future_steps), pred_np, 
                   label='Predicted', color='red', marker='s')
            
            ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            ax.legend()
            ax.set_title(f'Time Series Forecast - Epoch {epoch}')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Value')
            
            # Log plot
            wandb.log({f"forecast_plot_epoch_{epoch}": wandb.Image(fig)})
            plt.close(fig)
    
    def calculate_advanced_metrics(self):
        """Calculate advanced forecasting metrics"""
        self.model.eval()
        all_predictions = []
        all_actuals = []
        
        with torch.no_grad():
            for data, target in self.val_loader:
                output = self.model(data)
                
                all_predictions.extend(output.cpu().numpy().flatten())
                all_actuals.extend(target.cpu().numpy().flatten())
        
        # Calculate directional accuracy
        if len(all_predictions) > 1:
            directional_acc = directional_accuracy(all_actuals, all_predictions)
            theil_u = theil_u_statistic(all_actuals, all_predictions)
            
            return {
                'advanced_directional_accuracy': directional_acc,
                'advanced_theil_u': theil_u
            }
        
        return {}
    
    def get_lr(self):
        """Get current learning rate"""
        # This would depend on your optimizer setup
        return 0.001  # Placeholder

# Enhanced trainer with logging
class LoggingTimeSeriesTrainer(AdvancedTimeSeriesTrainer):
    def __init__(self, model, config, logging_callback=None):
        super().__init__(model, config)
        self.logging_callback = logging_callback
        
    def train(self, train_loader, val_loader):
        """Training loop with enhanced logging"""
        train_losses = []
        val_losses = []
        
        for epoch in range(self.config['num_epochs']):
            # Train
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            # Scheduler step
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Log metrics
            if self.logging_callback:
                self.logging_callback.on_epoch_end(epoch, train_loss, val_loss)
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pt')
                wandb.save('best_model.pt')
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.config['early_stopping_patience']:
                print(f"Early stopping at epoch {epoch}")
                break
        
        return train_losses, val_losses
```

### MLflow Integration
```python
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

mlflow.set_experiment("time-series-forecasting")

with mlflow.start_run():
    # Log parameters
    mlflow.log_params({
        "model_type": training_config['model_type'],
        "sequence_length": training_config['sequence_length'],
        "hidden_dim": training_config['hidden_dim'],
        "learning_rate": training_config['learning_rate'],
        "batch_size": training_config['batch_size'],
        "dataset": "m4_daily",
        "num_series": 1000
    })
    
    # Train model
    trainer = LoggingTimeSeriesTrainer(model, training_config)
    train_losses, val_losses = trainer.train(train_loader, val_loader)
    
    # Final evaluation
    evaluator = TimeSeriesEvaluator()
    test_results = evaluator.evaluate_forecast(test_actuals, test_predictions)
    
    # Log metrics
    for metric_name, value in test_results.items():
        mlflow.log_metric(metric_name, value)
    
    # Log training curves
    for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
    
    # Log model
    mlflow.pytorch.log_model(
        model,
        "time-series-model",
        registered_model_name="TimeSeriesForecaster"
    )
    
    # Log evaluation artifacts
    mlflow.log_artifacts("evaluation_plots/", "evaluation")
```

### Experiment Configuration
```yaml
# experiment_config.yaml
experiment:
  name: "lstm-m4-forecasting"
  description: "LSTM model for M4 competition daily series"
  tags: ["time-series", "lstm", "m4", "daily"]

model:
  type: "lstm"
  input_dim: 1
  hidden_dim: 128
  num_layers: 3
  dropout: 0.2
  bidirectional: false

data:
  dataset: "m4_daily"
  sequence_length: 60
  forecast_horizon: 14
  features: ["target", "dayofweek", "month", "is_weekend"]
  normalization: "standard"
  
training:
  batch_size: 64
  learning_rate: 0.001
  weight_decay: 1e-5
  num_epochs: 200
  early_stopping_patience: 20
  gradient_clip_norm: 1.0
  
validation:
  method: "time_series_split"
  n_splits: 5
  test_size: 0.2
  
evaluation:
  metrics: ["mse", "mae", "mape", "mase", "directional_accuracy"]
  horizons: [1, 7, 14]
  confidence_intervals: [80, 95]
  
reproducibility:
  random_seed: 42
  deterministic: true
  benchmark: false
```

## 8. Deployment Pathway

### Option 1: Real-time Forecasting API
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
import pandas as pd
from typing import List, Optional
import redis
import pickle

app = FastAPI(title="Time Series Forecasting API")

# Load trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('best_forecasting_model.pt', map_location=device)
model.eval()

# Redis for caching
redis_client = redis.Redis(host='localhost', port=6379, db=0)

class TimeSeriesRequest(BaseModel):
    series_id: str
    historical_data: List[float]
    features: Optional[List[List[float]]] = None
    forecast_horizon: int = 1
    include_confidence: bool = False

class ForecastResponse(BaseModel):
    series_id: str
    forecast: List[float]
    confidence_lower: Optional[List[float]] = None
    confidence_upper: Optional[List[float]] = None
    timestamp: float

class ForecastingService:
    def __init__(self, model, scaler=None, sequence_length=60):
        self.model = model
        self.scaler = scaler
        self.sequence_length = sequence_length
        
    def preprocess_data(self, historical_data, features=None):
        """Preprocess input data for model"""
        # Convert to numpy array
        data = np.array(historical_data)
        
        # Normalize if scaler is available
        if self.scaler:
            data = self.scaler.transform(data.reshape(-1, 1)).flatten()
        
        # Create sequences
        if len(data) < self.sequence_length:
            # Pad with zeros if not enough history
            padded_data = np.zeros(self.sequence_length)
            padded_data[-len(data):] = data
            data = padded_data
        else:
            # Take last sequence_length points
            data = data[-self.sequence_length:]
        
        # Add features if provided
        if features:
            features_array = np.array(features)
            if len(features_array) >= self.sequence_length:
                features_array = features_array[-self.sequence_length:]
                data = np.column_stack([data, features_array])
            else:
                # Handle case where features are shorter than sequence
                padded_features = np.zeros((self.sequence_length, features_array.shape[1]))
                padded_features[-len(features_array):] = features_array
                data = np.column_stack([data, padded_features])
        
        return torch.FloatTensor(data).unsqueeze(0)  # Add batch dimension
    
    def generate_forecast(self, series_id, historical_data, features=None, 
                         forecast_horizon=1, include_confidence=False):
        """Generate forecast for time series"""
        
        # Check cache
        cache_key = f"forecast:{series_id}:{len(historical_data)}:{forecast_horizon}"
        cached_result = redis_client.get(cache_key)
        if cached_result:
            return pickle.loads(cached_result)
        
        # Preprocess data
        input_data = self.preprocess_data(historical_data, features)
        
        # Generate forecast
        forecasts = []
        current_input = input_data.clone()
        
        with torch.no_grad():
            for _ in range(forecast_horizon):
                # Get next prediction
                next_pred = self.model(current_input)
                
                # Store prediction
                forecasts.append(next_pred.item())
                
                # Update input for multi-step forecasting
                if forecast_horizon > 1:
                    # Shift sequence and add prediction
                    current_input = current_input.clone()
                    current_input[0, :-1] = current_input[0, 1:]
                    current_input[0, -1, 0] = next_pred.item()  # Assuming first column is target
        
        # Inverse transform if scaler available
        if self.scaler:
            forecasts = self.scaler.inverse_transform(
                np.array(forecasts).reshape(-1, 1)
            ).flatten().tolist()
        
        result = {
            'forecasts': forecasts,
            'confidence_lower': None,
            'confidence_upper': None
        }
        
        # Generate confidence intervals if requested
        if include_confidence:
            # Simple approach using model variance (in practice, use quantile regression or bootstrap)
            std_estimate = np.std(historical_data[-20:]) if len(historical_data) >= 20 else 1.0
            confidence_lower = [f - 1.96 * std_estimate for f in forecasts]
            confidence_upper = [f + 1.96 * std_estimate for f in forecasts]
            
            result['confidence_lower'] = confidence_lower
            result['confidence_upper'] = confidence_upper
        
        # Cache result
        redis_client.setex(cache_key, 300, pickle.dumps(result))  # 5 minutes cache
        
        return result
    
    def batch_forecast(self, requests):
        """Generate forecasts for multiple series"""
        results = []
        
        for req in requests:
            try:
                result = self.generate_forecast(
                    req['series_id'],
                    req['historical_data'],
                    req.get('features'),
                    req.get('forecast_horizon', 1),
                    req.get('include_confidence', False)
                )
                
                results.append({
                    'series_id': req['series_id'],
                    'success': True,
                    'forecast': result['forecasts'],
                    'confidence_lower': result['confidence_lower'],
                    'confidence_upper': result['confidence_upper']
                })
                
            except Exception as e:
                results.append({
                    'series_id': req['series_id'],
                    'success': False,
                    'error': str(e)
                })
        
        return results

# Initialize service
forecasting_service = ForecastingService(model)

@app.post("/forecast", response_model=ForecastResponse)
async def generate_forecast(request: TimeSeriesRequest):
    """Generate forecast for single time series"""
    try:
        result = forecasting_service.generate_forecast(
            request.series_id,
            request.historical_data,
            request.features,
            request.forecast_horizon,
            request.include_confidence
        )
        
        return ForecastResponse(
            series_id=request.series_id,
            forecast=result['forecasts'],
            confidence_lower=result['confidence_lower'],
            confidence_upper=result['confidence_upper'],
            timestamp=time.time()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_forecast")
async def batch_forecast(requests: List[TimeSeriesRequest]):
    """Generate forecasts for multiple time series"""
    try:
        batch_requests = [
            {
                'series_id': req.series_id,
                'historical_data': req.historical_data,
                'features': req.features,
                'forecast_horizon': req.forecast_horizon,
                'include_confidence': req.include_confidence
            }
            for req in requests
        ]
        
        results = forecasting_service.batch_forecast(batch_requests)
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}

# Model monitoring endpoints
@app.get("/model/stats")
async def model_statistics():
    """Get model performance statistics"""
    # In practice, this would fetch from monitoring database
    return {
        "total_predictions": 10000,
        "average_latency_ms": 15,
        "error_rate": 0.02,
        "last_retrain": "2024-01-15T10:30:00Z"
    }
```

### Option 2: Streaming Forecasting Pipeline
```python
import apache_kafka
from kafka import KafkaConsumer, KafkaProducer
import json
import asyncio
from datetime import datetime, timedelta

class StreamingForecastingPipeline:
    def __init__(self, model, kafka_config):
        self.model = model
        self.kafka_config = kafka_config
        
        # Kafka setup
        self.consumer = KafkaConsumer(
            'time_series_data',
            bootstrap_servers=kafka_config['bootstrap_servers'],
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='latest'
        )
        
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_config['bootstrap_servers'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        # Data buffer for each series
        self.series_buffers = {}
        self.buffer_size = 100  # Keep last 100 points per series
        
    async def process_stream(self):
        """Process streaming time series data"""
        for message in self.consumer:
            try:
                data = message.value
                series_id = data['series_id']
                timestamp = data['timestamp']
                value = data['value']
                features = data.get('features', {})
                
                # Update buffer
                if series_id not in self.series_buffers:
                    self.series_buffers[series_id] = {
                        'values': [],
                        'timestamps': [],
                        'features': []
                    }
                
                buffer = self.series_buffers[series_id]
                buffer['values'].append(value)
                buffer['timestamps'].append(timestamp)
                buffer['features'].append(features)
                
                # Maintain buffer size
                if len(buffer['values']) > self.buffer_size:
                    buffer['values'] = buffer['values'][-self.buffer_size:]
                    buffer['timestamps'] = buffer['timestamps'][-self.buffer_size:]
                    buffer['features'] = buffer['features'][-self.buffer_size:]
                
                # Generate forecast if enough data
                if len(buffer['values']) >= 20:  # Minimum required
                    forecast = await self.generate_streaming_forecast(series_id)
                    
                    # Send forecast
                    forecast_message = {
                        'series_id': series_id,
                        'forecast_timestamp': timestamp,
                        'forecast_horizon': 1,
                        'forecast_value': forecast,
                        'generated_at': datetime.now().isoformat()
                    }
                    
                    self.producer.send('forecasts', forecast_message)
                
            except Exception as e:
                print(f"Error processing message: {e}")
                continue
    
    async def generate_streaming_forecast(self, series_id):
        """Generate forecast for streaming data"""
        buffer = self.series_buffers[series_id]
        
        # Prepare data for model
        historical_data = buffer['values']
        
        # Use forecasting service
        forecasting_service = ForecastingService(self.model)
        result = forecasting_service.generate_forecast(
            series_id, historical_data, forecast_horizon=1
        )
        
        return result['forecasts'][0]
    
    def start_processing(self):
        """Start the streaming pipeline"""
        asyncio.run(self.process_stream())

# Example usage with Apache Beam for batch processing
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

class ForecastDoFn(beam.DoFn):
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
    
    def setup(self):
        """Load model once per worker"""
        self.model = torch.load(self.model_path, map_location='cpu')
        self.model.eval()
    
    def process(self, element):
        """Process each time series"""
        series_id, historical_data = element
        
        # Generate forecast
        forecasting_service = ForecastingService(self.model)
        result = forecasting_service.generate_forecast(
            series_id, historical_data, forecast_horizon=7
        )
        
        yield {
            'series_id': series_id,
            'forecast': result['forecasts'],
            'timestamp': datetime.now().isoformat()
        }

# Beam pipeline for batch forecasting
def run_batch_forecasting_pipeline():
    """Run batch forecasting pipeline"""
    pipeline_options = PipelineOptions([
        '--runner=DataflowRunner',
        '--project=your-project-id',
        '--region=us-central1',
        '--temp_location=gs://your-bucket/temp'
    ])
    
    with beam.Pipeline(options=pipeline_options) as pipeline:
        # Read time series data
        time_series_data = (
            pipeline
            | 'Read Data' >> beam.io.ReadFromBigQuery(
                query='SELECT series_id, ARRAY_AGG(value ORDER BY timestamp) as values FROM `dataset.table` GROUP BY series_id'
            )
        )
        
        # Generate forecasts
        forecasts = (
            time_series_data
            | 'Generate Forecasts' >> beam.ParDo(ForecastDoFn('gs://your-bucket/model.pt'))
        )
        
        # Write results
        (
            forecasts
            | 'Write Forecasts' >> beam.io.WriteToBigQuery(
                table='project:dataset.forecasts',
                schema='series_id:STRING, forecast:REPEATED FLOAT64, timestamp:TIMESTAMP'
            )
        )
```

### Option 3: Automated Forecasting Dashboard
```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests

st.set_page_config(page_title="Time Series Forecasting Dashboard", layout="wide")

class ForecastingDashboard:
    def __init__(self, api_url="http://localhost:8000"):
        self.api_url = api_url
        
    def main(self):
        """Main dashboard interface"""
        st.title("🔮 Time Series Forecasting Dashboard")
        
        # Sidebar for controls
        with st.sidebar:
            st.header("Forecast Configuration")
            
            # Data input method
            input_method = st.selectbox(
                "Data Input Method",
                ["Upload CSV", "Manual Entry", "API Connection"]
            )
            
            # Forecast parameters
            forecast_horizon = st.slider("Forecast Horizon", 1, 30, 7)
            confidence_intervals = st.checkbox("Include Confidence Intervals", True)
            
            # Model selection
            model_type = st.selectbox(
                "Model Type",
                ["LSTM", "Transformer", "ARIMA", "Prophet"]
            )
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("Time Series Data & Forecasts")
            
            # Handle data input
            df = self.handle_data_input(input_method)
            
            if df is not None:
                # Display data preview
                st.subheader("Data Preview")
                st.dataframe(df.head())
                
                # Generate and display forecasts
                if st.button("Generate Forecast", type="primary"):
                    with st.spinner("Generating forecasts..."):
                        forecasts = self.generate_forecasts(
                            df, forecast_horizon, confidence_intervals, model_type
                        )
                        
                        if forecasts:
                            self.display_forecast_results(df, forecasts)
        
        with col2:
            st.header("Model Performance")
            
            # Display model metrics
            self.display_model_metrics()
            
            # Real-time monitoring
            self.display_realtime_monitoring()
    
    def handle_data_input(self, method):
        """Handle different data input methods"""
        if method == "Upload CSV":
            uploaded_file = st.file_uploader("Choose CSV file", type="csv")
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                
                # Date column selection
                date_col = st.selectbox("Select Date Column", df.columns)
                value_col = st.selectbox("Select Value Column", df.columns)
                
                # Process data
                df[date_col] = pd.to_datetime(df[date_col])
                df = df[[date_col, value_col]].rename(
                    columns={date_col: 'timestamp', value_col: 'value'}
                )
                df = df.sort_values('timestamp').reset_index(drop=True)
                
                return df
                
        elif method == "Manual Entry":
            # Simple manual data entry
            st.info("Enter comma-separated values")
            values_input = st.text_area("Values (comma-separated)")
            
            if values_input:
                values = [float(x.strip()) for x in values_input.split(',')]
                dates = pd.date_range(end=datetime.now(), periods=len(values), freq='D')
                
                df = pd.DataFrame({
                    'timestamp': dates,
                    'value': values
                })
                
                return df
                
        elif method == "API Connection":
            # Connect to external API
            api_endpoint = st.text_input("API Endpoint")
            
            if api_endpoint and st.button("Fetch Data"):
                try:
                    response = requests.get(api_endpoint)
                    data = response.json()
                    df = pd.DataFrame(data)
                    return df
                except Exception as e:
                    st.error(f"Error fetching data: {e}")
        
        return None
    
    def generate_forecasts(self, df, horizon, confidence, model_type):
        """Generate forecasts using API"""
        try:
            # Prepare API request
            request_data = {
                "series_id": "dashboard_series",
                "historical_data": df['value'].tolist(),
                "forecast_horizon": horizon,
                "include_confidence": confidence
            }
            
            # Call API
            response = requests.post(f"{self.api_url}/forecast", json=request_data)
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"API Error: {response.text}")
                return None
                
        except Exception as e:
            st.error(f"Error generating forecasts: {e}")
            return None
    
    def display_forecast_results(self, df, forecasts):
        """Display forecast results with interactive plots"""
        # Create forecast dataframe
        last_date = df['timestamp'].iloc[-1]
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=len(forecasts['forecast']),
            freq='D'
        )
        
        forecast_df = pd.DataFrame({
            'timestamp': forecast_dates,
            'forecast': forecasts['forecast'],
            'confidence_lower': forecasts.get('confidence_lower'),
            'confidence_upper': forecasts.get('confidence_upper')
        })
        
        # Create interactive plot
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['value'],
            mode='lines',
            name='Historical',
            line=dict(color='blue')
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast_df['timestamp'],
            y=forecast_df['forecast'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='red', dash='dash')
        ))
        
        # Confidence intervals
        if forecasts.get('confidence_lower'):
            fig.add_trace(go.Scatter(
                x=forecast_df['timestamp'].tolist() + forecast_df['timestamp'].tolist()[::-1],
                y=forecast_df['confidence_upper'].tolist() + forecast_df['confidence_lower'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval'
            ))
        
        fig.update_layout(
            title="Time Series Forecast",
            xaxis_title="Date",
            yaxis_title="Value",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display forecast table
        st.subheader("Forecast Values")
        st.dataframe(forecast_df)
    
    def display_model_metrics(self):
        """Display model performance metrics"""
        try:
            response = requests.get(f"{self.api_url}/model/stats")
            if response.status_code == 200:
                stats = response.json()
                
                st.metric("Total Predictions", stats['total_predictions'])
                st.metric("Avg Latency", f"{stats['average_latency_ms']} ms")
                st.metric("Error Rate", f"{stats['error_rate']:.2%}")
                
        except Exception as e:
            st.error(f"Error fetching model stats: {e}")
    
    def display_realtime_monitoring(self):
        """Display real-time monitoring"""
        st.subheader("Real-time Monitoring")
        
        # Placeholder for real-time charts
        # In practice, this would connect to monitoring system
        monitoring_data = np.random.randn(100).cumsum()
        
        fig = px.line(y=monitoring_data, title="Model Performance Over Time")
        st.plotly_chart(fig, use_container_width=True)

# Run dashboard
if __name__ == "__main__":
    dashboard = ForecastingDashboard()
    dashboard.main()
```

### Cloud Deployment Options
- **AWS SageMaker**: Real-time endpoints with auto-scaling
- **Google Cloud AI Platform**: Managed model serving
- **Azure ML**: Enterprise-grade deployment with monitoring
- **Kubernetes**: Container orchestration for microservices
- **Apache Kafka**: Streaming data processing
- **Apache Airflow**: Workflow orchestration for batch jobs

## 9. Extensions & Research Directions

### Advanced Techniques
1. **Hierarchical Forecasting**
   ```python
   class HierarchicalForecaster:
       def __init__(self, hierarchy_structure):
           self.hierarchy = hierarchy_structure
           self.models = {}
           
       def fit_hierarchical(self, data_dict):
           """Fit models for each level of hierarchy"""
           for level, series_ids in self.hierarchy.items():
               for series_id in series_ids:
                   # Fit model for this series
                   model = LSTMForecaster()
                   model.fit(data_dict[series_id])
                   self.models[series_id] = model
       
       def reconcile_forecasts(self, base_forecasts):
           """Reconcile forecasts to maintain hierarchy consistency"""
           # Implementation of MinT (Minimum Trace) reconciliation
           reconciled = {}
           
           # Bottom-up reconciliation
           for parent, children in self.hierarchy.items():
               if children:  # Has child series
                   reconciled[parent] = sum(base_forecasts[child] for child in children)
               else:  # Leaf node
                   reconciled[parent] = base_forecasts[parent]
           
           return reconciled
   ```

2. **Probabilistic Forecasting**
   ```python
   class ProbabilisticForecaster(nn.Module):
       def __init__(self, input_dim, hidden_dim=128):
           super().__init__()
           
           # Shared encoder
           self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
           
           # Mean and variance heads
           self.mean_head = nn.Linear(hidden_dim, 1)
           self.log_var_head = nn.Linear(hidden_dim, 1)
           
       def forward(self, x):
           encoded, _ = self.encoder(x)
           
           # Extract last timestep
           last_output = encoded[:, -1, :]
           
           # Predict mean and log variance
           mean = self.mean_head(last_output)
           log_var = self.log_var_head(last_output)
           
           return mean, log_var
       
       def sample_forecasts(self, x, n_samples=100):
           """Generate probabilistic forecasts"""
           mean, log_var = self.forward(x)
           std = torch.exp(0.5 * log_var)
           
           # Sample from normal distribution
           samples = torch.normal(
               mean.repeat(1, n_samples), 
               std.repeat(1, n_samples)
           )
           
           return samples
       
       def quantile_forecasts(self, x, quantiles=[0.1, 0.5, 0.9]):
           """Generate quantile forecasts"""
           samples = self.sample_forecasts(x)
           
           quantile_forecasts = []
           for q in quantiles:
               quantile_val = torch.quantile(samples, q, dim=1)
               quantile_forecasts.append(quantile_val)
           
           return quantile_forecasts
   ```

3. **Online Learning and Adaptation**
   ```python
   class OnlineForecaster:
       def __init__(self, base_model, adaptation_rate=0.01):
           self.base_model = base_model
           self.adaptation_rate = adaptation_rate
           self.optimizer = torch.optim.SGD(
               base_model.parameters(), 
               lr=adaptation_rate
           )
           
       def predict_and_update(self, x, y_true=None):
           """Predict and update model if true value is available"""
           # Make prediction
           prediction = self.base_model(x)
           
           # Update if true value is available
           if y_true is not None:
               loss = nn.MSELoss()(prediction, y_true)
               
               self.optimizer.zero_grad()
               loss.backward()
               self.optimizer.step()
           
           return prediction
       
       def detect_concept_drift(self, recent_errors, threshold=2.0):
           """Simple concept drift detection"""
           if len(recent_errors) > 20:
               recent_mean = np.mean(recent_errors[-10:])
               historical_mean = np.mean(recent_errors[:-10])
               
               if recent_mean > threshold * historical_mean:
                   return True
           
           return False
       
       def handle_concept_drift(self):
           """Handle detected concept drift"""
           # Increase learning rate temporarily
           for param_group in self.optimizer.param_groups:
               param_group['lr'] *= 2.0
           
           # Schedule learning rate decay
           scheduler = torch.optim.lr_scheduler.ExponentialLR(
               self.optimizer, gamma=0.95
           )
           
           return scheduler
   ```

4. **Multi-Scale Forecasting**
   ```python
   class MultiScaleForecaster(nn.Module):
       def __init__(self, input_dim, scales=[1, 7, 30]):
           super().__init__()
           
           self.scales = scales
           self.encoders = nn.ModuleList([
               nn.LSTM(input_dim, 64, batch_first=True) 
               for _ in scales
           ])
           
           # Attention mechanism to combine scales
           self.attention = nn.MultiheadAttention(64, num_heads=4, batch_first=True)
           
           # Output layer
           self.output = nn.Linear(64, 1)
           
       def multi_scale_encode(self, x):
           """Encode at multiple time scales"""
           scale_encodings = []
           
           for i, scale in enumerate(self.scales):
               # Downsample input for this scale
               if scale > 1:
                   downsampled = x[:, ::scale, :]
               else:
                   downsampled = x
               
               # Encode
               encoded, _ = self.encoders[i](downsampled)
               scale_encodings.append(encoded[:, -1:, :])  # Take last timestep
           
           return scale_encodings
       
       def forward(self, x):
           # Get multi-scale encodings
           scale_encodings = self.multi_scale_encode(x)
           
           # Concatenate scale encodings
           combined = torch.cat(scale_encodings, dim=1)  # Shape: (batch, n_scales, hidden)
           
           # Apply attention to combine scales
           attended, _ = self.attention(combined, combined, combined)
           
           # Global average pooling
           pooled = attended.mean(dim=1)
           
           # Final prediction
           prediction = self.output(pooled)
           
           return prediction
   ```

### Novel Experiments
- **Causal inference in time series**: Understanding true causal relationships
- **Meta-learning for few-shot forecasting**: Quick adaptation to new series
- **Graph neural networks**: Modeling relationships between multiple time series
- **Physics-informed forecasting**: Incorporating domain knowledge and constraints
- **Federated time series learning**: Training on distributed data while preserving privacy

### Emerging Applications
```python
# Climate forecasting with physics constraints
class PhysicsInformedForecaster(nn.Module):
    def __init__(self, input_dim, physics_constraints=None):
        super().__init__()
        
        self.neural_model = nn.LSTM(input_dim, 128, batch_first=True)
        self.physics_constraints = physics_constraints
        
    def apply_physics_constraints(self, predictions):
        """Apply physical constraints to predictions"""
        if self.physics_constraints:
            # Example: Energy conservation
            if 'energy_conservation' in self.physics_constraints:
                predictions = torch.clamp(predictions, min=0)  # Non-negative energy
            
            # Example: Mass conservation
            if 'mass_conservation' in self.physics_constraints:
                # Ensure total mass is conserved
                total_mass = torch.sum(predictions, dim=1, keepdim=True)
                predictions = predictions / total_mass * self.physics_constraints['total_mass']
        
        return predictions

# Financial forecasting with risk constraints
class RiskConstrainedForecaster:
    def __init__(self, base_model, risk_limit=0.05):
        self.base_model = base_model
        self.risk_limit = risk_limit
        
    def forecast_with_risk_control(self, x, portfolio_weights):
        """Generate forecasts with portfolio risk constraints"""
        # Get base forecast
        base_forecast = self.base_model(x)
        
        # Estimate portfolio risk
        forecast_returns = base_forecast / x[:, -1, 0] - 1  # Convert to returns
        portfolio_return = torch.sum(forecast_returns * portfolio_weights, dim=1)
        
        # Apply risk constraint
        if portfolio_return.std() > self.risk_limit:
            # Adjust forecast to meet risk constraint
            risk_adjusted_forecast = self.adjust_for_risk(base_forecast, portfolio_weights)
            return risk_adjusted_forecast
        
        return base_forecast
```

### Industry Applications
- **Supply Chain**: Demand forecasting with inventory constraints
- **Energy**: Load forecasting with renewable integration
- **Healthcare**: Patient flow and resource demand prediction
- **Finance**: Risk management and algorithmic trading
- **Manufacturing**: Predictive maintenance and quality control
- **Smart Cities**: Traffic flow and utility demand prediction

## 10. Portfolio Polish

### Documentation Structure
```
time_series_forecasting/
├── README.md                           # This file
├── notebooks/
│   ├── 01_Data_Exploration.ipynb       # M4 and ETT dataset analysis
│   ├── 02_Classical_Methods.ipynb      # ARIMA, ExponentialSmoothing
│   ├── 03_Deep_Learning.ipynb          # LSTM, GRU, Transformer
│   ├── 04_Advanced_Models.ipynb        # N-BEATS, DeepAR, Prophet
│   ├── 05_Evaluation_Study.ipynb       # Comprehensive evaluation
│   └── 06_Deployment_Demo.ipynb        # Production deployment
├── src/
│   ├── models/
│   │   ├── classical/
│   │   │   ├── arima.py
│   │   │   ├── exponential_smoothing.py
│   │   │   └── prophet_model.py
│   │   ├── deep_learning/
│   │   │   ├── lstm_forecaster.py
│   │   │   ├── transformer_forecaster.py
│   │   │   ├── nbeats.py
│   │   │   └── deepar.py
│   │   └── ensemble/
│   │       ├── model_combination.py
│   │       └── hierarchical.py
│   ├── data/
│   │   ├── preprocessing.py
│   │   ├── feature_engineering.py
│   │   ├── data_loaders.py
│   │   └── validation.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── cross_validation.py
│   │   ├── residual_analysis.py
│   │   └── statistical_tests.py
│   ├── training/
│   │   ├── trainers.py
│   │   ├── hyperparameter_tuning.py
│   │   ├── online_learning.py
│   │   └── meta_learning.py
│   ├── inference/
│   │   ├── forecasting_service.py
│   │   ├── batch_inference.py
│   │   └── streaming_inference.py
│   ├── train.py
│   ├── forecast.py
│   └── evaluate.py
├── configs/
│   ├── models/
│   │   ├── lstm_config.yaml
│   │   ├── transformer_config.yaml
│   │   ├── nbeats_config.yaml
│   │   └── arima_config.yaml
│   └── experiments/
├── api/
│   ├── fastapi_server.py
│   ├── streaming_service.py
│   ├── batch_service.py
│   └── requirements.txt
├── deployment/
│   ├── docker/
│   │   ├── Dockerfile
│   │   ├── docker-compose.yml
│   │   └── kubernetes/
│   ├── streaming/
│   │   ├── kafka_pipeline.py
│   │   └── beam_pipeline.py
│   ├── batch/
│   │   ├── airflow_dag.py
│   │   └── scheduled_jobs.py
│   └── monitoring/
├── dashboard/
│   ├── streamlit_app.py
│   ├── plotly_dashboard.py
│   └── monitoring_dashboard.py
├── evaluation/
│   ├── m4_evaluation.py
│   ├── cross_dataset_eval.py
│   ├── benchmark_comparison.py
│   └── statistical_analysis.py
├── datasets/
│   ├── m4_loader.py
│   ├── ett_loader.py
│   ├── custom_datasets.py
│   └── data_generators.py
├── tests/
│   ├── test_models.py
│   ├── test_preprocessing.py
│   ├── test_evaluation.py
│   └── test_api.py
├── requirements.txt
├── setup.py
├── Makefile
└── .github/workflows/
```

### Visualization Requirements
- **Time series plots**: Interactive historical data with forecasts
- **Forecast performance**: Error metrics across different horizons
- **Seasonal decomposition**: Trend, seasonal, and residual components
- **Residual analysis**: Diagnostic plots for model validation
- **Confidence intervals**: Uncertainty quantification visualization
- **Multi-series comparison**: Performance across different time series
- **Feature importance**: Contribution of different input features
- **Real-time monitoring**: Live forecast performance dashboards

### Blog Post Template
1. **The Forecasting Revolution**: From statistics to deep learning
2. **Dataset Deep-dive**: M4 Competition insights and challenges
3. **Classical vs Modern**: ARIMA, Prophet, and neural approaches
4. **Deep Learning Breakthroughs**: LSTMs, Transformers, and N-BEATS
5. **Production Challenges**: Scalability, latency, and continuous learning
6. **Evaluation Beyond Accuracy**: Statistical significance and business metrics
7. **Real-world Deployment**: From research to production systems
8. **Future Horizons**: Probabilistic, hierarchical, and causal forecasting

### Demo Video Script
- 1 minute: Time series forecasting importance across industries
- 1.5 minutes: Dataset exploration with seasonal patterns and trends
- 2.5 minutes: Model comparison from ARIMA to deep learning
- 3 minutes: Live forecasting demo with confidence intervals
- 1.5 minutes: Evaluation metrics and statistical testing
- 1 minute: Hierarchical and multi-variate forecasting
- 2 minutes: Production deployment and real-time monitoring
- 1 minute: Advanced techniques and future research

### GitHub README Essentials
```markdown
# Advanced Time Series Forecasting with Deep Learning

![Forecasting Demo](assets/forecasting_demo.gif)

## 🚀 Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Download M4 dataset
python src/data/download_m4.py --frequency daily

# Train LSTM model
python src/train.py --config configs/models/lstm_config.yaml

# Generate forecasts
python src/forecast.py --model models/best_lstm.pt --data data/test.csv

# Launch dashboard
streamlit run dashboard/streamlit_app.py
```

## 📊 Results
| Model | Dataset | MAPE | MASE | sMAPE | Directional Accuracy |
|-------|---------|------|------|-------|---------------------|
| ARIMA | M4-Daily | 12.3% | 1.08 | 11.7% | 68.4% |
| LSTM | M4-Daily | 8.9% | 0.94 | 8.2% | 72.1% |
| Transformer | M4-Daily | 7.6% | 0.89 | 7.1% | 74.3% |
| N-BEATS | M4-Daily | 6.8% | 0.85 | 6.4% | 76.8% |

## 🎯 Live Demo
Try the forecasting system: [Streamlit App](https://forecasting-demo.streamlit.app)

## 📈 API Usage
```python
import requests

response = requests.post("http://api.example.com/forecast", json={
    "series_id": "daily_sales",
    "historical_data": [100, 120, 98, 110, 105],
    "forecast_horizon": 7,
    "include_confidence": true
})

forecast = response.json()
print(f"7-day forecast: {forecast['forecast']}")
```

## 📚 Citation
```bibtex
@article{advanced_forecasting_2024,
  title={Advanced Time Series Forecasting: From Classical Methods to Deep Learning},
  author={Your Name},
  journal={Journal of Forecasting},
  year={2024}
}
```
```

### Performance Benchmarks
- **Forecast accuracy**: MAPE, MASE, sMAPE across different horizons and datasets
- **Computational efficiency**: Training time and inference speed by model type
- **Memory requirements**: RAM and storage needs for different model sizes
- **Scalability analysis**: Performance with increasing number of time series
- **Real-time latency**: End-to-end forecast generation time
- **Statistical significance**: Diebold-Mariano test results vs benchmarks
- **Robustness testing**: Performance under different data quality conditions
- **Cross-dataset generalization**: Transfer learning capabilities
- **Business impact**: Revenue/cost improvements from accurate forecasting