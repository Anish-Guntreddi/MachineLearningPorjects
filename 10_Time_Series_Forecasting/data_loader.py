"""
Data loading and preprocessing for Time Series Forecasting
"""
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesDataset(Dataset):
    """Dataset for time series forecasting"""
    
    def __init__(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        sequence_length: int = 24,
        prediction_length: int = 12,
        target_columns: Optional[List[str]] = None,
        feature_columns: Optional[List[str]] = None,
        scaler: Optional[object] = None,
        scale_data: bool = True,
        stride: int = 1
    ):
        """
        Args:
            data: Time series data (DataFrame or array)
            sequence_length: Length of input sequence
            prediction_length: Length of prediction horizon
            target_columns: Columns to predict
            feature_columns: Additional feature columns
            scaler: Pre-fitted scaler
            scale_data: Whether to scale the data
            stride: Stride for sliding window
        """
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.stride = stride
        self.scale_data = scale_data
        
        # Convert to DataFrame if needed
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        
        self.data = data
        
        # Identify columns
        if target_columns:
            self.target_columns = target_columns
        else:
            self.target_columns = [data.columns[0]] if len(data.columns) > 0 else []
        
        if feature_columns:
            self.feature_columns = feature_columns
        else:
            self.feature_columns = [col for col in data.columns if col not in self.target_columns]
        
        # Prepare data arrays
        self.target_data = data[self.target_columns].values if self.target_columns else data.values
        self.feature_data = data[self.feature_columns].values if self.feature_columns else None
        
        # Scale data
        if scale_data:
            if scaler is None:
                self.scaler = StandardScaler()
                self.target_data = self.scaler.fit_transform(self.target_data)
            else:
                self.scaler = scaler
                self.target_data = self.scaler.transform(self.target_data)
            
            if self.feature_data is not None:
                self.feature_scaler = StandardScaler()
                self.feature_data = self.feature_scaler.fit_transform(self.feature_data)
        else:
            self.scaler = None
            self.feature_scaler = None
        
        # Calculate valid indices
        self.indices = self._get_valid_indices()
    
    def _get_valid_indices(self) -> List[int]:
        """Get valid starting indices for sequences"""
        indices = []
        max_start = len(self.target_data) - self.sequence_length - self.prediction_length + 1
        
        for i in range(0, max_start, self.stride):
            indices.append(i)
        
        return indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        end_idx = start_idx + self.sequence_length
        pred_end_idx = end_idx + self.prediction_length
        
        # Input sequence
        input_seq = self.target_data[start_idx:end_idx]
        
        # Target sequence
        target_seq = self.target_data[end_idx:pred_end_idx]
        
        # Features if available
        if self.feature_data is not None:
            feature_seq = self.feature_data[start_idx:end_idx]
            # Concatenate target and features
            input_seq = np.concatenate([input_seq, feature_seq], axis=1)
        
        return {
            'input': torch.tensor(input_seq, dtype=torch.float32),
            'target': torch.tensor(target_seq, dtype=torch.float32),
            'time_idx': torch.tensor(start_idx, dtype=torch.long)
        }


class MultivariateTSDataset(Dataset):
    """Dataset for multivariate time series"""
    
    def __init__(
        self,
        data: pd.DataFrame,
        sequence_length: int = 24,
        prediction_length: int = 12,
        target_idx: Optional[List[int]] = None,
        time_features: bool = True,
        scale_data: bool = True
    ):
        """
        Args:
            data: Multivariate time series DataFrame
            sequence_length: Length of input sequence
            prediction_length: Length of prediction horizon
            target_idx: Indices of target variables
            time_features: Whether to add time-based features
            scale_data: Whether to scale the data
        """
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.time_features = time_features
        
        # Add time features if requested
        if time_features and isinstance(data.index, pd.DatetimeIndex):
            data = self._add_time_features(data)
        
        # Separate targets and features
        if target_idx is not None:
            self.target_data = data.iloc[:, target_idx].values
            feature_idx = [i for i in range(len(data.columns)) if i not in target_idx]
            self.feature_data = data.iloc[:, feature_idx].values if feature_idx else None
        else:
            self.target_data = data.values
            self.feature_data = None
        
        # Scale data
        if scale_data:
            self.target_scaler = StandardScaler()
            self.target_data = self.target_scaler.fit_transform(self.target_data)
            
            if self.feature_data is not None:
                self.feature_scaler = StandardScaler()
                self.feature_data = self.feature_scaler.fit_transform(self.feature_data)
        else:
            self.target_scaler = None
            self.feature_scaler = None
        
        # Store dimensions
        self.n_targets = self.target_data.shape[1]
        self.n_features = self.feature_data.shape[1] if self.feature_data is not None else 0
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        df = df.copy()
        
        # Extract time features
        df['hour'] = df.index.hour / 23.0
        df['day_of_week'] = df.index.dayofweek / 6.0
        df['day_of_month'] = df.index.day / 31.0
        df['month'] = df.index.month / 12.0
        
        # Add cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        
        return df
    
    def __len__(self):
        return len(self.target_data) - self.sequence_length - self.prediction_length + 1
    
    def __getitem__(self, idx):
        # Get sequences
        input_target = self.target_data[idx:idx + self.sequence_length]
        output_target = self.target_data[idx + self.sequence_length:idx + self.sequence_length + self.prediction_length]
        
        if self.feature_data is not None:
            input_features = self.feature_data[idx:idx + self.sequence_length]
            input_seq = np.concatenate([input_target, input_features], axis=1)
        else:
            input_seq = input_target
        
        return {
            'input': torch.tensor(input_seq, dtype=torch.float32),
            'target': torch.tensor(output_target, dtype=torch.float32)
        }


class StockDataset(Dataset):
    """Dataset for stock price forecasting"""
    
    def __init__(
        self,
        ticker: str = 'AAPL',
        start_date: str = '2010-01-01',
        end_date: str = '2023-12-31',
        sequence_length: int = 60,
        prediction_length: int = 5,
        features: List[str] = ['Open', 'High', 'Low', 'Close', 'Volume'],
        target: str = 'Close',
        add_technical_indicators: bool = True
    ):
        """
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for data
            end_date: End date for data
            sequence_length: Length of input sequence
            prediction_length: Length of prediction horizon
            features: Feature columns to use
            target: Target column
            add_technical_indicators: Whether to add technical indicators
        """
        self.ticker = ticker
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.target = target
        
        # Load stock data
        self.data = self._load_stock_data(ticker, start_date, end_date)
        
        # Add technical indicators
        if add_technical_indicators:
            self.data = self._add_technical_indicators(self.data)
        
        # Select features
        self.feature_columns = features
        if add_technical_indicators:
            self.feature_columns.extend(['SMA_20', 'SMA_50', 'RSI', 'MACD', 'BB_upper', 'BB_lower'])
        
        # Prepare data
        self.feature_data = self.data[self.feature_columns].values
        self.target_data = self.data[[target]].values
        
        # Scale data
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        
        self.feature_data = self.feature_scaler.fit_transform(self.feature_data)
        self.target_data = self.target_scaler.fit_transform(self.target_data)
    
    def _load_stock_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Load stock data (placeholder - would use yfinance in practice)"""
        # Generate dummy stock data for demonstration
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n = len(dates)
        
        # Simulate stock prices
        np.random.seed(42)
        price = 100
        prices = []
        
        for _ in range(n):
            change = np.random.randn() * 2
            price = max(price * (1 + change / 100), 10)
            prices.append(price)
        
        df = pd.DataFrame({
            'Date': dates,
            'Open': prices + np.random.randn(n) * 0.5,
            'High': prices + np.abs(np.random.randn(n) * 2),
            'Low': prices - np.abs(np.random.randn(n) * 2),
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, n)
        })
        
        df.set_index('Date', inplace=True)
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators"""
        df = df.copy()
        
        # Simple Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
        df['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
        
        # Relative Strength Index (RSI)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / (loss + 1e-10)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        
        # Bollinger Bands
        sma = df['Close'].rolling(window=20, min_periods=1).mean()
        std = df['Close'].rolling(window=20, min_periods=1).std()
        df['BB_upper'] = sma + (std * 2)
        df['BB_lower'] = sma - (std * 2)
        
        # Fill NaN values
        df.fillna(method='ffill', inplace=True)
        df.fillna(0, inplace=True)
        
        return df
    
    def __len__(self):
        return len(self.target_data) - self.sequence_length - self.prediction_length + 1
    
    def __getitem__(self, idx):
        # Get sequences
        feature_seq = self.feature_data[idx:idx + self.sequence_length]
        target_seq = self.target_data[idx + self.sequence_length:idx + self.sequence_length + self.prediction_length]
        
        return {
            'features': torch.tensor(feature_seq, dtype=torch.float32),
            'target': torch.tensor(target_seq, dtype=torch.float32)
        }


class TimeSeriesDataModule:
    """Data module for time series forecasting"""
    
    def __init__(
        self,
        dataset_name: str = 'synthetic',
        data_path: Optional[str] = None,
        sequence_length: int = 24,
        prediction_length: int = 12,
        batch_size: int = 32,
        num_workers: int = 4,
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15
    ):
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.scaler = None
    
    def setup(self):
        """Setup datasets"""
        if self.dataset_name == 'synthetic':
            data = self._generate_synthetic_data()
        elif self.dataset_name == 'stock':
            data = self._load_stock_data()
        else:
            data = self._load_custom_data()
        
        # Split data
        n = len(data)
        train_size = int(n * self.train_split)
        val_size = int(n * self.val_split)
        
        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]
        
        # Create datasets
        self.train_dataset = TimeSeriesDataset(
            train_data,
            sequence_length=self.sequence_length,
            prediction_length=self.prediction_length,
            scale_data=True
        )
        
        # Use the same scaler for validation and test
        self.scaler = self.train_dataset.scaler
        
        self.val_dataset = TimeSeriesDataset(
            val_data,
            sequence_length=self.sequence_length,
            prediction_length=self.prediction_length,
            scaler=self.scaler
        )
        
        self.test_dataset = TimeSeriesDataset(
            test_data,
            sequence_length=self.sequence_length,
            prediction_length=self.prediction_length,
            scaler=self.scaler
        )
    
    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic time series data"""
        n_points = 10000
        t = np.arange(n_points)
        
        # Create multiple patterns
        trend = t * 0.01
        seasonal1 = 10 * np.sin(2 * np.pi * t / 100)
        seasonal2 = 5 * np.sin(2 * np.pi * t / 50)
        noise = np.random.randn(n_points) * 2
        
        # Combine patterns
        series1 = trend + seasonal1 + noise
        series2 = trend * 0.5 + seasonal2 + noise * 0.5
        series3 = seasonal1 * 0.5 + seasonal2 * 0.5 + noise
        
        # Create DataFrame
        dates = pd.date_range(start='2020-01-01', periods=n_points, freq='H')
        df = pd.DataFrame({
            'series1': series1,
            'series2': series2,
            'series3': series3
        }, index=dates)
        
        return df
    
    def _load_stock_data(self) -> pd.DataFrame:
        """Load stock data"""
        dataset = StockDataset()
        return dataset.data
    
    def _load_custom_data(self) -> pd.DataFrame:
        """Load custom data from file"""
        if self.data_path:
            return pd.read_csv(self.data_path, index_col=0, parse_dates=True)
        else:
            return self._generate_synthetic_data()
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


def create_sliding_windows(
    data: np.ndarray,
    window_size: int,
    prediction_steps: int = 1,
    stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows for time series
    
    Args:
        data: Time series data
        window_size: Size of input window
        prediction_steps: Number of steps to predict
        stride: Stride for sliding window
    
    Returns:
        Input windows and targets
    """
    X, y = [], []
    
    for i in range(0, len(data) - window_size - prediction_steps + 1, stride):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size:i + window_size + prediction_steps])
    
    return np.array(X), np.array(y)


if __name__ == "__main__":
    # Test data loading
    print("Testing time series data loading...")
    
    # Test basic dataset
    data = pd.DataFrame({
        'value': np.sin(np.linspace(0, 100, 1000)) + np.random.randn(1000) * 0.1
    })
    
    dataset = TimeSeriesDataset(
        data,
        sequence_length=24,
        prediction_length=12
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"Input shape: {sample['input'].shape}")
    print(f"Target shape: {sample['target'].shape}")
    
    # Test data module
    dm = TimeSeriesDataModule(
        dataset_name='synthetic',
        batch_size=32
    )
    
    dm.setup()
    print(f"Train dataset size: {len(dm.train_dataset)}")
    print(f"Val dataset size: {len(dm.val_dataset)}")
    print(f"Test dataset size: {len(dm.test_dataset)}")
    
    # Test batch
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    print(f"Batch input shape: {batch['input'].shape}")
    print(f"Batch target shape: {batch['target'].shape}")