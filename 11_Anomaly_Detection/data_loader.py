"""
Data loading and preprocessing for Anomaly Detection
"""
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class AnomalyDataset(Dataset):
    """Dataset for anomaly detection"""
    
    def __init__(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        labels: Optional[np.ndarray] = None,
        window_size: Optional[int] = None,
        stride: int = 1,
        scale_data: bool = True,
        scaler: Optional[object] = None,
        contamination: float = 0.1
    ):
        """
        Args:
            data: Input data
            labels: Anomaly labels (0: normal, 1: anomaly)
            window_size: Window size for time series
            stride: Stride for sliding window
            scale_data: Whether to scale the data
            scaler: Pre-fitted scaler
            contamination: Expected proportion of anomalies
        """
        # Convert to numpy if needed
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        self.data = data
        self.labels = labels
        self.window_size = window_size
        self.stride = stride
        self.contamination = contamination
        
        # Scale data
        if scale_data:
            if scaler is None:
                self.scaler = StandardScaler()
                self.data = self.scaler.fit_transform(self.data)
            else:
                self.scaler = scaler
                self.data = self.scaler.transform(self.data)
        else:
            self.scaler = None
        
        # Create windows if window_size is specified
        if window_size:
            self.windows, self.window_labels = self._create_windows()
        else:
            self.windows = self.data
            self.window_labels = self.labels if self.labels is not None else np.zeros(len(self.data))
    
    def _create_windows(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding windows for time series"""
        windows = []
        window_labels = []
        
        for i in range(0, len(self.data) - self.window_size + 1, self.stride):
            window = self.data[i:i + self.window_size]
            windows.append(window)
            
            if self.labels is not None:
                # Label window as anomaly if any point in window is anomaly
                window_label = 1 if np.any(self.labels[i:i + self.window_size]) else 0
                window_labels.append(window_label)
            else:
                window_labels.append(0)
        
        return np.array(windows), np.array(window_labels)
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        window = self.windows[idx]
        label = self.window_labels[idx] if self.window_labels is not None else 0
        
        return {
            'data': torch.tensor(window, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long)
        }


class KDDCup99Dataset(Dataset):
    """KDD Cup 99 network intrusion detection dataset"""
    
    ATTACK_TYPES = {
        'normal': 0,
        'dos': 1,
        'u2r': 2,
        'r2l': 3,
        'probe': 4
    }
    
    def __init__(
        self,
        data_path: str,
        train: bool = True,
        scale_data: bool = True,
        binary: bool = True
    ):
        """
        Args:
            data_path: Path to KDD Cup 99 data
            train: Whether to load training data
            scale_data: Whether to scale features
            binary: Whether to use binary classification (normal vs attack)
        """
        self.data_path = Path(data_path)
        self.train = train
        self.binary = binary
        
        # Load data
        self.data, self.labels = self._load_data()
        
        # Preprocess features
        self.data = self._preprocess_features(self.data)
        
        # Scale data
        if scale_data:
            self.scaler = StandardScaler()
            self.data = self.scaler.fit_transform(self.data)
        else:
            self.scaler = None
    
    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load KDD Cup 99 data"""
        # This is a placeholder - in practice, download and load the actual dataset
        # Generate synthetic data for demonstration
        n_samples = 10000 if self.train else 2000
        n_features = 41
        
        # Generate normal data
        normal_data = np.random.randn(int(n_samples * 0.8), n_features)
        normal_labels = np.zeros(int(n_samples * 0.8))
        
        # Generate anomaly data (different distribution)
        anomaly_data = np.random.randn(int(n_samples * 0.2), n_features) * 2 + 1
        anomaly_labels = np.ones(int(n_samples * 0.2))
        
        # Combine
        data = np.vstack([normal_data, anomaly_data])
        labels = np.hstack([normal_labels, anomaly_labels])
        
        # Shuffle
        idx = np.random.permutation(len(data))
        data = data[idx]
        labels = labels[idx]
        
        return data, labels
    
    def _preprocess_features(self, data: np.ndarray) -> np.ndarray:
        """Preprocess features"""
        # Handle categorical features, etc.
        # This is simplified for demonstration
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'data': torch.tensor(self.data[idx], dtype=torch.float32),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


class CreditCardFraudDataset(Dataset):
    """Credit card fraud detection dataset"""
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        train: bool = True,
        scale_data: bool = True,
        balance_data: bool = False
    ):
        """
        Args:
            data_path: Path to credit card fraud data
            train: Whether to load training data
            scale_data: Whether to scale features
            balance_data: Whether to balance the dataset
        """
        self.train = train
        self.balance_data = balance_data
        
        # Load or generate data
        if data_path and Path(data_path).exists():
            self.data, self.labels = self._load_real_data(data_path)
        else:
            self.data, self.labels = self._generate_synthetic_data()
        
        # Balance data if requested
        if balance_data:
            self.data, self.labels = self._balance_dataset()
        
        # Scale data
        if scale_data:
            self.scaler = StandardScaler()
            self.data = self.scaler.fit_transform(self.data)
        else:
            self.scaler = None
    
    def _generate_synthetic_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic credit card fraud data"""
        n_samples = 5000 if self.train else 1000
        n_features = 30  # Similar to real credit card dataset
        
        # Normal transactions (95%)
        n_normal = int(n_samples * 0.95)
        normal_data = np.random.randn(n_normal, n_features)
        normal_data[:, 0] = np.abs(np.random.randn(n_normal) * 50)  # Transaction amount
        normal_labels = np.zeros(n_normal)
        
        # Fraudulent transactions (5%)
        n_fraud = n_samples - n_normal
        fraud_data = np.random.randn(n_fraud, n_features)
        fraud_data[:, 0] = np.abs(np.random.randn(n_fraud) * 200 + 100)  # Higher amounts
        fraud_data[:, 1:10] *= 3  # Different pattern in features
        fraud_labels = np.ones(n_fraud)
        
        # Combine
        data = np.vstack([normal_data, fraud_data])
        labels = np.hstack([normal_labels, fraud_labels])
        
        # Shuffle
        idx = np.random.permutation(len(data))
        data = data[idx]
        labels = labels[idx]
        
        return data, labels
    
    def _load_real_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load real credit card fraud data"""
        df = pd.read_csv(data_path)
        
        # Assume 'Class' column contains labels
        labels = df['Class'].values
        data = df.drop(['Class'], axis=1).values
        
        return data, labels
    
    def _balance_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Balance the dataset using undersampling"""
        normal_idx = np.where(self.labels == 0)[0]
        anomaly_idx = np.where(self.labels == 1)[0]
        
        # Undersample normal class
        n_anomalies = len(anomaly_idx)
        selected_normal = np.random.choice(normal_idx, n_anomalies, replace=False)
        
        # Combine
        selected_idx = np.concatenate([selected_normal, anomaly_idx])
        
        return self.data[selected_idx], self.labels[selected_idx]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'data': torch.tensor(self.data[idx], dtype=torch.float32),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


class TimeSeriesAnomalyDataset(Dataset):
    """Time series anomaly detection dataset"""
    
    def __init__(
        self,
        sequence_length: int = 100,
        n_features: int = 1,
        n_samples: int = 1000,
        anomaly_ratio: float = 0.1,
        anomaly_type: str = 'point'
    ):
        """
        Args:
            sequence_length: Length of time series sequences
            n_features: Number of features
            n_samples: Number of samples
            anomaly_ratio: Ratio of anomalies
            anomaly_type: Type of anomaly ('point', 'contextual', 'collective')
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_samples = n_samples
        self.anomaly_ratio = anomaly_ratio
        self.anomaly_type = anomaly_type
        
        # Generate data
        self.data, self.labels = self._generate_time_series()
    
    def _generate_time_series(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate time series with anomalies"""
        data = []
        labels = []
        
        n_anomalies = int(self.n_samples * self.anomaly_ratio)
        anomaly_indices = np.random.choice(self.n_samples, n_anomalies, replace=False)
        
        for i in range(self.n_samples):
            # Generate base time series
            t = np.linspace(0, 4 * np.pi, self.sequence_length)
            
            if self.n_features == 1:
                series = np.sin(t) + np.random.randn(self.sequence_length) * 0.1
                series = series.reshape(-1, 1)
            else:
                series = []
                for j in range(self.n_features):
                    component = np.sin(t * (j + 1)) + np.random.randn(self.sequence_length) * 0.1
                    series.append(component)
                series = np.array(series).T
            
            # Add anomalies
            if i in anomaly_indices:
                series = self._add_anomaly(series)
                labels.append(1)
            else:
                labels.append(0)
            
            data.append(series)
        
        return np.array(data), np.array(labels)
    
    def _add_anomaly(self, series: np.ndarray) -> np.ndarray:
        """Add anomaly to time series"""
        if self.anomaly_type == 'point':
            # Random point anomalies
            n_points = np.random.randint(1, 5)
            anomaly_points = np.random.choice(len(series), n_points, replace=False)
            series[anomaly_points] += np.random.randn(n_points, series.shape[1]) * 3
            
        elif self.anomaly_type == 'contextual':
            # Contextual anomaly (out of context)
            start = np.random.randint(0, len(series) - 20)
            series[start:start + 20] = -series[start:start + 20]
            
        elif self.anomaly_type == 'collective':
            # Collective anomaly (abnormal pattern)
            start = np.random.randint(0, len(series) - 30)
            t = np.linspace(0, 2 * np.pi, 30)
            abnormal_pattern = np.sin(t * 10) * 2
            series[start:start + 30, 0] = abnormal_pattern
        
        return series
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'data': torch.tensor(self.data[idx], dtype=torch.float32),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


class AnomalyDataModule:
    """Data module for anomaly detection"""
    
    def __init__(
        self,
        dataset_name: str = 'synthetic',
        data_path: Optional[str] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        window_size: Optional[int] = None,
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15
    ):
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.window_size = window_size
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.scaler = None
    
    def setup(self):
        """Setup datasets"""
        if self.dataset_name == 'kdd':
            # KDD Cup 99 dataset
            full_dataset = KDDCup99Dataset(
                data_path=self.data_path or './data/kdd',
                train=True
            )
        elif self.dataset_name == 'credit':
            # Credit card fraud dataset
            full_dataset = CreditCardFraudDataset(
                data_path=self.data_path,
                train=True
            )
        elif self.dataset_name == 'timeseries':
            # Time series anomaly dataset
            full_dataset = TimeSeriesAnomalyDataset(
                n_samples=5000,
                anomaly_ratio=0.1
            )
        else:
            # Synthetic dataset
            data, labels = self._generate_synthetic_data()
            full_dataset = AnomalyDataset(
                data,
                labels,
                window_size=self.window_size
            )
        
        # Split dataset
        n = len(full_dataset)
        train_size = int(n * self.train_split)
        val_size = int(n * self.val_split)
        test_size = n - train_size - val_size
        
        self.train_dataset, self.val_dataset, self.test_dataset = \
            torch.utils.data.random_split(
                full_dataset,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42)
            )
        
        # Store scaler if available
        if hasattr(full_dataset, 'scaler'):
            self.scaler = full_dataset.scaler
    
    def _generate_synthetic_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic anomaly detection data"""
        n_samples = 5000
        n_features = 10
        
        # Generate normal data (Gaussian)
        normal_data = np.random.randn(int(n_samples * 0.9), n_features)
        normal_labels = np.zeros(int(n_samples * 0.9))
        
        # Generate anomalies (different distributions)
        anomaly_data = []
        anomaly_labels = []
        
        # Type 1: Outliers
        outliers = np.random.randn(int(n_samples * 0.05), n_features) * 5
        anomaly_data.append(outliers)
        anomaly_labels.append(np.ones(len(outliers)))
        
        # Type 2: Different mean
        shifted = np.random.randn(int(n_samples * 0.05), n_features) + 3
        anomaly_data.append(shifted)
        anomaly_labels.append(np.ones(len(shifted)))
        
        # Combine all data
        anomaly_data = np.vstack(anomaly_data)
        anomaly_labels = np.hstack(anomaly_labels)
        
        data = np.vstack([normal_data, anomaly_data])
        labels = np.hstack([normal_labels, anomaly_labels])
        
        # Shuffle
        idx = np.random.permutation(len(data))
        data = data[idx]
        labels = labels[idx]
        
        return data, labels
    
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


if __name__ == "__main__":
    # Test data loading
    print("Testing anomaly detection data loading...")
    
    # Test basic dataset
    data = np.random.randn(1000, 10)
    labels = np.random.randint(0, 2, 1000)
    
    dataset = AnomalyDataset(data, labels)
    print(f"Dataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"Sample data shape: {sample['data'].shape}")
    print(f"Sample label: {sample['label']}")
    
    # Test data module
    dm = AnomalyDataModule(
        dataset_name='synthetic',
        batch_size=32
    )
    
    dm.setup()
    print(f"Train dataset size: {len(dm.train_dataset)}")
    print(f"Val dataset size: {len(dm.val_dataset)}")
    print(f"Test dataset size: {len(dm.test_dataset)}")
    
    # Test time series dataset
    ts_dataset = TimeSeriesAnomalyDataset(
        n_samples=100,
        anomaly_ratio=0.1
    )
    print(f"Time series dataset size: {len(ts_dataset)}")
    print(f"Number of anomalies: {np.sum(ts_dataset.labels)}")