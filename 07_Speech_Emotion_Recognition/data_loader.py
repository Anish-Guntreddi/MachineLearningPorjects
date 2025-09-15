"""
Data loading and preprocessing for Speech Emotion Recognition
"""
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import librosa
import soundfile as sf
from sklearn.preprocessing import LabelEncoder


class EmotionDataset(Dataset):
    """Dataset for emotion recognition from speech"""
    
    def __init__(
        self,
        data_path: str,
        csv_file: Optional[str] = None,
        sr: int = 16000,
        duration: float = 3.0,
        augment: bool = False,
        feature_type: str = 'raw'  # 'raw', 'mfcc', 'melspec', 'combined'
    ):
        """
        Args:
            data_path: Path to audio files
            csv_file: CSV file with file paths and labels
            sr: Sample rate
            duration: Duration to pad/crop audio
            augment: Whether to apply augmentation
            feature_type: Type of features to extract
        """
        self.data_path = Path(data_path)
        self.sr = sr
        self.duration = duration
        self.augment = augment
        self.feature_type = feature_type
        self.target_length = int(sr * duration)
        
        # Load metadata
        if csv_file:
            self.df = pd.read_csv(csv_file)
            self.file_paths = self.df['path'].tolist()
            self.labels = self.df['emotion'].tolist()
        else:
            # Assume folder structure: data_path/emotion/audio_file
            self.file_paths = []
            self.labels = []
            for emotion_dir in self.data_path.iterdir():
                if emotion_dir.is_dir():
                    for audio_file in emotion_dir.glob('*.wav'):
                        self.file_paths.append(str(audio_file))
                        self.labels.append(emotion_dir.name)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)
        self.num_classes = len(self.label_encoder.classes_)
        
        # Initialize transforms
        self._init_transforms()
    
    def _init_transforms(self):
        """Initialize audio transforms"""
        # Basic transforms
        self.resample = T.Resample(orig_freq=self.sr, new_freq=self.sr)
        
        # Feature extraction transforms
        self.mfcc_transform = T.MFCC(
            sample_rate=self.sr,
            n_mfcc=40,
            melkwargs={'n_mels': 128, 'n_fft': 2048, 'hop_length': 512}
        )
        
        self.melspec_transform = T.MelSpectrogram(
            sample_rate=self.sr,
            n_mels=128,
            n_fft=2048,
            hop_length=512
        )
        
        # Augmentation transforms
        if self.augment:
            self.time_stretch = T.TimeStretch()
            self.pitch_shift = lambda x: self._pitch_shift(x)
            self.add_noise = lambda x: self._add_noise(x)
    
    def _pitch_shift(self, waveform: torch.Tensor, shift_steps: int = None) -> torch.Tensor:
        """Apply pitch shifting"""
        if shift_steps is None:
            shift_steps = np.random.randint(-3, 4)
        
        # Use torchaudio's pitch shift if available, otherwise use librosa
        try:
            shifted = T.PitchShift(self.sr, shift_steps)(waveform)
        except:
            audio_np = waveform.numpy().squeeze()
            shifted_np = librosa.effects.pitch_shift(audio_np, sr=self.sr, n_steps=shift_steps)
            shifted = torch.from_numpy(shifted_np).unsqueeze(0)
        
        return shifted
    
    def _add_noise(self, waveform: torch.Tensor, noise_level: float = 0.005) -> torch.Tensor:
        """Add white noise to audio"""
        noise = torch.randn_like(waveform) * noise_level
        return waveform + noise
    
    def _pad_or_crop(self, waveform: torch.Tensor) -> torch.Tensor:
        """Pad or crop waveform to target length"""
        length = waveform.shape[-1]
        
        if length < self.target_length:
            # Pad with zeros
            padding = self.target_length - length
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif length > self.target_length:
            # Random crop
            if self.augment:
                start = np.random.randint(0, length - self.target_length)
            else:
                start = 0
            waveform = waveform[..., start:start + self.target_length]
        
        return waveform
    
    def _extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract features based on feature_type"""
        if self.feature_type == 'raw':
            return waveform
        
        elif self.feature_type == 'mfcc':
            mfcc = self.mfcc_transform(waveform)
            # Calculate deltas
            delta = torchaudio.functional.compute_deltas(mfcc)
            delta2 = torchaudio.functional.compute_deltas(delta)
            features = torch.cat([mfcc, delta, delta2], dim=0)
            return features
        
        elif self.feature_type == 'melspec':
            melspec = self.melspec_transform(waveform)
            melspec_db = torchaudio.functional.amplitude_to_db(melspec)
            return melspec_db
        
        elif self.feature_type == 'combined':
            # Combine multiple features
            mfcc = self.mfcc_transform(waveform)
            melspec = self.melspec_transform(waveform)
            melspec_db = torchaudio.functional.amplitude_to_db(melspec)
            
            # Compute statistics
            mfcc_mean = mfcc.mean(dim=-1, keepdim=True)
            mfcc_std = mfcc.std(dim=-1, keepdim=True)
            
            # Concatenate features
            features = torch.cat([
                mfcc,
                melspec_db[:40],  # Use first 40 mel bands
                mfcc_mean.expand(-1, -1, mfcc.shape[-1]),
                mfcc_std.expand(-1, -1, mfcc.shape[-1])
            ], dim=0)
            
            return features
        
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # Load audio
        audio_path = self.file_paths[idx]
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if necessary
        if sr != self.sr:
            waveform = self.resample(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Pad or crop
        waveform = self._pad_or_crop(waveform)
        
        # Apply augmentation
        if self.augment and np.random.random() > 0.5:
            if np.random.random() > 0.5:
                waveform = self._pitch_shift(waveform)
            if np.random.random() > 0.5:
                waveform = self._add_noise(waveform)
        
        # Extract features
        features = self._extract_features(waveform)
        
        # Get label
        label = self.encoded_labels[idx]
        
        return {
            'features': features.float(),
            'label': torch.tensor(label, dtype=torch.long),
            'audio_path': audio_path
        }


class RAVDESSDataset(EmotionDataset):
    """RAVDESS dataset specific loader"""
    
    EMOTIONS = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }
    
    def __init__(self, data_path: str, **kwargs):
        super().__init__(data_path, **kwargs)
        self._parse_ravdess_labels()
    
    def _parse_ravdess_labels(self):
        """Parse RAVDESS filename format for labels"""
        parsed_labels = []
        for file_path in self.file_paths:
            filename = Path(file_path).stem
            parts = filename.split('-')
            if len(parts) >= 3:
                emotion_code = parts[2]
                emotion = self.EMOTIONS.get(emotion_code, 'unknown')
                parsed_labels.append(emotion)
            else:
                parsed_labels.append('unknown')
        
        self.labels = parsed_labels
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)


class EmotionDataModule:
    """Data module for emotion recognition"""
    
    def __init__(
        self,
        dataset_name: str = 'ravdess',
        data_path: str = './data',
        batch_size: int = 32,
        num_workers: int = 4,
        feature_type: str = 'mfcc',
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15
    ):
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.feature_type = feature_type
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self):
        """Setup datasets"""
        if self.dataset_name == 'ravdess':
            dataset_class = RAVDESSDataset
        else:
            dataset_class = EmotionDataset
        
        # Load full dataset
        full_dataset = dataset_class(
            self.data_path,
            feature_type=self.feature_type,
            augment=False
        )
        
        # Split dataset
        total_size = len(full_dataset)
        train_size = int(self.train_split * total_size)
        val_size = int(self.val_split * total_size)
        test_size = total_size - train_size - val_size
        
        train_data, val_data, test_data = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create augmented training dataset
        self.train_dataset = dataset_class(
            self.data_path,
            feature_type=self.feature_type,
            augment=True
        )
        self.train_dataset.file_paths = [full_dataset.file_paths[i] for i in train_data.indices]
        self.train_dataset.encoded_labels = [full_dataset.encoded_labels[i] for i in train_data.indices]
        
        self.val_dataset = val_data
        self.test_dataset = test_data
        
        # Store label encoder
        self.label_encoder = full_dataset.label_encoder
        self.num_classes = full_dataset.num_classes
    
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
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function for variable length sequences"""
    features = []
    labels = []
    paths = []
    
    for item in batch:
        features.append(item['features'])
        labels.append(item['label'])
        paths.append(item['audio_path'])
    
    # Pad features to same length
    max_len = max(f.shape[-1] for f in features)
    padded_features = []
    
    for f in features:
        if f.shape[-1] < max_len:
            padding = max_len - f.shape[-1]
            f = torch.nn.functional.pad(f, (0, padding))
        padded_features.append(f)
    
    return {
        'features': torch.stack(padded_features),
        'labels': torch.stack(labels),
        'paths': paths
    }


if __name__ == "__main__":
    # Test data loading
    dataset = EmotionDataset(
        data_path="./data/ravdess",
        feature_type='mfcc',
        augment=True
    )
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Features shape: {sample['features'].shape}")
        print(f"Label: {sample['label']}")
        print(f"Audio path: {sample['audio_path']}")
    
    # Test data module
    dm = EmotionDataModule(
        dataset_name='ravdess',
        data_path='./data/ravdess',
        batch_size=16,
        feature_type='combined'
    )
    
    dm.setup()
    print(f"Number of classes: {dm.num_classes}")
    print(f"Training samples: {len(dm.train_dataset)}")