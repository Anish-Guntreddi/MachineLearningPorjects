"""
Utility functions for Speech Emotion Recognition
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import librosa
import librosa.display
from typing import List, Tuple, Optional, Dict
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


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    title: str = 'Confusion Matrix'
):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Percentage'}
    )
    
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def plot_waveform(
    waveform: np.ndarray,
    sr: int,
    title: str = 'Waveform',
    save_path: Optional[str] = None
):
    """Plot audio waveform"""
    plt.figure(figsize=(12, 4))
    
    time = np.arange(0, len(waveform)) / sr
    plt.plot(time, waveform)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def plot_spectrogram(
    waveform: np.ndarray,
    sr: int,
    title: str = 'Spectrogram',
    save_path: Optional[str] = None
):
    """Plot spectrogram"""
    plt.figure(figsize=(12, 6))
    
    # Compute spectrogram
    D = librosa.stft(waveform)
    D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    # Plot
    librosa.display.specshow(
        D_db,
        sr=sr,
        x_axis='time',
        y_axis='hz',
        cmap='viridis'
    )
    
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def plot_mel_spectrogram(
    waveform: np.ndarray,
    sr: int,
    n_mels: int = 128,
    title: str = 'Mel Spectrogram',
    save_path: Optional[str] = None
):
    """Plot mel spectrogram"""
    plt.figure(figsize=(12, 6))
    
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_mels=n_mels
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Plot
    librosa.display.specshow(
        mel_spec_db,
        sr=sr,
        x_axis='time',
        y_axis='mel',
        cmap='viridis'
    )
    
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def plot_mfcc(
    waveform: np.ndarray,
    sr: int,
    n_mfcc: int = 40,
    title: str = 'MFCC',
    save_path: Optional[str] = None
):
    """Plot MFCC features"""
    plt.figure(figsize=(12, 6))
    
    # Compute MFCC
    mfcc = librosa.feature.mfcc(
        y=waveform,
        sr=sr,
        n_mfcc=n_mfcc
    )
    
    # Plot
    librosa.display.specshow(
        mfcc,
        sr=sr,
        x_axis='time',
        cmap='viridis'
    )
    
    plt.colorbar()
    plt.title(title)
    plt.ylabel('MFCC Coefficients')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def save_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: Optional[str] = None
):
    """Save training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-', label='Train Acc')
    ax2.plot(epochs, val_accs, 'r-', label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def extract_audio_features(
    waveform: np.ndarray,
    sr: int,
    feature_types: List[str] = ['mfcc', 'spectral', 'prosodic']
) -> Dict[str, np.ndarray]:
    """
    Extract various audio features
    
    Args:
        waveform: Audio waveform
        sr: Sample rate
        feature_types: List of feature types to extract
    
    Returns:
        Dictionary of features
    """
    features = {}
    
    if 'mfcc' in feature_types:
        # MFCC features
        mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=40)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        features['mfcc'] = mfcc
        features['mfcc_delta'] = mfcc_delta
        features['mfcc_delta2'] = mfcc_delta2
        features['mfcc_mean'] = np.mean(mfcc, axis=1)
        features['mfcc_std'] = np.std(mfcc, axis=1)
    
    if 'spectral' in feature_types:
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=waveform, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=waveform, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=waveform, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(waveform)
        
        features['spectral_centroid'] = np.mean(spectral_centroid)
        features['spectral_bandwidth'] = np.mean(spectral_bandwidth)
        features['spectral_rolloff'] = np.mean(spectral_rolloff)
        features['spectral_contrast'] = np.mean(spectral_contrast, axis=1)
        features['zero_crossing_rate'] = np.mean(zero_crossing_rate)
    
    if 'prosodic' in feature_types:
        # Prosodic features (pitch, energy)
        pitches, magnitudes = librosa.piptrack(y=waveform, sr=sr)
        
        # Get pitch values
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if pitch_values:
            features['pitch_mean'] = np.mean(pitch_values)
            features['pitch_std'] = np.std(pitch_values)
            features['pitch_min'] = np.min(pitch_values)
            features['pitch_max'] = np.max(pitch_values)
        
        # Energy
        energy = np.sum(waveform ** 2) / len(waveform)
        features['energy'] = energy
        
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=waveform, sr=sr)
        features['tempo'] = tempo
    
    return features


def augment_audio(
    waveform: np.ndarray,
    sr: int,
    augmentation_type: str = 'pitch_shift',
    **kwargs
) -> np.ndarray:
    """
    Apply audio augmentation
    
    Args:
        waveform: Audio waveform
        sr: Sample rate
        augmentation_type: Type of augmentation
        **kwargs: Additional parameters
    
    Returns:
        Augmented waveform
    """
    if augmentation_type == 'pitch_shift':
        n_steps = kwargs.get('n_steps', np.random.randint(-3, 4))
        return librosa.effects.pitch_shift(waveform, sr=sr, n_steps=n_steps)
    
    elif augmentation_type == 'time_stretch':
        rate = kwargs.get('rate', np.random.uniform(0.8, 1.2))
        return librosa.effects.time_stretch(waveform, rate=rate)
    
    elif augmentation_type == 'add_noise':
        noise_level = kwargs.get('noise_level', 0.005)
        noise = np.random.randn(len(waveform)) * noise_level
        return waveform + noise
    
    elif augmentation_type == 'shift':
        shift_max = kwargs.get('shift_max', int(sr * 0.2))
        shift = np.random.randint(-shift_max, shift_max)
        return np.roll(waveform, shift)
    
    else:
        return waveform


def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """Compute class weights for imbalanced datasets"""
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(labels)
    weights = compute_class_weight(
        'balanced',
        classes=classes,
        y=labels
    )
    
    return torch.tensor(weights, dtype=torch.float32)


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    top_k: int = 20,
    save_path: Optional[str] = None
):
    """Plot feature importance"""
    # Sort by importance
    indices = np.argsort(importances)[::-1][:top_k]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(top_k), importances[indices])
    plt.xticks(range(top_k), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title(f'Top {top_k} Feature Importances')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Generate dummy audio
    sr = 16000
    duration = 3
    t = np.linspace(0, duration, sr * duration)
    waveform = np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))
    
    # Test feature extraction
    features = extract_audio_features(waveform, sr)
    print(f"Extracted features: {list(features.keys())}")
    
    # Test augmentation
    augmented = augment_audio(waveform, sr, 'pitch_shift', n_steps=2)
    print(f"Original shape: {waveform.shape}, Augmented shape: {augmented.shape}")
    
    # Test early stopping
    early_stop = EarlyStopping(patience=3, mode='max')
    scores = [0.7, 0.75, 0.74, 0.73, 0.72]
    for i, score in enumerate(scores):
        if early_stop(score):
            print(f"Early stopping at iteration {i+1}")
            break