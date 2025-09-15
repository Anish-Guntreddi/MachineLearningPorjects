"""
Utility functions for Automatic Speech Recognition
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
from typing import List, Tuple, Optional, Dict
from jiwer import wer, cer, mer
import torchaudio
import torchaudio.transforms as T
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


def compute_features(
    waveform: torch.Tensor,
    sr: int = 16000,
    feature_type: str = 'melspec',
    n_mels: int = 80,
    n_mfcc: int = 40,
    n_fft: int = 400,
    hop_length: int = 160
) -> torch.Tensor:
    """
    Compute acoustic features from waveform
    
    Args:
        waveform: Audio waveform tensor
        sr: Sample rate
        feature_type: Type of features ('melspec', 'mfcc', 'spectrogram')
        n_mels: Number of mel bands
        n_mfcc: Number of MFCC coefficients
        n_fft: FFT size
        hop_length: Hop length
    
    Returns:
        Feature tensor
    """
    if feature_type == 'melspec':
        transform = T.MelSpectrogram(
            sample_rate=sr,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length
        )
        features = transform(waveform)
        features = T.AmplitudeToDB()(features)
        
    elif feature_type == 'mfcc':
        transform = T.MFCC(
            sample_rate=sr,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_mels': n_mels,
                'n_fft': n_fft,
                'hop_length': hop_length
            }
        )
        features = transform(waveform)
        
    elif feature_type == 'spectrogram':
        transform = T.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length
        )
        features = transform(waveform)
        features = T.AmplitudeToDB()(features)
        
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")
    
    # Transpose to (time, features)
    features = features.transpose(-1, -2)
    
    return features


def plot_waveform(
    waveform: np.ndarray,
    sr: int,
    title: str = 'Waveform',
    save_path: Optional[str] = None
):
    """Plot audio waveform"""
    plt.figure(figsize=(14, 4))
    
    time = np.arange(0, len(waveform)) / sr
    plt.plot(time, waveform, linewidth=0.5)
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
    plt.figure(figsize=(14, 6))
    
    D = librosa.stft(waveform, n_fft=2048, hop_length=512)
    D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
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
    n_mels: int = 80,
    title: str = 'Mel Spectrogram',
    save_path: Optional[str] = None
):
    """Plot mel spectrogram"""
    plt.figure(figsize=(14, 6))
    
    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_mels=n_mels,
        n_fft=2048,
        hop_length=512
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
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


def plot_attention_weights(
    attention_weights: np.ndarray,
    input_labels: Optional[List[str]] = None,
    output_labels: Optional[List[str]] = None,
    title: str = 'Attention Weights',
    save_path: Optional[str] = None
):
    """Plot attention weights"""
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(
        attention_weights,
        cmap='Blues',
        cbar_kws={'label': 'Attention Weight'},
        xticklabels=input_labels if input_labels else False,
        yticklabels=output_labels if output_labels else False
    )
    
    plt.title(title)
    plt.xlabel('Input Sequence')
    plt.ylabel('Output Sequence')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def save_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_metrics: List[float],
    val_metrics: List[float],
    save_path: Optional[str] = None,
    metric_name: str = 'WER'
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
    
    # Metric plot
    ax2.plot(epochs, train_metrics, 'b-', label=f'Train {metric_name}')
    ax2.plot(epochs, val_metrics, 'r-', label=f'Val {metric_name}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel(metric_name)
    ax2.set_title(f'Training and Validation {metric_name}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def calculate_metrics(
    predictions: List[str],
    references: List[str]
) -> Dict[str, float]:
    """
    Calculate ASR metrics
    
    Args:
        predictions: List of predicted transcripts
        references: List of reference transcripts
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'wer': wer(references, predictions),
        'cer': cer(references, predictions),
        'mer': mer(references, predictions)
    }
    
    return metrics


def beam_search_decode(
    log_probs: torch.Tensor,
    vocab: List[str],
    beam_width: int = 5,
    blank_idx: int = 0,
    use_lm: bool = False,
    lm_weight: float = 0.1
) -> List[str]:
    """
    Beam search decoding for CTC models
    
    Args:
        log_probs: Log probabilities from model (time, vocab_size)
        vocab: Vocabulary list
        beam_width: Beam width
        blank_idx: Index of blank token
        use_lm: Whether to use language model
        lm_weight: Language model weight
    
    Returns:
        List of decoded strings
    """
    T, V = log_probs.shape
    
    # Initialize beams
    beams = [([], 0.0)]  # (prefix, score)
    
    for t in range(T):
        new_beams = []
        
        for prefix, score in beams:
            for v in range(V):
                if v == blank_idx:
                    # Add blank
                    new_score = score + log_probs[t, v].item()
                    new_beams.append((prefix, new_score))
                else:
                    # Add character
                    char = vocab[v]
                    
                    # Check for repetition
                    if prefix and prefix[-1] == char:
                        # Same character, don't add
                        new_score = score + log_probs[t, v].item()
                        new_beams.append((prefix, new_score))
                    else:
                        # New character
                        new_prefix = prefix + [char]
                        new_score = score + log_probs[t, v].item()
                        
                        # Apply language model if available
                        if use_lm:
                            # Placeholder for LM score
                            lm_score = 0.0
                            new_score += lm_weight * lm_score
                        
                        new_beams.append((new_prefix, new_score))
        
        # Prune beams
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_width]
    
    # Get best beam
    best_prefix, _ = beams[0]
    return ''.join(best_prefix)


def greedy_decode(
    log_probs: torch.Tensor,
    vocab: List[str],
    blank_idx: int = 0
) -> str:
    """
    Greedy decoding for CTC models
    
    Args:
        log_probs: Log probabilities from model (time, vocab_size)
        vocab: Vocabulary list
        blank_idx: Index of blank token
    
    Returns:
        Decoded string
    """
    # Get most likely tokens
    tokens = log_probs.argmax(dim=-1)
    
    # Remove blanks and repetitions
    decoded = []
    prev_token = None
    
    for token in tokens:
        if token != blank_idx and token != prev_token:
            decoded.append(vocab[token])
        prev_token = token
    
    return ''.join(decoded)


def apply_spec_augment(
    spectrogram: torch.Tensor,
    time_mask_param: int = 10,
    freq_mask_param: int = 10,
    time_masks: int = 2,
    freq_masks: int = 2
) -> torch.Tensor:
    """
    Apply SpecAugment to spectrogram
    
    Args:
        spectrogram: Spectrogram tensor (time, freq)
        time_mask_param: Maximum time mask length
        freq_mask_param: Maximum frequency mask length
        time_masks: Number of time masks
        freq_masks: Number of frequency masks
    
    Returns:
        Augmented spectrogram
    """
    augmented = spectrogram.clone()
    T, F = augmented.shape
    
    # Time masking
    for _ in range(time_masks):
        t = np.random.randint(0, time_mask_param)
        t0 = np.random.randint(0, max(1, T - t))
        augmented[t0:t0 + t, :] = 0
    
    # Frequency masking
    for _ in range(freq_masks):
        f = np.random.randint(0, freq_mask_param)
        f0 = np.random.randint(0, max(1, F - f))
        augmented[:, f0:f0 + f] = 0
    
    return augmented


def normalize_audio(
    waveform: np.ndarray,
    target_db: float = -20.0
) -> np.ndarray:
    """
    Normalize audio to target dB level
    
    Args:
        waveform: Audio waveform
        target_db: Target dB level
    
    Returns:
        Normalized waveform
    """
    # Calculate current dB
    rms = np.sqrt(np.mean(waveform ** 2))
    current_db = 20 * np.log10(rms + 1e-10)
    
    # Calculate gain
    gain_db = target_db - current_db
    gain = 10 ** (gain_db / 20)
    
    # Apply gain
    normalized = waveform * gain
    
    # Clip to prevent overflow
    normalized = np.clip(normalized, -1.0, 1.0)
    
    return normalized


def remove_silence(
    waveform: np.ndarray,
    sr: int,
    top_db: int = 30
) -> np.ndarray:
    """
    Remove silence from audio
    
    Args:
        waveform: Audio waveform
        sr: Sample rate
        top_db: Threshold in dB below reference
    
    Returns:
        Trimmed waveform
    """
    trimmed, _ = librosa.effects.trim(
        waveform,
        top_db=top_db
    )
    
    return trimmed


if __name__ == "__main__":
    # Test utilities
    print("Testing ASR utility functions...")
    
    # Generate dummy audio
    sr = 16000
    duration = 3
    t = np.linspace(0, duration, sr * duration)
    waveform = np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))
    
    # Test feature computation
    waveform_tensor = torch.from_numpy(waveform).float().unsqueeze(0)
    melspec = compute_features(waveform_tensor, sr, 'melspec')
    print(f"Mel spectrogram shape: {melspec.shape}")
    
    # Test metrics
    predictions = ["hello world", "speech recognition"]
    references = ["hello world", "automatic speech recognition"]
    metrics = calculate_metrics(predictions, references)
    print(f"Metrics: {metrics}")
    
    # Test early stopping
    early_stop = EarlyStopping(patience=3, mode='min')
    scores = [0.5, 0.45, 0.46, 0.47, 0.48]
    for i, score in enumerate(scores):
        if early_stop(score):
            print(f"Early stopping at iteration {i+1}")
            break