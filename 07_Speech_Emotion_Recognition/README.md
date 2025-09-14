# Speech Emotion Recognition Project - RAVDESS / CREMA-D

## 1. Problem Definition & Use Case

**Problem:** Automatically identify and classify human emotions from speech audio signals, accounting for speaker variability, recording conditions, and cultural differences in emotional expression.

**Use Case:** Speech emotion recognition enables empathetic AI systems across:
- Mental health monitoring and therapy assistance
- Customer service quality assessment and routing
- Human-computer interaction with emotional awareness
- Educational technology for learning assessment
- Automotive safety systems (driver stress detection)
- Smart home assistants with emotional context
- Gaming and entertainment with responsive characters
- Call center analytics and agent training

**Business Impact:** Emotion-aware systems improve customer satisfaction by 40%, reduce support escalations by 25%, and enable personalized experiences that increase user engagement by 60%.

## 2. Dataset Acquisition & Preprocessing

### Primary Datasets
- **RAVDESS (Ryerson Audio-Visual Database)**: 24 actors, 8 emotions, 1440 audio files
  ```python
  import kaggle
  kaggle.api.dataset_download_files('uwrfkaggler/ravdess-emotional-speech-audio')
  ```
- **CREMA-D (Crowdsourced Emotional Multimodal Actors Dataset)**: 91 actors, 6 emotions, 7442 audio clips
  ```python
  import requests
  url = "https://github.com/CheyneyComputerScience/CREMA-D/releases/download/v1.3/AudioWAV.zip"
  response = requests.get(url)
  ```
- **IEMOCAP**: Multimodal emotional database with 12 hours of audio
  ```bash
  # Requires academic license
  wget https://sail.usc.edu/iemocap/iemocap_release.tar.gz
  ```
- **EmoDB**: German emotional speech database
  ```python
  from datasets import load_dataset
  emodb = load_dataset('superb', 'er')
  ```

### Data Schema
```python
{
    'audio': {
        'path': str,           # File path to audio
        'array': np.ndarray,   # Raw audio waveform
        'sampling_rate': int,  # Sample rate (typically 16kHz)
    },
    'emotion': str,            # Emotion label (happy, sad, angry, etc.)
    'emotion_id': int,         # Numeric emotion class
    'intensity': str,          # Emotion intensity (normal, strong)
    'statement': str,          # Spoken text content
    'repetition': int,         # Repetition number
    'actor_id': int,          # Speaker identifier
    'gender': str,            # Speaker gender
    'duration': float,        # Audio duration in seconds
}
```

### Preprocessing Pipeline
```python
import librosa
import numpy as np
import pandas as pd
from scipy import signal
import torch
import torchaudio
from transformers import Wav2Vec2Processor

def load_and_preprocess_audio(file_path, target_sr=16000, duration=None):
    """Load and standardize audio file"""
    # Load audio
    audio, sr = librosa.load(file_path, sr=target_sr)
    
    # Normalize audio amplitude
    audio = librosa.util.normalize(audio)
    
    # Trim silence
    audio, _ = librosa.effects.trim(audio, top_db=20)
    
    # Pad or truncate to fixed duration if specified
    if duration:
        target_length = int(duration * target_sr)
        if len(audio) < target_length:
            # Pad with zeros
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            # Truncate
            audio = audio[:target_length]
    
    return audio, target_sr

def extract_acoustic_features(audio, sr):
    """Extract comprehensive acoustic features"""
    features = {}
    
    # Mel-frequency cepstral coefficients (MFCCs)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    features['mfcc_mean'] = np.mean(mfccs, axis=1)
    features['mfcc_std'] = np.std(mfccs, axis=1)
    
    # Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    features['spectral_centroid_mean'] = np.mean(spectral_centroids)
    features['spectral_centroid_std'] = np.std(spectral_centroids)
    
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
    features['spectral_rolloff_std'] = np.std(spectral_rolloff)
    
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_std'] = np.std(zcr)
    
    # Chroma features
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    features['chroma_mean'] = np.mean(chroma, axis=1)
    features['chroma_std'] = np.std(chroma, axis=1)
    
    # Tonnetz (harmonic features)
    tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
    features['tonnetz_mean'] = np.mean(tonnetz, axis=1)
    features['tonnetz_std'] = np.std(tonnetz, axis=1)
    
    return features

def preprocess_dataset(data_path, emotion_mapping):
    """Preprocess entire dataset"""
    processed_data = []
    
    for file_path in data_path.glob('*.wav'):
        # Extract emotion from filename (RAVDESS/CREMA-D specific)
        emotion_label = extract_emotion_from_filename(file_path.name)
        
        # Load and process audio
        audio, sr = load_and_preprocess_audio(file_path, duration=3.0)
        
        # Extract features
        features = extract_acoustic_features(audio, sr)
        
        # Create sample
        sample = {
            'audio_path': str(file_path),
            'audio_array': audio,
            'emotion': emotion_label,
            'emotion_id': emotion_mapping[emotion_label],
            'features': features,
            'duration': len(audio) / sr
        }
        
        processed_data.append(sample)
    
    return processed_data

# Emotion mappings
RAVDESS_EMOTIONS = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

CREMA_EMOTIONS = {
    'NEU': 'neutral', 'HAP': 'happy', 'SAD': 'sad', 
    'ANG': 'angry', 'FEA': 'fearful', 'DIS': 'disgust'
}
```

### Feature Engineering
- **Prosodic features**: Pitch, tempo, rhythm, stress patterns
- **Spectral features**: Spectral centroid, bandwidth, rolloff
- **Voice quality**: Jitter, shimmer, harmonics-to-noise ratio
- **Linguistic features**: Text sentiment, word embeddings
- **Temporal dynamics**: Feature trajectories and delta coefficients

## 3. Baseline Models

### Traditional Machine Learning Baseline
```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def create_traditional_ml_baseline(features, labels):
    """Traditional ML baseline with hand-crafted features"""
    
    # Prepare feature matrix
    X = np.array([sample['features'] for sample in features])
    y = np.array(labels)
    
    # Flatten feature dictionaries
    feature_vectors = []
    for sample in features:
        vector = []
        for key, value in sample.items():
            if isinstance(value, np.ndarray):
                vector.extend(value.flatten())
            else:
                vector.append(value)
        feature_vectors.append(vector)
    
    X = np.array(feature_vectors)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred),
            'model': model
        }
    
    return results, scaler
```
**Expected Performance:** 60-70% accuracy on RAVDESS, 55-65% on CREMA-D

### CNN-based Audio Classification
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

class AudioCNN(nn.Module):
    def __init__(self, num_classes=8, input_length=48000):
        super(AudioCNN, self).__init__()
        
        # Mel-spectrogram transformation
        self.mel_transform = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            hop_length=512,
            n_mels=128
        )
        
        # CNN layers for spectrogram processing
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        
        # Calculate flattened size
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 128, 94)  # Approximate mel-spec size
            conv_output = self.conv_layers(dummy_input)
            self.flattened_size = conv_output.numel()
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # Convert audio to mel-spectrogram
        x = self.mel_transform(x)
        x = torch.log(x + 1e-8)  # Log scale
        
        # Add channel dimension
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        
        # CNN processing
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        return x

# Training function
def train_cnn_model(model, train_loader, val_loader, num_epochs=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    
    train_losses, val_losses = [], []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        val_accuracy = 100. * correct / len(val_loader.dataset)
        
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}: '
              f'Train Loss: {train_losses[-1]:.4f}, '
              f'Val Loss: {val_losses[-1]:.4f}, '
              f'Val Acc: {val_accuracy:.2f}%')
    
    return model, train_losses, val_losses
```
**Expected Performance:** 70-80% accuracy with data augmentation

## 4. Advanced/Stretch Models

### State-of-the-Art Architectures

1. **Wav2Vec2 for Emotion Recognition**
```python
from transformers import (
    Wav2Vec2Model, Wav2Vec2Processor, 
    Wav2Vec2ForSequenceClassification, Trainer, TrainingArguments
)

class Wav2Vec2EmotionClassifier(nn.Module):
    def __init__(self, model_name='facebook/wav2vec2-base', num_labels=8):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.wav2vec2.config.hidden_size, num_labels)
        
    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec2(
            input_values=input_values,
            attention_mask=attention_mask
        )
        
        # Global average pooling over time dimension
        hidden_states = outputs.last_hidden_state
        if attention_mask is not None:
            # Mask out padded positions
            hidden_states = hidden_states * attention_mask.unsqueeze(-1)
            lengths = attention_mask.sum(dim=1, keepdim=True)
            pooled = hidden_states.sum(dim=1) / lengths
        else:
            pooled = hidden_states.mean(dim=1)
        
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits

# Fine-tuning setup
processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')
model = Wav2Vec2EmotionClassifier(num_labels=8)

def preprocess_wav2vec2(batch):
    """Preprocess audio for Wav2Vec2"""
    inputs = processor(
        batch['audio_array'],
        sampling_rate=16000,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=16000 * 6  # 6 seconds max
    )
    batch['input_values'] = inputs.input_values
    batch['attention_mask'] = inputs.attention_mask
    return batch
```

2. **Multimodal Emotion Recognition (Audio + Text)**
```python
from transformers import BertModel, BertTokenizer

class MultimodalEmotionRecognizer(nn.Module):
    def __init__(self, audio_model, text_model, num_classes=8):
        super().__init__()
        self.audio_encoder = audio_model
        self.text_encoder = text_model
        
        # Fusion layers
        self.audio_projection = nn.Linear(768, 256)  # Wav2Vec2 hidden size
        self.text_projection = nn.Linear(768, 256)   # BERT hidden size
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=256, num_heads=8, batch_first=True
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),  # Concatenated audio + text features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, audio_input, text_input, audio_mask=None, text_mask=None):
        # Extract audio features
        audio_features = self.audio_encoder(
            input_values=audio_input,
            attention_mask=audio_mask
        )  # Shape: [batch, seq_len, hidden_size]
        
        # Extract text features
        text_features = self.text_encoder(
            input_ids=text_input,
            attention_mask=text_mask
        ).last_hidden_state  # Shape: [batch, seq_len, hidden_size]
        
        # Project to common dimension
        audio_projected = self.audio_projection(audio_features)
        text_projected = self.text_projection(text_features)
        
        # Cross-modal attention
        audio_attended, _ = self.cross_attention(
            audio_projected, text_projected, text_projected
        )
        text_attended, _ = self.cross_attention(
            text_projected, audio_projected, audio_projected
        )
        
        # Global pooling
        audio_pooled = audio_attended.mean(dim=1)
        text_pooled = text_attended.mean(dim=1)
        
        # Concatenate and classify
        fused_features = torch.cat([audio_pooled, text_pooled], dim=1)
        logits = self.classifier(fused_features)
        
        return logits
```

3. **Transformer-based Spectrogram Analysis**
```python
import timm
from transformers import ViTModel, ViTConfig

class SpectrogramTransformer(nn.Module):
    def __init__(self, num_classes=8, image_size=224):
        super().__init__()
        
        # Use Vision Transformer for spectrogram analysis
        config = ViTConfig(
            image_size=image_size,
            patch_size=16,
            num_channels=1,  # Grayscale spectrogram
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            num_labels=num_classes
        )
        
        self.vit = ViTModel(config)
        self.classifier = nn.Linear(768, num_classes)
        
        # Mel-spectrogram transformation
        self.mel_transform = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            hop_length=512,
            n_mels=224  # Match image size
        )
        
    def audio_to_spectrogram(self, audio):
        """Convert audio to spectrogram image"""
        mel_spec = self.mel_transform(audio)
        mel_spec = torch.log(mel_spec + 1e-8)
        
        # Normalize to [0, 1]
        mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min())
        
        # Resize to fixed size
        if mel_spec.shape[-1] != 224:
            mel_spec = F.interpolate(
                mel_spec.unsqueeze(1), 
                size=(224, 224), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(1)
        
        return mel_spec.unsqueeze(1)  # Add channel dimension
    
    def forward(self, audio):
        # Convert to spectrogram
        spectrogram = self.audio_to_spectrogram(audio)
        
        # Process with ViT
        outputs = self.vit(pixel_values=spectrogram)
        pooled_output = outputs.pooler_output
        
        # Classification
        logits = self.classifier(pooled_output)
        return logits
```

**Target Performance:** 85%+ accuracy with multimodal fusion

## 5. Training Details

### Input Pipeline
```python
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio

class EmotionDataset(Dataset):
    def __init__(self, data, processor=None, augment=False):
        self.data = data
        self.processor = processor
        self.augment = augment
        
        # Audio augmentation transforms
        if augment:
            self.time_stretch = T.TimeStretch(fixed_rate=1.2)
            self.pitch_shift = T.PitchShift(sample_rate=16000, n_steps=2)
            self.add_noise = self._add_noise
    
    def _add_noise(self, audio, noise_factor=0.005):
        """Add gaussian noise to audio"""
        noise = torch.randn_like(audio) * noise_factor
        return audio + noise
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Load audio
        audio_path = sample['audio_path']
        audio, sr = torchaudio.load(audio_path)
        
        # Resample if necessary
        if sr != 16000:
            resampler = T.Resample(sr, 16000)
            audio = resampler(audio)
        
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        # Apply augmentation during training
        if self.augment and torch.rand(1) > 0.5:
            aug_choice = torch.randint(0, 3, (1,)).item()
            if aug_choice == 0:
                audio = self.time_stretch(audio)
            elif aug_choice == 1:
                audio = self.pitch_shift(audio)
            else:
                audio = self.add_noise(audio)
        
        # Process with Wav2Vec2 processor if provided
        if self.processor:
            inputs = self.processor(
                audio.squeeze(),
                sampling_rate=16000,
                return_tensors='pt',
                padding='max_length',
                max_length=16000 * 6,  # 6 seconds
                truncation=True
            )
            return {
                'input_values': inputs.input_values.squeeze(),
                'attention_mask': inputs.attention_mask.squeeze(),
                'labels': torch.tensor(sample['emotion_id'], dtype=torch.long)
            }
        else:
            return {
                'audio': audio.squeeze(),
                'labels': torch.tensor(sample['emotion_id'], dtype=torch.long)
            }

# Data loading setup
def create_data_loaders(train_data, val_data, processor=None, batch_size=16):
    train_dataset = EmotionDataset(train_data, processor, augment=True)
    val_dataset = EmotionDataset(val_data, processor, augment=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader
```

### Training Configuration
```python
training_config = {
    'model_name': 'wav2vec2-emotion-classifier',
    'num_epochs': 100,
    'batch_size': 16,
    'learning_rate': 5e-5,
    'weight_decay': 0.01,
    'warmup_ratio': 0.1,
    'gradient_accumulation_steps': 2,
    'fp16': True,
    'max_grad_norm': 1.0,
    'save_strategy': 'steps',
    'save_steps': 500,
    'eval_strategy': 'steps',
    'eval_steps': 250,
    'logging_steps': 50,
    'load_best_model_at_end': True,
    'metric_for_best_model': 'eval_f1',
    'greater_is_better': True,
    'early_stopping_patience': 10,
    'dataloader_num_workers': 4,
    'remove_unused_columns': False,
}

# Custom trainer with emotion-specific metrics
from transformers import Trainer

class EmotionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        
        # Weighted loss for imbalanced classes
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(outputs, labels)
        
        return (loss, outputs) if return_outputs else loss
    
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
```

### Advanced Training Techniques
```python
# Mixup augmentation for audio
def mixup_data(x, y, alpha=1.0):
    """Apply mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# Focal loss for handling class imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduce:
            return torch.mean(focal_loss)
        else:
            return focal_loss
```

## 6. Evaluation Metrics & Validation Strategy

### Core Metrics
```python
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt

def compute_emotion_metrics(predictions, labels, emotion_names):
    """Compute comprehensive emotion recognition metrics"""
    
    # Basic metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average=None
    )
    
    # Macro and weighted averages
    macro_f1 = f1.mean()
    weighted_f1 = np.average(f1, weights=support)
    
    # Per-class metrics
    class_metrics = {}
    for i, emotion in enumerate(emotion_names):
        class_metrics[emotion] = {
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i],
            'support': support[i]
        }
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'class_metrics': class_metrics,
        'confusion_matrix': cm
    }

def plot_confusion_matrix(cm, emotion_names, title='Confusion Matrix'):
    """Plot confusion matrix heatmap"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=emotion_names,
        yticklabels=emotion_names
    )
    plt.title(title)
    plt.xlabel('Predicted Emotion')
    plt.ylabel('True Emotion')
    plt.tight_layout()
    return plt.gcf()

# Emotion-specific evaluation
def evaluate_emotion_arousal_valence(predictions, labels, emotion_to_av):
    """Evaluate in arousal-valence space"""
    pred_arousal = [emotion_to_av[pred]['arousal'] for pred in predictions]
    pred_valence = [emotion_to_av[pred]['valence'] for pred in predictions]
    
    true_arousal = [emotion_to_av[label]['arousal'] for label in labels]
    true_valence = [emotion_to_av[label]['valence'] for label in labels]
    
    arousal_mae = np.mean(np.abs(np.array(pred_arousal) - np.array(true_arousal)))
    valence_mae = np.mean(np.abs(np.array(pred_valence) - np.array(true_valence)))
    
    return arousal_mae, valence_mae
```

### Validation Strategy
- **Cross-validation**: Leave-one-speaker-out (LOSO) validation
- **Temporal splits**: Train on early sessions, test on later ones
- **Domain adaptation**: Train on one dataset, test on another
- **Speaker-independent**: Ensure no speaker overlap between splits
- **Emotion balancing**: Stratified sampling for balanced evaluation

### Advanced Evaluation
```python
# Speaker-independent evaluation
def speaker_independent_split(data, test_speakers_ratio=0.2):
    """Split data ensuring no speaker overlap"""
    speakers = list(set([sample['actor_id'] for sample in data]))
    n_test_speakers = int(len(speakers) * test_speakers_ratio)
    
    test_speakers = np.random.choice(speakers, n_test_speakers, replace=False)
    
    train_data = [s for s in data if s['actor_id'] not in test_speakers]
    test_data = [s for s in data if s['actor_id'] in test_speakers]
    
    return train_data, test_data

# Cross-dataset evaluation
def cross_dataset_evaluation(model, source_dataset, target_dataset):
    """Evaluate model trained on one dataset on another"""
    # Train on source
    model.fit(source_dataset)
    
    # Test on target
    predictions = model.predict(target_dataset)
    return compute_emotion_metrics(predictions, target_dataset.labels)

# Confidence calibration
def calibration_plot(predictions, confidences, labels, n_bins=10):
    """Plot reliability diagram for confidence calibration"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    accuracies = []
    confidences_binned = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = (predictions[in_bin] == labels[in_bin]).mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            accuracies.append(accuracy_in_bin)
            confidences_binned.append(avg_confidence_in_bin)
    
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.plot(confidences_binned, accuracies, 'o-', label='Model')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Reliability Diagram')
    plt.legend()
    return plt.gcf()
```

## 7. Experiment Tracking & Reproducibility

### Weights & Biases Integration
```python
import wandb
from transformers import TrainingArguments

# Initialize experiment tracking
wandb.init(
    project="speech-emotion-recognition",
    config=training_config,
    tags=["wav2vec2", "ravdess", "crema-d", "emotion-recognition"]
)

class EmotionLoggingCallback:
    def __init__(self, eval_dataset, processor, emotion_names):
        self.eval_dataset = eval_dataset
        self.processor = processor
        self.emotion_names = emotion_names
    
    def on_evaluate(self, trainer, model, tokenizer=None):
        # Get predictions on evaluation set
        predictions = trainer.predict(self.eval_dataset)
        pred_labels = np.argmax(predictions.predictions, axis=1)
        true_labels = predictions.label_ids
        
        # Compute metrics
        metrics = compute_emotion_metrics(
            pred_labels, true_labels, self.emotion_names
        )
        
        # Log confusion matrix
        cm_fig = plot_confusion_matrix(
            metrics['confusion_matrix'], 
            self.emotion_names,
            'Emotion Recognition Confusion Matrix'
        )
        
        # Log sample audio predictions
        sample_indices = np.random.choice(len(self.eval_dataset), 5, replace=False)
        audio_samples = []
        
        for idx in sample_indices:
            sample = self.eval_dataset[idx]
            true_emotion = self.emotion_names[sample['labels']]
            pred_emotion = self.emotion_names[pred_labels[idx]]
            
            # Create audio sample for logging
            audio_samples.append([
                wandb.Audio(sample['input_values'].numpy(), sample_rate=16000),
                true_emotion,
                pred_emotion
            ])
        
        # Log to wandb
        wandb.log({
            'confusion_matrix': wandb.Image(cm_fig),
            'audio_predictions': wandb.Table(
                columns=['audio', 'true_emotion', 'predicted_emotion'],
                data=audio_samples
            ),
            'per_class_f1': {
                emotion: metrics['class_metrics'][emotion]['f1'] 
                for emotion in self.emotion_names
            }
        })
        
        plt.close(cm_fig)

# Add callback to trainer
callback = EmotionLoggingCallback(eval_dataset, processor, emotion_names)
trainer.add_callback(callback)
```

### MLflow Tracking
```python
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

mlflow.set_experiment("speech-emotion-recognition")

with mlflow.start_run():
    # Log parameters
    mlflow.log_params({
        "model_type": "wav2vec2",
        "dataset": "ravdess+crema-d",
        "learning_rate": training_config['learning_rate'],
        "batch_size": training_config['batch_size'],
        "num_epochs": training_config['num_epochs'],
        "augmentation": True,
        "speaker_independent": True
    })
    
    # Train model
    trainer.train()
    
    # Evaluate and log metrics
    eval_results = trainer.evaluate()
    for key, value in eval_results.items():
        mlflow.log_metric(key, value)
    
    # Log model artifacts
    mlflow.pytorch.log_model(
        model,
        "emotion-recognition-model",
        registered_model_name="SpeechEmotionRecognizer"
    )
    
    # Log audio samples
    mlflow.log_artifacts("audio_samples/", "sample_predictions")
```

### Experiment Configuration
```yaml
# experiment_config.yaml
experiment:
  name: "wav2vec2-emotion-recognition"
  tags: ["speech", "emotion", "wav2vec2", "multimodal"]
  
model:
  name: "facebook/wav2vec2-base"
  num_labels: 8
  freeze_feature_extractor: false
  dropout: 0.1

data:
  datasets: ["ravdess", "crema-d"]
  sample_rate: 16000
  max_duration: 6.0
  augmentation:
    time_stretch: true
    pitch_shift: true
    noise_injection: true
    mixup_alpha: 0.2

training:
  num_epochs: 100
  batch_size: 16
  learning_rate: 5e-5
  weight_decay: 0.01
  warmup_ratio: 0.1
  gradient_accumulation_steps: 2
  fp16: true
  early_stopping_patience: 10

evaluation:
  strategy: "speaker_independent"
  cross_dataset_validation: true
  metrics: ["accuracy", "f1_macro", "f1_weighted"]
  calibration_analysis: true
```

## 8. Deployment Pathway

### Option 1: Real-time Emotion Detection API
```python
from fastapi import FastAPI, File, UploadFile, HTTPException
import torch
import torchaudio
import tempfile
import os
from pydantic import BaseModel

app = FastAPI(title="Speech Emotion Recognition API")

# Load trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('emotion_model.pt', map_location=device)
processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')

emotion_labels = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised', 'calm']

class EmotionResponse(BaseModel):
    emotion: str
    confidence: float
    probabilities: dict
    duration: float
    processing_time: float

@app.post("/predict_emotion", response_model=EmotionResponse)
async def predict_emotion(audio_file: UploadFile = File(...)):
    """Predict emotion from uploaded audio file"""
    import time
    start_time = time.time()
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            content = await audio_file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Load and process audio
        audio, sr = torchaudio.load(tmp_file_path)
        
        # Resample if necessary
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            audio = resampler(audio)
        
        # Convert to mono
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        # Process with Wav2Vec2
        inputs = processor(
            audio.squeeze(),
            sampling_rate=16000,
            return_tensors='pt',
            padding=True
        )
        
        # Predict emotion
        model.eval()
        with torch.no_grad():
            outputs = model(inputs.input_values.to(device))
            probabilities = torch.softmax(outputs, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Create response
        prob_dict = {
            emotion: prob.item() 
            for emotion, prob in zip(emotion_labels, probabilities[0])
        }
        
        processing_time = time.time() - start_time
        duration = audio.shape[1] / 16000
        
        response = EmotionResponse(
            emotion=emotion_labels[predicted_class],
            confidence=confidence,
            probabilities=prob_dict,
            duration=duration,
            processing_time=processing_time
        )
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_realtime")
async def predict_realtime_emotion(audio_data: bytes):
    """Predict emotion from real-time audio stream"""
    # Implementation for streaming audio processing
    pass

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

# WebSocket endpoint for real-time streaming
from fastapi import WebSocket
import json

@app.websocket("/ws/emotion")
async def websocket_emotion_detection(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            # Receive audio data
            data = await websocket.receive_bytes()
            
            # Process audio (implementation depends on audio format)
            # ... audio processing logic ...
            
            # Send prediction back
            emotion_result = {
                "emotion": "happy",
                "confidence": 0.85,
                "timestamp": time.time()
            }
            await websocket.send_text(json.dumps(emotion_result))
            
    except Exception as e:
        await websocket.close(code=1000)
```

### Option 2: Gradio Interactive Demo
```python
import gradio as gr
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt

# Load model
model = torch.load('emotion_model.pt', map_location='cpu')
processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')
emotion_labels = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised', 'calm']

def predict_emotion(audio_file):
    """Predict emotion from audio file"""
    if audio_file is None:
        return "Please upload an audio file.", None, None
    
    # Load audio
    audio, sr = torchaudio.load(audio_file)
    
    # Resample and convert to mono
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        audio = resampler(audio)
    
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    
    # Process with model
    inputs = processor(
        audio.squeeze(),
        sampling_rate=16000,
        return_tensors='pt',
        padding=True
    )
    
    model.eval()
    with torch.no_grad():
        outputs = model(inputs.input_values)
        probabilities = torch.softmax(outputs, dim=-1).squeeze()
    
    # Create results
    results = {
        emotion: prob.item() 
        for emotion, prob in zip(emotion_labels, probabilities)
    }
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot waveform
    time_axis = np.linspace(0, len(audio.squeeze()) / 16000, len(audio.squeeze()))
    ax1.plot(time_axis, audio.squeeze().numpy())
    ax1.set_title('Audio Waveform')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    
    # Plot emotion probabilities
    emotions = list(results.keys())
    probs = list(results.values())
    bars = ax2.bar(emotions, probs)
    ax2.set_title('Emotion Predictions')
    ax2.set_ylabel('Probability')
    ax2.set_xticklabels(emotions, rotation=45)
    
    # Highlight top prediction
    max_idx = np.argmax(probs)
    bars[max_idx].set_color('red')
    
    plt.tight_layout()
    
    # Get top prediction
    top_emotion = max(results, key=results.get)
    confidence = results[top_emotion]
    
    prediction_text = f"Predicted Emotion: **{top_emotion.upper()}** (Confidence: {confidence:.2%})"
    
    return prediction_text, fig, results

def record_and_predict(audio):
    """Predict emotion from recorded audio"""
    if audio is None:
        return "Please record some audio.", None, None
    
    # Audio is already in the correct format from Gradio
    sr, audio_data = audio
    
    # Convert to torch tensor
    audio_tensor = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0)
    
    # Process similar to file upload
    inputs = processor(
        audio_tensor.squeeze(),
        sampling_rate=sr,
        return_tensors='pt',
        padding=True
    )
    
    model.eval()
    with torch.no_grad():
        outputs = model(inputs.input_values)
        probabilities = torch.softmax(outputs, dim=-1).squeeze()
    
    results = {
        emotion: prob.item() 
        for emotion, prob in zip(emotion_labels, probabilities)
    }
    
    top_emotion = max(results, key=results.get)
    confidence = results[top_emotion]
    
    return f"Predicted Emotion: **{top_emotion.upper()}** (Confidence: {confidence:.2%})", None, results

# Create Gradio interface
with gr.Blocks(title="Speech Emotion Recognition") as demo:
    gr.Markdown("# 🎭 Speech Emotion Recognition System")
    gr.Markdown("Upload an audio file or record your voice to detect emotions!")
    
    with gr.Tab("Upload Audio File"):
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(
                    type="filepath",
                    label="Upload Audio File"
                )
                analyze_btn = gr.Button("Analyze Emotion", variant="primary")
            
            with gr.Column():
                prediction_output = gr.Markdown(label="Prediction")
                plot_output = gr.Plot(label="Analysis")
        
        analyze_btn.click(
            predict_emotion,
            inputs=audio_input,
            outputs=[prediction_output, plot_output, gr.State()]
        )
    
    with gr.Tab("Record Audio"):
        with gr.Row():
            with gr.Column():
                audio_record = gr.Audio(
                    source="microphone",
                    type="numpy",
                    label="Record Your Voice"
                )
                record_btn = gr.Button("Analyze Recording", variant="primary")
            
            with gr.Column():
                record_prediction = gr.Markdown(label="Prediction")
                record_probabilities = gr.JSON(label="All Probabilities")
        
        record_btn.click(
            record_and_predict,
            inputs=audio_record,
            outputs=[record_prediction, gr.State(), record_probabilities]
        )
    
    # Examples
    gr.Examples(
        examples=[
            ["samples/happy_sample.wav"],
            ["samples/sad_sample.wav"],
            ["samples/angry_sample.wav"],
        ],
        inputs=audio_input
    )

demo.launch(share=True)
```

### Option 3: Mobile App Integration
```python
# Flask backend for mobile app
from flask import Flask, request, jsonify
import torch
import torchaudio
import base64
import io

app = Flask(__name__)

@app.route('/predict_emotion', methods=['POST'])
def mobile_emotion_prediction():
    """Endpoint for mobile app emotion prediction"""
    try:
        # Get base64 encoded audio from mobile app
        audio_b64 = request.json['audio_data']
        audio_bytes = base64.b64decode(audio_b64)
        
        # Convert to audio tensor
        audio_buffer = io.BytesIO(audio_bytes)
        audio, sr = torchaudio.load(audio_buffer)
        
        # Process and predict (similar to above)
        # ... processing logic ...
        
        response = {
            'emotion': predicted_emotion,
            'confidence': confidence_score,
            'processing_time': processing_time,
            'status': 'success'
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Cloud Deployment Options
- **AWS SageMaker**: Real-time inference endpoints
- **Google Cloud AI Platform**: AutoML and custom model deployment
- **Azure Cognitive Services**: Integration with Speech Services
- **Heroku**: Quick deployment for demos and prototypes

## 9. Extensions & Research Directions

### Advanced Techniques
1. **Cross-cultural Emotion Recognition**
   - Multi-language emotion datasets
   - Cultural adaptation layers
   - Cross-lingual emotion transfer

2. **Continuous Emotion Recognition**
   ```python
   # Sliding window approach for continuous emotion tracking
   def continuous_emotion_detection(audio_stream, window_size=3.0, hop_size=1.0):
       """Detect emotions in continuous audio stream"""
       emotions_timeline = []
       
       for start_time in range(0, len(audio_stream), int(hop_size * 16000)):
           window_audio = audio_stream[start_time:start_time + int(window_size * 16000)]
           
           if len(window_audio) >= int(window_size * 16000):
               emotion = predict_emotion(window_audio)
               emotions_timeline.append({
                   'timestamp': start_time / 16000,
                   'emotion': emotion,
                   'window_duration': window_size
               })
       
       return emotions_timeline
   ```

3. **Multimodal Emotion Fusion**
   - Audio + visual (facial expressions)
   - Audio + text (sentiment analysis)
   - Physiological signals integration

4. **Personalized Emotion Recognition**
   - Speaker adaptation techniques
   - Few-shot learning for individual users
   - Continual learning from user feedback

### Novel Experiments
- **Emotion intensity prediction**: Beyond classification to regression
- **Mixed emotion detection**: Multiple emotions in single utterance
- **Sarcasm and irony detection**: Complex emotional states
- **Pathological speech analysis**: Depression, anxiety detection
- **Child emotion recognition**: Age-adapted models

### Robustness Enhancements
```python
# Domain adaptation for noisy environments
class DomainAdaptiveEmotionRecognizer(nn.Module):
    def __init__(self, base_model, num_domains=3):
        super().__init__()
        self.base_model = base_model
        self.domain_classifier = nn.Linear(768, num_domains)
        self.gradient_reversal = GradientReversalLayer()
    
    def forward(self, x, domain_adaptation=True):
        features = self.base_model.extract_features(x)
        emotion_logits = self.base_model.classifier(features)
        
        if domain_adaptation:
            reversed_features = self.gradient_reversal(features)
            domain_logits = self.domain_classifier(reversed_features)
            return emotion_logits, domain_logits
        
        return emotion_logits

# Adversarial training for robustness
def adversarial_training_step(model, audio, labels, epsilon=0.001):
    """Adversarial training to improve robustness"""
    audio.requires_grad = True
    
    # Forward pass
    outputs = model(audio)
    loss = F.cross_entropy(outputs, labels)
    
    # Generate adversarial examples
    loss.backward()
    adversarial_audio = audio + epsilon * audio.grad.sign()
    
    # Train on adversarial examples
    model.zero_grad()
    adv_outputs = model(adversarial_audio.detach())
    adv_loss = F.cross_entropy(adv_outputs, labels)
    
    return adv_loss
```

### Industry Applications
- **Healthcare**: Mental health monitoring, therapy assistance
- **Automotive**: Driver emotional state monitoring
- **Education**: Student engagement and stress detection
- **Customer service**: Automated emotion-aware routing
- **Entertainment**: Adaptive content based on user emotions
- **Security**: Stress detection in high-stakes environments

## 10. Portfolio Polish

### Documentation Structure
```
speech_emotion_recognition/
├── README.md                        # This file
├── notebooks/
│   ├── 01_Dataset_Exploration.ipynb # EDA of RAVDESS and CREMA-D
│   ├── 02_Feature_Engineering.ipynb # Acoustic feature analysis
│   ├── 03_Baseline_Models.ipynb    # Traditional ML approaches
│   ├── 04_Deep_Learning.ipynb      # CNN and Wav2Vec2 models
│   ├── 05_Multimodal_Fusion.ipynb  # Audio + text modeling
│   └── 06_Evaluation_Analysis.ipynb # Comprehensive evaluation
├── src/
│   ├── models/
│   │   ├── wav2vec2_emotion.py
│   │   ├── cnn_models.py
│   │   ├── multimodal_fusion.py
│   │   └── baseline_ml.py
│   ├── data/
│   │   ├── preprocessing.py
│   │   ├── feature_extraction.py
│   │   ├── augmentation.py
│   │   └── data_loaders.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── cross_dataset_eval.py
│   │   └── speaker_independent.py
│   ├── training/
│   │   ├── trainer.py
│   │   ├── adversarial_training.py
│   │   └── curriculum_learning.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
├── configs/
│   ├── wav2vec2_config.yaml
│   ├── cnn_config.yaml
│   ├── multimodal_config.yaml
│   └── training_configs/
├── api/
│   ├── fastapi_server.py
│   ├── flask_mobile_api.py
│   ├── websocket_realtime.py
│   └── requirements.txt
├── demo/
│   ├── gradio_app.py
│   ├── streamlit_dashboard.py
│   ├── mobile_demo/
│   └── web_interface/
├── deployment/
│   ├── docker/
│   │   ├── Dockerfile
│   │   └── docker-compose.yml
│   ├── kubernetes/
│   └── cloud_deployment/
├── evaluation/
│   ├── speaker_independent_eval.py
│   ├── cross_dataset_validation.py
│   ├── human_evaluation/
│   └── robustness_testing/
├── datasets/
│   ├── ravdess_loader.py
│   ├── crema_d_loader.py
│   ├── iemocap_loader.py
│   └── combined_dataset.py
├── tests/
│   ├── test_models.py
│   ├── test_preprocessing.py
│   ├── test_api.py
│   └── test_evaluation.py
├── requirements.txt
├── setup.py
├── Makefile
└── .github/workflows/
```

### Visualization Requirements
- **Audio waveforms**: Time-domain signal visualization
- **Spectrograms**: Mel-spectrograms and raw spectrograms
- **Feature distributions**: MFCC, spectral features by emotion
- **Confusion matrices**: Multi-class emotion classification results
- **ROC curves**: Per-class performance analysis
- **t-SNE plots**: Emotion feature space visualization
- **Attention heatmaps**: Transformer attention patterns
- **Training curves**: Loss and accuracy progression
- **Cross-dataset performance**: Generalization analysis

### Blog Post Template
1. **The Emotional AI Revolution**: Why machines need to understand human emotions
2. **Dataset Deep-dive**: Exploring RAVDESS and CREMA-D emotional speech databases
3. **From Acoustics to AI**: Evolution of speech emotion recognition techniques
4. **Deep Learning Breakthrough**: Wav2Vec2 and transformer-based approaches
5. **The Multimodal Advantage**: Combining audio, text, and visual cues
6. **Evaluation Challenges**: Speaker-independent and cross-cultural validation
7. **Real-world Deployment**: Building production emotion recognition systems
8. **Ethical Considerations**: Privacy, bias, and responsible emotion AI
9. **Future Horizons**: Personalized, continuous, and culturally-aware emotion AI

### Demo Video Script
- 1 minute: Emotional communication importance and AI applications
- 1.5 minutes: Dataset exploration with audio samples from different emotions
- 2 minutes: Model architecture walkthrough (traditional → deep learning)
- 2.5 minutes: Live emotion recognition demo with real-time audio
- 1.5 minutes: Performance analysis and cross-dataset validation
- 1 minute: Multimodal fusion demonstration
- 1.5 minutes: Production deployment and real-world applications
- 1 minute: Ethical considerations and future research directions

### GitHub README Essentials
```markdown
# Speech Emotion Recognition with Deep Learning

![Emotion Demo](assets/emotion_recognition_demo.gif)

## 🚀 Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Download datasets
python src/data/download_datasets.py --datasets ravdess crema-d

# Train Wav2Vec2 model
python src/train.py --config configs/wav2vec2_config.yaml

# Predict emotion from audio
python src/predict.py --model models/best_model.pt --audio sample.wav

# Launch interactive demo
python demo/gradio_app.py
```

## 📊 Results
| Model | Dataset | Accuracy | F1-Score | Cross-Dataset |
|-------|---------|----------|----------|---------------|
| CNN | RAVDESS | 76.3% | 0.754 | 62.1% |
| Wav2Vec2 | RAVDESS | 87.2% | 0.864 | 71.8% |
| Multimodal | Combined | 91.5% | 0.908 | 78.3% |

## 🎭 Live Demo
Try emotion recognition: [Hugging Face Space](https://huggingface.co/spaces/username/emotion-recognition)

## 📚 Citation
```bibtex
@article{speech_emotion_2024,
  title={Deep Learning for Speech Emotion Recognition: A Comprehensive Study},
  author={Your Name},
  journal={IEEE Transactions on Affective Computing},
  year={2024}
}
```
```

### Performance Benchmarks
- **Real-time performance**: Processing latency for streaming audio
- **Memory requirements**: RAM and VRAM usage by model complexity
- **Accuracy metrics**: Per-emotion and overall classification performance
- **Robustness analysis**: Performance under noise, different microphones
- **Cross-dataset generalization**: Transfer learning capabilities
- **Speaker-independent performance**: Generalization to unseen speakers
- **Computational efficiency**: FLOPs and inference time comparisons
- **Mobile deployment**: Performance on edge devices and smartphones