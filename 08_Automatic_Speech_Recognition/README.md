# Automatic Speech Recognition Project - LibriSpeech / Common Voice

## 1. Problem Definition & Use Case

**Problem:** Convert spoken language into written text accurately across different speakers, accents, languages, and acoustic conditions while handling background noise, overlapping speech, and domain-specific vocabulary.

**Use Case:** Automatic Speech Recognition (ASR) powers essential applications including:
- Voice assistants and smart speakers (Alexa, Siri, Google Assistant)
- Real-time transcription for meetings and conferences
- Accessibility tools for hearing-impaired users
- Voice-controlled interfaces and hands-free computing
- Call center analytics and quality monitoring
- Medical transcription and clinical documentation
- Language learning applications with pronunciation feedback
- Broadcast media captioning and archival

**Business Impact:** ASR technology reduces transcription costs by 90%, enables hands-free interactions improving accessibility, and processes 500+ billion hours of audio annually across global markets worth $30+ billion.

## 2. Dataset Acquisition & Preprocessing

### Primary Datasets
- **LibriSpeech**: 1000+ hours of English audiobooks, clean and noisy variants
  ```python
  from datasets import load_dataset
  librispeech = load_dataset("librispeech_asr", "clean", split="train.100")
  ```
- **Common Voice**: Mozilla's crowdsourced multilingual dataset, 100+ languages
  ```python
  common_voice = load_dataset("common_voice", "en", split="train")
  ```
- **VCTK Corpus**: 110 English speakers with different accents
  ```bash
  wget https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip
  ```
- **TED-LIUM 3**: TED talks with transcriptions, 452 hours
  ```python
  tedlium = load_dataset("LIUM/tedlium", "release3", split="train")
  ```

### Data Schema
```python
{
    'audio': {
        'path': str,           # Path to audio file
        'array': np.ndarray,   # Raw waveform
        'sampling_rate': int,  # Sample rate (16kHz standard)
    },
    'text': str,               # Ground truth transcription
    'speaker_id': str,         # Speaker identifier
    'chapter_id': str,         # Chapter/session identifier
    'id': str,                 # Unique sample ID
    'duration': float,         # Audio duration in seconds
    'num_samples': int,        # Number of audio samples
}
```

### Preprocessing Pipeline
```python
import librosa
import torch
import torchaudio
from transformers import Wav2Vec2Processor
import re
import string

def normalize_text(text):
    """Normalize transcription text"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation except apostrophes
    text = re.sub(r"[^\w\s']", "", text)
    
    # Normalize contractions
    contractions = {
        "won't": "will not",
        "can't": "cannot",
        "n't": " not",
        "'re": " are",
        "'ve": " have",
        "'ll": " will",
        "'d": " would",
        "'m": " am",
    }
    
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    return text

def preprocess_audio(audio_path, target_sr=16000):
    """Load and preprocess audio file"""
    # Load audio
    audio, sr = librosa.load(audio_path, sr=target_sr)
    
    # Normalize amplitude
    audio = librosa.util.normalize(audio)
    
    # Remove silence at start and end
    audio, _ = librosa.effects.trim(audio, top_db=20)
    
    return audio, target_sr

def extract_features(audio, sr):
    """Extract acoustic features for ASR"""
    features = {}
    
    # Log mel-spectrogram (common for ASR)
    mel_spec = librosa.feature.melspectrogram(
        y=audio, 
        sr=sr, 
        n_mels=80, 
        n_fft=1024, 
        hop_length=256
    )
    log_mel = librosa.power_to_db(mel_spec)
    features['log_mel'] = log_mel
    
    # MFCC features
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    features['mfcc'] = mfcc
    
    # Delta and delta-delta features
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features['mfcc_delta'] = delta
    features['mfcc_delta2'] = delta2
    
    return features

class ASRDataProcessor:
    def __init__(self, processor_name="facebook/wav2vec2-base-960h"):
        self.processor = Wav2Vec2Processor.from_pretrained(processor_name)
        
    def prepare_dataset(self, batch):
        """Prepare batch for training"""
        # Process audio
        audio_arrays = [sample["array"] for sample in batch["audio"]]
        
        # Batch process audio
        inputs = self.processor(
            audio_arrays,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=320000,  # ~20 seconds at 16kHz
        )
        
        # Process text targets
        with self.processor.as_target_processor():
            labels = self.processor(
                batch["text"],
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
        
        # Replace padding with -100 for loss computation
        labels["input_ids"].masked_fill_(
            labels.attention_mask.ne(1), -100
        )
        
        batch["input_values"] = inputs.input_values
        batch["attention_mask"] = inputs.attention_mask
        batch["labels"] = labels["input_ids"]
        
        return batch

# Data augmentation for robustness
class AudioAugmentation:
    def __init__(self):
        self.noise_factor = 0.005
        self.time_stretch_rates = [0.8, 0.9, 1.1, 1.2]
        self.pitch_shifts = [-2, -1, 1, 2]
    
    def add_noise(self, audio):
        """Add gaussian noise"""
        noise = torch.randn_like(audio) * self.noise_factor
        return audio + noise
    
    def time_stretch(self, audio, rate=None):
        """Time stretching without pitch change"""
        if rate is None:
            rate = torch.choice(self.time_stretch_rates)
        
        return torchaudio.functional.time_stretch(
            audio.unsqueeze(0), rate
        ).squeeze(0)
    
    def pitch_shift(self, audio, sr=16000, n_steps=None):
        """Pitch shift without tempo change"""
        if n_steps is None:
            n_steps = torch.choice(self.pitch_shifts)
        
        return torchaudio.functional.pitch_shift(audio, sr, n_steps)
    
    def spec_augment(self, spectrogram):
        """SpecAugment: time and frequency masking"""
        # Frequency masking
        freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)
        spectrogram = freq_mask(spectrogram)
        
        # Time masking
        time_mask = torchaudio.transforms.TimeMasking(time_mask_param=35)
        spectrogram = time_mask(spectrogram)
        
        return spectrogram
```

### Feature Engineering
- **Acoustic modeling**: Log mel-spectrograms, MFCC coefficients
- **Temporal features**: Delta and delta-delta coefficients
- **Voice activity detection**: Remove silence segments
- **Speaker normalization**: Cepstral mean normalization (CMN)
- **Data augmentation**: Speed perturbation, noise injection, SpecAugment

## 3. Baseline Models

### Hidden Markov Models (HMM) + GMM
```python
import numpy as np
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm

class GMMHMMASRModel:
    def __init__(self, n_components=16, n_states=5):
        self.n_components = n_components
        self.n_states = n_states
        self.models = {}  # One model per phoneme/word
        
    def train_phoneme_model(self, features, phoneme_labels):
        """Train HMM-GMM model for each phoneme"""
        unique_phonemes = np.unique(phoneme_labels)
        
        for phoneme in unique_phonemes:
            # Get features for this phoneme
            phoneme_features = features[phoneme_labels == phoneme]
            
            # Create GMM-HMM model
            model = hmm.GMMHMM(
                n_components=self.n_states,
                n_mix=self.n_components,
                covariance_type="diag"
            )
            
            # Train model
            model.fit(phoneme_features)
            self.models[phoneme] = model
    
    def decode(self, features):
        """Decode audio features to phoneme sequence"""
        best_score = -np.inf
        best_sequence = []
        
        for phoneme, model in self.models.items():
            score = model.score(features)
            if score > best_score:
                best_score = score
                best_sequence.append(phoneme)
        
        return best_sequence

# Traditional feature extraction
def extract_mfcc_features(audio, sr=16000, n_mfcc=13):
    """Extract MFCC features for traditional ASR"""
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    
    # Add delta and delta-delta features
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    
    # Concatenate features
    features = np.concatenate([mfcc, delta, delta2], axis=0)
    return features.T  # Time x Features
```
**Expected Performance:** WER 15-25% on clean speech, 35-50% on noisy speech

### Deep Neural Network Acoustic Model
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DNNAcousticModel(nn.Module):
    def __init__(self, input_dim=39, hidden_dim=512, num_classes=42):
        """
        DNN acoustic model for phoneme classification
        input_dim: MFCC + delta features (13*3 = 39)
        num_classes: Number of phonemes (typically ~40-50)
        """
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        return self.layers(x)

# Training function
def train_dnn_acoustic_model(model, train_loader, val_loader, num_epochs=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    
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
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_accuracy:.2f}%')
```
**Expected Performance:** WER 10-15% on clean speech with proper language modeling

## 4. Advanced/Stretch Models

### State-of-the-Art Architectures

1. **Wav2Vec2 Fine-tuning**
```python
from transformers import (
    Wav2Vec2ForCTC, 
    Wav2Vec2Processor, 
    Trainer, 
    TrainingArguments
)
import torch

class Wav2Vec2ASRModel:
    def __init__(self, model_name="facebook/wav2vec2-base-960h"):
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        
        # Freeze feature extractor for fine-tuning
        self.model.freeze_feature_extractor()
    
    def prepare_dataset(self, batch):
        """Prepare data for Wav2Vec2 training"""
        # Process audio
        audio_arrays = [sample["array"] for sample in batch["audio"]]
        inputs = self.processor(
            audio_arrays,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        
        # Process text labels
        with self.processor.as_target_processor():
            labels = self.processor(batch["text"], return_tensors="pt", padding=True)
        
        # Replace padding with -100 for CTC loss
        labels["input_ids"].masked_fill_(labels.attention_mask.ne(1), -100)
        
        batch["input_values"] = inputs.input_values
        batch["attention_mask"] = inputs.attention_mask
        batch["labels"] = labels["input_ids"]
        
        return batch
    
    def transcribe(self, audio_path):
        """Transcribe audio file"""
        # Load audio
        audio, _ = librosa.load(audio_path, sr=16000)
        
        # Process with Wav2Vec2
        inputs = self.processor(
            audio, 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding=True
        )
        
        # Generate logits
        with torch.no_grad():
            logits = self.model(inputs.input_values).logits
        
        # Decode predictions
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.decode(predicted_ids[0])
        
        return transcription

# Custom training with WER metric
from evaluate import load
wer_metric = load("wer")

def compute_metrics(pred):
    """Compute WER metric during training"""
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    
    # Replace -100 with pad token id
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    
    # Decode predictions and labels
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    
    # Compute WER
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}
```

2. **Whisper Fine-tuning**
```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration

class WhisperASRModel:
    def __init__(self, model_name="openai/whisper-base"):
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        
        # Configure for ASR task
        self.model.generation_config.language = "en"
        self.model.generation_config.task = "transcribe"
        
    def prepare_batch(self, batch):
        """Prepare batch for Whisper training"""
        # Process audio to log-mel spectrograms
        audio_arrays = [sample["array"] for sample in batch["audio"]]
        inputs = self.processor.feature_extractor(
            audio_arrays, 
            sampling_rate=16000, 
            return_tensors="pt"
        )
        
        # Process text labels
        labels = self.processor.tokenizer(
            batch["text"],
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Replace padding with -100
        labels["input_ids"].masked_fill_(
            labels["input_ids"] == self.processor.tokenizer.pad_token_id, -100
        )
        
        batch["input_features"] = inputs.input_features
        batch["labels"] = labels["input_ids"]
        
        return batch
    
    def transcribe_with_timestamps(self, audio_path):
        """Transcribe with word-level timestamps"""
        audio, _ = librosa.load(audio_path, sr=16000)
        
        # Process audio
        input_features = self.processor(
            audio, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features
        
        # Generate with return_timestamps
        predicted_ids = self.model.generate(
            input_features,
            return_timestamps=True,
            task="transcribe",
            language="en"
        )
        
        # Decode with timestamps
        result = self.processor.batch_decode(
            predicted_ids, 
            skip_special_tokens=True,
            decode_with_timestamps=True
        )
        
        return result
```

3. **Conformer Architecture**
```python
import torch.nn as nn

class ConformerBlock(nn.Module):
    """Conformer block combining CNN and self-attention"""
    def __init__(self, dim, num_heads=8, ff_mult=4, conv_kernel_size=31):
        super().__init__()
        
        # Feed-forward module (first half)
        self.ff1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * ff_mult),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(0.1)
        )
        
        # Multi-head self-attention
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=0.1, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(dim)
        
        # Convolution module
        self.conv = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Conv1d(dim, dim * 2, 1),
            nn.GLU(dim=1),
            nn.Conv1d(dim, dim, conv_kernel_size, padding=conv_kernel_size//2, groups=dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, 1),
            nn.Dropout(0.1)
        )
        
        # Feed-forward module (second half)
        self.ff2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * ff_mult),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(0.1)
        )
        
    def forward(self, x, mask=None):
        # First feed-forward (half step)
        x = x + 0.5 * self.ff1(x)
        
        # Self-attention
        x_norm = self.attn_norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=mask)
        x = x + attn_out
        
        # Convolution
        x_norm = x.transpose(1, 2)  # (B, D, T)
        conv_out = self.conv(x_norm).transpose(1, 2)
        x = x + conv_out
        
        # Second feed-forward (half step)
        x = x + 0.5 * self.ff2(x)
        
        return x

class ConformerASR(nn.Module):
    def __init__(self, input_dim=80, dim=512, num_layers=12, num_classes=32):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(5000, dim))
        
        # Conformer blocks
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(dim) for _ in range(num_layers)
        ])
        
        # Output projection for CTC
        self.output_proj = nn.Linear(dim, num_classes)
        
    def forward(self, x, lengths=None):
        # x shape: (batch, time, mel_bins)
        B, T, _ = x.shape
        
        # Input projection
        x = self.input_proj(x)
        
        # Add positional encoding
        x = x + self.pos_encoding[:T]
        
        # Create attention mask from lengths
        mask = None
        if lengths is not None:
            mask = torch.arange(T).expand(B, T) >= lengths.unsqueeze(1)
        
        # Apply Conformer blocks
        for block in self.conformer_blocks:
            x = block(x, mask)
        
        # Output projection
        logits = self.output_proj(x)
        
        return logits
```

4. **End-to-End Speech Recognition with RNN-T**
```python
class RNNTModel(nn.Module):
    """RNN-Transducer for streaming ASR"""
    def __init__(self, vocab_size, encoder_dim=512, prediction_dim=512, joint_dim=512):
        super().__init__()
        
        # Encoder (acoustic model)
        self.encoder = nn.LSTM(
            input_size=80,  # Mel features
            hidden_size=encoder_dim,
            num_layers=8,
            batch_first=True,
            bidirectional=True
        )
        self.encoder_proj = nn.Linear(encoder_dim * 2, encoder_dim)
        
        # Prediction network (language model)
        self.predictor = nn.LSTM(
            input_size=vocab_size,
            hidden_size=prediction_dim,
            num_layers=2,
            batch_first=True
        )
        
        # Joint network
        self.joint = nn.Sequential(
            nn.Linear(encoder_dim + prediction_dim, joint_dim),
            nn.Tanh(),
            nn.Linear(joint_dim, vocab_size)
        )
        
    def forward(self, audio_features, text_targets=None):
        # Encode audio
        encoder_out, _ = self.encoder(audio_features)
        encoder_out = self.encoder_proj(encoder_out)
        
        if text_targets is not None:
            # Training mode: use ground truth text
            predictor_out, _ = self.predictor(text_targets)
            
            # Joint network
            # Expand dimensions for broadcasting
            encoder_exp = encoder_out.unsqueeze(2)  # (B, T, 1, D)
            predictor_exp = predictor_out.unsqueeze(1)  # (B, 1, U, D)
            
            # Joint computation
            joint_input = torch.cat([
                encoder_exp.expand(-1, -1, predictor_out.size(1), -1),
                predictor_exp.expand(-1, encoder_out.size(1), -1, -1)
            ], dim=-1)
            
            logits = self.joint(joint_input)
            return logits
        else:
            # Inference mode: beam search decoding
            return self.beam_search_decode(encoder_out)
    
    def beam_search_decode(self, encoder_out, beam_size=10):
        """Beam search decoding for inference"""
        # Implementation of beam search for RNN-T
        # This is a simplified version
        batch_size, max_time = encoder_out.size(0), encoder_out.size(1)
        
        # Initialize beams
        beams = [[([], 0.0)] for _ in range(batch_size)]
        
        for t in range(max_time):
            new_beams = [[] for _ in range(batch_size)]
            
            for b in range(batch_size):
                for hypothesis, score in beams[b]:
                    # Predict next token
                    # ... beam search logic ...
                    pass
            
            beams = new_beams
        
        # Return best hypotheses
        return [beam[0][0] for beam in beams]
```

**Target Performance:** WER 2-5% on LibriSpeech test-clean, 5-8% on test-other

## 5. Training Details

### Input Pipeline
```python
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

class ASRDataset(Dataset):
    def __init__(self, data, processor, augment=False):
        self.data = data
        self.processor = processor
        self.augment = augment
        self.augmentation = AudioAugmentation() if augment else None
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Load audio
        audio = sample['audio']['array']
        
        # Apply augmentation
        if self.augment and torch.rand(1) > 0.5:
            audio = self.augmentation.add_noise(torch.tensor(audio)).numpy()
        
        # Process with processor
        inputs = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        
        # Process text
        with self.processor.as_target_processor():
            labels = self.processor(
                sample['text'],
                return_tensors="pt",
                padding=True
            )
        
        return {
            'input_values': inputs.input_values.squeeze(),
            'attention_mask': inputs.attention_mask.squeeze(),
            'labels': labels.input_ids.squeeze()
        }

def collate_fn(batch):
    """Custom collate function for variable length sequences"""
    input_values = [item['input_values'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    # Pad sequences
    input_values = pad_sequence(input_values, batch_first=True)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    
    # Create attention masks
    attention_mask = torch.ones_like(input_values)
    
    return {
        'input_values': input_values,
        'attention_mask': attention_mask,
        'labels': labels
    }

# Create data loaders
def create_asr_dataloaders(train_data, val_data, processor, batch_size=8):
    train_dataset = ASRDataset(train_data, processor, augment=True)
    val_dataset = ASRDataset(val_data, processor, augment=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader
```

### Training Configuration
```python
from transformers import TrainingArguments

training_config = {
    'output_dir': './wav2vec2-asr-finetuned',
    'group_by_length': True,  # Group samples by length for efficiency
    'per_device_train_batch_size': 8,
    'per_device_eval_batch_size': 8,
    'gradient_accumulation_steps': 4,
    'evaluation_strategy': 'steps',
    'eval_steps': 1000,
    'save_steps': 1000,
    'logging_steps': 100,
    'learning_rate': 3e-4,
    'weight_decay': 0.005,
    'warmup_steps': 2000,
    'num_train_epochs': 30,
    'save_total_limit': 2,
    'fp16': True,
    'push_to_hub': False,
    'load_best_model_at_end': True,
    'metric_for_best_model': 'wer',
    'greater_is_better': False,
    'dataloader_num_workers': 4,
    'remove_unused_columns': False,
    'dataloader_pin_memory': True,
}

training_args = TrainingArguments(**training_config)
```

### Advanced Training Techniques
```python
# Curriculum learning: start with shorter utterances
def curriculum_learning_scheduler(epoch, dataset):
    """Gradually increase maximum sequence length"""
    max_lengths = {
        0: 5,    # 5 seconds
        5: 10,   # 10 seconds
        10: 15,  # 15 seconds
        15: 20   # 20 seconds (full length)
    }
    
    max_length = max_lengths.get(epoch, 20)
    filtered_dataset = dataset.filter(
        lambda x: len(x['audio']['array']) / 16000 <= max_length
    )
    
    return filtered_dataset

# Knowledge distillation from larger model
class ASRKnowledgeDistillation:
    def __init__(self, teacher_model, student_model, alpha=0.7, temperature=4):
        self.teacher = teacher_model
        self.student = student_model
        self.alpha = alpha
        self.temperature = temperature
        
    def distillation_loss(self, student_logits, teacher_logits, labels):
        """Compute knowledge distillation loss"""
        # Soft targets from teacher
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_prob = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        # KL divergence loss
        distillation_loss = F.kl_div(
            soft_prob, soft_targets, reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Hard target loss
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # Combined loss
        total_loss = (
            self.alpha * distillation_loss + 
            (1 - self.alpha) * hard_loss
        )
        
        return total_loss

# CTC loss implementation
class CTCTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute CTC loss"""
        labels = inputs.pop("labels")
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Compute sequence lengths
        input_lengths = inputs["attention_mask"].sum(-1)
        label_lengths = (labels != -100).sum(-1)
        
        # CTC loss
        log_probs = F.log_softmax(logits, dim=-1)
        loss = F.ctc_loss(
            log_probs.transpose(0, 1),  # (T, B, C)
            labels,
            input_lengths,
            label_lengths,
            blank=model.config.pad_token_id,
            reduction='mean',
            zero_infinity=True
        )
        
        return (loss, outputs) if return_outputs else loss
```

## 6. Evaluation Metrics & Validation Strategy

### Core Metrics
```python
import jiwer
from evaluate import load

# Load evaluation metrics
wer_metric = load("wer")
cer_metric = load("cer")

def compute_asr_metrics(predictions, references):
    """Compute comprehensive ASR evaluation metrics"""
    
    # Clean and normalize text
    predictions = [normalize_text(pred) for pred in predictions]
    references = [normalize_text(ref) for ref in references]
    
    # Word Error Rate (WER)
    wer = wer_metric.compute(predictions=predictions, references=references)
    
    # Character Error Rate (CER)
    cer = cer_metric.compute(predictions=predictions, references=references)
    
    # Additional metrics using jiwer
    measures = jiwer.compute_measures(references, predictions)
    
    results = {
        'wer': wer,
        'cer': cer,
        'substitution_rate': measures['substitution_rate'],
        'deletion_rate': measures['deletion_rate'],
        'insertion_rate': measures['insertion_rate'],
        'hits': measures['hits'],
        'substitutions': measures['substitutions'],
        'deletions': measures['deletions'],
        'insertions': measures['insertions'],
    }
    
    return results

def evaluate_by_duration(predictions, references, durations):
    """Evaluate performance by utterance duration"""
    duration_bins = {
        'short': (0, 5),      # 0-5 seconds
        'medium': (5, 10),    # 5-10 seconds
        'long': (10, float('inf'))  # 10+ seconds
    }
    
    results = {}
    
    for bin_name, (min_dur, max_dur) in duration_bins.items():
        # Filter samples by duration
        indices = [
            i for i, dur in enumerate(durations) 
            if min_dur <= dur < max_dur
        ]
        
        if indices:
            bin_predictions = [predictions[i] for i in indices]
            bin_references = [references[i] for i in indices]
            
            bin_metrics = compute_asr_metrics(bin_predictions, bin_references)
            results[bin_name] = bin_metrics
    
    return results

def evaluate_by_speaker(predictions, references, speaker_ids):
    """Evaluate performance by speaker"""
    unique_speakers = set(speaker_ids)
    results = {}
    
    for speaker in unique_speakers:
        # Filter samples by speaker
        indices = [i for i, spk in enumerate(speaker_ids) if spk == speaker]
        
        if len(indices) > 5:  # Only evaluate speakers with enough samples
            spk_predictions = [predictions[i] for i in indices]
            spk_references = [references[i] for i in indices]
            
            spk_metrics = compute_asr_metrics(spk_predictions, spk_references)
            results[speaker] = spk_metrics
    
    return results

# Real-time factor (RTF) evaluation
import time

def measure_inference_speed(model, processor, audio_samples):
    """Measure real-time factor for inference"""
    total_audio_duration = 0
    total_processing_time = 0
    
    model.eval()
    with torch.no_grad():
        for audio in audio_samples:
            audio_duration = len(audio) / 16000  # Duration in seconds
            
            # Measure processing time
            start_time = time.time()
            
            inputs = processor(
                audio, 
                sampling_rate=16000, 
                return_tensors="pt"
            )
            outputs = model(**inputs)
            
            processing_time = time.time() - start_time
            
            total_audio_duration += audio_duration
            total_processing_time += processing_time
    
    rtf = total_processing_time / total_audio_duration
    return rtf

# Confidence scoring
def compute_confidence_scores(logits, predictions):
    """Compute confidence scores for predictions"""
    probs = F.softmax(logits, dim=-1)
    
    # Maximum probability confidence
    max_probs = torch.max(probs, dim=-1)[0]
    avg_confidence = torch.mean(max_probs, dim=-1)
    
    # Entropy-based confidence
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
    normalized_entropy = entropy / torch.log(torch.tensor(probs.size(-1)))
    entropy_confidence = 1 - torch.mean(normalized_entropy, dim=-1)
    
    return avg_confidence, entropy_confidence
```

### Validation Strategy
- **Hold-out validation**: Standard train/dev/test splits
- **Cross-domain evaluation**: Train on audiobooks, test on conversational speech
- **Speaker-independent validation**: No speaker overlap between splits
- **Noise robustness**: Evaluation on noisy and reverberant speech
- **Language adaptation**: Cross-lingual and multilingual evaluation

### Advanced Evaluation
```python
# Error analysis by phoneme/word frequency
def phoneme_error_analysis(predictions, references, phoneme_dict):
    """Analyze errors by phoneme frequency"""
    phoneme_errors = {}
    
    for pred, ref in zip(predictions, references):
        # Convert to phonemes (requires phoneme converter)
        pred_phonemes = text_to_phonemes(pred, phoneme_dict)
        ref_phonemes = text_to_phonemes(ref, phoneme_dict)
        
        # Align sequences and count errors
        alignment = align_sequences(pred_phonemes, ref_phonemes)
        
        for pred_ph, ref_ph in alignment:
            if ref_ph not in phoneme_errors:
                phoneme_errors[ref_ph] = {'total': 0, 'errors': 0}
            
            phoneme_errors[ref_ph]['total'] += 1
            if pred_ph != ref_ph:
                phoneme_errors[ref_ph]['errors'] += 1
    
    # Calculate error rates
    phoneme_error_rates = {
        ph: stats['errors'] / stats['total'] 
        for ph, stats in phoneme_errors.items()
    }
    
    return phoneme_error_rates

# Out-of-vocabulary (OOV) analysis
def oov_analysis(predictions, references, vocabulary):
    """Analyze out-of-vocabulary word performance"""
    oov_stats = {'iv_wer': [], 'oov_wer': []}
    
    for pred, ref in zip(predictions, references):
        pred_words = pred.split()
        ref_words = ref.split()
        
        # Separate in-vocabulary and OOV words
        iv_pred = [w for w in pred_words if w in vocabulary]
        iv_ref = [w for w in ref_words if w in vocabulary]
        
        oov_pred = [w for w in pred_words if w not in vocabulary]
        oov_ref = [w for w in ref_words if w not in vocabulary]
        
        # Calculate WER for each category
        if iv_ref:
            iv_wer = jiwer.wer(iv_ref, iv_pred)
            oov_stats['iv_wer'].append(iv_wer)
        
        if oov_ref:
            oov_wer = jiwer.wer(oov_ref, oov_pred)
            oov_stats['oov_wer'].append(oov_wer)
    
    return {
        'avg_iv_wer': np.mean(oov_stats['iv_wer']) if oov_stats['iv_wer'] else 0,
        'avg_oov_wer': np.mean(oov_stats['oov_wer']) if oov_stats['oov_wer'] else 0
    }
```

## 7. Experiment Tracking & Reproducibility

### Weights & Biases Integration
```python
import wandb
from transformers import TrainingArguments

# Initialize experiment tracking
wandb.init(
    project="automatic-speech-recognition",
    config=training_config,
    tags=["wav2vec2", "librispeech", "asr", "ctc"]
)

class ASRLoggingCallback:
    def __init__(self, eval_dataset, processor):
        self.eval_dataset = eval_dataset
        self.processor = processor
    
    def on_evaluate(self, trainer, model):
        # Sample predictions for qualitative analysis
        sample_indices = np.random.choice(len(self.eval_dataset), 10, replace=False)
        
        predictions_table = []
        for idx in sample_indices:
            sample = self.eval_dataset[idx]
            
            # Get model prediction
            inputs = self.processor(
                sample['audio']['array'],
                sampling_rate=16000,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                logits = model(**inputs).logits
                pred_ids = torch.argmax(logits, dim=-1)
                prediction = self.processor.decode(pred_ids[0])
            
            # Create audio sample for logging
            audio_sample = wandb.Audio(
                sample['audio']['array'],
                sample_rate=16000
            )
            
            predictions_table.append([
                audio_sample,
                sample['text'],
                prediction,
                jiwer.wer([sample['text']], [prediction])
            ])
        
        # Log predictions table
        wandb.log({
            "predictions": wandb.Table(
                columns=["audio", "reference", "prediction", "wer"],
                data=predictions_table
            )
        })
        
        # Log WER distribution
        wer_scores = [row[3] for row in predictions_table]
        wandb.log({
            "wer_distribution": wandb.Histogram(wer_scores),
            "avg_sample_wer": np.mean(wer_scores)
        })

# Add callback to trainer
trainer.add_callback(ASRLoggingCallback(eval_dataset, processor))
```

### MLflow Integration
```python
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

mlflow.set_experiment("librispeech-asr")

with mlflow.start_run():
    # Log parameters
    mlflow.log_params({
        "model_name": "wav2vec2-base-960h",
        "dataset": "librispeech",
        "learning_rate": training_config['learning_rate'],
        "batch_size": training_config['per_device_train_batch_size'],
        "num_epochs": training_config['num_train_epochs'],
        "gradient_accumulation": training_config['gradient_accumulation_steps'],
        "warmup_steps": training_config['warmup_steps']
    })
    
    # Train model
    trainer.train()
    
    # Evaluate on test set
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    
    # Log metrics
    mlflow.log_metrics({
        "final_wer": test_results['eval_wer'],
        "final_loss": test_results['eval_loss'],
        "training_time": trainer.state.log_history[-1]['train_runtime']
    })
    
    # Log model
    mlflow.pytorch.log_model(
        model,
        "asr-model",
        registered_model_name="LibriSpeechASR"
    )
    
    # Log audio samples
    mlflow.log_artifacts("sample_predictions/", "evaluation_samples")
```

### Experiment Configuration Management
```yaml
# experiment_config.yaml
experiment:
  name: "wav2vec2-librispeech-asr"
  description: "Fine-tuning Wav2Vec2 on LibriSpeech for English ASR"
  tags: ["asr", "wav2vec2", "librispeech", "english"]

model:
  name: "facebook/wav2vec2-base-960h"
  freeze_feature_extractor: true
  ctc_loss_reduction: "mean"
  
data:
  train_dataset: "librispeech_asr/train.clean.100"
  dev_dataset: "librispeech_asr/validation.clean"
  test_dataset: "librispeech_asr/test.clean"
  sampling_rate: 16000
  max_duration: 20.0
  min_duration: 1.0
  
preprocessing:
  normalize_text: true
  remove_punctuation: true
  lowercase: true
  
augmentation:
  speed_perturb: [0.9, 1.1]
  noise_injection: 0.005
  spec_augment:
    freq_mask: 15
    time_mask: 35
    
training:
  num_epochs: 30
  batch_size: 8
  gradient_accumulation_steps: 4
  learning_rate: 3e-4
  weight_decay: 0.005
  warmup_steps: 2000
  lr_scheduler: "linear"
  fp16: true
  
evaluation:
  metrics: ["wer", "cer"]
  eval_steps: 1000
  save_best_model: true
  early_stopping_patience: 5
```

## 8. Deployment Pathway

### Option 1: Real-time Streaming ASR Service
```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio
import json
import torch
import torchaudio
from collections import deque

app = FastAPI(title="Real-time ASR API")

# Global model loading
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Wav2Vec2ForCTC.from_pretrained('./fine-tuned-wav2vec2').to(device)
processor = Wav2Vec2Processor.from_pretrained('./fine-tuned-wav2vec2')

class StreamingASR:
    def __init__(self, model, processor, chunk_duration=1.0):
        self.model = model
        self.processor = processor
        self.chunk_duration = chunk_duration
        self.sample_rate = 16000
        self.chunk_samples = int(chunk_duration * self.sample_rate)
        self.buffer = deque(maxlen=5)  # 5-second context window
        
    def process_chunk(self, audio_chunk):
        """Process audio chunk and return transcription"""
        # Add chunk to buffer
        self.buffer.append(audio_chunk)
        
        # Concatenate buffered audio
        if len(self.buffer) > 1:
            audio = torch.cat(list(self.buffer))
        else:
            audio = audio_chunk
        
        # Process with model
        inputs = self.processor(
            audio.numpy(),
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True
        )
        
        with torch.no_grad():
            logits = self.model(inputs.input_values.to(device)).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.decode(predicted_ids[0])
        
        return transcription

streaming_asr = StreamingASR(model, processor)

@app.websocket("/ws/transcribe")
async def websocket_transcription(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            # Receive audio data (base64 encoded)
            data = await websocket.receive_text()
            audio_data = json.loads(data)
            
            # Decode audio (assuming it's sent as base64)
            import base64
            audio_bytes = base64.b64decode(audio_data['audio'])
            
            # Convert to tensor (this depends on your audio format)
            audio_tensor = torch.frombuffer(audio_bytes, dtype=torch.float32)
            
            # Process chunk
            transcription = streaming_asr.process_chunk(audio_tensor)
            
            # Send back transcription
            response = {
                'transcription': transcription,
                'timestamp': audio_data.get('timestamp'),
                'is_final': audio_data.get('is_final', False)
            }
            
            await websocket.send_text(json.dumps(response))
            
    except WebSocketDisconnect:
        print("WebSocket disconnected")

@app.post("/transcribe")
async def transcribe_file(audio_file: UploadFile = File(...)):
    """Transcribe uploaded audio file"""
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            content = await audio_file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Load and process audio
        audio, sr = torchaudio.load(tmp_path)
        
        # Resample if necessary
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            audio = resampler(audio)
        
        # Convert to mono
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        # Transcribe
        inputs = processor(
            audio.squeeze().numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            logits = model(inputs.input_values.to(device)).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.decode(predicted_ids[0])
        
        # Clean up
        os.unlink(tmp_path)
        
        return {
            'transcription': transcription,
            'duration': audio.shape[1] / 16000,
            'confidence': torch.max(torch.softmax(logits, dim=-1)).item()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Option 2: Gradio Interface
```python
import gradio as gr
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt

# Load model
model = Wav2Vec2ForCTC.from_pretrained('./fine-tuned-wav2vec2')
processor = Wav2Vec2Processor.from_pretrained('./fine-tuned-wav2vec2')

def transcribe_audio(audio_file, enable_timestamps=False):
    """Transcribe audio file with optional timestamps"""
    if audio_file is None:
        return "Please upload an audio file.", None
    
    # Load audio
    audio, sr = torchaudio.load(audio_file)
    
    # Resample and convert to mono
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        audio = resampler(audio)
    
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    
    # Transcribe
    inputs = processor(
        audio.squeeze().numpy(),
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )
    
    model.eval()
    with torch.no_grad():
        logits = model(inputs.input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        
        if enable_timestamps:
            # Get word-level timestamps (simplified)
            transcription = processor.decode(predicted_ids[0])
            # In practice, you'd use a more sophisticated alignment method
            timestamps = [(word, i*0.02, (i+len(word))*0.02) 
                         for i, word in enumerate(transcription.split())]
            result = f"Transcription: {transcription}\n\nWord-level timestamps:\n"
            for word, start, end in timestamps:
                result += f"{word}: {start:.2f}s - {end:.2f}s\n"
        else:
            transcription = processor.decode(predicted_ids[0])
            result = transcription
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot waveform
    time_axis = np.linspace(0, len(audio.squeeze()) / 16000, len(audio.squeeze()))
    ax1.plot(time_axis, audio.squeeze().numpy())
    ax1.set_title('Audio Waveform')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    
    # Plot attention/alignment (simplified visualization)
    attention_weights = torch.softmax(logits, dim=-1).squeeze().numpy()
    im = ax2.imshow(attention_weights.T, aspect='auto', origin='lower')
    ax2.set_title('Model Attention Weights')
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Tokens')
    plt.colorbar(im, ax=ax2)
    
    plt.tight_layout()
    
    return result, fig

def record_and_transcribe(audio):
    """Transcribe recorded audio"""
    if audio is None:
        return "Please record some audio.", None
    
    sr, audio_data = audio
    
    # Convert to torch tensor
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32) / 32768.0
    
    audio_tensor = torch.tensor(audio_data).unsqueeze(0)
    
    # Resample if needed
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        audio_tensor = resampler(audio_tensor)
    
    # Transcribe
    inputs = processor(
        audio_tensor.squeeze().numpy(),
        sampling_rate=16000,
        return_tensors="pt"
    )
    
    model.eval()
    with torch.no_grad():
        logits = model(inputs.input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0])
    
    return transcription, None

# Create Gradio interface
with gr.Blocks(title="Automatic Speech Recognition") as demo:
    gr.Markdown("# 🎤 Automatic Speech Recognition System")
    gr.Markdown("Upload an audio file or record your voice to get transcription!")
    
    with gr.Tab("Upload Audio File"):
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(
                    type="filepath",
                    label="Upload Audio File"
                )
                timestamps_checkbox = gr.Checkbox(
                    label="Enable Word Timestamps",
                    value=False
                )
                transcribe_btn = gr.Button("Transcribe", variant="primary")
            
            with gr.Column():
                transcription_output = gr.Textbox(
                    label="Transcription",
                    lines=10
                )
                visualization_output = gr.Plot(label="Audio Analysis")
        
        transcribe_btn.click(
            transcribe_audio,
            inputs=[audio_input, timestamps_checkbox],
            outputs=[transcription_output, visualization_output]
        )
    
    with gr.Tab("Record Audio"):
        with gr.Row():
            with gr.Column():
                audio_recorder = gr.Audio(
                    source="microphone",
                    type="numpy",
                    label="Record Your Speech"
                )
                record_btn = gr.Button("Transcribe Recording", variant="primary")
            
            with gr.Column():
                recording_output = gr.Textbox(
                    label="Transcription",
                    lines=5
                )
        
        record_btn.click(
            record_and_transcribe,
            inputs=audio_recorder,
            outputs=[recording_output, gr.State()]
        )
    
    # Examples
    gr.Examples(
        examples=[
            ["samples/clean_speech.wav", False],
            ["samples/noisy_speech.wav", False],
            ["samples/accented_speech.wav", True],
        ],
        inputs=[audio_input, timestamps_checkbox]
    )

demo.launch(share=True)
```

### Option 3: Mobile SDK Integration
```python
# Flask backend optimized for mobile
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchaudio
import base64
import io

app = Flask(__name__)
CORS(app)

# Optimized model for mobile inference
model = torch.jit.script(model)  # TorchScript for faster inference
model.eval()

@app.route('/transcribe_mobile', methods=['POST'])
def mobile_transcription():
    """Optimized endpoint for mobile applications"""
    try:
        data = request.get_json()
        
        # Decode base64 audio
        audio_b64 = data['audio_data']
        audio_bytes = base64.b64decode(audio_b64)
        
        # Load audio
        audio_buffer = io.BytesIO(audio_bytes)
        audio, sr = torchaudio.load(audio_buffer)
        
        # Quick preprocessing for mobile
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)
        
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        # Limit audio length for mobile (max 30 seconds)
        max_samples = 30 * 16000
        if audio.shape[1] > max_samples:
            audio = audio[:, :max_samples]
        
        # Process and transcribe
        inputs = processor(
            audio.squeeze().numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            start_time = time.time()
            logits = model(inputs.input_values).logits
            inference_time = time.time() - start_time
            
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.decode(predicted_ids[0])
        
        # Calculate confidence score
        probabilities = torch.softmax(logits, dim=-1)
        confidence = torch.mean(torch.max(probabilities, dim=-1)[0]).item()
        
        response = {
            'transcription': transcription,
            'confidence': confidence,
            'inference_time': inference_time,
            'audio_duration': audio.shape[1] / 16000,
            'real_time_factor': inference_time / (audio.shape[1] / 16000),
            'status': 'success'
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model': 'loaded'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

### Cloud Deployment Options
- **AWS Transcribe**: Managed ASR service integration
- **Google Cloud Speech-to-Text**: Multi-language support
- **Azure Speech Services**: Custom model deployment
- **NVIDIA Riva**: GPU-accelerated inference
- **Kubernetes**: Scalable container orchestration

## 9. Extensions & Research Directions

### Advanced Techniques
1. **Multilingual and Code-switching ASR**
   ```python
   # Language identification and switching
   class MultilingualASR:
       def __init__(self, language_models):
           self.models = language_models
           self.language_detector = LanguageDetector()
       
       def transcribe_multilingual(self, audio):
           # Detect language segments
           language_segments = self.language_detector.segment(audio)
           
           transcriptions = []
           for segment, language in language_segments:
               model = self.models[language]
               transcription = model.transcribe(segment)
               transcriptions.append((transcription, language))
           
           return transcriptions
   ```

2. **Speaker Diarization + ASR**
   ```python
   from pyannote.audio import Pipeline
   
   class SpeakerAwareASR:
       def __init__(self, asr_model, diarization_pipeline):
           self.asr_model = asr_model
           self.diarization = diarization_pipeline
       
       def transcribe_with_speakers(self, audio_file):
           # Speaker diarization
           diarization = self.diarization(audio_file)
           
           # Transcribe each speaker segment
           results = []
           for segment, _, speaker in diarization.itertracks(yield_label=True):
               audio_segment = audio_file[segment]
               transcription = self.asr_model.transcribe(audio_segment)
               results.append({
                   'speaker': speaker,
                   'start': segment.start,
                   'end': segment.end,
                   'text': transcription
               })
           
           return results
   ```

3. **Domain Adaptation**
   ```python
   # Medical ASR adaptation
   class DomainAdaptiveASR:
       def __init__(self, base_model, domain_vocab):
           self.base_model = base_model
           self.domain_vocab = domain_vocab
           self.adaptation_layer = nn.Linear(base_model.config.hidden_size, len(domain_vocab))
       
       def adapt_to_domain(self, domain_data):
           # Fine-tune on domain-specific data
           for batch in domain_data:
               # Standard fine-tuning with domain vocabulary
               outputs = self.base_model(**batch)
               adapted_logits = self.adaptation_layer(outputs.hidden_states[-1])
               loss = F.cross_entropy(adapted_logits.view(-1, len(self.domain_vocab)), 
                                    batch['domain_labels'].view(-1))
               loss.backward()
   ```

4. **Contextual Biasing**
   ```python
   # Bias ASR towards expected vocabulary
   class ContextualASR:
       def __init__(self, base_model, contextual_biaser):
           self.base_model = base_model
           self.biaser = contextual_biaser
       
       def transcribe_with_context(self, audio, context_words):
           # Get base predictions
           logits = self.base_model(audio).logits
           
           # Apply contextual biasing
           biased_logits = self.biaser.bias_logits(logits, context_words)
           
           # Decode with biased probabilities
           predicted_ids = torch.argmax(biased_logits, dim=-1)
           return self.processor.decode(predicted_ids[0])
   ```

### Novel Experiments
- **Few-shot ASR**: Adaptation to new speakers/accents with minimal data
- **Noise-robust ASR**: Training on synthetic noise and room impulse responses
- **Emotion-aware ASR**: Joint emotion and speech recognition
- **Visual ASR**: Lip-reading integration for noisy environments
- **Continual learning**: Adapting to new vocabulary without forgetting

### Performance Optimization
```python
# Model quantization for mobile deployment
def quantize_asr_model(model):
    """Quantize model for mobile deployment"""
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8
    )
    return quantized_model

# Knowledge distillation for model compression
class ASRDistillation:
    def __init__(self, teacher_model, student_model):
        self.teacher = teacher_model
        self.student = student_model
    
    def distill_knowledge(self, training_data):
        for batch in training_data:
            # Teacher predictions
            with torch.no_grad():
                teacher_logits = self.teacher(**batch).logits
            
            # Student predictions
            student_logits = self.student(**batch).logits
            
            # Distillation loss
            distill_loss = F.kl_div(
                F.log_softmax(student_logits / self.temperature, dim=-1),
                F.softmax(teacher_logits / self.temperature, dim=-1),
                reduction='batchmean'
            )
            
            distill_loss.backward()

# Streaming optimization
class StreamingOptimizer:
    def __init__(self, model, chunk_size=1600):  # 0.1s chunks
        self.model = model
        self.chunk_size = chunk_size
        self.context_buffer = []
    
    def process_streaming(self, audio_stream):
        """Process streaming audio efficiently"""
        results = []
        
        for chunk in audio_stream:
            # Maintain context window
            self.context_buffer.append(chunk)
            if len(self.context_buffer) > 10:  # 1 second context
                self.context_buffer.pop(0)
            
            # Process chunk with context
            context_audio = torch.cat(self.context_buffer)
            prediction = self.model.predict_chunk(context_audio)
            
            results.append(prediction)
        
        return results
```

### Industry Applications
- **Healthcare**: Medical transcription, clinical notes
- **Legal**: Court reporting, deposition transcription
- **Education**: Lecture transcription, language learning
- **Media**: Broadcast captioning, podcast transcription
- **Accessibility**: Real-time captioning for hearing impaired
- **Automotive**: Voice commands, hands-free interfaces

## 10. Portfolio Polish

### Documentation Structure
```
automatic_speech_recognition/
├── README.md                          # This file
├── notebooks/
│   ├── 01_Dataset_Analysis.ipynb      # LibriSpeech exploration
│   ├── 02_Baseline_Models.ipynb       # HMM-GMM and DNN baselines
│   ├── 03_Wav2Vec2_Training.ipynb     # Transformer fine-tuning
│   ├── 04_Whisper_Experiments.ipynb   # OpenAI Whisper adaptation
│   ├── 05_Evaluation_Analysis.ipynb   # Comprehensive evaluation
│   └── 06_Error_Analysis.ipynb        # Detailed error studies
├── src/
│   ├── models/
│   │   ├── wav2vec2_asr.py
│   │   ├── whisper_asr.py
│   │   ├── conformer.py
│   │   ├── rnn_transducer.py
│   │   └── baseline_models.py
│   ├── data/
│   │   ├── preprocessing.py
│   │   ├── augmentation.py
│   │   ├── feature_extraction.py
│   │   └── data_loaders.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── error_analysis.py
│   │   ├── speaker_evaluation.py
│   │   └── noise_robustness.py
│   ├── training/
│   │   ├── ctc_trainer.py
│   │   ├── attention_trainer.py
│   │   ├── curriculum_learning.py
│   │   └── knowledge_distillation.py
│   ├── inference/
│   │   ├── streaming_asr.py
│   │   ├── batch_inference.py
│   │   └── mobile_optimized.py
│   ├── train.py
│   ├── transcribe.py
│   └── evaluate.py
├── configs/
│   ├── wav2vec2_config.yaml
│   ├── whisper_config.yaml
│   ├── conformer_config.yaml
│   └── training_configs/
├── api/
│   ├── fastapi_server.py
│   ├── websocket_streaming.py
│   ├── mobile_flask_api.py
│   └── requirements.txt
├── demo/
│   ├── gradio_app.py
│   ├── streamlit_dashboard.py
│   ├── real_time_demo.py
│   └── mobile_demo/
├── deployment/
│   ├── docker/
│   │   ├── Dockerfile
│   │   ├── docker-compose.yml
│   │   └── nginx.conf
│   ├── kubernetes/
│   ├── mobile/
│   │   ├── android/
│   │   └── ios/
│   └── cloud_deployment/
├── evaluation/
│   ├── benchmark_scripts/
│   ├── cross_dataset_eval.py
│   ├── noise_robustness_test.py
│   └── real_time_performance.py
├── datasets/
│   ├── librispeech_loader.py
│   ├── common_voice_loader.py
│   ├── custom_dataset.py
│   └── data_validation.py
├── tests/
│   ├── test_models.py
│   ├── test_preprocessing.py
│   ├── test_streaming.py
│   └── test_api.py
├── requirements.txt
├── setup.py
├── Makefile
└── .github/workflows/
```

### Visualization Requirements
- **Audio waveforms**: Time-domain signal analysis
- **Spectrograms**: Log mel-spectrograms with overlaid text
- **Attention patterns**: Transformer attention weight visualization
- **Error analysis**: Confusion matrices for phonemes/words
- **Performance curves**: WER vs training steps
- **Real-time metrics**: Latency and throughput analysis
- **Speaker analysis**: Performance distribution across speakers
- **Noise robustness**: WER vs SNR curves
- **Word clouds**: Most frequent recognition errors

### Blog Post Template
1. **The ASR Revolution**: From dictation software to conversational AI
2. **Dataset Deep-dive**: LibriSpeech characteristics and challenges
3. **Model Evolution**: From HMM-GMM to end-to-end neural networks
4. **Wav2Vec2 Breakthrough**: Self-supervised learning for speech
5. **Production Challenges**: Real-time processing and edge deployment
6. **Error Analysis**: Understanding and fixing recognition failures
7. **Multilingual ASR**: Scaling to global languages and accents
8. **Future Frontiers**: Contextual understanding and speaker adaptation

### Demo Video Script
- 1 minute: ASR importance and applications showcase
- 1.5 minutes: Dataset exploration with audio samples and transcriptions
- 2 minutes: Model architecture progression (HMM → Neural → Transformer)
- 3 minutes: Live transcription demo with various audio conditions
- 1.5 minutes: Performance analysis and error case studies
- 1 minute: Real-time streaming demonstration
- 1.5 minutes: Mobile app and deployment showcase
- 1 minute: Multilingual capabilities and future developments

### GitHub README Essentials
```markdown
# Automatic Speech Recognition with Deep Learning

![ASR Demo](assets/asr_demo.gif)

## 🚀 Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Download LibriSpeech dataset
python src/data/download_librispeech.py --subset clean-100

# Train Wav2Vec2 model
python src/train.py --config configs/wav2vec2_config.yaml

# Transcribe audio file
python src/transcribe.py --model models/best_model.pt --audio sample.wav

# Launch real-time demo
python demo/gradio_app.py
```

## 📊 Results
| Model | Dataset | WER | CER | RTF |
|-------|---------|-----|-----|-----|
| Wav2Vec2-Base | LibriSpeech-clean | 3.2% | 1.1% | 0.15 |
| Wav2Vec2-Large | LibriSpeech-clean | 2.1% | 0.7% | 0.28 |
| Custom Conformer | LibriSpeech-other | 5.8% | 2.3% | 0.22 |

## 🎤 Live Demo
Try the ASR system: [Hugging Face Space](https://huggingface.co/spaces/username/asr-demo)

## 📱 Mobile App
Download from [App Store](link) | [Google Play](link)

## 📚 Citation
```bibtex
@article{asr_deep_learning_2024,
  title={End-to-End Automatic Speech Recognition with Transformers},
  author={Your Name},
  journal={INTERSPEECH 2024},
  year={2024}
}
```
```

### Performance Benchmarks
- **Word Error Rate (WER)**: Performance across different datasets and conditions
- **Real-time Factor (RTF)**: Processing speed relative to audio duration
- **Memory usage**: RAM and VRAM requirements by model size
- **Latency analysis**: End-to-end transcription delay
- **Throughput**: Concurrent requests handling capability
- **Mobile performance**: On-device inference metrics
- **Robustness testing**: Performance under various noise conditions
- **Cross-domain evaluation**: Generalization to different audio types
- **Speaker adaptation**: Performance variation across demographics